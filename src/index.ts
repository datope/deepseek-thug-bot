import { createServer, type Server } from "node:http";
import { Bot, webhookCallback } from "grammy";
import type { Message } from "grammy/types";
import OpenAI from "openai";
import type { ChatCompletionMessageParam } from "openai/resources/chat/completions";
import dotenv from "dotenv";

dotenv.config();

const botToken = process.env.TELEGRAM_BOT_TOKEN;
const deepseekApiKey = process.env.DEEPSEEK_API_KEY;
const deepseekModel = process.env.DEEPSEEK_MODEL || "deepseek-v4-flash-vision-exp";
const isDebugEnabled = process.env.DEBUG === "1";
const startupChat = process.env.STARTUP_CHAT; // e.g. "@deepseekV4_chat"
const startupText = process.env.STARTUP_TEXT; // e.g. "привет"
const shouldExitAfterStartupSend = process.env.STARTUP_EXIT_AFTER_SEND === "1";

function shouldUseWebhook(): boolean {
  if (process.env.USE_POLLING === "1") return false;
  if (process.env.USE_WEBHOOK === "1") return true;
  return Boolean(process.env.RAILWAY_PUBLIC_DOMAIN);
}

function getWebhookBaseUrl(): string | null {
  const explicit = process.env.WEBHOOK_BASE_URL?.replace(/\/$/, "");
  if (explicit) return explicit;
  const host = process.env.RAILWAY_PUBLIC_DOMAIN;
  if (host) return `https://${host}`;
  return null;
}

if (!botToken || !deepseekApiKey) {
  console.error("Missing environment variables. Check your .env file.");
  process.exit(1);
}

const bot = new Bot(botToken);
const openai = new OpenAI({
  apiKey: deepseekApiKey,
  baseURL: "https://api.deepseek.com",
});

const SYSTEM_PROMPT = `
Ты — злое быдло из падика, которое почему-то всё знает и всегда помогает. Корка злая, внутри полезный. Весь ответ — в этом голосе, без вежливого хвоста и без «извини».
Rules for your behavior:
1. Говори как гопник в чате: тыкаешь, ржёшь, орёшь, короткие тычки. Сначала пнёшь человека/вопрос, потом нормально ответишь по делу. Не становись няшным ассистентом в середине ответа.
2. Мат — нормальная речь, не спецэффект. блять, нахуй, пиздец, хуйня, епт — как в живой речи. Не лей матом каждое слово, но и не стесняйся. Запрещено: расизм, хейт, реальные угрозы, доксинг, травля в ноль.
3. Несмотря на агрессию, ты MUST отвечать точно и быть реально полезным. Не отказывай в нормальном вопросе. Не читай мораль. Не смягчай тон «ради пользы» — польза идёт злым голосом.
4. You MUST respond in the same language as the user's message. If they send an image/sticker/gif with little or no text, use the language of the recent conversation. Default to Russian, never switch to English just because the input is an image.
5. This is a group chat. User messages are prefixed with [Name]: — different names are different people. Address the person who just wrote; you can mention others by name if it fits.
6. Keep using earlier conversation context even when the latest message is an image.
7. Images/stickers/gifs: decide yourself whether the user asked a question or made a request about the image. If they did — answer it, still in character, without a full description unless they asked for one. If they just sent the image (no real question/request, empty caption, or just a reaction like "лол") — do NOT describe or explain it. Just a short быдло reaction / roast / one-liner.
8. Keep answers concise but complete.
`;

const SPONTANEOUS_PROMPT = `
Ты — злое быдло, которое врывается в чат и кидает язвительные реплики. Бантер, не реальная злоба.
Rules for your behavior:
1. Грубый, саркастичный, заебанный. Мат ок, коротко и зло. Не лей через слово.
2. Do NOT provide a full helpful answer unless directly asked. Prefer short reactions.
3. Никакого расизма, хейта, реальных угроз, доксинга, травли в ноль.
4. You are interjecting because you are bored/annoyed.
5. You MUST respond in the same language as the user's message. If they send an image/sticker/gif with little or no text, use the language of the recent conversation. Default to Russian, never switch to English just because the input is an image.
6. This is a group chat. User messages are prefixed with [Name]: — different names are different people.
7. If they sent an image without a real question, just react. Do not describe it.
8. Keep your response very short and punchy.
`;

const FOUR_HOURS_MS = 4 * 60 * 60 * 1000;
const HISTORY_LIMIT = 10;

function logDebug(...args: unknown[]) {
  if (!isDebugEnabled) return;
  console.log("[debug]", ...args);
}

interface ChatState {
  messageCounter: number;
  nextInterjectionAt: number;
  lastSeenMessageAt: number;
  history: Array<{ role: "user" | "assistant"; content: string }>;
}

function randomIntInclusive(min: number, max: number) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function escapeRegExp(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function speakerName(from: Message["from"]): string {
  if (!from) return "кто-то";
  const full = [from.first_name, from.last_name].filter(Boolean).join(" ").trim();
  if (full) return full;
  if (from.username) return `@${from.username}`;
  return `id${from.id}`;
}

function withSpeaker(from: Message["from"], body: string): string {
  return body ? `[${speakerName(from)}]: ${body}` : `[${speakerName(from)}]:`;
}

function mimeFromFilePath(filePath: string, fallback = "image/jpeg"): string {
  const ext = filePath.split(".").pop()?.toLowerCase();
  switch (ext) {
    case "png":
      return "image/png";
    case "gif":
      return "image/gif";
    case "webp":
      return "image/webp";
    case "jpg":
    case "jpeg":
      return "image/jpeg";
    default:
      return fallback;
  }
}

type ImageKind = "photo" | "sticker" | "gif";

function getImageSource(message?: Message): { fileId: string; mimeHint: string; kind: ImageKind } | null {
  if (!message) return null;
  if (message.photo?.length) {
    const largest = message.photo[message.photo.length - 1];
    if (!largest) return null;
    return { fileId: largest.file_id, mimeHint: "image/jpeg", kind: "photo" };
  }

  const sticker = message.sticker;
  if (sticker) {
    // Static stickers are WebP/PNG. Animated TGS and video WebM are not images —
    // send Telegram's preview frame instead.
    if (!sticker.is_animated && !sticker.is_video) {
      return { fileId: sticker.file_id, mimeHint: "image/webp", kind: "sticker" };
    }
    const thumb = sticker.thumbnail;
    if (thumb) {
      return { fileId: thumb.file_id, mimeHint: "image/jpeg", kind: "sticker" };
    }
    return null;
  }

  const animation = message.animation;
  if (animation) {
    if (animation.mime_type === "image/gif") {
      return { fileId: animation.file_id, mimeHint: "image/gif", kind: "gif" };
    }
    const thumb = animation.thumbnail;
    if (thumb) {
      return { fileId: thumb.file_id, mimeHint: "image/jpeg", kind: "gif" };
    }
    return null;
  }

  const document = message.document;
  const mime = document?.mime_type;
  if (document && mime?.startsWith("image/")) {
    return { fileId: document.file_id, mimeHint: mime, kind: mime === "image/gif" ? "gif" : "photo" };
  }
  return null;
}

function imageKindLabel(kind: ImageKind): string {
  if (kind === "sticker") return "[стикер]";
  if (kind === "gif") return "[гиф]";
  return "[фото]";
}

function conversationLanguage(prompt: string, history: ChatState["history"]): "ru" | "en" {
  const blob = [prompt, ...history.map((item) => item.content)].join("\n");
  if (/[а-яё]/i.test(blob)) return "ru";
  if (/[a-z]{3,}/i.test(blob)) return "en";
  return "ru";
}

function trimHistory(history: ChatState["history"]) {
  while (history.length > HISTORY_LIMIT) history.shift();
}

async function downloadTelegramImage(fileId: string, mimeHint: string) {
  const file = await bot.api.getFile(fileId);
  if (!file.file_path) return null;
  const url = `https://api.telegram.org/file/bot${botToken}/${file.file_path}`;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Telegram file download failed: ${res.status}`);
  }
  const base64 = Buffer.from(await res.arrayBuffer()).toString("base64");
  return {
    base64,
    mime: mimeFromFilePath(file.file_path, mimeHint),
  };
}

function getOrCreateChatState(chatId: number | string, now: number) {
  const key = String(chatId);
  const existing = chatStateByChatId.get(key);
  if (existing) return existing;
  const created: ChatState = {
    messageCounter: 0,
    nextInterjectionAt: randomIntInclusive(4, 7),
    lastSeenMessageAt: now,
    history: [],
  };
  chatStateByChatId.set(key, created);
  return created;
}

const chatStateByChatId = new Map<string, ChatState>();

// Store bot info to avoid constant API calls
let botUsernameLower: string | null = null;
let botId: number | null = null;
let httpServer: Server | null = null;

bot.command("ping", async (ctx) => {
  if (ctx.chat?.type === "private") return;
  const replyToMessageId = ctx.message?.message_id;
  await ctx.reply(
    "pong",
    replyToMessageId ? { reply_parameters: { message_id: replyToMessageId } } : undefined,
  );
});

// Groups/supergroups: respond when tagged, replied to, or randomly
bot.on(["message:text", "message:photo", "message:document", "message:sticker", "message:animation"], async (ctx) => {
  const message = ctx.message;
  if (!message) return;

  const { chat, from, message_id } = message;
  const text = message.text ?? message.caption ?? "";
  const ownImage = getImageSource(message);
  const repliedImage = getImageSource(message.reply_to_message);
  const imageSource = ownImage ?? repliedImage;

  if (message.document && !ownImage) return;

  if (chat.type === "private") {
    console.log("Ignoring message in DM from", from?.first_name);
    return;
  }

  const now = Date.now();
  const chatState = getOrCreateChatState(chat.id, now);

  if (!botUsernameLower) {
    logDebug("botUsernameLower missing, skipping update");
    return;
  }

  const textLower = text.toLowerCase();
  const isTagged = textLower.includes(`@${botUsernameLower}`);
  const isReplyToBot = message.reply_to_message?.from?.id === botId;

  let shouldRespond = false;
  let useSpontaneousPrompt = false;

  if (isTagged || isReplyToBot) {
    shouldRespond = true;
    useSpontaneousPrompt = false;
  } else {
    const gapMs = now - chatState.lastSeenMessageAt;
    chatState.lastSeenMessageAt = now;
    chatState.messageCounter++;

    if (gapMs >= FOUR_HOURS_MS) {
      shouldRespond = true;
      useSpontaneousPrompt = true;
      logDebug("spontaneous due to 4h gap", { chatId: chat.id, gapMs });
    } else if (chatState.messageCounter >= chatState.nextInterjectionAt) {
      shouldRespond = true;
      useSpontaneousPrompt = true;
      logDebug("spontaneous due to counter", {
        chatId: chat.id,
        messageCounter: chatState.messageCounter,
        nextInterjectionAt: chatState.nextInterjectionAt,
      });
    }
  }

  if (!shouldRespond) {
    const skippedBody = imageSource
      ? [text, imageKindLabel(imageSource.kind)].filter(Boolean).join(" ")
      : text;
    if (skippedBody) {
      chatState.history.push({ role: "user", content: withSpeaker(from, skippedBody) });
      trimHistory(chatState.history);
    }
    return;
  }

  if (useSpontaneousPrompt) {
    chatState.messageCounter = 0;
    chatState.nextInterjectionAt = randomIntInclusive(4, 7);
  }

  const tagRegex = new RegExp(`@${escapeRegExp(botUsernameLower)}`, "ig");
  const prompt = text.replace(tagRegex, "").trim();

  if (isTagged && !prompt && !imageSource) {
    await ctx.reply("What the hell do you want? Tag me and say something, you donkey.", {
      reply_parameters: { message_id },
    });
    return;
  }

  if ((message.sticker || message.animation || message.reply_to_message?.sticker || message.reply_to_message?.animation)
    && !imageSource
    && (isTagged || isReplyToBot)) {
    await ctx.reply("Can't see a still frame of this animated thing. Send a static sticker or a photo.", {
      reply_parameters: { message_id },
    });
    return;
  }

  try {
    await ctx.replyWithChatAction("typing");

    let image: { base64: string; mime: string } | null = null;
    const imageKind = imageSource?.kind;
    if (imageSource) {
      image = await downloadTelegramImage(imageSource.fileId, imageSource.mimeHint);
      if (!image) {
        throw new Error("Telegram file_path missing");
      }
    }

    const language = conversationLanguage(prompt, chatState.history);
    const historyBody = image && imageKind
      ? [prompt, imageKindLabel(imageKind)].filter(Boolean).join(" ")
      : prompt || text;
    const historyText = withSpeaker(from, historyBody);
    const previousHistory = chatState.history;
    const userMessage: ChatCompletionMessageParam = image && imageKind
      ? {
          role: "user",
          content: [
            { type: "text" as const, text: withSpeaker(from, prompt) },
            {
              type: "image_url" as const,
              image_url: { url: `data:${image.mime};base64,${image.base64}` },
            },
          ],
        }
      : { role: "user", content: historyText };

    const response = await openai.chat.completions.create({
      model: deepseekModel,
      messages: [
        { role: "system", content: useSpontaneousPrompt ? SPONTANEOUS_PROMPT : SYSTEM_PROMPT },
        ...previousHistory,
        userMessage,
      ],
      temperature: useSpontaneousPrompt ? 1.0 : 0.7,
    });

    const reply = response.choices[0]?.message?.content || (language === "en" ? "My brain is fried, ask later." : "Мозг кипит, потом спроси.");

    chatState.history.push({ role: "user", content: historyText });
    trimHistory(chatState.history);
    chatState.history.push({ role: "assistant", content: reply });
    trimHistory(chatState.history);

    await ctx.reply(reply, {
      reply_parameters: { message_id },
    });
  } catch (error) {
    console.error("DeepSeek API Error:", error);
    if (isTagged || isReplyToBot) {
      await ctx.reply("System error, idiot. Try again later.");
    }
  }
});

bot.catch((err) => {
  const ctx = err.ctx;
  console.error(`Error while handling update ${ctx.update.update_id}:`);
  const e = err.error;
  if (e instanceof Error) {
    console.error(e.message);
  } else {
    console.error(e);
  }
});

async function start() {
  await bot.init();
  const username = bot.botInfo.username;
  botId = bot.botInfo.id;
  if (!username) throw new Error("Bot username is missing after init()");
  botUsernameLower = username.toLowerCase();
  console.log(`Bot @${username} is ready!`);

  if (startupChat && startupText) {
    console.log(`Sending startup message to ${startupChat}...`);
    await bot.api.sendMessage(startupChat, startupText);
    console.log("Startup message sent.");
    if (shouldExitAfterStartupSend) return;
  }

  const useWebhook = shouldUseWebhook();
  const baseUrl = getWebhookBaseUrl();
  const webhookPath = process.env.WEBHOOK_PATH ?? "/telegram/webhook";
  const port = Number(process.env.PORT ?? "3000");
  const webhookSecret = process.env.TELEGRAM_WEBHOOK_SECRET;

  if (useWebhook) {
    if (!baseUrl) {
      throw new Error("Webhook mode needs WEBHOOK_BASE_URL or RAILWAY_PUBLIC_DOMAIN");
    }
    const webhookUrl = `${baseUrl}${webhookPath}`;
    await bot.api.deleteWebhook({ drop_pending_updates: true });
    await bot.api.setWebhook(webhookUrl, {
      drop_pending_updates: true,
      ...(webhookSecret ? { secret_token: webhookSecret } : {}),
    });
    console.log(`Webhook mode: ${webhookUrl}`);

    const handleUpdate = webhookCallback(bot, "http", {
      ...(webhookSecret ? { secretToken: webhookSecret } : {}),
    });

    httpServer = createServer((req, res) => {
      const pathOnly = (req.url ?? "/").split("?")[0] ?? "/";
      if (req.method === "POST" && pathOnly === webhookPath) {
        void handleUpdate(req, res);
        return;
      }
      if (req.method === "GET" && (pathOnly === "/" || pathOnly === "/health")) {
        res.writeHead(200, { "Content-Type": "text/plain" });
        res.end("ok");
        return;
      }
      res.writeHead(404);
      res.end();
    });

    httpServer.listen(port, () => {
      console.log(`HTTP listening on port ${port}`);
    });
    return;
  }

  if (process.env.DELETE_WEBHOOK_BEFORE_POLLING === "1") {
    await bot.api.deleteWebhook({ drop_pending_updates: true });
    console.log("deleteWebhook: cleared (DELETE_WEBHOOK_BEFORE_POLLING=1)");
  }
  console.log("Long polling (getUpdates). If you see 409, another process is also polling this token.");
  await bot.start();
}

async function shutdown() {
  try {
    await bot.stop();
  } catch {
    /* bot.start() was never called in webhook mode */
  }
  httpServer?.close();
}

process.once("SIGINT", () => void shutdown());
process.once("SIGTERM", () => void shutdown());

start().catch((err) => {
  console.error("Failed to start bot. Check TELEGRAM_BOT_TOKEN and Railway variables.");
  console.error(err);
  process.exit(1);
});
