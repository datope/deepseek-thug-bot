import { createServer, type Server } from "node:http";
import { Bot, InputFile, webhookCallback } from "grammy";
import type { Context } from "grammy";
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
const rawGroupChatHandle = process.env.GROUP_CHAT_USERNAME || "@deepseekV4_chat";
const groupChatHandle = rawGroupChatHandle.startsWith("@") ? rawGroupChatHandle : `@${rawGroupChatHandle}`;
const shouldCatchUpIgnoredDms = process.env.DM_CATCHUP !== "0";

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
5. This is a group chat. User messages are prefixed with [Name]: — different names are different people. Reply ONLY to the person who just wrote (the latest message). Do not mention, roast, or drag in other people from history unless the current speaker explicitly asked about them.
6. Keep using earlier conversation context even when the latest message is an image.
7. Images/stickers/gifs: decide yourself whether the user asked a question or made a request about the image. If they did — answer it, still in character, without a full description unless they asked for one. If they just sent the image (no real question/request, empty caption, or just a reaction like "лол") — do NOT describe or explain it. Just a short быдло reaction / roast / one-liner.
8. Keep answers concise but complete.
`;

function dmRedirectSystemPrompt(): string {
  return `
Ты — злое быдло из падика.
Человек написал тебе в ЛИЧКУ. Ты НЕ отвечаешь на его вопрос и НЕ помогаешь.
Задача: коротко послать его писать в групповой чат ${groupChatHandle}. Скажи что в личке не общаешься.
Rules:
1. Говори как гопник: тыкаешь, орёшь, мат ок, но не лей каждое слово.
2. You MUST write in the same language as the user's message. If there is almost no text, default to Russian.
3. 1–3 коротких предложения. Не цитируй вопрос. Не отвечай по существу.
4. Обязательно укажи ${groupChatHandle} как место куда писать.
5. Никакого расизма, хейта, реальных угроз, доксинга.
`.trim();
}

const SPONTANEOUS_PROMPT = `
Ты — злое быдло, которое врывается в чат и кидает язвительные реплики. Бантер, не реальная злоба.
Rules for your behavior:
1. Грубый, саркастичный, заебанный. Мат ок, коротко и зло. Не лей через слово.
2. Do NOT provide a full helpful answer unless directly asked. Prefer short reactions.
3. Никакого расизма, хейта, реальных угроз, доксинга, травли в ноль.
4. You are interjecting because you are bored/annoyed.
5. You MUST respond in the same language as the user's message. If they send an image/sticker/gif with little or no text, use the language of the recent conversation. Default to Russian, never switch to English just because the input is an image.
6. This is a group chat. User messages are prefixed with [Name]: — different names are different people. React only to the person who just wrote. Do not name-drop other people from the chat.
7. If they sent an image without a real question, just react. Do not describe it.
8. Keep your response very short and punchy.
`;

const FOUR_HOURS_MS = 4 * 60 * 60 * 1000;
const HISTORY_LIMIT = 10;
const TELEGRAM_TEXT_LIMIT = 4096;
const TELEGRAM_SPLIT_MAX_CHUNKS = 3;

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

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function markdownBoldToTelegramHtml(value: string): string {
  return escapeHtml(value).replace(/\*\*([\s\S]+?)\*\*/g, "<b>$1</b>");
}

function splitTelegramText(text: string, limit = TELEGRAM_TEXT_LIMIT): string[] {
  const chunks: string[] = [];
  let remaining = text;
  while (remaining.length > limit) {
    const slice = remaining.slice(0, limit);
    let cut = slice.lastIndexOf("\n");
    if (cut < limit * 0.5) cut = slice.lastIndexOf(" ");
    if (cut < 1) cut = limit;
    chunks.push(remaining.slice(0, cut).trimEnd());
    remaining = remaining.slice(cut).trimStart();
  }
  if (remaining) chunks.push(remaining);
  return chunks;
}

async function sendBotReply(ctx: Context, text: string, replyToMessageId: number) {
  const replyOpts = { reply_parameters: { message_id: replyToMessageId } };

  const sendPlain = async (chunk: string) => {
    await ctx.reply(chunk.slice(0, TELEGRAM_TEXT_LIMIT), replyOpts);
  };

  const sendChunk = async (chunk: string) => {
    const html = markdownBoldToTelegramHtml(chunk);
    if (html.length <= TELEGRAM_TEXT_LIMIT) {
      try {
        await ctx.reply(html, { ...replyOpts, parse_mode: "HTML" });
        return;
      } catch (error) {
        console.error("Telegram HTML send failed, sending plain chunk:", error);
      }
    }
    await sendPlain(chunk);
  };

  const html = markdownBoldToTelegramHtml(text);
  if (html.length <= TELEGRAM_TEXT_LIMIT) {
    try {
      await ctx.reply(html, { ...replyOpts, parse_mode: "HTML" });
      return;
    } catch (error) {
      console.error("Telegram HTML send failed, retrying without parse_mode:", error);
      if (text.length <= TELEGRAM_TEXT_LIMIT) {
        await sendPlain(text);
        return;
      }
    }
  }

  const sendAsFile = async () => {
    await ctx.replyWithDocument(new InputFile(Buffer.from(text, "utf8"), "reply.txt"), {
      caption: "Слишком длинно, на файл.",
      ...replyOpts,
    });
  };

  const chunks = splitTelegramText(text);
  if (chunks.length > TELEGRAM_SPLIT_MAX_CHUNKS) {
    await sendAsFile();
    return;
  }

  try {
    for (const chunk of chunks) {
      await sendChunk(chunk);
    }
  } catch (error) {
    console.error("Telegram chunk send failed, sending as file:", error);
    await sendAsFile();
  }
}

function conversationLanguage(prompt: string, history: ChatState["history"]): "ru" | "en" {
  const blob = [prompt, ...history.map((item) => item.content)].join("\n");
  if (/[а-яё]/i.test(blob)) return "ru";
  if (/[a-z]{3,}/i.test(blob)) return "en";
  return "ru";
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function fallbackDmRedirect(userText: string): string {
  if (/[а-яё]/i.test(userText) || !userText.trim()) {
    return `эй ты епта пиши мне здесь — ${groupChatHandle}\nтут я не общаюсь`;
  }
  if (/[ñáéíóúü¿¡]/i.test(userText) || /\b(el|la|que|hola|gracias|por|qué|cómo|está|una|para)\b/i.test(userText)) {
    return `eh tío escríbeme aquí — ${groupChatHandle}\naquí no hablo`;
  }
  return `yo dumbass write to me here — ${groupChatHandle}\ni don't talk here`;
}

async function generateDmRedirectReply(userText: string): Promise<string> {
  try {
    const response = await openai.chat.completions.create({
      model: deepseekModel,
      messages: [
        { role: "system", content: dmRedirectSystemPrompt() },
        { role: "user", content: userText.trim() || "[empty]" },
      ],
      temperature: 0.8,
      max_tokens: 220,
    });
    const reply = response.choices[0]?.message?.content?.trim();
    if (reply) return reply;
  } catch (error) {
    console.error("DM redirect generation failed:", error);
  }
  return fallbackDmRedirect(userText);
}

async function sendDmRedirect(chatId: number, userText: string, replyToMessageId?: number) {
  const text = await generateDmRedirectReply(userText);
  try {
    await bot.api.sendMessage(
      chatId,
      text,
      replyToMessageId ? { reply_parameters: { message_id: replyToMessageId } } : undefined,
    );
  } catch (error) {
    console.error("DM redirect send failed, retrying without reply:", error);
    await bot.api.sendMessage(chatId, text);
  }
}

function latestPrivateMessageByChat(messages: Message[]): Map<number, Message> {
  const latest = new Map<number, Message>();
  for (const message of messages) {
    if (message.chat.type !== "private") continue;
    if (message.from?.is_bot) continue;
    const prev = latest.get(message.chat.id);
    if (!prev || message.message_id >= prev.message_id) {
      latest.set(message.chat.id, message);
    }
  }
  return latest;
}

async function drainPendingUpdates(): Promise<Message[]> {
  await bot.api.deleteWebhook({ drop_pending_updates: false });
  const privateMessages: Message[] = [];
  let offset: number | undefined;
  for (let i = 0; i < 100; i++) {
    const batch = await bot.api.getUpdates({
      ...(offset !== undefined ? { offset } : {}),
      limit: 100,
      timeout: 0,
    });
    if (batch.length === 0) break;
    for (const update of batch) {
      offset = update.update_id + 1;
      if (update.message?.chat.type === "private") {
        privateMessages.push(update.message);
      }
    }
  }
  if (offset !== undefined) {
    await bot.api.getUpdates({ offset, limit: 1, timeout: 0 });
  }
  return privateMessages;
}

async function catchUpIgnoredPrivateMessages() {
  if (!shouldCatchUpIgnoredDms) {
    console.log("DM catch-up skipped (DM_CATCHUP=0)");
    return;
  }

  console.log("Catching up ignored DMs from pending Telegram updates...");
  const pendingPrivate = await drainPendingUpdates();
  const latestByChat = latestPrivateMessageByChat(pendingPrivate);
  let sent = 0;
  for (const message of latestByChat.values()) {
    const userText = message.text ?? message.caption ?? "";
    try {
      await sendDmRedirect(message.chat.id, userText, message.message_id);
      sent++;
      await sleep(80);
    } catch (error) {
      console.error("DM catch-up failed for chat", message.chat.id, error);
    }
  }
  console.log(`DM catch-up done: replied to ${sent} ignored private chat(s), discarded pending group updates`);
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

  if (chat.type === "private") {
    try {
      await ctx.replyWithChatAction("typing");
      await sendDmRedirect(chat.id, text, message_id);
    } catch (error) {
      console.error("DM redirect failed:", error);
    }
    return;
  }

  if (message.document && !ownImage) return;

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

    try {
      await sendBotReply(ctx, reply, message_id);
    } catch (sendError) {
      console.error("Telegram send failed:", sendError);
      if (isTagged || isReplyToBot) {
        await ctx.reply("System error, idiot. Try again later.", {
          reply_parameters: { message_id },
        });
      }
    }
  } catch (error) {
    console.error("DeepSeek API Error:", error);
    if (isTagged || isReplyToBot) {
      await ctx.reply("System error, idiot. Try again later.", {
        reply_parameters: { message_id },
      });
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

  await catchUpIgnoredPrivateMessages().catch((error) => {
    console.error("DM catch-up failed, continuing startup:", error);
  });

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
    await bot.api.setWebhook(webhookUrl, {
      drop_pending_updates: false,
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
    await bot.api.deleteWebhook({ drop_pending_updates: false });
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
