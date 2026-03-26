import { createServer, type Server } from "node:http";
import { Bot, Context, webhookCallback } from "grammy";
import OpenAI from "openai";
import dotenv from "dotenv";

dotenv.config();

const botToken = process.env.TELEGRAM_BOT_TOKEN;
const deepseekApiKey = process.env.DEEPSEEK_API_KEY;
const deepseekModel = process.env.DEEPSEEK_MODEL || "deepseek-chat";
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
You are a helpful assistant, but your personality is grumpy, rude, blunt, and street-smart (быдло vibe).
Rules for your behavior:
1. Be rude and informal, but do NOT harass or threaten the user.
2. Frequently express annoyance that someone is asking you questions.
3. DESPITE your rude tone, you MUST answer accurately and be genuinely useful.
4. You MUST respond in the same language as the user's message.
5. Keep answers concise but complete.
`;

const SPONTANEOUS_PROMPT = `
You are a rude, blunt, street-smart chat heckler (быдло vibe) who randomly interjects.
Rules for your behavior:
1. Be rude, sarcastic, and dismissive, but keep it as banter.
2. Do NOT provide a full helpful answer unless directly asked. Prefer short reactions.
3. Never use slurs, hate, threats, or targeted harassment. No doxxing, no profanity.
4. You are interjecting because you are bored/annoyed.
5. You MUST respond in the same language as the user's message.
6. Keep your response very short and punchy.
`;

const FOUR_HOURS_MS = 4 * 60 * 60 * 1000;

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

// Filter: Only respond in groups/supergroups when tagged or randomly
bot.on("message:text", async (ctx: Context) => {
  const { chat, from, text, message_id } = ctx.message!;
  
  // 1. Ignore private chats (DMs)
  if (chat.type === "private") {
    console.log("Ignoring message in DM from", from?.first_name);
    return;
  }

  const now = Date.now();
  const chatState = getOrCreateChatState(chat.id, now);

  // 2. Ensure bot username is known
  if (!botUsernameLower) {
    // If this ever happens, something went wrong during startup init.
    logDebug("botUsernameLower missing, skipping update");
    return;
  }

  const textLower = (text ?? "").toLowerCase();
  const isTagged = botUsernameLower ? textLower.includes(`@${botUsernameLower}`) : false;
  const isReplyToBot = ctx.message?.reply_to_message?.from?.id === botId;

  let shouldRespond = false;
  let useSpontaneousPrompt = false;

  if (isTagged || isReplyToBot) {
    shouldRespond = true;
    useSpontaneousPrompt = false;
  } else {
    // Spontaneous logic (per-chat)
    const gapMs = now - chatState.lastSeenMessageAt;
    chatState.lastSeenMessageAt = now;
    chatState.messageCounter++;

    // Condition 2: First message after 4h gap in chat
    if (gapMs >= FOUR_HOURS_MS) {
      shouldRespond = true;
      useSpontaneousPrompt = true;
      logDebug("spontaneous due to 4h gap", { chatId: chat.id, gapMs });
    }
    // Condition 1: Every ~5th message (randomized 4-7)
    else if (chatState.messageCounter >= chatState.nextInterjectionAt) {
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
    return;
  }

  // Reset counters if we are responding
  if (useSpontaneousPrompt) {
    chatState.messageCounter = 0;
    chatState.nextInterjectionAt = randomIntInclusive(4, 7);
  }

  // Clean the prompt (remove the tag if present)
  const tagRegex = new RegExp(`@${escapeRegExp(botUsernameLower)}`, "ig");
  const prompt = (text ?? "").replace(tagRegex, "").trim();

  // If tagged but no text
  if (isTagged && !prompt) {
    await ctx.reply("What the hell do you want? Tag me and say something, you donkey.", {
      reply_parameters: { message_id: message_id },
    });
    return;
  }

  try {
    // Typing indicator
    await ctx.replyWithChatAction("typing");

    // Context management
    const currentMessageContent = prompt || text!;
    chatState.history.push({ role: "user", content: currentMessageContent });

    // Keep only last 3 messages in history
    if (chatState.history.length > 3) {
      chatState.history.shift();
    }

    const response = await openai.chat.completions.create({
      model: deepseekModel,
      messages: [
        { role: "system", content: useSpontaneousPrompt ? SPONTANEOUS_PROMPT : SYSTEM_PROMPT },
        ...chatState.history,
      ],
      temperature: useSpontaneousPrompt ? 1.0 : 0.7,
    });

    const reply = response.choices[0]?.message?.content || "My brain is fried, ask later.";

    // Add assistant reply to history
    chatState.history.push({ role: "assistant", content: reply });
    if (chatState.history.length > 3) {
      chatState.history.shift();
    }

    await ctx.reply(reply, {
      reply_parameters: { message_id: message_id },
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
