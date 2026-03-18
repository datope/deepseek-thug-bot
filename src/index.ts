import { Bot, Context } from "grammy";
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
  };
  chatStateByChatId.set(key, created);
  return created;
}

const chatStateByChatId = new Map<string, ChatState>();

// Store bot info to avoid constant API calls
let botUsernameLower: string | null = null;
let botId: number | null = null;

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

    const response = await openai.chat.completions.create({
      model: deepseekModel,
      messages: [
        { role: "system", content: useSpontaneousPrompt ? SPONTANEOUS_PROMPT : SYSTEM_PROMPT },
        { role: "user", content: prompt || text! },
      ],
      temperature: useSpontaneousPrompt ? 1.0 : 0.7,
    });

    const reply = response.choices[0]?.message?.content || "My brain is fried, ask later.";

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

  bot.start();
}

start().catch((err) => {
  console.error("Failed to start bot. Check TELEGRAM_BOT_TOKEN and Railway variables.");
  console.error(err);
  process.exit(1);
});
