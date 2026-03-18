import { Bot, Context } from "grammy";
import OpenAI from "openai";
import dotenv from "dotenv";

dotenv.config();

const botToken = process.env.TELEGRAM_BOT_TOKEN;
const deepseekApiKey = process.env.DEEPSEEK_API_KEY;
const deepseekModel = process.env.DEEPSEEK_MODEL || "deepseek-chat";

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
You are a helpful assistant, but your personality is that of a rude, aggressive, and street-smart "thug" (быдло style).
Rules for your behavior:
1. Be extremely rude and use informal language.
2. Frequently express annoyance that someone is asking you questions.
3. DESPITE your rude personality, you MUST answer the user's question accurately and be genuinely useful.
4. You MUST respond in the same language as the user's message.
5. Keep your answers concise but complete.
`;

// Filter: Only respond in groups/supergroups when tagged
bot.on("message:text", async (ctx: Context) => {
  const { chat, from, text, message_id } = ctx.message!;
  
  // 1. Ignore private chats (DMs)
  if (chat.type === "private") {
    console.log("Ignoring message in DM from", from?.first_name);
    return;
  }

  // 2. Check if the bot is tagged
  const botInfo = await ctx.api.getMe();
  const botUsername = botInfo.username;
  const isTagged = text?.includes(`@${botUsername}`);

  // 3. Only respond if tagged
  if (!isTagged) {
    return;
  }

  // Clean the prompt (remove the tag)
  const prompt = text!.replace(`@${botUsername}`, "").trim();

  if (!prompt) {
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
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: prompt },
      ],
      temperature: 0.9,
    });

    const reply = response.choices[0]?.message?.content || "My brain is fried, ask later.";

    await ctx.reply(reply, {
      reply_parameters: { message_id: message_id },
    });
  } catch (error) {
    console.error("DeepSeek API Error:", error);
    await ctx.reply("System error, idiot. Try again later.");
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

console.log("Bot is running...");
bot.start();
