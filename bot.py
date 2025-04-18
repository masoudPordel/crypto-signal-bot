import os
from telegram import Bot

BOT_TOKEN = os.getenv("BOT_TOKEN")
USER_ID = 632886964  # Masoud Pordel

bot = Bot(token=BOT_TOKEN)
bot.send_message(chat_id=USER_ID, text="✅ ربات وصله و کار می‌کنه!")
