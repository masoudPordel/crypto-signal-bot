# -*- coding: utf-8 -*-

import asyncio
from telegram import Bot
from strategy_engine import generate_signals

API_TOKEN = "8111192844:AAHuVZYs6RolBhdqPpTWW9g7ksGRaq3p0WA"
USER_ID = 632886964  # آیدی عددی تلگرام شما

bot = Bot(token=API_TOKEN)

async def main():
    while True:
        signals = generate_signals()
        for sig in signals:
            message = f"""
📊 سیگنال جدید:
جفت ارز: {sig['pair']}
قیمت فعلی: {sig['price']}
هدف اول: {sig['target1']}
هدف دوم: {sig['target2']}
حد ضرر: {sig['stop_loss']}
درصد اطمینان: {sig['confidence']}٪
            """
            await bot.send_message(chat_id=USER_ID, text=message)
        await asyncio.sleep(300)  # ارسال هر ۵ دقیقه

if __name__ == "__main__":
    asyncio.run(main())
