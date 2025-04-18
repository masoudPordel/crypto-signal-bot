import os
import asyncio
from telegram import Bot
from strategy_engine import generate_signals

BOT_TOKEN = os.getenv("BOT_TOKEN")
USER_ID = 632886964  # Masoud Pordel

bot = Bot(token=BOT_TOKEN)

async def send_signals():
    while True:
        try:
            signals = generate_signals()
            for sig in signals:
                message = f"""📊 سیگنال جدید:

نماد: {sig['symbol']}
ورود: {sig['entry']}
هدف: {sig['tp']}
حد ضرر: {sig['sl']}
نوسان مورد انتظار: {sig.get('volatility', '-')}٪
اطمینان: {sig['confidence']}٪
تحلیل: {sig['analysis']}
"""
                await bot.send_message(chat_id=USER_ID, text=message)
        except Exception as e:
            await bot.send_message(chat_id=USER_ID, text=f"خطا در ارسال سیگنال: {str(e)}")
        await asyncio.sleep(300)

if __name__ == "__main__":
    asyncio.run(send_signals())