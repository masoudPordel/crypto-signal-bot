import asyncio
from telegram import Bot
from strategy_engine import generate_signals

API_TOKEN = "8111192844:AAHuVZYs6RolBhdqPpTWW9g7ksGRaq3p0WA"
USER_ID = 632886964  # Masoud Pordel

bot = Bot(token=API_TOKEN)

async def send_signals():
    while True:
        try:
            signals = generate_signals()
            for sig in signals:
                message = f"""📊 سیگنال جدید:
سیمبل: {sig['symbol']}
ورود: {sig['entry']}
حد سود: {sig['tp']}
حد ضرر: {sig['sl']}
اطمینان: {sig['confidence']}٪
تحلیل: {sig['analysis']}
"""
                await bot.send_message(chat_id=USER_ID, text=message)
        except Exception as e:
            await bot.send_message(chat_id=USER_ID, text=f"خطا در دریافت سیگنال: {str(e)}")
        await asyncio.sleep(300)  # هر ۵ دقیقه

if __name__ == "__main__":
    asyncio.run(send_signals())