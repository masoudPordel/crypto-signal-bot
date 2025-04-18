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
                if "/" in sig["symbol"]:  # برای فارکس (مثال: USD/EUR)
                    message = f"""💱 سیگنال فارکس:
جفت ارز: {sig['symbol']}
قیمت لحظه‌ای: {sig['entry']}
هدف: {sig['tp']}
حد ضرر: {sig['sl']}
اطمینان: {sig['confidence']}٪
تحلیل: {sig['analysis']}
"""
                else:  # برای ارز دیجیتال
                    message = f"""📊 سیگنال کریپتو:
نماد: {sig['symbol']}
ورود: {sig['entry']}
حد سود: {sig['tp']}
حد ضرر: {sig['sl']}
اطمینان: {sig['confidence']}٪
تحلیل تکنیکال: {sig['analysis']}
"""
                await bot.send_message(chat_id=USER_ID, text=message)
        except Exception as e:
            await bot.send_message(chat_id=USER_ID, text=f"خطا در دریافت سیگنال: {str(e)}")

        await asyncio.sleep(300)  # هر ۵ دقیقه

if __name__ == "__main__":
    asyncio.run(send_signals())