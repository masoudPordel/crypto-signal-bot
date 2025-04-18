import time
from telegram.bot import Bot
from strategy_engine import generate_signals

TELEGRAM_TOKEN = "8111192844:AAHuVZYs6RolBhdqPpTWW9g7ksGRaq3p0WA"
CHAT_ID = 632886964

bot = Bot(token=TELEGRAM_TOKEN)

def send_signals():
    signals = generate_signals()
    if not signals:
        bot.send_message(chat_id=CHAT_ID, text="فعلاً سیگنالی یافت نشد.")
        return

    for sig in signals:
        market_type = "کریپتو" if "USDT" in sig['symbol'] else "فارکس"
        message = f"""📊 سیگنال جدید ({market_type}):

نماد: {sig['symbol']}
تایم‌فریم: {sig['timeframe']}
قیمت ورود: {sig.get('entry', '-')}
حد سود: {sig.get('tp', '-')}
حد ضرر: {sig.get('sl', '-')}
نوسان: {sig.get('volatility', '-')}٪
اعتماد: {sig.get('confidence', '-')}٪
تحلیل: {sig.get('analysis', '-')}
"""
        bot.send_message(chat_id=CHAT_ID, text=message)

if __name__ == "__main__":
    while True:
        try:
            send_signals()
            time.sleep(60 * 15)
        except Exception as e:
            bot.send_message(chat_id=CHAT_ID, text=f"خطا: {e}")
            time.sleep(60 * 5)
