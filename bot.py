import time
import asyncio
import telegram
from analyzer import scan_all_crypto_symbols, scan_all_forex_symbols

# اطلاعات کاربر و توکن
BOT_TOKEN = "8111192844:AAHuVZYs6RolBhdqPpTWW9g7ksGRaq3p0WA"
CHAT_ID = 632886964         # همونی که دادی

bot = telegram.Bot(token=BOT_TOKEN)
sent_signals = set()  # برای جلوگیری از ارسال سیگنال‌های تکراری

def format_signal(signal):
    return (
        f"💠 <b>{signal['symbol']}</b> | تایم‌فریم: {signal['tf']}\n"
        f"🎯 ورود: <code>{signal['entry']}</code>\n"
        f"✅ حد سود: <code>{signal['tp']}</code>\n"
        f"❌ حد ضرر: <code>{signal['sl']}</code>\n"
        f"⚡️ قدرت سیگنال: <b>{signal['confidence']}%</b>\n"
        f"📊 تحلیل: {signal['analysis']}\n"
        f"📉 نوسان: {signal['volatility']}%\n"
    )

async def send_signals():
    print("در حال بررسی بازار...")

    crypto_signals = await scan_all_crypto_symbols()
    forex_signals = await scan_all_forex_symbols()

    all_signals = crypto_signals + forex_signals
    new_signals = []

    for signal in all_signals:
        signal_id = (signal["symbol"], signal["tf"], signal["entry"])
        if signal_id not in sent_signals:
            new_signals.append(signal)
            sent_signals.add(signal_id)

    for signal in new_signals:
        text = format_signal(signal)
        await bot.send_message(chat_id=CHAT_ID, text=text, parse_mode=telegram.constants.ParseMode.HTML)

async def main():
    while True:
        await send_signals()
        time.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
