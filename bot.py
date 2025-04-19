import time
import asyncio
import telegram
from analyzer import scan_all_crypto_symbols, scan_all_forex_symbols

# اطلاعات کاربر و توکن
BOT_TOKEN = "8111192844:AAHuVZYs6Ro1BhdqPpTWW9g7ksGRaq3p0WA"
CHAT_ID = 632886964  # آیدی چت تلگرام

bot = telegram.Bot(token=BOT_TOKEN)
sent_signals = set()  # برای جلوگیری از ارسال دوباره‌ی سیگنال‌ها

async def send_signals():
    print("در حال بررسی بازار...")

    crypto_signals = await scan_all_crypto_symbols()
    forex_signals = await scan_all_forex_symbols()

    print("تعداد سیگنال‌های کریپتو:", len(crypto_signals))
    print("تعداد سیگنال‌های فارکس:", len(forex_signals))

    all_signals = crypto_signals + forex_signals

    for signal in all_signals:
        if all(k in signal for k in ("نماد", "قیمت ورود", "تایم‌فریم")):
            signal_id = (signal["نماد"], signal["تایم‌فریم"], signal["قیمت ورود"])
            if signal_id not in sent_signals:
                sent_signals.add(signal_id)

                message = (
                    f"نماد: {signal['نماد']}\n"
                    f"تایم‌فریم: {signal['تایم‌فریم']}\n"
                    f"قیمت ورود: {signal['قیمت ورود']}\n"
                    f"حد ضرر: {signal['حد ضرر']}\n"
                    f"هدف سود: {signal['هدف سود']}\n"
                    f"سطح اطمینان: {signal['سطح اطمینان']}%\n"
                    f"\nتحلیل:\n{signal['تحلیل']}"
                )

                await bot.send_message(chat_id=CHAT_ID, text=message)
        else:
            print("فرمت سیگنال نامعتبر:", signal)

async def main():
    while True:
        await send_signals()
        await asyncio.sleep(300)  # هر ۵ دقیقه یک‌بار

if __name__ == "__main__":
    asyncio.run(main())