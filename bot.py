import time
import asyncio
import telegram
import logging
from analyzer import scan_all_crypto_symbols, scan_all_forex_symbols

# تنظیمات لاگ
logging.basicConfig(level=logging.INFO)

BOT_TOKEN = "8111192844:AAHuVZYs6RolBhdqPpTWW9g7ksGRaq3p0WA"
CHAT_ID = 632886964

bot = telegram.Bot(token=BOT_TOKEN)
sent_signals = set()

async def send_signals():
    logging.info("در حال بررسی بازار...")
    try:
        crypto_signals = await scan_all_crypto_symbols()
        forex_signals = await scan_all_forex_symbols()

        all_signals = crypto_signals + forex_signals

        for signal in all_signals:
            # چک می‌کنیم فیلدهای اصلی تکنیکال موجود باشن
            required_keys = (
                "نماد", "قیمت ورود", "تایم‌فریم",
                "هدف سود", "حد ضرر", "سطح اطمینان",
                "ریسک به ریوارد", "تحلیل"
            )
            if all(k in signal for k in required_keys):
                signal_id = (signal["نماد"], signal["تایم‌فریم"], signal["قیمت ورود"])
                if signal_id not in sent_signals:
                    sent_signals.add(signal_id)

                    # پیام پایه
                    message = (
                        f"نماد: {signal['نماد']}\n"
                        f"تایم‌فریم: {signal['تایم‌فریم']}\n"
                        f"قیمت ورود: {signal['قیمت ورود']}\n"
                        f"هدف سود: {signal['هدف سود']}\n"
                        f"حد ضرر: {signal['حد ضرر']}\n"
                        f"سطح اطمینان: {signal['سطح اطمینان']}%\n"
                        f"ریسک به ریوارد: {signal['ریسک به ریوارد']}\n"
                        f"تحلیل:\n{signal['تحلیل']}"
                    )

                    # اضافه کردن تحلیل فاندامنتال اگر موجود باشه
                    if "رتبه فاندامنتال" in signal:
                        message += f"\nرتبه فاندامنتال: {signal['رتبه فاندامنتال']}"
                    if "امتیاز توسعه‌دهنده" in signal:
                        message += f"\nامتیاز توسعه‌دهنده: {signal['امتیاز توسعه‌دهنده']}"
                    if "امتیاز جامعه" in signal:
                        message += f"\nامتیاز جامعه: {signal['امتیاز جامعه']}"

                    await bot.send_message(chat_id=CHAT_ID, text=message)
            else:
                logging.warning("فرمت سیگنال ناقص: %s", signal)
    except Exception as e:
        logging.error("خطا در ارسال سیگنال‌ها: %s", e)

async def main():
    while True:
        await send_signals()
        await asyncio.sleep(300)  # ۵ دقیقه بین هر بررسی

if __name__ == "__main__":
    asyncio.run(main())