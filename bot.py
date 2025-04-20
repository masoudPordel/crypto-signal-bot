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

        # هندل کردن None
        crypto_signals = crypto_signals or []
        forex_signals = forex_signals or []

        all_signals = crypto_signals + forex_signals

        for signal in all_signals:
            if all(k in signal for k in (
                "نماد", "قیمت ورود", "تایم‌فریم", "هدف سود", "حد ضرر",
                "سطح اطمینان", "تحلیل", "ریسک به ریوارد", "فاندامنتال"
            )):
                signal_id = (signal["نماد"], signal["تایم‌فریم"], signal["قیمت ورود"])
                if signal_id not in sent_signals:
                    sent_signals.add(signal_id)

                    message = f"""نماد: {signal['نماد']}
تایم‌فریم: {signal['تایم‌فریم']}
قیمت ورود: {signal['قیمت ورود']}
هدف سود: {signal['هدف سود']}
حد ضرر: {signal['حد ضرر']}
سطح اطمینان: {signal['سطح اطمینان']}%
ریسک به ریوارد: {signal['ریسک به ریوارد']}

تحلیل تکنیکال:
{signal['تحلیل']}

تحلیل فاندامنتال:
{signal['فاندامنتال']}"""

                    await bot.send_message(chat_id=CHAT_ID, text=message)

            else:
                logging.warning("فرمت سیگنال ناقص: %s", signal)

    except Exception as e:
        logging.error("خطا در ارسال سیگنال‌ها: %s", e)

async def main():
    while True:
        await send_signals()
        await asyncio.sleep(300)  # بررسی هر ۵ دقیقه

if __name__ == "__main__":
    asyncio.run(main())