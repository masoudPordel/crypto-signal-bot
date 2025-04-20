import time
import asyncio
import telegram
import logging
import os
import sys
from analyzer import scan_all_crypto_symbols  # فقط کریپتو

# تنظیمات لاگ
logging.basicConfig(level=logging.INFO)

BOT_TOKEN = "8111192844:AAHuVZYs6RolBhdqPpTWW9g7ksGRaq3p0WA"
CHAT_ID = 632886964
LOCK_FILE = "bot.lock"

bot = telegram.Bot(token=BOT_TOKEN)

def check_already_running():
    if os.path.exists(LOCK_FILE):
        logging.error("ربات در حال اجراست. ابتدا آن را متوقف کن.")
        sys.exit()
    with open(LOCK_FILE, "w") as f:
        f.write(str(os.getpid()))

def remove_lock():
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)

async def send_signals():
    logging.info("در حال بررسی بازار...")

    try:
        crypto_signals = await scan_all_crypto_symbols()
        all_signals = crypto_signals

        for signal in all_signals:
            if all(k in signal for k in ("نماد", "قیمت ورود", "تایم‌فریم", "هدف سود", "حد ضرر", "سطح اطمینان", "تحلیل", "ریسک به ریوارد")):
                entry_price = float(signal["قیمت ورود"])
                tp = float(signal["هدف سود"])
                sl = float(signal["حد ضرر"])
                confidence = float(signal["سطح اطمینان"])
                rr = float(signal["ریسک به ریوارد"])
                signal_type = "خرید" if tp > entry_price else "فروش"
                fundamental = signal.get("فاندامنتال", "ندارد")

                message = f"""📢 سیگنال {signal_type.upper()}

نماد: {signal['نماد']}
تایم‌فریم: {signal['تایم‌فریم']}
قیمت ورود: {entry_price}
هدف سود: {tp}
حد ضرر: {sl}
سطح اطمینان: {confidence}%
ریسک به ریوارد: {rr}

تحلیل تکنیکال:
{signal['تحلیل']}

تحلیل فاندامنتال:
{fundamental}"""

                logging.info("در حال ارسال سیگنال به تلگرام:\n%s", message)

                try:
                    await bot.send_message(chat_id=CHAT_ID, text=message)
                    logging.info("سیگنال با موفقیت ارسال شد.")
                except Exception as e:
                    logging.error("خطا در ارسال پیام تلگرام: %s", e)
            else:
                logging.warning("فرمت سیگنال ناقص: %s", signal)
    except Exception as e:
        logging.error("خطا در دریافت یا پردازش سیگنال‌ها: %s", e)

async def main():
    while True:
        await send_signals()
        await asyncio.sleep(300)  # هر ۵ دقیقه بررسی می‌کنه

if __name__ == "__main__":
    check_already_running()
    try:
        asyncio.run(main())
    finally:
        remove_lock()