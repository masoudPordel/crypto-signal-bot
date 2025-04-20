import time
import asyncio
import telegram
import logging
import os
import sys
from analyzer import scan_all_crypto_symbols  # مطمئن شو analyzer.py در همون پوشه هست

# --- تنظیمات لاگ ---
logging.basicConfig(level=logging.INFO)

# --- اطلاعات ربات تلگرام ---
BOT_TOKEN = "8111192844:AAHuVZYs6RolBhdqPpTWW9g7ksGRaq3p0WA"
CHAT_ID = 632886964
# --- قفل اجرا برای جلوگیری از دوبار اجرا شدن ---
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

# --- ارسال سیگنال‌ها به تلگرام ---
async def send_signals():
    logging.info("در حال بررسی بازار...")
    try:
        all_signals = await scan_all_crypto_symbols()

        for signal in all_signals:
            logging.debug("سیگنال دریافتی: %s", signal)

            # بررسی وجود مقادیر ضروری
            required_keys = ["نماد", "قیمت ورود", "هدف سود", "حد ضرر"]
            if all(k in signal for k in required_keys):
                try:
                    entry_price = float(signal["قیمت ورود"])
                    tp = float(signal["هدف سود"])
                    sl = float(signal["حد ضرر"])
                    confidence = int(signal.get("سطح اطمینان", 0))
                    rr = float(signal.get("ریسک به ریوارد", 0))
                    tf = signal.get("تایم‌فریم", "نامشخص")
                    ta = signal.get("تحلیل", "ندارد")
                    fa = signal.get("فاندامنتال", "ندارد")

                    signal_type = "خرید" if tp > entry_price else "فروش"

                    message = f"""📢 سیگنال {signal_type.upper()}

نماد: {signal.get('نماد')}
تایم‌فریم: {tf}
قیمت ورود: {entry_price}
هدف سود: {tp}
حد ضرر: {sl}
سطح اطمینان: {confidence}%
ریسک به ریوارد: {rr}

تحلیل تکنیکال:
{ta}

تحلیل فاندامنتال:
{fa}"""

                    await bot.send_message(chat_id=CHAT_ID, text=message)
                    logging.info(">>> سیگنال با موفقیت ارسال شد به تلگرام.")

                except Exception as e:
                    logging.error(">>> خطا در پردازش یا ارسال پیام: %s", e)
            else:
                logging.warning(">>> سیگنال ناقص: %s", signal)

    except Exception as e:
        logging.error(">>> خطا در دریافت یا ارسال سیگنال‌ها: %s", e)

# --- حلقه‌ی اصلی ---
async def main():
    while True:
        await send_signals()
        await asyncio.sleep(300)  # هر ۵ دقیقه یک‌بار بررسی کن

# --- اجرای برنامه ---
if __name__ == "__main__":
    check_already_running()
    try:
        asyncio.run(main())
    finally:
        remove_lock()