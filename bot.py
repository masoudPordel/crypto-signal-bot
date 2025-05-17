import time
import asyncio
import telegram
import logging
import os
import sys
from datetime import datetime, timedelta
from analyzer import scan_all_crypto_symbols

# تنظیمات logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bot.log", encoding="utf-8"),
        logging.StreamHandler()
    ],
    force=True
)

# تنظیمات ربات تلگرام
BOT_TOKEN = "8111192844:AAHuVZYs6RolBhdqPpTWW9g7ksGRaq3p0WA"
CHAT_ID = 632886964
LOCK_FILE = "bot.lock"
bot = telegram.Bot(token=BOT_TOKEN)

# بررسی اینکه آیا ربات در حال اجرا است با زمان‌بندی
def check_already_running():
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE, "r") as f:
                content = f.read().strip()
                if not content:
                    logging.warning("فایل قفل خالی است، قفل نادیده گرفته می‌شود.")
                    return
                pid, timestamp = content.split(":")
                timestamp = datetime.fromisoformat(timestamp)
                if datetime.now() - timestamp > timedelta(hours=1):
                    logging.warning("فایل قفل قدیمی است، قفل نادیده گرفته می‌شود.")
                    remove_lock()
                    return
            logging.error("ربات در حال اجراست.")
            sys.exit(1)
        except Exception as e:
            logging.error(f"خطا در بررسی فایل قفل: {e}")
            sys.exit(1)
    with open(LOCK_FILE, "w") as f:
        f.write(f"{os.getpid()}:{datetime.now().isoformat()}")

# حذف فایل قفل
def remove_lock():
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)
        logging.info("فایل قفل حذف شد.")

# تابع ارسال سیگنال‌ها
async def send_signals():
    logging.info("شروع اسکن بازار...")

    # ارسال پیام اولیه برای اطمینان از فعال بودن ربات
    try:
        await bot.send_message(chat_id=CHAT_ID, text="ربات فعال شد.")
    except Exception as e:
        logging.error(f"ارسال پیام تستی ناموفق: {e}")
        # ادامه دادن به جای return

    # تابع برای مدیریت سیگنال‌ها و ارسال به تلگرام
    async def on_signal(signal):
        try:
            entry = float(signal["قیمت ورود"])
            tp = float(signal["هدف سود"])
            sl = float(signal["حد ضرر"])
            trade_type = signal["نوع معامله"]

            msg = f"""📢 سیگنال {trade_type.upper()}

نماد: {signal['نماد']}
تایم‌فریم: {signal['تایم‌فریم']}
قیمت ورود: {entry}
هدف سود: {tp}
حد ضرر: {sl}
سطح اطمینان: {signal['سطح اطمینان']}%
امتیاز: {signal['امتیاز']}
قدرت سیگنال: {signal['قدرت سیگنال']}
ریسک به ریوارد: {signal['ریسک به ریوارد']}

تحلیل تکنیکال:
{signal['تحلیل']}

روانشناسی بازار:
{signal['روانشناسی']}

روند بازار:
{signal['روند بازار']}

تحلیل فاندامنتال:
{signal['فاندامنتال']}
"""
            await bot.send_message(chat_id=CHAT_ID, text=msg)
            logging.info(f"سیگنال برای {signal['نماد']} @ {signal['تایم‌فریم']} ارسال شد.")
        except Exception as e:
            logging.error(f"ارسال سیگنال {signal.get('نماد', 'Unknown')} ناموفق: {e}")

    # اسکن بازار و ارسال سیگنال‌ها
    try:
        await scan_all_crypto_symbols(on_signal=on_signal)
    except Exception as e:
        logging.error(f"خطا در اسکن بازار: {e}")

# تابع اصلی برای اجرای ربات
async def main():
    while True:
        try:
            await send_signals()
            logging.info("اسکن بازار تکمیل شد، در انتظار اسکن بعدی (5 دقیقه)...")
            await asyncio.sleep(300)  # انتظار 5 دقیقه
        except Exception as e:
            logging.error(f"خطا در حلقه اصلی: {e}")
            await asyncio.sleep(60)  # در صورت خطا، 1 دقیقه صبر کنید و دوباره تلاش کنید

# اجرای ربات
if __name__ == "__main__":
    logging.info("شروع ربات...")
    check_already_running()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("ربات توسط کاربر متوقف شد.")
    except Exception as e:
        logging.error(f"خطای غیرمنتظره در اجرای ربات: {e}")
    finally:
        remove_lock()
        logging.info("ربات متوقف شد.")