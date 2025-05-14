import time
import asyncio
import telegram
import logging
import os
import sys
from analyzer import scan_all_crypto_symbols

# تنظیمات logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
    force=True
)

# تنظیمات ربات تلگرام
BOT_TOKEN = "8111192844:AAHuVZYs6RolBhdqPpTWW9g7ksGRaq3p0WA"
CHAT_ID = 632886964
LOCK_FILE = "bot.lock"
bot = telegram.Bot(token=BOT_TOKEN)

# بررسی اینکه آیا ربات در حال اجرا است
def check_already_running():
    if os.path.exists(LOCK_FILE):
        logging.error("ربات در حال اجراست.")
        sys.exit()
    with open(LOCK_FILE, "w") as f:
        f.write(str(os.getpid()))

# حذف فایل قفل در زمان اتمام
def remove_lock():
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)

# تابع ارسال سیگنال‌ها
async def send_signals():
    logging.info("شروع اسکن بازار...")

    # ارسال پیام اولیه برای اطمینان از فعال بودن ربات
    try:
        await bot.send_message(chat_id=CHAT_ID, text="ربات فعال شد.")
    except Exception as e:
        logging.error(f"ارسال پیام تستی ناموفق: {e}")
        return

    # تابع برای مدیریت سیگنال‌ها و ارسال به تلگرام
    async def on_signal(signal):
        entry = float(signal["قیمت ورود"])
        tp = float(signal["هدف سود"])
        sl = float(signal["حد ضرر"])
        trade_type = signal["نوع معامله"]  # مستقیماً نوع معامله رو از سیگنال می‌گیرم

        # پیام با فرمت جدید شامل نوع معامله (برای پشتیبانی از رگرسیون)
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
        try:
            await bot.send_message(chat_id=CHAT_ID, text=msg)
            logging.info(f"سیگنال برای {signal['نماد']} @ {signal['تایم‌فریم']} ارسال شد.")
        except Exception as e:
            logging.error(f"ارسال سیگنال {signal['نماد']} ناموفق: {e}")

    # اسکن بازار و ارسال سیگنال‌ها
    await scan_all_crypto_symbols(on_signal=on_signal)

# تابع اصلی برای اجرای ربات
async def main():
    while True:
        await send_signals()
        await asyncio.sleep(max(0, 300 - (time.time() % 300)))

# اجرای ربات
if __name__ == "__main__":
    check_already_running()
    try:
        asyncio.run(main())
    finally:
        remove_lock()
