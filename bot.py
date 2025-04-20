import time
import asyncio
import telegram
import logging
import os
import sys
import requests
from datetime import datetime
from analyzer import scan_all_crypto_symbols  # فقط ایمپورت کریپتو

# تنظیمات لاگ
logging.basicConfig(level=logging.INFO)

BOT_TOKEN = "8111192844:AAHuVZYs6RolBhdqPpTWW9g7ksGRaq3p0WA"
CHAT_ID = 632886964
LOCK_FILE = "bot.lock"

bot = telegram.Bot(token=BOT_TOKEN)
sent_signals = set()

def clear_webhook():
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/deleteWebhook"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            logging.info("Webhook با موفقیت حذف شد.")
        else:
            logging.warning(f"خطا در حذف Webhook: {response.text}")
    except Exception as e:
        logging.error(f"خطا در اتصال برای حذف Webhook: {e}")

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
        all_signals = await scan_all_crypto_symbols()  # فقط کریپتو

        for signal in all_signals:
            required_keys = ("نماد", "قیمت ورود", "تایم‌فریم", "هدف سود", "حد ضرر", "سطح اطمینان", "تحلیل", "ریسک به ریوارد")
            if all(k in signal for k in required_keys):
                signal_id = (signal["نماد"], signal["تایم‌فریم"], signal["قیمت ورود"])
                if signal_id not in sent_signals:
                    sent_signals.add(signal_id)

                    # تعیین نوع سیگنال
                    signal_type = "Buy" if float(signal["قیمت ورود"]) < float(signal["هدف سود"]) else "Sell"

                    # ساخت دیکشنری نهایی سیگنال
                    final_signal = {
                        "نماد": signal["نماد"],
                        "تایم‌فریم": signal["تایم‌فریم"],
                        "نوع": signal_type,
                        "قیمت ورود": float(signal["قیمت ورود"]),
                        "هدف سود": float(signal["هدف سود"]),
                        "حد ضرر": float(signal["حد ضرر"]),
                        "ریسک به ریوارد": float(signal["ریسک به ریوارد"]),
                        "سطح اطمینان": int(signal["سطح اطمینان"]),
                        "تحلیل": signal["تحلیل"],
                        "فاندامنتال": signal.get("فاندامنتال", "ندارد"),
                        "تاریخ تولید": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "سیستم": "Elliott + EMA + MACD + Volume Filter",
                        "هشدار": "این سیگنال صرفاً برای اهداف آموزشی بوده و نباید به‌تنهایی مبنای خرید یا فروش قرار گیرد."
                    }

                    # پیام تلگرام با فرمت حرفه‌ای
                    message = f"""
{final_signal['نوع']} سیگنال - {final_signal['نماد']} [{final_signal['تایم‌فریم']}]
----------------------------------------
قیمت ورود: {final_signal['قیمت ورود']}
هدف سود: {final_signal['هدف سود']}
حد ضرر: {final_signal['حد ضرر']}
ریسک به ریوارد: {final_signal['ریسک به ریوارد']}
سطح اطمینان: {final_signal['سطح اطمینان']}%

تحلیل تکنیکال:
{final_signal['تحلیل']}

تحلیل فاندامنتال:
{final_signal['فاندامنتال']}

تاریخ: {final_signal['تاریخ تولید']}
سیستم: {final_signal['سیستم']}

{final_signal['هشدار']}
""".strip()

                    await bot.send_message(chat_id=CHAT_ID, text=message)
            else:
                logging.warning("فرمت سیگنال ناقص: %s", signal)
    except Exception as e:
        logging.error("خطا در ارسال سیگنال‌ها: %s", e)

async def main():
    while True:
        await send_signals()
        await asyncio.sleep(300)

if __name__ == "__main__":
    clear_webhook()
    check_already_running()
    try:
        asyncio.run(main())
    finally:
        remove_lock()