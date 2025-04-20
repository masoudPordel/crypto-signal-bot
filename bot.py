# bot.py

import time
import asyncio
import telegram
import logging
import os
import sys
from analyzer import scan_all_crypto_symbols

logging.basicConfig(level=logging.INFO)

BOT_TOKEN ="8111192844:AAHuVZYs6RolBhdqPpTWW9g7ksGRaq3p0WA"
CHAT_ID = 632886964

LOCK_FILE = "bot.lock"
bot = telegram.Bot(token=BOT_TOKEN)

def check_already_running():
    if os.path.exists(LOCK_FILE):
        logging.error("ربات در حال اجراست.")
        sys.exit()
    with open(LOCK_FILE, "w") as f:
        f.write(str(os.getpid()))

def remove_lock():
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)

async def send_signals():
    try:
        all_signals = await scan_all_crypto_symbols()
        for signal in all_signals:
            logging.info(f"بررسی سیگنال برای ارسال: {signal}")
            required_keys = ["نماد", "قیمت ورود", "هدف سود", "حد ضرر"]

            if all(k in signal for k in required_keys):
                entry_price = float(signal["قیمت ورود"])
                tp = float(signal["هدف سود"])
                sl = float(signal["حد ضرر"])
                signal_type = "خرید" if tp > entry_price else "فروش"

                message = f"""📢 سیگنال {signal_type.upper()}

نماد: {signal.get('نماد')}
تایم‌فریم: {signal.get('تایم‌فریم')}
قیمت ورود: {entry_price}
هدف سود: {tp}
حد ضرر: {sl}
سطح اطمینان: {signal.get('سطح اطمینان')}%
ریسک به ریوارد: {signal.get('ریسک به ریوارد')}

تحلیل تکنیکال:
{signal.get('تحلیل')}
"""
                try:
                    await bot.send_message(chat_id=CHAT_ID, text=message)
                    logging.info("پیام ارسال شد.")
                except Exception as e:
                    logging.error(f"خطا در ارسال پیام: {e}")
            else:
                logging.warning(f"سیگنال ناقص: {signal}")
    except Exception as e:
        logging.error(f"خطا در پردازش: {e}")

async def main():
    while True:
        await send_signals()
        await asyncio.sleep(300)

if __name__ == "__main__":
    check_already_running()
    try:
        asyncio.run(main())
    finally:
        remove_lock()