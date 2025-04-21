import time
import asyncio
import telegram
import logging
import os
import sys
from analyzer import scan_all_crypto_symbols

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
    force=True
)

BOT_TOKEN = "8111192844:AAHuVZYs6RolBhdqPpTWW9g7ksGRaq3p0WA"
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
    logging.info("شروع اسکن بازار...")

    try:
        await bot.send_message(chat_id=CHAT_ID, text="ربات فعال شد.")
    except Exception as e:
        logging.error(f"ارسال پیام تستی ناموفق: {e}")
        return

    async def on_signal(signal):
        entry = float(signal["قیمت ورود"])
        tp = float(signal["هدف سود"])
        sl = float(signal["حد ضرر"])
        typ = "خرید" if tp > entry else "فروش"

        msg = f"""📢 سیگنال {typ.upper()}

نماد: {signal['نماد']}
تایم‌فریم: {signal['تایم‌فریم']}
قیمت ورود: {entry}
هدف سود: {tp}
حد ضرر: {sl}
سطح اطمینان: {signal['سطح اطمینان']}%
ریسک به ریوارد: {signal['ریسک به ریوارد']}

تحلیل تکنیکال:
{signal['تحلیل']}

تحلیل فاندامنتال:
{signal['فاندامنتال']}
"""
        try:
            await bot.send_message(chat_id=CHAT_ID, text=msg)
        except Exception as e:
            logging.error(f"ارسال سیگنال {signal['نماد']} ناموفق: {e}")

    await scan_all_crypto_symbols(on_signal=on_signal)

async def main():
    while True:
        await send_signals()
        await asyncio.sleep(max(0, 300 - (time.time() % 300)))

if __name__ == "__main__":
    check_already_running()
    try:
        asyncio.run(main())
    finally:
        remove_lock()