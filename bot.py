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
        logging.error("ربات در حال اجراست. ابتدا آن را متوقف کن.")
        sys.exit()
    with open(LOCK_FILE, "w") as f:
        f.write(str(os.getpid()))

def remove_lock():
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)

async def send_signals():
    logging.info("شروع بررسی بازار...")

    # پیام تستی
    try:
        await bot.send_message(chat_id=CHAT_ID, text="ربات آماده به کار است.")
    except Exception as e:
        logging.error(f"خطا در ارسال پیام تستی: {e}")
        return

    start_time = time.time()

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
سطح اطمینان: {signal.get('سطح اطمینان', 0)}%
ریسک به ریوارد: {signal.get('ریسک به ریوارد', 0)}

تحلیل تکنیکال:
{signal.get('تحلیل', '')}

تحلیل فاندامنتال:
{signal.get('فاندامنتال', 'ندارد')}
"""
        logging.info(f"ارسال فوری سیگنال {signal['نماد']}")
        try:
            await bot.send_message(chat_id=CHAT_ID, text=msg)
            await asyncio.sleep(1.2)
        except Exception as e:
            logging.error(f"خطا در ارسال سیگنال {signal['نماد']}: {e}")

    await scan_all_crypto_symbols(on_signal=on_signal)

    elapsed = time.time() - start_time
    logging.info(f"اسکن کامل شد در {elapsed:.2f} ثانیه")

async def main():
    while True:
        await send_signals()
        # محاسبه‌ی باقیمانده تا ۵ دقیقه
        await asyncio.sleep(max(0, 300 - (time.time() % 300)))

if __name__ == "__main__":
    check_already_running()
    try:
        asyncio.run(main())
    finally:
        remove_lock()