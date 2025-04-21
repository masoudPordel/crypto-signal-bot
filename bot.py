import time
import asyncio
import telegram
import logging
import os
import sys
from analyzer import scan_all_crypto_symbols

# --- لاگ ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
    force=True
)

BOT_TOKEN = "8111192844:AAHuVZYs6RolBhdqPpTWW9g7ksGRaq3p0WA"
CHAT_ID    = 632886964
LOCK_FILE  = "bot.lock"

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
    logging.info("🔄 در حال بررسی سیگنال‌های بازار...")
    try:
        signals = await scan_all_crypto_symbols()
        for sig in signals:
            # کلیدهای اصلی مورد نیاز
            required = ["نماد", "قیمت ورود", "هدف سود", "حد ضرر"]
            if all(k in sig for k in required):
                entry_price = float(sig["قیمت ورود"])
                tp          = float(sig["هدف سود"])
                sl          = float(sig["حد ضرر"])
                stype       = "خرید" if tp > entry_price else "فروش"

                message = f"""📢 سیگنال {stype.upper()}

نماد: {sig.get('نماد')}
تایم‌فریم: {sig.get('تایم‌فریم', 'نامشخص')}
قیمت ورود: {entry_price}
هدف سود: {tp}
حد ضرر: {sl}
سطح اطمینان: {sig.get('سطح اطمینان', 'نامشخص')}%
ریسک به ریوارد: {sig.get('ریسک به ریوارد', 'نامشخص')}

تحلیل تکنیکال:
{sig.get('تحلیل', 'ندارد')}"""

                try:
                    await bot.send_message(chat_id=CHAT_ID, text=message)
                    logging.info("✅ سیگنال با موفقیت ارسال شد.")
                except Exception as e:
                    logging.error("❌ خطا در ارسال پیام تلگرام: %s", e)
            else:
                logging.warning("⚠️ سیگنال ناقص: %s", sig)
    except Exception as e:
        logging.error("❌ خطا در ارسال سیگنال‌ها: %s", e)

async def main():
    while True:
        await send_signals()
        await asyncio.sleep(300)  # هر ۵ دقیقه یک‌بار

if __name__ == "__main__":
    check_already_running()
    try:
        asyncio.run(main())
    finally:
        remove_lock()