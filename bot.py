import time
import asyncio
import telegram
from analyzer import scan_all_crypto_symbols, scan_all_forex_symbols

# اطلاعات ربات
BOT_TOKEN = "8111192844:AAHuVZYs6Ro1BhdqPpTWW9g7ksGRaq3p0WA"
CHAT_ID = 632886964  # آیدی کاربر تلگرام

# ایجاد ربات تلگرام
bot = telegram.Bot(token=BOT_TOKEN)
sent_signals = set()  # جلوگیری از ارسال سیگنال تکراری

async def send_signals():
    print("در حال بررسی بازار...")
    
    # دریافت سیگنال‌ها
    crypto_signals = await scan_all_crypto_symbols()
    forex_signals = await scan_all_forex_symbols()
    
    all_signals = crypto_signals + forex_signals

    for signal in all_signals:
        if signal and all(k in signal for k in ("نماد", "تایم‌فریم", "قیمت ورود")):
            signal_id = (signal["نماد"], signal["تایم‌فریم"], signal["قیمت ورود"])
            
            if signal_id not in sent_signals:
                sent_signals.add(signal_id)
                
                message = (
                    f"📊 **سیگنال معاملاتی جدید**\n"
                    f"--------------------------\n"
                    f"نماد: {signal['نماد']}\n"
                    f"تایم‌فریم: {signal['تایم‌فریم']}\n"
                    f"قیمت ورود: {signal['قیمت ورود']}\n"
                    f"هدف سود (TP): {signal['هدف سود']}\n"
                    f"حد ضرر (SL): {signal['حد ضرر']}\n"
                    f"سطح اطمینان: {signal['سطح اطمینان']}%\n"
                    f"--------------------------\n"
                    f"تحلیل:\n{signal['تحلیل']}"
                )
                
                try:
                    await bot.send_message(chat_id=CHAT_ID, text=message, parse_mode=telegram.constants.ParseMode.MARKDOWN)
                    print(f"سیگنال ارسال شد: {signal['نماد']}")
                except Exception as e:
                    print(f"خطا در ارسال پیام تلگرام: {e}")
        else:
            print("سیگنال نامعتبر:", signal)

async def main():
    while True:
        await send_signals()
        await asyncio.sleep(300)  # هر ۵ دقیقه

if __name__ == "__main__":
    asyncio.run(main())