import time

import telegram

from analyzer import scan_all_crypto_symbols, scan_all_forex_symbols



# اطلاعات کاربر و توکن

BOT_TOKEN = "8111192844:AAHuVZYs6RolBhdqPpTWW9g7ksGRaq3p0WA"


CHAT_ID = 632886964         # همونی که دادی



bot = telegram.Bot(token=BOT_TOKEN)

sent_signals = set()  # برای جلوگیری از ارسال سیگنال‌های تکراری



def format_signal(signal):

    return (

        f"💠 <b>{signal['symbol']}</b> | تایم‌فریم: {signal['tf']}\n"

        f"🎯 ورود: <code>{signal['entry']}</code>\n"

        f"✅ حد سود: <code>{signal['tp']}</code>\n"

        f"❌ حد ضرر: <code>{signal['sl']}</code>\n"

        f"⚡️ قدرت سیگنال: <b>{signal['confidence']}%</b>\n"

        f"📊 تحلیل: {signal['analysis']}\n"

        f"📉 نوسان: {signal['volatility']}%\n"

    )



def send_signals():

    print("در حال بررسی بازار...")



    crypto_signals = scan_all_crypto_symbols()

    forex_signals = scan_all_forex_symbols()



    all_signals = crypto_signals + forex_signals

    new_signals = []



    for signal in all_signals:

        unique_id = f"{signal['symbol']}_{signal['tf']}_{signal['entry']}"

        if unique_id not in sent_signals:

            sent_signals.add(unique_id)

            new_signals.append(signal)



    for sig in new_signals:

        msg = format_signal(sig)

        try:

            bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode=telegram.ParseMode.HTML)

            print(f"سیگنال ارسال شد: {sig['symbol']} - {sig['tf']}")

        except Exception as e:

            print(f"خطا در ارسال سیگنال: {e}")



    if not new_signals:

        print("سیگنال جدیدی نبود.")



# اجرای دائمی با فاصله ۲ دقیقه

if __name__ == "__main__":

    while True:

        send_signals()

        time.sleep(120)  # هر ۲ دقیقه

