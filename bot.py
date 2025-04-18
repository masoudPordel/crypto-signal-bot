import requests

from analyzer import scan_all_crypto_symbols, scan_all_forex_symbols

import time



# === تنظیمات تلگرام ===

TELEGRAM_TOKEN = "8111192844:AAHuVZYs6RolBhdqPpTWW9g7ksGRaq3p0WA"

CHAT_ID = "632886964"



def send_to_telegram(message):

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

    payload = {

        "chat_id": CHAT_ID,

        "text": message,

        "parse_mode": "HTML"

    }

    try:

        response = requests.post(url, data=payload)

        if response.status_code != 200:

            print("ارسال به تلگرام ناموفق بود:", response.text)

    except Exception as e:

        print("خطا در ارسال پیام تلگرام:", e)



def format_signal_message(sig):

    market = "کریپتو" if "USDT" in sig["symbol"] else "فارکس"

    return (

        f"✅ <b>سیگنال جدید ({market})</b>\n\n"

        f"<b>نماد:</b> {sig['symbol']}\n"

        f"<b>بازه:</b> {sig['tf']}\n"

        f"<b>ورود:</b> {sig['entry']}\n"

        f"<b>TP:</b> {sig['tp']} | <b>SL:</b> {sig['sl']}\n"

        f"<b>اعتماد:</b> {sig['confidence']}%\n"

        f"<b>نوسان:</b> {sig['volatility']}%\n"

        f"<b>تحلیل:</b> {sig['analysis']}"

    )



def send_signals():

    crypto_signals = scan_all_crypto_symbols()

    forex_signals = scan_all_forex_symbols()



    all_signals = crypto_signals + forex_signals

    print(f"\n>> تعداد سیگنال‌ها: {len(all_signals)}\n")



    for sig in all_signals:

        if not sig:

            continue

        try:

            message = format_signal_message(sig)

            print(message)

            send_to_telegram(message)

        except Exception as e:

            print(f"خطا در ارسال سیگنال {sig['symbol']}: {e}")



# اجرای زمان‌بندی‌شده ساده

if __name__ == "__main__":

    while True:

        print("\n--- شروع اسکن ---\n")

        send_signals()

        time.sleep(600)

