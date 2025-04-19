import time
import asyncio
import telegram
from analyzer import scan_all_crypto_symbols, scan_all_forex_symbols

# اطلاعات کاربر و توکن
BOT_TOKEN = "8111192844:AAHuVZYs6Ro1BhdqPpTWW9g7ksGRaq3p0WA"
CHAT_ID = 632886964  # همونی که دادی

bot = telegram.Bot(token=BOT_TOKEN)
sent_signals = set()  # جلوگیری از ارسال سیگنال تکراری

async def send_signals():
    print("در حال بررسی بازار...")
    crypto_signals = await scan_all_crypto_symbols()
    forex_signals = await scan_all_forex_symbols()
    all_signals = crypto_signals + forex_signals

    for signal in all_signals:
        if all(k in signal for k in ("symbol", "tf", "entry")):
            signal_id = (signal["symbol"], signal["tf"], signal["entry"])
            if signal_id not in sent_signals:
                sent_signals.add(signal_id)
                message = (
                    f"Symbol: {signal['symbol']}\n"
                    f"TF: {signal['tf']}\n"
                    f"Entry: {signal['entry']}\n"
                    f"SL: {signal['sl']}\n"
                    f"TP: {signal['tp']}\n"
                    f"Type: {signal['type']}"
                )
                await bot.send_message(chat_id=CHAT_ID, text=message)
        else:
            print("Invalid signal format:", signal)

async def main():
    while True:
        await send_signals()
        await asyncio.sleep(300)  # هر 5 دقیقه

if __name__ == "__main__":
    asyncio.run(main())
