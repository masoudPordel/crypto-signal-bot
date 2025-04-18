# -*- coding: utf-8 -*-

import asyncio
from telegram import Bot
from strategy_engine import get_crypto_price, get_forex_rate

API_TOKEN = "8111192844:AAHuVZYs6RolBhdqPpTWW9g7ksGRaq3p0WA"
USER_ID = 632886964  # Masoud Pordel

bot = Bot(token=API_TOKEN)

async def send_signals():
    while True:
        try:
            message = "📊 سیگنال خودکار:

"

            # ارز دیجیتال
            btc = get_crypto_price("BTCUSDT")
            eth = get_crypto_price("ETHUSDT")
            message += f"BTC/USDT: {btc}
ETH/USDT: {eth}

"

            # فارکس
            eurusd = get_forex_rate("USD", "EUR")
            gbpusd = get_forex_rate("USD", "GBP")
            usdjpy = get_forex_rate("USD", "JPY")
            message += f"USD/EUR: {eurusd}
USD/GBP: {gbpusd}
USD/JPY: {usdjpy}
"

            await bot.send_message(chat_id=USER_ID, text=message)
        except Exception as e:
            await bot.send_message(chat_id=USER_ID, text=f"خطا در دریافت سیگنال: {str(e)}")

        await asyncio.sleep(300)  # هر 5 دقیقه

if __name__ == "__main__":
    asyncio.run(send_signals())
