import ccxt
import pandas as pd
import numpy as np
import logging
from aiogram import Bot, Dispatcher, types

API_TOKEN = '8111192844:AAHuVZYs6RolBhdqPpTWW9g7ksGRaq3p0WA'
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

user_id = 632886964  # MasoudPordel

def get_signal(symbol='BTC/USDT', timeframe='5m'):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=150)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()

    last_close = df['close'].iloc[-1]
    ema_20 = df['ema_20'].iloc[-1]
    ema_50 = df['ema_50'].iloc[-1]

    signal = "Neutral"
    if ema_20 > ema_50 and last_close > ema_20:
        signal = "Buy"
    elif ema_20 < ema_50 and last_close < ema_20:
        signal = "Sell"

    tp = round(last_close * 1.02, 2)
    sl = round(last_close * 0.98, 2)

    return f"Signal: {signal}\nSymbol: {symbol}\nPrice: {last_close}\nTP: {tp}, SL: {sl}"

async def run_bot():
    await bot.send_message(chat_id=user_id, text="Signal bot is running...")

    from asyncio import sleep
    while True:
        try:
            message = get_signal('BTC/USDT')
            await bot.send_message(chat_id=user_id, text=message)
        except Exception as e:
            logging.error(f"Error: {e}")
        await sleep(300)
