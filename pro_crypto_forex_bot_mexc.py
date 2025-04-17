
import ccxt
import pandas as pd
import pandas_ta as ta
import time
from telegram import Bot
import logging

TELEGRAM_TOKEN = '8111192844:AAHuVZYs6RolBhdqPpTWW9g7ksGRaq3p0WA'
CHAT_ID = 632886964

bot = Bot(token=TELEGRAM_TOKEN)
exchange = ccxt.mexc()

symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
timeframes = ['5m', '15m']

logging.basicConfig(level=logging.INFO)

def get_signal(symbol, timeframe):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    df['EMA20'] = ta.ema(df['close'], length=20)
    df['EMA50'] = ta.ema(df['close'], length=50)
    df['RSI'] = ta.rsi(df['close'], length=14)
    macd = ta.macd(df['close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['Signal'] = macd['MACDs_12_26_9']

    latest = df.iloc[-1]
    signal = None
    sl = tp = None

    if (
        latest['EMA20'] > latest['EMA50'] and 
        latest['MACD'] > latest['Signal'] and 
        latest['RSI'] < 70
    ):
        signal = 'BUY'
        sl = latest['close'] * 0.97
        tp = latest['close'] * 1.03

    elif (
        latest['EMA20'] < latest['EMA50'] and 
        latest['MACD'] < latest['Signal'] and 
        latest['RSI'] > 30
    ):
        signal = 'SELL'
        sl = latest['close'] * 1.03
        tp = latest['close'] * 0.97

    return signal, latest['close'], sl, tp

def send_signals():
    for symbol in symbols:
        for tf in timeframes:
            try:
                signal, price, sl, tp = get_signal(symbol, tf)
                if signal:
                    msg = (
                        f"{signal} SIGNAL for {symbol} ({tf})\n"
                        f"Price: {price:.2f}\n"
                        f"Take Profit: {tp:.2f}\n"
                        f"Stop Loss: {sl:.2f}"
                    )
                    bot.send_message(chat_id=CHAT_ID, text=msg)
            except Exception as e:
                err_msg = f"Error for {symbol} ({tf}): {str(e)}"
                logging.error(err_msg)
                try:
                    bot.send_message(chat_id=CHAT_ID, text=err_msg)
                except Exception as telegram_error:
                    logging.error(f"Telegram error: {telegram_error}")

while True:
    send_signals()
    time.sleep(300)
