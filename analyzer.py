
import ccxt
import pandas as pd
import time
import logging
from ta.trend import EMAIndicator, MACD

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def fetch_ohlcv(symbol="BTC/USDT", timeframe="1h", limit=100):
    exchange = ccxt.kucoin()
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def check_for_signal(df):
    if df.empty:
        logging.warning("Empty dataframe, skipping signal check.")
        return None

    df["ema_20"] = EMAIndicator(df["close"], window=20).ema_indicator()
    macd = MACD(close=df["close"])
    df["macd_diff"] = macd.macd_diff()

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    signal = None
    if latest["close"] > latest["ema_20"] and latest["macd_diff"] > 0 and prev["macd_diff"] <= 0:
        signal = "BUY"
    elif latest["close"] < latest["ema_20"] and latest["macd_diff"] < 0 and prev["macd_diff"] >= 0:
        signal = "SELL"

    logging.info(f"Latest Price: {latest['close']} - Signal: {signal}")
    return signal

def main():
    symbol = "BTC/USDT"
    df = fetch_ohlcv(symbol)
    signal = check_for_signal(df)
    if signal:
        logging.info(f"{symbol} Signal: {signal}")
    else:
        logging.info("No new signal.")

if __name__ == "__main__":
    main()
