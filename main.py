
from kucoin_client import fetch_kucoin_ohlcv
from signal_analyzer import analyze
from logger import logger
import schedule
import time
import pandas as pd
import os

def save_signal(symbol, signal, timestamp):
    file = "signals.csv"
    row = {"symbol": symbol, "signal": signal, "time": timestamp}
    if os.path.exists(file):
        df = pd.read_csv(file)
        df = df.append(row, ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(file, index=False)

def run():
    symbols = ["BTC-USDT", "ETH-USDT"]
    for symbol in symbols:
        df = fetch_kucoin_ohlcv(symbol)
        if df is not None:
            signal = analyze(df, symbol)
            if signal in ["BUY", "SELL"]:
                save_signal(symbol, signal, pd.Timestamp.now())
            logger.info(f"{symbol} → سیگنال: {signal}")
        else:
            logger.warning(f"دیتا یافت نشد برای {symbol}")

# زمان‌بندی هر 5 دقیقه
schedule.every(5).minutes.do(run)

if __name__ == "__main__":
    print("ربات فعال است. هر ۵ دقیقه اجرا خواهد شد.")
    while True:
        schedule.run_pending()
        time.sleep(1)
