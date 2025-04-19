
import pandas as pd
import numpy as np

def compute_indicators(df):
    df["EMA12"] = df["close"].ewm(span=12).mean()
    df["EMA26"] = df["close"].ewm(span=26).mean()
    df["RSI"] = compute_rsi(df["close"])
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def detect_elliott_wave(df):
    # placeholder الگوریتم الیوت (قابل ارتقا)
    return "روند پنج موجی" if df["close"].iloc[-1] > df["close"].mean() else None

def detect_price_action(df):
    last = df.iloc[-1]
    body = abs(last["close"] - last["open"])
    range_ = last["high"] - last["low"]
    if body < range_ * 0.2:
        return "دوجی"
    elif last["close"] > last["open"]:
        return "کندل صعودی"
    else:
        return "کندل نزولی"
