import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

def compute_rsi(df, period=14):
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss.where(loss != 0, 1e-10)
    return 100 - (100 / (1 + rs))

def compute_atr(df, period=14):
    tr = pd.concat([
        df["high"] - df["low"],
        abs(df["high"] - df["close"].shift()),
        abs(df["low"] - df["close"].shift())
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def compute_bollinger_bands(df, period=20, std_dev=2):
    sma = df["close"].rolling(window=period).mean()
    std = df["close"].rolling(window=period).std()
    return sma + std_dev * std, sma - std_dev * std

def detect_pin_bar(df):
    body = (df["close"] - df["open"]).abs()
    candle_range = df["high"] - df["low"]
    max_oc = pd.concat([df["open"], df["close"]], axis=1).max(axis=1)
    min_oc = pd.concat([df["open"], df["close"]], axis=1).min(axis=1)
    upper_shadow = df["high"] - max_oc
    lower_shadow = min_oc - df["low"]
    condition = (
        (body < 0.3 * candle_range) &
        ((upper_shadow > 2 * body) | (lower_shadow > 2 * body))
    )
    df["PinBar"] = condition
    return df

def detect_engulfing(df):
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    condition = (
        ((df["close"] > df["open"]) & (prev_close < prev_open) &
         (df["close"] > prev_open) & (df["open"] < prev_close)) |
        ((df["close"] < df["open"]) & (prev_close > prev_open) &
         (df["close"] < prev_open) & (df["open"] > prev_close))
    )
    df["Engulfing"] = condition
    return df

def detect_elliott_wave(df):
    local_max = argrelextrema(df['close'].values, np.greater, order=5)[0]
    local_min = argrelextrema(df['close'].values, np.less, order=5)[0]
    df["WavePoint"] = np.nan
    df.loc[local_max, "WavePoint"] = df.loc[local_max, "close"]
    df.loc[local_min, "WavePoint"] = df.loc[local_min, "close"]
    return df

def backtest_ema_strategy(df):
    df["TradeSignal"] = 0
    df.loc[df["EMA12"] > df["EMA26"], "TradeSignal"] = 1
    df.loc[df["EMA12"] < df["EMA26"], "TradeSignal"] = -1
    df["Return"] = df["close"].pct_change()
    df["StrategyReturn"] = df["TradeSignal"].shift() * df["Return"]
    df["EquityCurve"] = (1 + df["StrategyReturn"]).cumprod()
    return df

def compute_indicators(df):
    df["EMA12"] = df["close"].ewm(span=12).mean()
    df["EMA26"] = df["close"].ewm(span=26).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    df["RSI"] = compute_rsi(df)
    df["ATR"] = compute_atr(df)
    df["BB_upper"], df["BB_lower"] = compute_bollinger_bands(df)
    df = detect_pin_bar(df)
    df = detect_engulfing(df)
    df = detect_elliott_wave(df)
    df = backtest_ema_strategy(df)
    return df