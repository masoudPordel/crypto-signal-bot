import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import ccxt
import time

# === لیست تایم‌فریم‌هایی که بررسی می‌شن ===
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]

# === اندیکاتورها ===
def compute_rsi(df, period=14):
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss.replace(0, 1e-10)
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

# === پرایس اکشن ===
def detect_pin_bar(df):
    df = df.copy()
    df["body"] = abs(df["close"] - df["open"])
    df["candle_range"] = df["high"] - df["low"]
    df["upper_shadow"] = df["high"] - df[["close", "open"]].max(axis=1)
    df["lower_shadow"] = df[["close", "open"]].min(axis=1) - df["low"]

    condition = (
        (df["body"] < 0.3 * df["candle_range"]) &
        ((df["upper_shadow"] > 2 * df["body"]) | (df["lower_shadow"] > 2 * df["body"]))
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

# === الیوت ساده ===
def detect_elliott_wave(df):
    local_max = argrelextrema(df['close'].values, np.greater, order=5)[0]
    local_min = argrelextrema(df['close'].values, np.less, order=5)[0]

    df["WavePoint"] = np.nan
    df.loc[df.index[local_max], "WavePoint"] = df.loc[df.index[local_max], "close"]
    df.loc[df.index[local_min], "WavePoint"] = df.loc[df.index[local_min], "close"]
    return df

# === کراس EMA ===
def backtest_ema_strategy(df):
    df["TradeSignal"] = 0
    df.loc[df["EMA12"] > df["EMA26"], "TradeSignal"] = 1
    df.loc[df["EMA12"] < df["EMA26"], "TradeSignal"] = -1

    df["Return"] = df["close"].pct_change()
    df["StrategyReturn"] = df["TradeSignal"].shift() * df["Return"]
    df["EquityCurve"] = (1 + df["StrategyReturn"]).cumprod()
    return df

# === اجرای تحلیل کامل ===
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

# === تحلیل یک نماد در یک تایم‌فریم ===
def analyze_symbol(symbol, timeframe="1h", limit=100):
    exchange = ccxt.binance()
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        df = compute_indicators(df)
        last = df.iloc[-1]

        if last["Engulfing"] or last["PinBar"]:
            print(f"Signal on {symbol} ({timeframe}) - PinBar: {last['PinBar']} - Engulfing: {last['Engulfing']}")

        return df
    except Exception as e:
        print(f"Error analyzing {symbol} ({timeframe}):", e)

# === اسکن کل بازار کریپتو ===
def scan_all_crypto_symbols():
    exchange = ccxt.binance()
    markets = exchange.load_markets()
    crypto_symbols = [s for s in markets if s.endswith("/USDT")]

    for symbol in crypto_symbols:
        for tf in TIMEFRAMES:
            analyze_symbol(symbol, tf)
            time.sleep(0.2)  # جلوگیری از rate limit

# === اسکن بازار فارکس (مثلاً با OANDA یا بروکر دیگر) ===
def scan_all_forex_symbols():
    pass  # برای بروکرهای واقعی نیاز به API هست