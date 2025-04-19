import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import ccxt
import time

# تایم‌فریم‌هایی که بررسی می‌شن
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]

# --- اندیکاتورها ---
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

# --- پرایس اکشن ---
def detect_pin_bar(df):
    df = df.copy()
    df["body"] = abs(df["close"] - df["open"])
    df["range"] = df["high"] - df["low"]
    df["upper_shadow"] = df["high"] - df[["close","open"]].max(axis=1)
    df["lower_shadow"] = df[["close","open"]].min(axis=1) - df["low"]
    df["PinBar"] = (df["body"] < 0.3*df["range"]) & (
        (df["upper_shadow"]>2*df["body"]) | (df["lower_shadow"]>2*df["body"])
    )
    return df

def detect_engulfing(df):
    df = df.copy()
    o0, c0 = df["open"], df["close"]
    o1, c1 = o0.shift(1), c0.shift(1)
    df["Engulfing"] = (
        ((c0>o0)&(c1<o1)&(c0>o1)&(o0<c1)) |
        ((c0<o0)&(c1>o1)&(c0<o1)&(o0>c1))
    )
    return df

# --- امواج الیوت ساده ---
def detect_elliott_wave(df):
    df = df.copy()
    vals = df["close"].values
    local_max = argrelextrema(vals, np.greater, order=5)[0]
    local_min = argrelextrema(vals, np.less,    order=5)[0]
    df["ElliottPeak"] = np.nan
    df.loc[df.index[local_max], "ElliottPeak"] = df.loc[df.index[local_max], "close"]
    df.loc[df.index[local_min], "ElliottPeak"] = df.loc[df.index[local_min], "close"]
    return df

# --- کراس EMA و MACD ---
def detect_ema_cross(df):
    return (df["EMA12"].iloc[-2] < df["EMA26"].iloc[-2]) and (df["EMA12"].iloc[-1] > df["EMA26"].iloc[-1])

def detect_macd_cross(df):
    return (df["MACD"].iloc[-2] < df["Signal"].iloc[-2]) and (df["MACD"].iloc[-1] > df["Signal"].iloc[-1])

# --- بک‌تست EMA کراس (برای خروج) ---
def backtest_ema_strategy(df):
    # اینجا صرفاً برای محاسبه equity curve
    df["SignalLine"] = np.where(df["EMA12"]>df["EMA26"],1,-1)
    df["Return"] = df["close"].pct_change()
    df["StratRet"] = df["SignalLine"].shift()*df["Return"]
    df["Equity"] = (1+df["StratRet"]).cumprod()
    return df

# --- محاسبهٔ همهٔ اندیکاتورها و پرایس‌اکشن‌ها ---
def compute_indicators(df):
    df["EMA12"] = df["close"].ewm(span=12).mean()
    df["EMA26"] = df["close"].ewm(span=26).mean()
    df["MACD"]  = df["EMA12"] - df["EMA26"]
    df["Signal"]= df["MACD"].ewm(span=9).mean()
    df["RSI"]   = compute_rsi(df)
    df["ATR"]   = compute_atr(df)
    df["BB_up"], df["BB_lo"] = compute_bollinger_bands(df)
    df = detect_pin_bar(df)
    df = detect_engulfing(df)
    df = detect_elliott_wave(df)
    df = backtest_ema_strategy(df)
    return df

# --- تحلیل یک نماد در یک تایم‌فریم ---
def analyze_symbol(symbol, timeframe="1h", limit=100):
    exchange = ccxt.binance()
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"],unit="ms")
        df.set_index("timestamp", inplace=True)

        df = compute_indicators(df)
        last = df.iloc[-1]

        # محاسبهٔ نقطه‌ای بودن هر شرط
        conds = {
            "PinBar":    last["PinBar"],
            "Engulfing": last["Engulfing"],
            "EMAcross":  detect_ema_cross(df),
            "MACDcross": detect_macd_cross(df),
            "RSI<30":    last["RSI"]<30,
            "BBbreak":   last["close"]<last["BB_lo"],
            "ATRbreak":  last["close"] > (df["high"].rolling(14).max().iloc[-2])
        }
        score = sum(conds.values())
        if score >= 2:
            return {
                "نماد": symbol,
                "تایم‌فریم": timeframe,
                "قیمت ورود": round(last["close"],4),
                "هدف سود":   round(last["close"]+2*last["ATR"],4),
                "حد ضرر":    round(last["close"]-1.5*last["ATR"],4),
                "سطح اطمینان": min(score*15, 100),
                "تحلیل": " | ".join([k for k,v in conds.items() if v])
            }

    except Exception as e:
        print(f"Error analyzing {symbol} {timeframe}:", e)

    return None

# --- اسکن کل بازار کریپتو ---
def scan_all_crypto_symbols():
    exchange = ccxt.binance()
    exchange.load_markets()
    symbols = [s for s in exchange.symbols if s.endswith("/USDT")]
    signals = []
    for sym in symbols:
        for tf in TIMEFRAMES:
            sig = analyze_symbol(sym, tf)
            if sig:
                signals.append(sig)
            time.sleep(0.2)
    return signals

# --- اسکن فارکس (این‌جا خالیه، برای یاهوفایننس می‌تونی مشابه پیادش کنی) ---
def scan_all_forex_symbols():
    return []