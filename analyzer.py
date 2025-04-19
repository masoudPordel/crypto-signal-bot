import pandas as pd
import numpy as np
import ccxt
import time
from scipy.signal import argrelextrema

# تایم‌فریم‌هایی که برای اسکن بازار استفاده می‌شوند
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]

# ===== اندیکاتورها =====
def compute_rsi(df, period=14):
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))

def compute_atr(df, period=14):
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift()).abs()
    tr3 = (df["low"]  - df["close"].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_bollinger_bands(df, period=20, std_dev=2):
    sma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    return sma + std_dev * std, sma - std_dev * std

# ===== پرایس‌اکشن =====
def detect_pin_bar(df):
    df = df.copy()
    body    = (df["close"] - df["open"]).abs()
    rng     = df["high"] - df["low"]
    upper   = df["high"] - df[["open","close"]].max(axis=1)
    lower   = df[["open","close"]].min(axis=1) - df["low"]
    cond = (body < 0.3*rng) & ((upper > 2*body) | (lower > 2*body))
    df["PinBar"] = cond
    return df

def detect_engulfing(df):
    df = df.copy()
    o0 = df["open"].shift(1); c0 = df["close"].shift(1)
    cond = (
        (df["close"]>df["open"]) & (c0<o0) & (df["close"]>o0) & (df["open"]<c0)
    ) | (
        (df["close"]<df["open"]) & (c0>o0) & (df["close"]<o0) & (df["open"]>c0)
    )
    df["Engulfing"] = cond
    return df

# ===== امواج الیوت ساده =====
def detect_elliott_wave(df, order=5):
    df = df.copy()
    idx_max = argrelextrema(df["close"].values, np.greater, order=order)[0]
    idx_min = argrelextrema(df["close"].values, np.less,    order=order)[0]
    df["WavePoint"] = np.nan
    df.iloc[idx_max, df.columns.get_loc("WavePoint")] = df["close"].iloc[idx_max]
    df.iloc[idx_min, df.columns.get_loc("WavePoint")] = df["close"].iloc[idx_min]
    return df

# ===== بک‌تست EMA کراس (اگر نیاز دارید) =====
def backtest_ema_strategy(df):
    df = df.copy()
    df["TradeSignal"] = 0
    df["EMA12"] = df["close"].ewm(span=12).mean()
    df["EMA26"] = df["close"].ewm(span=26).mean()
    df.loc[df["EMA12"]>df["EMA26"], "TradeSignal"] = 1
    df.loc[df["EMA12"]<df["EMA26"], "TradeSignal"] = -1
    df["Return"] = df["close"].pct_change()
    df["StrategyReturn"] = df["TradeSignal"].shift() * df["Return"]
    df["EquityCurve"] = (1+df["StrategyReturn"]).cumprod()
    return df

# ===== ترکیب همه اندیکاتورها =====
def compute_indicators(df):
    df = df.copy()
    # EMA & MACD
    df["EMA12"] = df["close"].ewm(span=12).mean()
    df["EMA26"] = df["close"].ewm(span=26).mean()
    df["MACD"]  = df["EMA12"] - df["EMA26"]
    df["Signal"]= df["MACD"].ewm(span=9).mean()
    # بقیه اندیکاتورها
    df["RSI"] = compute_rsi(df)
    df["ATR"] = compute_atr(df)
    df["BB_upper"], df["BB_lower"] = compute_bollinger_bands(df)
    # پرایس‌اکشن و الیوت
    df = detect_pin_bar(df)
    df = detect_engulfing(df)
    df = detect_elliott_wave(df)
    return df

# ===== آنالیز یک نماد در یک تایم‌فریم =====
def analyze_symbol(symbol, timeframe="1h", limit=100):
    exchange = ccxt.binance()
    try:
        # 1) داده‌های تایم‌فریم جاری
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df    = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = compute_indicators(df)
        last= df.iloc[-1]

        # 2) داده‌های H4 برای سنجش روند
        if timeframe != "4h":
            h4 = exchange.fetch_ohlcv(symbol, timeframe="4h", limit=limit)
            df4= pd.DataFrame(h4, columns=["timestamp","open","high","low","close","volume"])
            df4["timestamp"] = pd.to_datetime(df4["timestamp"], unit="ms")
            df4.set_index("timestamp", inplace=True)
            df4 = compute_indicators(df4)
            trend_ok = df4["EMA12"].iloc[-1] > df4["EMA26"].iloc[-1]
        else:
            trend_ok = last["EMA12"] > last["EMA26"]

        # 3) امتیازدهی شرایط
        score = 0
        if trend_ok:                 score += 1
        if last["PinBar"] or last["Engulfing"]: score += 1
        if last["RSI"] < 30:         score += 1

        # 4) آستانه سیگنال
        if score >= 2:
            # ساخت پیغام سیگنال
            entry = last["close"]
            return {
                "نماد": symbol,
                "تایم‌فریم": timeframe,
                "قیمت ورود": round(entry, 4),
                "هدف سود": round(entry * 1.02, 4),
                "حد ضرر": round(entry * 0.98, 4),
                "سطح اطمینان": int(score/3*100),
                "تحلیل": (
                    f"روند H4 {'صعودی' if trend_ok else 'نزولی'}، "
                    f"PinBar={last['PinBar']}، Engulfing={last['Engulfing']}، "
                    f"RSI={round(last['RSI'],1)}"
                )
            }

    except Exception as e:
        print(f"Error analyzing {symbol} {timeframe}:", e)

    return None

# ===== اسکن کل بازار کریپتو =====
def scan_all_crypto_symbols():
    exchange = ccxt.binance()
    markets = exchange.load_markets()
    symbols = [s for s in markets if s.endswith("/USDT")]
    signals = []
    for sym in symbols:
        for tf in TIMEFRAMES:
            sig = analyze_symbol(sym, tf)
            if sig:
                signals.append(sig)
            time.sleep(0.2)  # جلوگیری از rate limit
    return signals

# ===== اسکن فارکس (در صورت نیاز) =====
def scan_all_forex_symbols():
    return []