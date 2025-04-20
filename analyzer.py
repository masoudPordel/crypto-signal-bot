import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
import pandas as pd
import numpy as np
import requests
from scipy.signal import argrelextrema
import ccxt.async_support as ccxt
import asyncio
import time
import logging
from datetime import datetime

# --- تنظیمات ---
TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]
CACHE = {}
CACHE_TTL = 60  # ثانیه
VOLUME_THRESHOLD = 10000
VOLUME_SPIKE_MULTIPLIER = 1.5
SL_FACTOR = 1.5
TP_FACTOR = 2.0
SIGNAL_LOG = "signals.csv"

logging.basicConfig(
    level=logging.INFO,
    filename="errors.log",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- اندیکاتورها ---
def compute_rsi(df, period=14):
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))

def compute_atr(df, period=14):
    tr = pd.concat([
        df["high"] - df["low"],
        abs(df["high"] - df["close"].shift()),
        abs(df["low"] - df["close"].shift())
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_bollinger_bands(df, period=20, std_dev=2):
    sma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    return sma + std_dev * std, sma - std_dev * std

def compute_adx(df, period=14):
    df["up"] = df["high"].diff()
    df["down"] = -df["low"].diff()
    df["+DM"] = np.where((df["up"] > df["down"]) & (df["up"] > 0), df["up"], 0.0)
    df["-DM"] = np.where((df["down"] > df["up"]) & (df["down"] > 0), df["down"], 0.0)
    tr = pd.concat([
        df["high"] - df["low"],
        abs(df["high"] - df["close"].shift()),
        abs(df["low"] - df["close"].shift())
    ], axis=1).max(axis=1)
    tr_smooth = tr.rolling(window=period).sum()
    plus_di = 100 * (df["+DM"].rolling(window=period).sum() / tr_smooth)
    minus_di = 100 * (df["-DM"].rolling(window=period).sum() / tr_smooth)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(window=period).mean()

# --- الگوها ---
def detect_pin_bar(df):
    df["body"] = abs(df["close"] - df["open"])
    df["range"] = df["high"] - df["low"]
    df["upper"] = df["high"] - df[["close", "open"]].max(axis=1)
    df["lower"] = df[["close", "open"]].min(axis=1) - df["low"]
    return (df["body"] < 0.3 * df["range"]) & (
        (df["upper"] > 2 * df["body"]) | (df["lower"] > 2 * df["body"])
    )

def detect_engulfing(df):
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    return (
        ((df["close"] > df["open"]) & (prev_close < prev_open) &
         (df["close"] > prev_open) & (df["open"] < prev_close)) |
        ((df["close"] < df["open"]) & (prev_close > prev_open) &
         (df["close"] < prev_open) & (df["open"] > prev_close))
    )

def detect_elliott_wave(df):
    df["ElliottHigh"] = False
    df["ElliottLow"]  = False
    highs = argrelextrema(df["close"].values, np.greater, order=5)[0]
    lows  = argrelextrema(df["close"].values, np.less,   order=5)[0]
    df.loc[df.index[highs], "ElliottHigh"] = True
    df.loc[df.index[lows],  "ElliottLow"]  = True
    return df

# --- فاندامنتال (CoinGecko) ---
def get_symbol_id_map():
    try:
        url = "https://api.coingecko.com/api/v3/coins/list"
        data = requests.get(url).json()
        return {coin["symbol"].upper(): coin["id"] for coin in data}
    except Exception as e:
        logging.error(f"Error fetching symbol map: {e}")
        return {}

def fetch_fundamental_data(symbol_id):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{symbol_id}"
        data = requests.get(url).json()
        md = data["market_data"]
        return {
            "market_cap": md["market_cap"]["usd"],
            "rank": data.get("market_cap_rank"),
            "community_score": data.get("community_score"),
            "developer_score": data.get("developer_score")
        }
    except Exception as e:
        logging.error(f"Error fetching fundamental data for {symbol_id}: {e}")
        return None

# --- محاسبهٔ اندیکاتورها ---
def compute_indicators(df):
    df["EMA12"] = df["close"].ewm(span=12).mean()
    df["EMA26"] = df["close"].ewm(span=26).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    df["RSI"] = compute_rsi(df)
    df["ATR"] = compute_atr(df)
    df["ADX"] = compute_adx(df)
    df["BB_upper"], df["BB_lower"] = compute_bollinger_bands(df)
    df["PinBar"] = detect_pin_bar(df)
    df["Engulfing"] = detect_engulfing(df)
    df = detect_elliott_wave(df)
    return df

# --- کش کندل‌ها ---
async def get_ohlcv_cached(exchange, symbol, tf, limit=200):
    key = f"{symbol}_{tf}"
    now = time.time()
    if key in CACHE and now - CACHE[key]["time"] < CACHE_TTL:
        return CACHE[key]["data"]
    try:
        data = await exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
        df = pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        CACHE[key] = {"data": df.copy(), "time": now}
        return df
    except Exception as e:
        logging.error(f"Fetch error: {symbol}-{tf}: {e}")
        return None

# --- بررسی سیگنال ---
async def analyze_symbol(exchange, symbol, tf):
    # ۱) دیتای اصلی
    df = await get_ohlcv_cached(exchange, symbol, tf, limit=100)
    if df is None or len(df) < 50:
        return None

    # ۲) فیلتر حجم و اسپایک
    avg_vol = df["volume"].rolling(20).mean().iloc[-1]
    if df["volume"].iloc[-1] < VOLUME_THRESHOLD or df["volume"].iloc[-1] < avg_vol * VOLUME_SPIKE_MULTIPLIER:
        return None

    # ۳) فیلتر روند کلان (EMA200 روزانه)
    df_d = await get_ohlcv_cached(exchange, symbol, "1d", limit=250)
    if df_d is None or len(df_d) < 50:
        return None
    ema200 = df_d["close"].ewm(span=50).mean().iloc[-1]
    if df_d["close"].iloc[-1] < ema50:
        return None

    # ۴) محاسبهٔ اندیکاتورها
    df = compute_indicators(df)
    last = df.iloc[-1]

    # ۵) شروط تکنیکال (حالا شامل ElliottHigh و ElliottLow)
    conds = {
        "PinBar": bool(last["PinBar"]),
        "Engulfing": bool(last["Engulfing"]),
        "EMA_Cross": df["EMA12"].iloc[-2] < df["EMA26"].iloc[-2] and df["EMA12"].iloc[-1] > df["EMA26"].iloc[-1],
        "MACD_Cross": df["MACD"].iloc[-2] < df["Signal"].iloc[-2] and df["MACD"].iloc[-1] > df["Signal"].iloc[-1],
        "RSI_Oversold": last["RSI"] < 30,
        "ADX_StrongTrend": last["ADX"] > 25,
        "ElliottHigh": bool(last["ElliottHigh"]),
        "ElliottLow":  bool(last["ElliottLow"])
    }
    score = sum(conds.values())
    if score < 2:
        return None

    # ۶) تأیید در TF بالاتر
    idx = TIMEFRAMES.index(tf)
    if idx < len(TIMEFRAMES) - 1:
        higher_tf = TIMEFRAMES[idx+1]
        df_h = await get_ohlcv_cached(exchange, symbol, higher_tf, limit=100)
        if df_h is not None:
            df_h = compute_indicators(df_h)
            ema_cross_h = df_h["EMA12"].iloc[-2] < df_h["EMA26"].iloc[-2] and df_h["EMA12"].iloc[-1] > df_h["EMA26"].iloc[-1]
            macd_cross_h = df_h["MACD"].iloc[-2] < df_h["Signal"].iloc[-2] and df_h["MACD"].iloc[-1] > df_h["Signal"].iloc[-1]
            if not (ema_cross_h or macd_cross_h):
                return None

    # ۷) SL/TP و RR
    sl = last["close"] - SL_FACTOR * last["ATR"]
    tp = last["close"] + TP_FACTOR * last["ATR"]
    rr = round((tp - last["close"]) / (last["close"] - sl), 2)

    # ۸) ساخت سیگنال
    signal = {
        "نماد": symbol,
        "تایم‌فریم": tf,
        "قیمت ورود": round(last["close"], 6),
        "حد ضرر": round(sl, 6),
        "هدف سود": round(tp, 6),
        "ریسک به ریوارد": rr,
        "سطح اطمینان": min(score * 12.5, 100),  # حالا 8 شرط، هرکدوم 12.5 امتیاز
        "تحلیل": " | ".join([k for k,v in conds.items() if v])
    }

    # ۹) افزودن فاندامنتال
    symbol_map = get_symbol_id_map()
    base = symbol.split("/")[0].upper()
    symbol_id = symbol_map.get(base)
    if symbol_id:
        f = fetch_fundamental_data(symbol_id)
        if f:
            signal["رتبه فاندامنتال"] = f"Rank: {f['rank']}, Market Cap: ${f['market_cap']:,}"
            signal["امتیاز توسعه‌دهنده"] = f["developer_score"]
            signal["امتیاز جامعه"] = f["community_score"]

    # ۱۰) ذخیره و بازگشت
    df_out = pd.DataFrame([signal])
    df_out["زمان"] = datetime.utcnow()
    df_out.to_csv(
        SIGNAL_LOG,
        mode='a',
        index=False,
        header=not pd.io.common.file_exists(SIGNAL_LOG)
    )
    return signal

# --- اسکن کریپتو ---
async def scan_all_crypto_symbols():
    exchange = ccxt.kucoin()
    await exchange.load_markets()
    syms = [s for s in exchange.symbols if s.endswith("/USDT")]
    tasks = []
    for sym in syms:
        for tf in TIMEFRAMES:
            tasks.append(analyze_symbol(exchange, sym, tf))
            await asyncio.sleep(0.1)
    res = await asyncio.gather(*tasks)
    await exchange.close()
    return [r for r in res if r]

# --- اسکن فارکس (خالی) ---
async def scan_all_forex_symbols():
    return []
