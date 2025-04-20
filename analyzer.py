# نسخه حرفه‌ای با کنترل throttle و اجرای موازی محدود

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import ccxt.async_support as ccxt
import asyncio
import time
import logging
from datetime import datetime

# --- لاگ ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- تنظیمات ---
TIMEFRAMES = ["5m", "15m", "1h", "4h"]
CACHE = {}
CACHE_TTL = 60
VOLUME_THRESHOLD = 1000
SIGNAL_LOG = "signals.csv"
EMA_FILTER_PERIOD = 50

MAX_CONCURRENT_REQUESTS = 2
WAIT_BETWEEN_REQUESTS = 2  # ثانیه تأخیر بین درخواست‌های داخل semaphore
WAIT_BETWEEN_CHUNKS = 60  # مکث بین هر دسته از نمادها

# اندیکاتورها
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

def detect_pin_bar(df):
    df["body"] = abs(df["close"] - df["open"])
    df["range"] = df["high"] - df["low"]
    df["upper"] = df["high"] - df[["close", "open"]].max(axis=1)
    df["lower"] = df[["close", "open"]].min(axis=1) - df["low"]
    return (df["body"] < 0.3 * df["range"]) & ((df["upper"] > 2 * df["body"]) | (df["lower"] > 2 * df["body"]))

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
    df["WavePoint"] = np.nan
    highs = argrelextrema(df['close'].values, np.greater, order=5)[0]
    lows = argrelextrema(df['close'].values, np.less, order=5)[0]
    df.loc[df.index[highs], "WavePoint"] = df.loc[df.index[highs], "close"]
    df.loc[df.index[lows], "WavePoint"] = df.loc[df.index[lows], "close"]
    return df

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

# کنترل همزمانی
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

async def get_ohlcv_cached(exchange, symbol, tf, limit=100):
    async with semaphore:
        await asyncio.sleep(WAIT_BETWEEN_REQUESTS)
        key = f"{symbol}_{tf}"
        now = time.time()
        if key in CACHE and now - CACHE[key]["time"] < CACHE_TTL:
            return CACHE[key]["data"]
        try:
            data = await exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            CACHE[key] = {"data": df.copy(), "time": now}
            return df
        except Exception as e:
            logging.error(f"خطا در دریافت داده: {symbol}-{tf}: {e}")
            return None

async def analyze_symbol(exchange, symbol, tf):
    df = await get_ohlcv_cached(exchange, symbol, tf)
    if df is None or len(df) < 50:
        logging.info(f"[{symbol}-{tf}] داده نامعتبر")
        return None
    if df["volume"].iloc[-1] < VOLUME_THRESHOLD:
        logging.info(f"[{symbol}-{tf}] حجم پایین")
        return None

    df_d = await get_ohlcv_cached(exchange, symbol, "1d", limit=200)
    if df_d is None or len(df_d) < EMA_FILTER_PERIOD:
        logging.info(f"[{symbol}] داده روزانه کافی نیست.")
        return None
    ema_daily = df_d["close"].ewm(span=EMA_FILTER_PERIOD).mean().iloc[-1]
    close_daily = df_d["close"].iloc[-1]
    if close_daily < ema_daily:
        logging.info(f"[{symbol}] زیر EMA روزانه.")
        return None

    df = compute_indicators(df)
    last = df.iloc[-1]

    conds = {
        "PinBar": bool(last["PinBar"]),
        "Engulfing": bool(last["Engulfing"]),
        "EMA_Cross": df["EMA12"].iloc[-2] < df["EMA26"].iloc[-2] and df["EMA12"].iloc[-1] > df["EMA26"].iloc[-1],
        "MACD_Cross": df["MACD"].iloc[-2] < df["Signal"].iloc[-2] and df["MACD"].iloc[-1] > df["Signal"].iloc[-1],
        "RSI_Oversold": last["RSI"] < 30,
        "ADX_StrongTrend": last["ADX"] > 25,
    }

    score = sum(conds.values())
    logging.info(f"[{symbol}-{tf}] شرایط: {conds} - امتیاز: {score}")
    if score >= 2:
        sl = last["close"] - 1.5 * last["ATR"]
        tp = last["close"] + 2 * last["ATR"]
        rr = round((tp - last["close"]) / (last["close"] - sl), 2)
        signal = {
            "نماد": symbol,
            "تایم‌فریم": tf,
            "قیمت ورود": round(last["close"], 4),
            "هدف سود": round(tp, 4),
            "حد ضرر": round(sl, 4),
            "سطح اطمینان": min(score * 20, 100),
            "تحلیل": " | ".join([k for k, v in conds.items() if v]),
            "ریسک به ریوارد": rr
        }
        logging.info(f"سیگنال نهایی: {signal}")
        return signal
    return None

async def scan_all_crypto_symbols():
    exchange = ccxt.kucoin({
        'enableRateLimit': True,
        'rateLimit': 1500
    })

    await exchange.load_markets()
    symbols = [s for s in exchange.symbols if s.endswith("/USDT")]
    timeframes = TIMEFRAMES
    results = []

    chunk_size = 10
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i + chunk_size]
        tasks = []
        for symbol in chunk:
            for tf in timeframes:
                tasks.append(analyze_symbol(exchange, symbol, tf))
        chunk_results = await asyncio.gather(*tasks)
        results.extend([res for res in chunk_results if res])
        logging.info(f"اسکن {i + chunk_size}/{len(symbols)} نماد کامل شد.")
        await asyncio.sleep(WAIT_BETWEEN_CHUNKS)

    await exchange.close()
    return results