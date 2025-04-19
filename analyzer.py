import requests
import pandas as pd
import aiohttp
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
from functools import lru_cache
import logging
import numpy as np
import backtrader as bt

# تنظیم لاگ‌ها
logging.basicConfig(filename="trading_errors.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@lru_cache(maxsize=1)
def get_all_symbols_kucoin():
    try:
        url = "https://api.kucoin.com/api/v1/symbols"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return [item["symbol"].replace("-", "") for item in data["data"] if item["symbol"].endswith("-USDT")]
    except Exception as e:
        logging.error(f"خطا در دریافت نمادها: {e}")
        return []

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def fetch_ohlcv_kucoin_async(symbol, interval="5min", limit=200):
    async with aiohttp.ClientSession() as session:
        url = f"https://api.kucoin.com/api/v1/market/candles?type={interval}&symbol={symbol[:-4]}-USDT"
        async with session.get(url) as response:
            if response.status != 200:
                return None
            data = await response.json()
            if not data["data"]:
                return None
            df = pd.DataFrame(data["data"], columns=["timestamp", "open", "close", "high", "low", "volume", "turnover"])
            df = df.iloc[::-1]
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
            return df

# --- Indicators ---
def compute_rsi(df, period=14):
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss.where(loss != 0, 1e-10)
    return 100 - (100 / (1 + rs))

def compute_atr(df, period=14):
    tr = pd.concat([
        df["high"] - df["low"],
        np.abs(df["high"] - df["close"].shift()),
        np.abs(df["low"] - df["close"].shift())
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def compute_bollinger_bands(df, period=20, std_dev=2):
    sma = df["close"].rolling(window=period).mean()
    std = df["close"].rolling(window=period).std()
    return sma + std_dev * std, sma - std_dev * std

def compute_indicators(df):
    df["EMA12"] = df["close"].ewm(span=12).mean()
    df["EMA26"] = df["close"].ewm(span=26).mean()
    df["RSI"] = compute_rsi(df)
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    df["ATR"] = compute_atr(df)
    df["BB_upper"], df["BB_lower"] = compute_bollinger_bands(df)
    return df

# --- Pattern Detection ---
def detect_engulfing(df):
    last, prev = df.iloc[-1], df.iloc[-2]
    if last["close"] > last["open"] and prev["close"] < prev["open"] and last["open"] < prev["close"] and last["close"] > prev["open"]:
        return "الگوی پوشای صعودی"
    elif last["close"] < last["open"] and prev["close"] > prev["open"] and last["open"] > prev["close"] and last["close"] < prev["open"]:
        return "الگوی پوشای نزولی"
    return None

def detect_advanced_price_action(df):
    last = df.iloc[-1]
    body = abs(last["close"] - last["open"])
    wick = last["high"] - last["low"]
    upper = last["high"] - max(last["close"], last["open"])
    lower = min(last["close"], last["open"]) - last["low"]
    if body < wick * 0.2:
        return "دوجی"
    elif lower > body * 2:
        return "پین بار صعودی"
    elif upper > body * 2:
        return "پین بار نزولی"
    return None

def detect_trend(df):
    highs = df["high"].rolling(20).max()
    lows = df["low"].rolling(20).min()
    if df["close"].iloc[-1] > highs.iloc[-2]:
        return "روند صعودی"
    elif df["close"].iloc[-1] < lows.iloc[-2]:
        return "روند نزولی"
    return "بدون روند"

def breakout_strategy(df):
    last, prev = df.iloc[-1], df.iloc[-2]
    resistance = df["high"].rolling(20).max().iloc[-2]
    if last["close"] > resistance and prev["close"] <= resistance:
        return "شکست صعودی"
    return None

def bollinger_strategy(df):
    last = df.iloc[-1]
    if last["close"] < last["BB_lower"]:
        return "نزدیک باند پایینی بولینگر"
    return None

# --- Generate Signal ---
def generate_signal(symbol, df, interval="5min", min_confidence=40):
    if df is None or len(df) < 50:
        return None
    df = compute_indicators(df)

    rsi = df["RSI"].iloc[-1]
    macd, signal_line = df["MACD"].iloc[-1], df["Signal"].iloc[-1]
    ema_cross = df["EMA12"].iloc[-2] < df["EMA26"].iloc[-2] and df["EMA12"].iloc[-1] > df["EMA26"].iloc[-1]
    volume_spike = df["volume"].iloc[-1] > df["volume"].rolling(20).mean().iloc[-1] * 1.2
    atr = df["ATR"].iloc[-1]
    close = df["close"].iloc[-1]

    score = sum([
        rsi < 45,
        macd > signal_line,
        ema_cross,
        bool(detect_engulfing(df) or detect_advanced_price_action(df)),
        volume_spike,
        bool(breakout_strategy(df)),
        bool(bollinger_strategy(df))
    ])

    confidence = int((score / 7) * 100)
    if confidence < min_confidence:
        return None

    return {
        "نماد": symbol,
        "قیمت ورود": round(close, 5),
        "هدف سود": round(close + 2 * atr, 5),
        "حد ضرر": round(close - 1.5 * atr, 5),
        "سطح اطمینان": confidence,
        "تحلیل": f"RSI={round(rsi,1)}, EMA کراس={ema_cross}, MACD={'مثبت' if macd > signal_line else 'منفی'}, "
                 f"الگو={detect_engulfing(df) or detect_advanced_price_action(df)}, {detect_trend(df)}, "
                 f"{breakout_strategy(df) or '-'}, {bollinger_strategy(df) or '-'}, "
                 f"حجم={'بالا' if volume_spike else 'نرمال'}",
        "تایم‌فریم": interval
    }