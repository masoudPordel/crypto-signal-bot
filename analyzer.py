import logging
import aiohttp
import yfinance as yf
import pandas as pd
from strategy_engine import compute_indicators

from datetime import datetime

async def fetch_ohlcv_kucoin_async(symbol, interval="5min"):
    async with aiohttp.ClientSession() as session:
        url = f"https://api.kucoin.com/api/v1/market/candles?type={interval}&symbol={symbol}-USDT"
        async with session.get(url) as response:
            data = await response.json()
            if "data" not in data:
                return None
            df = pd.DataFrame(data["data"], columns=["timestamp", "open", "close", "high", "low", "volume", "turnover"])
            df = df.iloc[::-1]
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
            return df

def fetch_forex_ohlcv(symbol):
    df = yf.download(symbol, period="1d", interval="5m")
    if df.empty:
        return None
    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
    })[["open", "high", "low", "close", "volume"]]
    return df

def generate_signal(symbol, df, interval="5min", is_crypto=True):
    if df is None or len(df) < 50:
        return None
    df = compute_indicators(df)
    last = df.iloc[-1]
    atr = df["ATR"].iloc[-1]
    confidence = 70
    return {
        "نماد": symbol,
        "قیمت ورود": round(last["close"], 5),
        "هدف سود": round(last["close"] + 2 * atr, 5),
        "حد ضرر": round(last["close"] - 1.5 * atr, 5),
        "سطح اطمینان": confidence,
        "تحلیل": f"RSI={round(last['RSI'],1)}, MACD={round(last['MACD'], 2)}",
        "تایم‌فریم": interval
    }

async def scan_all_crypto_symbols():
    symbols = ["BTC", "ETH", "XRP", "LTC", "TRX"]
    results = []
    for symbol in symbols:
        df = await fetch_ohlcv_kucoin_async(symbol)
        signal = generate_signal(symbol, df, is_crypto=True)
        if signal:
            results.append(signal)
    return results

async def scan_all_forex_symbols():
    pairs = ["EURUSD=X", "GBPUSD=X"]
    results = []
    for symbol in pairs:
        df = fetch_forex_ohlcv(symbol)
        signal = generate_signal(symbol, df, is_crypto=False)
        if signal:
            results.append(signal)
    return results