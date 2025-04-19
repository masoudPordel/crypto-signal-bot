
import requests
import pandas as pd
import aiohttp
import asyncio
from ratelimit import limits, sleep_and_retry
from tenacity import retry, stop_after_attempt, wait_exponential
from functools import lru_cache
import logging
import numpy as np

# تنظیم لاگ‌ها
logging.basicConfig(filename="trading_combined.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ALPHA_VANTAGE_API_KEY = "8VL54YT3N656MW5T"

# --- دریافت نمادها ---
@lru_cache(maxsize=1)
def get_all_symbols_kucoin():
    try:
        url = "https://api.kucoin.com/api/v1/symbols"
        response = requests.get(url)
        data = response.json()
        return [item["symbol"].replace("-", "") for item in data["data"] if item["symbol"].endswith("-USDT")]
    except Exception as e:
        logging.error(f"خطا در دریافت نمادهای کوکوین: {e}")
        return []

@lru_cache(maxsize=1)
def get_all_symbols_mexc():
    try:
        url = "https://api.mexc.com/api/v3/exchangeInfo"
        response = requests.get(url)
        data = response.json()
        return [s["symbol"] for s in data["symbols"] if s["quoteAsset"] == "USDT"]
    except Exception as e:
        logging.error(f"خطا در دریافت نمادهای مکسی: {e}")
        return []

# --- دیتا کریپتو ---
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=8))
async def fetch_ohlcv_kucoin(symbol, interval="5min"):
    url = f"https://api.kucoin.com/api/v1/market/candles?type={interval}&symbol={symbol[:-4]}-USDT"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.json()
            if not data["data"]:
                return None
            df = pd.DataFrame(data["data"], columns=["timestamp", "open", "close", "high", "low", "volume", "turnover"])
            df = df.iloc[::-1]
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            df.set_index("timestamp", inplace=True)
            df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
            return df

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=8))
async def fetch_ohlcv_mexc(symbol, interval="5m"):
    url = f"https://api.mexc.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=200"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.json()
            if not data:
                return None
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset", "trades", "taker_base", "taker_quote", "ignore"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            df = df[["open", "high", "low", "close", "volume"]].astype(float)
            return df

# --- دیتا فارکس ---
@sleep_and_retry
@limits(calls=5, period=60)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=4, max=10))
def fetch_forex_ohlcv(from_symbol, to_symbol="USD", interval="5min"):
    try:
        url = (
            f"https://www.alphavantage.co/query?function=FX_INTRADAY"
            f"&from_symbol={from_symbol}&to_symbol={to_symbol}"
            f"&interval={interval}&outputsize=compact&apikey={ALPHA_VANTAGE_API_KEY}"
        )
        response = requests.get(url)
        data = response.json()
        ts_key = [k for k in data if "Time Series" in k]
        if not ts_key:
            return None
        df = pd.DataFrame.from_dict(data[ts_key[0]], orient="index").sort_index()
        df = df.rename(columns={
            "1. open": "open", "2. high": "high", "3. low": "low", "4. close": "close"
        }).astype(float)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        logging.error(f"خطا در دریافت دیتا فارکس: {e}")
        return None

# ادامه کد شامل تحلیل تکنیکال، سیگنال‌دهی و غیره می‌تونه بعداً اضافه بشه
