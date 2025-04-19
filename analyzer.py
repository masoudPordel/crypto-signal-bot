import requests
import pandas as pd
import aiohttp
import asyncio
import numpy as np
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from ratelimit import limits, sleep_and_retry
from functools import lru_cache

# لاگ‌ها
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ALPHA_VANTAGE_API_KEY = "8VL54YT3N656MW5T"

# --- دریافت نمادها ---
@lru_cache(maxsize=1)
def get_all_symbols_kucoin():
    try:
        url = "https://api.kucoin.com/api/v1/symbols"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        symbols = [item["symbol"].replace("-", "") for item in data["data"] if item["symbol"].endswith("-USDT")]
        logging.info(f"تعداد نمادهای KuCoin: {len(symbols)}")
        return symbols
    except Exception as e:
        logging.error(f"خطا در دریافت نمادهای کوکوین: {e}")
        return []

def get_all_symbols_mexc():
    try:
        url = "https://api.mexc.com/api/v3/exchangeInfo"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        symbols = [s["symbol"] for s in data["symbols"] if s["quoteAsset"] == "USDT"]
        logging.info(f"تعداد نمادهای MEXC: {len(symbols)}")
        return symbols
    except Exception as e:
        logging.error(f"خطا در دریافت نمادهای مکسی: {e}")
        return []

# --- دریافت دیتا Async از کوکوین ---
@retry(stop=stop_after_attempt(3), wait=wait_exponential())
async def fetch_ohlcv_kucoin_async(symbol, interval="5min"):
    try:
        async with aiohttp.ClientSession() as session:
            url = f"https://api.kucoin.com/api/v1/market/candles?type={interval}&symbol={symbol[:-4]}-USDT"
            async with session.get(url) as response:
                if response.status != 200:
                    logging.warning(f"{symbol} وضعیت {response.status}")
                    return None
                data = await response.json()
                if not data["data"]:
                    logging.warning(f"{symbol} دیتایی ندارد")
                    return None
                df = pd.DataFrame(data["data"], columns=["timestamp", "open", "close", "high", "low", "volume", "turnover"])
                df = df.iloc[::-1]
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
                print(f"[KuCoin] {symbol}: {len(df)} کندل دریافت شد.")
                print(df.tail(2))
                return df
    except Exception as e:
        logging.error(f"خطا در دریافت OHLCV کوکوین برای {symbol}: {e}")
        return None

# --- دریافت دیتا از فارکس ---
@sleep_and_retry
@limits(calls=5, period=60)
@retry(stop=stop_after_attempt(3), wait=wait_exponential())
def fetch_forex_ohlcv(from_symbol, to_symbol="USD", interval="5min"):
    try:
        url = (
            f"https://www.alphavantage.co/query?function=FX_INTRADAY"
            f"&from_symbol={from_symbol}&to_symbol={to_symbol}"
            f"&interval={interval}&outputsize=compact&apikey={ALPHA_VANTAGE_API_KEY}"
        )
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        ts_key = [k for k in data if "Time Series" in k]
        if not ts_key:
            logging.warning(f"No time series in response for {from_symbol}/{to_symbol}")
            return None
        df = pd.DataFrame.from_dict(data[ts_key[0]], orient="index").sort_index()
        df = df.rename(columns={
            "1. open": "open", "2. high": "high", "3. low": "low", "4. close": "close"
        }).astype(float)
        df.index = pd.to_datetime(df.index)
        print(f"[Forex] {from_symbol}/{to_symbol}: {len(df)} کندل دریافت شد.")
        print(df.tail(2))
        return df
    except Exception as e:
        logging.error(f"خطا در دریافت دیتا از فارکس: {e}")
        return None

# --- اجرای تست ساده ---
async def test():
    # تست کوکوین
    kucoin_symbols = get_all_symbols_kucoin()[:3]
    print(f"نمادهای تست کوکوین: {kucoin_symbols}")
    for symbol in kucoin_symbols:
        await fetch_ohlcv_kucoin_async(symbol)

    # تست مکسی
    mexc_symbols = get_all_symbols_mexc()[:3]
    print(f"نمادهای تست مکسی: {mexc_symbols}")
    # داده‌های MEXC مشابه کوکوین خوانده می‌شه؟ بستگی داره اگه لازم بود پیاده‌سازی می‌کنم.

    # تست فارکس
    for pair in [("USD", "JPY"), ("AUD", "USD")]:
        fetch_forex_ohlcv(pair[0], pair[1])

# اجرای async
if __name__ == "__main__":
    asyncio.run(test())