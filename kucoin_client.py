
import requests
import pandas as pd
from config import BASE_URL

def fetch_kucoin_ohlcv(symbol: str, interval: str = "5min", limit: int = 200):
    url = f"{BASE_URL}/api/v1/market/candles?type={interval}&symbol={symbol}"
    resp = requests.get(url)
    data = resp.json()
    if not data["data"]:
        return None
    df = pd.DataFrame(data["data"], columns=["time", "open", "close", "high", "low", "volume", "turnover"])
    df = df.iloc[::-1]
    df[["open", "close", "high", "low", "volume"]] = df[["open", "close", "high", "low", "volume"]].astype(float)
    df["time"] = pd.to_datetime(df["time"], unit='s')
    df.set_index("time", inplace=True)
    return df
