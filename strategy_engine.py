import yfinance as yf
import pandas as pd
import json
from kucoin.client import Client

def fetch_yahoo_data(symbol):
    df = yf.download(symbol, period="1d", interval="5m")
    return df

def fetch_kucoin_data(symbol):
    from datetime import datetime, timedelta
    import requests

    end = int(datetime.utcnow().timestamp())
    start = int((datetime.utcnow() - timedelta(days=1)).timestamp())

    url = f"https://api.kucoin.com/api/v1/market/candles?type=5min&symbol={symbol}&startAt={start}&endAt={end}"
    response = requests.get(url).json()
    if response["code"] != "200":
        return pd.DataFrame()

    df = pd.DataFrame(response["data"], columns=["time", "open", "close", "high", "low", "volume", "turnover"])
    df = df.iloc[::-1]
    df[["open", "close", "high", "low", "volume"]] = df[["open", "close", "high", "low", "volume"]].astype(float)
    return df

def advanced_price_action(df):
    if len(df) < 5:
        return None
    last_candle = df.iloc[-1]
    if last_candle["close"] > last_candle["open"] and last_candle["volume"] > df["volume"].mean():
        entry = last_candle["close"]
        stop_loss = entry * 0.98
        take_profit = entry * 1.05
        return f"ورود: {entry:.2f} | حد ضرر: {stop_loss:.2f} | حد سود: {take_profit:.2f} (پرایس اکشن پیشرفته)"
    return None

def ema_cross(df):
    df["ema_fast"] = df["close"].ewm(span=5).mean()
    df["ema_slow"] = df["close"].ewm(span=20).mean()
    if df["ema_fast"].iloc[-2] < df["ema_slow"].iloc[-2] and df["ema_fast"].iloc[-1] > df["ema_slow"].iloc[-1]:
        entry = df["close"].iloc[-1]
        stop_loss = entry * 0.97
        take_profit = entry * 1.06
        return f"ورود: {entry:.2f} | حد ضرر: {stop_loss:.2f} | حد سود: {take_profit:.2f} (EMA کراس)"
    return None

def rsi_strategy(df):
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    if rsi.iloc[-1] < 30:
        entry = df["close"].iloc[-1]
        stop_loss = entry * 0.95
        take_profit = entry * 1.1
        return f"ورود: {entry:.2f} | حد ضرر: {stop_loss:.2f} | حد سود: {take_profit:.2f} (RSI oversold)"
    return None

def run_strategies():
    signals = []
    with open("symbols.json") as f:
        symbols = json.load(f)

    for asset in symbols:
        if asset["type"] == "forex":
            df = fetch_yahoo_data(asset["symbol"])
        elif asset["type"] == "crypto":
            df = fetch_kucoin_data(asset["symbol"])
        else:
            continue

        if df is None or df.empty:
            continue

        for strat in [advanced_price_action, ema_cross, rsi_strategy]:
            result = strat(df)
            if result:
                signals.append(f"{asset['symbol']}:
{result}")

    return signals