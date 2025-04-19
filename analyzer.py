import requests
import pandas as pd
import aiohttp
import asyncio
from ratelimit import limits, sleep_and_retry
from tenacity import retry, stop_after_attempt, wait_exponential
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import logging
import numpy as np
import backtrader as bt
import os

# Configure logging
logging.basicConfig(filename="trading_errors.log", level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

# Alpha Vantage API key (hardcoded as requested)
ALPHA_VANTAGE_API_KEY = "8VL54YT3N656MW5T"

# ---------- Fetch KuCoin crypto symbols ----------
@lru_cache(maxsize=1)
def get_all_symbols_kucoin():
    url = "https://api.kucoin.com/api/v1/symbols"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return [item["symbol"].replace("-", "") for item in data["data"] if item["symbol"].endswith("-USDT")]
    except Exception as e:
        logging.error(f"Failed to fetch KuCoin symbols: {e}")
        return []

# ---------- Fetch crypto OHLCV data (async) ----------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def fetch_ohlcv_kucoin_async(symbol, interval="5min", limit=100):
    async with aiohttp.ClientSession() as session:
        url = f"https://api.kucoin.com/api/v1/market/candles?type={interval}&symbol={symbol[:len(symbol)-4]}-USDT"
        async with session.get(url) as response:
            if response.status != 200:
                logging.error(f"Failed to fetch KuCoin data for {symbol}: {response.status}")
                return None
            raw_data = await response.json()
            if not raw_data["data"]:
                return None
            df = pd.DataFrame(raw_data["data"], columns=[
                "timestamp", "open", "close", "high", "low", "volume", "turnover"
            ])
            df = df.iloc[::-1]
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
            return df

# ---------- Fetch forex OHLCV data ----------
@sleep_and_retry
@limits(calls=5, period=60)  # 5 requests per minute
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_forex_ohlcv(from_symbol, to_symbol="USD", interval="5min"):
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=FX_INTRADAY"
        f"&from_symbol={from_symbol}"
        f"&to_symbol={to_symbol}"
        f"&interval={interval}"
        f"&outputsize=compact"
        f"&apikey={ALPHA_VANTAGE_API_KEY}"
    )
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        ts_key = [k for k in data if "Time Series" in k]
        if not ts_key:
            logging.error(f"Invalid forex data for {from_symbol}{to_symbol}")
            return None
        df = pd.DataFrame.from_dict(data[ts_key[0]], orient="index").sort_index()
        df = df.rename(columns={
            "1. open": "open", "2. high": "high", "3. low": "low", "4. close": "close"
        }).astype(float)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        logging.error(f"Failed to fetch forex data for {from_symbol}{to_symbol}: {e}")
        return None

# ---------- Technical Indicators ----------
def compute_indicators(df):
    df["EMA20"] = df["close"].ewm(span=20).mean()
    df["EMA50"] = df["close"].ewm(span=50).mean()
    df["RSI"] = compute_rsi(df)
    df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    df["ATR"] = compute_atr(df)
    return df

def compute_rsi(df, period=14):
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss.where(loss != 0, 1e-10)  # Prevent division by zero
    return 100 - (100 / (1 + rs))

def compute_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

# ---------- Advanced Price Action ----------
def detect_advanced_price_action(df):
    last = df.iloc[-1]
    body = abs(last["close"] - last["open"])
    wick = last["high"] - last["low"]
    upper_shadow = last["high"] - max(last["close"], last["open"])
    lower_shadow = min(last["close"], last["open"]) - last["low"]

    if body < wick * 0.2:
        return "Doji"
    elif lower_shadow > body * 2 and upper_shadow < body:
gry: "Bullish Pin Bar"
    elif upper_shadow > body * 2 and lower_shadow < body:
        return "Bearish Pin Bar"
    return None

def detect_engulfing(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    if (last["close"] > last["open"] and prev["close"] < prev["open"] and
            last["open"] < prev["close"] and last["close"] > prev["open"]):
        return "Bullish Engulfing"
    elif (last["close"] < last["open"] and prev["close"] > prev["open"] and
          last["open"] > prev["close"] and last["close"] < prev["open"]):
        return "Bearish Engulfing"
    return None

def detect_fake_breakout(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    resistance = df["high"].rolling(20).max().iloc[-2]
    support = df["low"].rolling(20).min().iloc[-2]
    if last["high"] > resistance and last["close"] < resistance:
        return "Fake Bullish Breakout"
    if last["low"] < support and last["close"] > support:
        return "Fake Bearish Breakout"
    return None

def detect_trend(df):
    highs = df["high"].rolling(20).max()
    lows = df["low"].rolling(20).min()
    if df["close"].iloc[-1] > highs.iloc[-2] and df["close"].iloc[-2] > highs.iloc[-3]:
        return "Uptrend"
    elif df["close"].iloc[-1] < lows.iloc[-2] and df["close"].iloc[-2] < lows.iloc[-3]:
        return "Downtrend"
    return "No Trend"

def detect_key_levels(df):
    support = df["low"].rolling(20).min().iloc[-1]
    resistance = df["high"].rolling(20).max().iloc[-1]
    last = df.iloc[-1]
    if abs(last["close"] - support) < last["close"] * 0.01:
        return "Near Support"
    elif abs(last["close"] - resistance) < last["close"] * 0.01:
        return "Near Resistance"
    return None

# ---------- Multi-Timeframe Analysis ----------
async def check_higher_tf_trend(symbol, higher_tf="1hour"):
    df_higher = await fetch_ohlcv_kucoin_async(symbol, interval=higher_tf)
    if df_higher is None:
        return False
    df_higher = compute_indicators(df_higher)
    return df_higher["close"].iloc[-1] > df_higher["EMA20"].iloc[-1]

# ---------- Generate Signal ----------
def generate_signal(symbol, df, interval="5min", is_crypto=True):
    if df is None or len(df) < 50:
        return None
    df = compute_indicators(df)

    rsi = df["RSI"].iloc[-1]
    macd = df["MACD"].iloc[-1]
    signal = df["Signal"].iloc[-1]
    ema_cross = df["EMA20"].iloc[-2] < df["EMA50"].iloc[-2] and df["EMA20"].iloc[-1] > df["EMA50"].iloc[-1]
    volume_spike = df["volume"].iloc[-1] > df["volume"].rolling(20).mean().iloc[-1]
    atr = df["ATR"].iloc[-1]

    price_action = detect_engulfing(df) or detect_advanced_price_action(df)
    fake_breakout = detect_fake_breakout(df)
    key_level = detect_key_levels(df)
    trend = detect_trend(df)

    score = 0
    if rsi < 35:
        score += 1
    if macd > signal:
        score += 1
    if ema_cross:
        score += 1
    if price_action:
        score += 1
    if volume_spike:
        score += 1
    if key_level == "Near Support":
        score += 1
    if fake_breakout:
        score -= 1
    if trend == "Uptrend":
        score += 1

    confidence = int((score / 7) * 100)
    if confidence < 80:
        return None

    close_price = df["close"].iloc[-1]
    signal_data = {
        "symbol": symbol,
        "entry": round(close_price, 5),
        "tp": round(close_price + 2 * atr, 5),
        "sl": round(close_price - 1.5 * atr, 5),
        "confidence": confidence,
        "analysis": (
            f"RSI: {round(rsi,1)} | EMA Crossover: {ema_cross} | MACD: {'Positive' if macd > signal else 'Negative'} | "
            f"Price Action: {price_action or '-'} | {key_level or '-'} | {trend} | "
            f"Volume: {'High' if volume_spike else 'Normal'} | {fake_breakout or '-'}"
        ),
        "tf": interval
    }

    if is_crypto:
        loop = asyncio.get_event_loop()
        if not loop.run_until_complete(check_higher_tf_trend(symbol)):
            return None

    return signal_data

# ---------- Scan Crypto Symbols ----------
async def scan_all_crypto_symbols():
    TIMEFRAMES = ["5min", "15min", "1hour"]
    all_symbols = get_all_symbols_kucoin()
    signals = []

    async def scan_symbol(symbol, tf):
        try:
            df = await fetch_ohlcv_kucoin_async(symbol, interval=tf)
            signal = generate_signal(symbol, df, tf, is_crypto=True)
            return signal
        except Exception as e:
            logging.error(f"Error for {symbol}-{tf}: {e}")
            return None

    tasks = [scan_symbol(symbol, tf) for symbol in all_symbols for tf in TIMEFRAMES]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    signals = [r for r in results if r and not isinstance(r, Exception)]
    return signals

# ---------- Scan Forex Symbols ----------
def scan_all_forex_symbols():
    pairs = [
        ("EUR", "USD"), ("GBP", "USD"), ("USD", "JPY"), ("USD", "CAD"), ("USD", "CHF"),
        ("NZD", "USD"), ("AUD", "USD"), ("AUD", "NZD"), ("AUD", "CAD"), ("AUD", "CHF"), ("AUD", "JPY"),
        ("CAD", "CHF"), ("CAD", "JPY"), ("CHF", "JPY"), ("EUR", "AUD"), ("EUR", "CAD"),
        ("EUR", "CHF"), ("EUR", "GBP"), ("EUR", "JPY"), ("EUR", "NZD"), ("GBP", "CHF"),
        ("GBP", "JPY"), ("NZD", "JPY"), ("XAG", "USD"), ("XAU", "USD")
    ]
    signals = []

    def scan_forex_pair(pair):
        try:
            base, quote = pair
            df = fetch_forex_ohlcv(base, quote)
            symbol = base + quote
            signal = generate_signal(symbol, df, is_crypto=False)
            return signal
        except Exception as e:
            logging.error(f"Error for {base}{quote}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(scan_forex_pair, pairs))
    signals = [r for r in results if r]
    return signals

# ---------- Backtest Strategy ----------
class SignalStrategy(bt.Strategy):
    params = (('signals', None),)

    def __init__(self):
        self.signal_index = 0
        self.order = None

    def next(self):
        date = self.datas[0].datetime.datetime(0)
        if self.signal_index >= len(self.p.signals):
            return

        signal = self.p.signals.iloc[self.signal_index]
        signal_time = pd.to_datetime(signal['timestamp'])

        if date >= signal_time:
            if self.order:
                return
            self.order = self.buy(size=1000, price=signal['entry'], exectype=bt.Order.Limit)
            self.order_add = self.sell(size=1000, price=signal['tp'], exectype=bt.Order.Limit, parent=self.order)
            self.order_add = self.sell(size=1000, price=signal['sl'], exectype=bt.Order.Stop, parent=self.order)
            self.signal_index += 1

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None

def run_backtest(symbol, df, signals):
    cerebro = bt.Cerebro()
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    cerebro.addstrategy(SignalStrategy, signals=signals)
    cerebro.broker.setcash(100000)
    cerebro.addsizer(bt.sizers.FixedSize, stake=1000)
    cerebro.run()
    return cerebro.broker.getvalue()

# ---------- Main Execution ----------
if __name__ == "__main__":
    # Scan crypto symbols
    loop = asyncio.get_event_loop()
    crypto_signals = loop.run_until_complete(scan_all_crypto_symbols())
    
    # Scan forex symbols
    forex_signals = scan_all_forex_symbols()
    
    # Combine signals
    all_signals = crypto_signals + forex_signals
    
    # Save signals to CSV
    if all_signals:
        signals_df = pd.DataFrame(all_signals)
        signals_df['timestamp'] = pd.Timestamp.now()
        signals_df.to_csv("trading_signals.csv", index=False)
    
    # Backtest (example: BTCUSDT)
    try:
        btc_df = loop.run_until_complete(fetch_ohlcv_kucoin_async("BTCUSDT", interval="5min"))
        btc_signals = pd.DataFrame([s for s in all_signals if s['symbol'] == 'BTCUSDT'])
        if not btc_signals.empty:
            final_value = run_backtest("BTCUSDT", btc_df, btc_signals)
            print(f"Backtest Result (BTCUSDT): Final Portfolio Value = {final_value}")
    except Exception as e:
        logging.error(f"Backtest Failed: {e}")
    
    # Display signals
    for s in all_signals:
        print(s)