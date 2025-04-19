import requests
import pandas as pd
import aiohttp
import asyncio
from ratelimit import limits, sleep_and_retry
from tenacity import retry, stop_after_attempt, wait_exponential
from functools import lru_cache
import logging
import numpy as np
import backtrader as bt
import os

# تنظیم لاگ‌ها برای دیباگ دقیق‌تر
logging.basicConfig(filename="trading_errors.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# تنظیم API Key از متغیر محیطی
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "8VL54YT3N656MW5T")

# تنظیم پروکسی برای دور زدن فیلتر (در صورت نیاز)
PROXY = os.getenv("HTTP_PROXY", None)  # مثال: "http://user:pass@proxy:port"

@lru_cache(maxsize=1)
def get_all_symbols_kucoin(_=None):
    try:
        url = "https://api.kucoin.com/api/v1/symbols"
        proxies = {"http": PROXY, "https": PROXY} if PROXY else None
        response = requests.get(url, proxies=proxies)
        response.raise_for_status()
        data = response.json()
        if not data.get("data"):
            logging.error("No symbols data returned from KuCoin")
            return []
        symbols = [item["symbol"].replace("-", "") for item in data["data"] if item["symbol"].endswith("-USDT")]
        logging.info(f"Fetched {len(symbols)} symbols from KuCoin")
        return symbols
    except Exception as e:
        logging.error(f"خطا در دریافت نمادها از KuCoin: {e}")
        return []

@sleep_and_retry
@limits(calls=10, period=1)  # محدودیت KuCoin: 10 درخواست در ثانیه
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
async def fetch_ohlcv_kucoin_async(symbol, interval="5min", limit=200):
    async with aiohttp.ClientSession() as session:
        url = f"https://api.kucoin.com/api/v1/market/candles?type={interval}&symbol={symbol[:-4]}-USDT"
        proxy = PROXY if PROXY else None
        async with session.get(url, proxy=proxy) as response:
            if response.status != 200:
                logging.error(f"خطا در دریافت داده از KuCoin برای {symbol}: {response.status}")
                return None
            data = await response.json()
            if not data.get("data"):
                logging.error(f"داده‌ای برای {symbol} از KuCoin دریافت نشد")
                return None
            df = pd.DataFrame(data["data"], columns=["timestamp", "open", "close", "high", "low", "volume", "turnover"])
            df = df.iloc[::-1]
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
            logging.info(f"Fetched {len(df)} candles for {symbol} from KuCoin")
            return df

@sleep_and_retry
@limits(calls=5, period=60)  # محدودیت Alpha Vantage: 5 درخواست در دقیقه
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_forex_ohlcv(from_symbol, to_symbol="USD", interval="5min"):
    try:
        url = (
            f"https://www.alphavantage.co/query?function=FX_INTRADAY"
            f"&from_symbol={from_symbol}&to_symbol={to_symbol}"
            f"&interval={interval}&outputsize=compact&apikey={ALPHA_VANTAGE_API_KEY}"
        )
        proxies = {"http": PROXY, "https": PROXY} if PROXY else None
        response = requests.get(url, proxies=proxies)
        response.raise_for_status()
        data = response.json()
        ts_key = [k for k in data if "Time Series" in k]
        if not ts_key:
            logging.error(f"داده زمانی برای {from_symbol}/{to_symbol} از Alpha Vantage دریافت نشد")
            return None
        df = pd.DataFrame.from_dict(data[ts_key[0]], orient="index").sort_index()
        df = df.rename(columns={
            "1. open": "open", "2. high": "high", "3. low": "low", "4. close": "close"
        }).astype(float)
        df.index = pd.to_datetime(df.index)
        logging.info(f"Fetched {len(df)} candles for {from_symbol}/{to_symbol} from Alpha Vantage")
        return df
    except Exception as e:
        logging.error(f"خطا در دریافت داده فارکس برای {from_symbol}/{to_symbol}: {e}")
        return None

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
def generate_signal(symbol, df, interval="5min", is_crypto=True, min_confidence=20):  # کاهش سطح اطمینان
    if df is None:
        logging.error(f"داده‌ای برای {symbol} دریافت نشد")
        return None
    if len(df) < 50:
        logging.error(f"تعداد کندل‌ها برای {symbol} کمتر از 50 است: {len(df)}")
        return None
    df = compute_indicators(df)

    rsi = df["RSI"].iloc[-1]
    macd, signal = df["MACD"].iloc[-1], df["Signal"].iloc[-1]
    ema_cross = df["EMA12"].iloc[-2] < df["EMA26"].iloc[-2] and df["EMA12"].iloc[-1] > df["EMA26"].iloc[-1]
    volume_spike = df["volume"].iloc[-1] > df["volume"].rolling(20).mean().iloc[-1] * 1.2 if "volume" in df.columns else False
    atr = df["ATR"].iloc[-1]
    close = df["close"].iloc[-1]

    score = sum([
        rsi < 45,
        macd > signal,
        ema_cross,
        bool(detect_engulfing(df) or detect_advanced_price_action(df)),
        volume_spike,
        bool(breakout_strategy(df)),
        bool(bollinger_strategy(df))
    ])

    confidence = int((score / 7) * 100)
    if confidence < min_confidence:
        logging.info(f"سطح اطمینان برای {symbol} کمتر از حداقل است: {confidence}%")
        return None

    signal = {
        "نماد": symbol,
        "قیمت ورود": round(close, 5),
        "هدف سود": round(close + 2 * atr, 5),
        "حد ضرر": round(close - 1.5 * atr, 5),
        "سطح اطمینان": confidence,
        "تحلیل": f"RSI={round(rsi,1)}, EMA کراس={ema_cross}, MACD={'مثبت' if macd > signal else 'منفی'}, "
                 f"الگو={detect_engulfing(df) or detect_advanced_price_action(df) or '-'}, {detect_trend(df)}, "
                 f"{breakout_strategy(df) or '-'}, {bollinger_strategy(df) or '-'}, "
                 f"حجم={'بالا' if volume_spike else 'نرمال'}",
        "تایم‌فریم": interval
    }
    logging.info(f"سیگنال برای {symbol} با اطمینان {confidence}% تولید شد")
    return signal

# --- Backtest ---
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
        if pd.isnull(signal).any():
            self.signal_index += 1
            return
        try:
            signal_time = pd.to_datetime(signal['زمان'])
            if date >= signal_time:
                if not self.order:
                    self.order = self.buy(size=1000, price=signal['قیمت ورود'], exectype=bt.Order.Limit)
                    self.sell(size=1000, price=signal['هدف سود'], exectype=bt.Order.Limit, parent=self.order)
                    self.sell(size=1000, price=signal['حد ضرر'], exectype=bt.Order.Stop, parent=self.order)
                    self.signal_index += 1
        except Exception as e:
            logging.error(f"خطا در بک‌تست برای سیگنال {self.signal_index}: {e}")
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
    final_value = cerebro.broker.getvalue()
    logging.info(f"ارزش نهایی پرتفوی برای {symbol}: {final_value}")
    return final_value

# --- تابع برای دریافت داده و تولید سیگنال برای فارکس ---
async def analyze_forex_pair(from_symbol, to_symbol="USD", interval="5min"):
    symbol = f"{from_symbol}/{to_symbol}"
    df = fetch_forex_ohlcv(from_symbol, to_symbol, interval)
    if df is None:
        return None
    signal = generate_signal(symbol, df, interval=interval, is_crypto=False, min_confidence=20)
    return signal

# --- تست کد برای EUR/USD ---
if __name__ == "__main__":
    # تست برای جفت‌ارز EUR/USD
    symbol = "EUR/USD"
    loop = asyncio.get_event_loop()
    signal = loop.run_until_complete(analyze_forex_pair("EUR", "USD", "5min"))
    if signal:
        print(f"سیگنال برای {symbol}: {signal}")
        # بک‌تست
        df = fetch_forex_ohlcv("EUR", "USD", "5min")
        signals = pd.DataFrame([signal])
        signals['زمان'] = df.index[-1]
        final_value = run_backtest(symbol, df, signals)
        print(f"ارزش نهایی پرتفوی: {final_value}")
    else:
        print(f"هیچ سیگنالی برای {symbol} تولید نشد")