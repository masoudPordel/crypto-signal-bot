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

# تنظیم لاگ‌ها
logging.basicConfig(filename="trading_errors.log", level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

# کلید API Alpha Vantage (سخت‌کد شده به درخواست شما)
ALPHA_VANTAGE_API_KEY = "8VL54YT3N656MW5T"

# ---------- دریافت نمادهای کریپتو از KuCoin ----------
@lru_cache(maxsize=1)
def get_all_symbols_kucoin():
    url = "https://api.kucoin.com/api/v1/symbols"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return [item["symbol"].replace("-", "") for item in data["data"] if item["symbol"].endswith("-USDT")]
    except Exception as e:
        logging.error(f"خطا در دریافت نمادهای KuCoin: {e}")
        return []

# ---------- دریافت داده‌های OHLCV کریپتو (ناهمگام) ----------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def fetch_ohlcv_kucoin_async(symbol, interval="5min", limit=100):
    async with aiohttp.ClientSession() as session:
        url = f"https://api.kucoin.com/api/v1/market/candles?type={interval}&symbol={symbol[:len(symbol)-4]}-USDT"
        async with session.get(url) as response:
            if response.status != 200:
                logging.error(f"خطا در دریافت داده‌های KuCoin برای {symbol}: {response.status}")
                return None
            raw_data = await response.json()
            if not raw_data["data"]:
                return None
            df = pd.DataFrame(raw_data["data"], columns=[
                "timestamp", "open", "close", "high", "low", "volume", "turnover"
            ])
            df = df.iloc[::-1]  # معکوس کردن برای ترتیب زمانی
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
            return df

# ---------- دریافت داده‌های OHLCV فارکس ----------
@sleep_and_retry
@limits(calls=5, period=60)  # 5 درخواست در دقیقه
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
            logging.error(f"داده‌های فارکس برای {from_symbol}{to_symbol} نامعتبر است")
            return None
        df = pd.DataFrame.from_dict(data[ts_key[0]], orient="index").sort_index()
        df = df.rename(columns={
            "1. open": "open", "2. high": "high", "3. low": "low", "4. close": "close"
        }).astype(float)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        logging.error(f"خطا در دریافت داده‌های فارکس برای {from_symbol}{to_symbol}: {e}")
        return None

# ---------- محاسبه اندیکاتورهای تکنیکال ----------
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
    rs = gain / loss.where(loss != 0, 1e-10)  # جلوگیری از تقسیم بر صفر
    return 100 - (100 / (1 + rs))

def compute_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

# ---------- تشخیص الگوهای پرایس اکشن پیشرفته ----------
def detect_advanced_price_action(df):
    last = df.iloc[-1]
    body = abs(last["close"] - last["open"])
    wick = last["high"] - last["low"]
    upper_shadow = last["high"] - max(last["close"], last["open"])
    lower_shadow = min(last["close"], last["open"]) - last["low"]

    if body < wick * 0.2:
        return "دوجی"
    elif lower_shadow > body * 2 and upper_shadow < body:
        return "پین بار صعودی"
    elif upper_shadow > body * 2 and lower_shadow < body:
        return "پین بار نزولی"
    return None

def detect_engulfing(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    if (last["close"] > last["open"] and prev["close"] < prev["open"] and
            last["open"] < prev["close"] and last["close"] > prev["open"]):
        return "الگوی پوشای صعودی"
    elif (last["close"] < last["open"] and prev["close"] > prev["open"] and
          last["open"] > prev["close"] and last["close"] < prev["open"]):
        return "الگوی پوشای نزولی"
    return None

def detect_trend(df):
    highs = df["high"].rolling(20).max()
    lows = df["low"].rolling(20).min()
    if df["close"].iloc[-1] > highs.iloc[-2] and df["close"].iloc[-2] > highs.iloc[-3]:
        return "روند صعودی"
    elif df["close"].iloc[-1] < lows.iloc[-2] and df["close"].iloc[-2] < lows.iloc[-3]:
        return "روند نزولی"
    return "بدون روند"

def detect_key_levels(df):
    support = df["low"].rolling(20).min().iloc[-1]
    resistance = df["high"].rolling(20).max().iloc[-1]
    last = df.iloc[-1]
    if abs(last["close"] - support) < last["close"] * 0.01:
        return "نزدیک به حمایت"
    elif abs(last["close"] - resistance) < last["close"] * 0.01:
        return "نزدیک به مقاومت"
    return None

# ---------- تحلیل چند تایم‌فریمی ----------
async def check_higher_tf_trend(symbol, higher_tf="1hour"):
    df_higher = await fetch_ohlcv_kucoin_async(symbol, interval=higher_tf)
    if df_higher is None:
        return False
    df_higher = compute_indicators(df_higher)
    return df_higher["close"].iloc[-1] > df_higher["EMA20"].iloc[-1]

# ---------- تولید سیگنال ----------
def generate_signal(symbol, df, interval="5min", is_crypto=True, min_confidence=70):
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
    if key_level == "نزدیک به حمایت":
        score += 1
    if trend == "روند صعودی":
        score += 1

    confidence = int((score / 7) * 100)
    if confidence < min_confidence:  # حداقل اطمینان قابل تنظیم
        return None

    close_price = df["close"].iloc[-1]
    signal_data = {
        "نماد": symbol,
        "قیمت ورود": round(close_price, 5),
        "هدف سود": round(close_price + 2 * atr, 5),
        "حد ضرر": round(close_price - 1.5 * atr, 5),
        "سطح اطمینان": confidence,
        "تحلیل": (
            f"RSI: {round(rsi,1)} | تقاطع EMA: {ema_cross} | MACD: {'مثبت' if macd > signal else 'منفی'} | "
            f"پرایس اکشن: {price_action or '-'} | {key_level or '-'} | {trend} | "
            f"حجم: {'بالا' if volume_spike else 'عادی'}"
        ),
        "تایم‌فریم": interval
    }

    if is_crypto:
        loop = asyncio.get_event_loop()
        if not loop.run_until_complete(check_higher_tf_trend(symbol)):
            return None

    return signal_data

# ---------- اسکن نمادهای کریپتو ----------
async def scan_all_crypto_symbols(min_confidence=70):
    TIMEFRAMES = ["5min", "15min", "1hour"]
    all_symbols = get_all_symbols_kucoin()
    signals = []

    async def scan_symbol(symbol, tf):
        try:
            df = await fetch_ohlcv_kucoin_async(symbol, interval=tf)
            signal = generate_signal(symbol, df, tf, is_crypto=True, min_confidence=min_confidence)
            return signal
        except Exception as e:
            logging.error(f"خطا برای {symbol}-{tf}: {e}")
            return None

    tasks = [scan_symbol(symbol, tf) for symbol in all_symbols for tf in TIMEFRAMES]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    signals = [r for r in results if r and not isinstance(r, Exception)]
    return signals

# ---------- اسکن نمادهای فارکس ----------
def scan_all_forex_symbols(min_confidence=70):
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
            signal = generate_signal(symbol, df, is_crypto=False, min_confidence=min_confidence)
            return signal
        except Exception as e:
            logging.error(f"خطا برای {base}{quote}: {e}")
            return None

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(scan_forex_pair, pairs))
    signals = [r for r in results if r]
    return signals

# ---------- بک‌تست استراتژی ----------
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
            self.order = self.buy(size=1000, price=signal['قیمت ورود'], exectype=bt.Order.Limit)
            self.order_add = self.sell(size=1000, price=signal['هدف سود'], exectype=bt.Order.Limit, parent=self.order)
            self.order_add = self.sell(size=1000, price=signal['حد ضرر'], exectype=bt.Order.Stop, parent=self.order)
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

# ---------- اجرای اصلی ----------
if __name__ == "__main__":
    # تنظیم حداقل اطمینان برای تولید سیگنال (کاهش به 70 برای سیگنال‌های بیشتر)
    MIN_CONFIDENCE = 70

    print("در حال اسکن نمادهای کریپتو...")
    loop = asyncio.get_event_loop()
    crypto_signals = loop.run_until_complete(scan_all_crypto_symbols(min_confidence=MIN_CONFIDENCE))
    
    print("در حال اسکن نمادهای فارکس...")
    forex_signals = scan_all_forex_symbols(min_confidence=MIN_CONFIDENCE)
    
    # ترکیب سیگنال‌ها
    all_signals = crypto_signals + forex_signals
    
    # ذخیره سیگنال‌ها در CSV
    if all_signals:
        signals_df = pd.DataFrame(all_signals)
        signals_df['زمان'] = pd.Timestamp.now()
        signals_df.to_csv("trading_signals.csv", index=False)
        print(f"تعداد سیگنال‌های تولیدشده: {len(all_signals)}")
    
    # بک‌تست (مثال: BTCUSDT)
    try:
        btc_df = loop.run_until_complete(fetch_ohlcv_kucoin_async("BTCUSDT", interval="5min"))
        btc_signals = pd.DataFrame([s for s in all_signals if s['نماد'] == 'BTCUSDT'])
        if not btc_signals.empty:
            final_value = run_backtest("BTCUSDT", btc_df, btc_signals)
            print(f"نتیجه بک‌تست (BTCUSDT): ارزش نهایی حساب = {final_value}")
    except Exception as e:
        logging.error(f"خطا در بک‌تست: {e}")
    
    # نمایش سیگنال‌ها
    if all_signals:
        for s in all_signals:
            print("\n📈 سیگنال جدید:")
            print(f"نماد: {s['نماد']}")
            print(f"تایم‌فریم: {s['تایم‌فریم']}")
            print(f"قیمت ورود: {s['قیمت ورود']}")
            print(f"هدف سود: {s['هدف سود']}")
            print(f"حد ضرر: {s['حد ضرر']}")
            print(f"سطح اطمینان: {s['سطح اطمینان']}%")
            print(f"تحلیل: {s['تحلیل']}")
    else:
        print("هیچ سیگنالی تولید نشد! ممکن است داده‌ها ناکافی باشند یا فیلترها بیش از حد سخت‌گیرانه باشند.")