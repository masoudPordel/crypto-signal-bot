import os
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["LC_ALL"] = "en_US.UTF-8"

import requests
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import ccxt.async_support as ccxt
import asyncio
import time
import logging
from datetime import datetime, timedelta
from sklearn.tree import DecisionTreeClassifier  # برای Decision Tree

# تنظیمات لاگ
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
    force=True,
    encoding="utf-8"
)

# کلیدهای API
CMC_API_KEY = "7fc7dc4d-2d30-4c83-9836-875f9e0f74c7"
COINGECKO_API_KEY = "CG-cnXmskNzo7Bi2Lzj3j3QY6Gu"  # کلید خودت رو جایگزین کن
TIMEFRAMES = ["30m", "1h", "4h", "1d"]  # حذف 5m و 15m

# پارامترهای اصلی
VOLUME_WINDOW = 15
S_R_BUFFER = 0.01  # کاهش به 0.015
ADX_THRESHOLD = 30
ADX_TREND_THRESHOLD = 25
CACHE = {}
CACHE_TTL = 300
VOLUME_THRESHOLD = 2  # کاهش از 5 به 3
MAX_CONCURRENT_REQUESTS = 20
WAIT_BETWEEN_REQUESTS = 0.5  # افزایش به 1 ثانیه
WAIT_BETWEEN_CHUNKS = 3
VOLATILITY_THRESHOLD = 0.004
LIQUIDITY_SPREAD_THRESHOLD = 0.005

# ضرایب مقیاس‌پذیری حجم
VOLUME_SCALING = {
    "30m": 0.05,  # کاهش از 0.1
    "1h": 0.08,
    "4h": 0.2,   # کاهش از 0.3
    "1d": 0.3    # کاهش از 0.4
}

# دریافت ۵۰۰ نماد برتر از CoinMarketCap
def get_top_500_symbols_from_cmc():
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    headers = {'Accepts': 'application/json', 'X-CMC_PRO_API_KEY': CMC_API_KEY}
    params = {'start': '1', 'limit': '500', 'convert': 'USD'}  # 500 نماد
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        data = resp.json()
        return [entry['symbol'] for entry in data['data']]
    except Exception as e:
        logging.error(f"خطا در دریافت از CMC: {e}")
        return []

# کلاس برای مدیریت اندیکاتورها
class IndicatorCalculator:
    @staticmethod
    def compute_rsi(df, period=14):
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        rs = gain / loss.replace(0, 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def compute_atr(df, period=14):
        tr = pd.concat([df["high"] - df["low"], abs(df["high"] - df["close"].shift()), abs(df["low"] - df["close"].shift())], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def compute_bollinger_bands(df, period=20, std_dev=2):
        sma = df["close"].rolling(period).mean()
        std = df["close"].rolling(period).std()
        return sma + std_dev * std, sma - std_dev * std

    @staticmethod
    def compute_adx(df, period=14):
        df["up"] = df["high"].diff()
        df["down"] = -df["low"].diff()
        df["+DM"] = np.where((df["up"] > df["down"]) & (df["up"] > 0), df["up"], 0.0)
        df["-DM"] = np.where((df["down"] > df["up"]) & (df["down"] > 0), df["down"], 0.0)
        tr = pd.concat([df["high"] - df["low"], abs(df["high"] - df["close"].shift()), abs(df["low"] - df["close"].shift())], axis=1).max(axis=1)
        tr_smooth = tr.rolling(window=period).sum()
        plus_di = 100 * (df["+DM"].rolling(window=period).sum() / tr_smooth)
        minus_di = 100 * (df["-DM"].rolling(window=period).sum() / tr_smooth)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        return dx.rolling(window=period).mean()

    @staticmethod
    def compute_stochastic(df, period=14):
        low_min = df["low"].rolling(window=period).min()
        high_max = df["high"].rolling(window=period).max()
        k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, 1e-10)
        return k

# تشخیص الگوها
class PatternDetector:
    @staticmethod
    def detect_pin_bar(df):
        df["body"] = abs(df["close"] - df["open"])
        df["range"] = df["high"] - df["low"]
        df["upper"] = df["high"] - df[["close", "open"]].max(axis=1)
        df["lower"] = df[["close", "open"]].min(axis=1) - df["low"]
        return (df["body"] < 0.3 * df["range"]) & ((df["upper"] > 2 * df["body"]) | (df["lower"] > 2 * df["body"]))

    @staticmethod
    def detect_engulfing(df):
        prev_o = df["open"].shift(1)
        prev_c = df["close"].shift(1)
        return (((df["close"] > df["open"]) & (prev_c < prev_o) & (df["close"] > prev_o) & (df["open"] < prev_c)) |
                ((df["close"] < df["open"]) & (prev_c > prev_o) & (df["close"] < prev_o) & (df["open"] > prev_c)))

    @staticmethod
    def detect_elliott_wave(df):
        df["WavePoint"] = np.nan
        highs = argrelextrema(df['close'].values, np.greater, order=5)[0]
        lows = argrelextrema(df['close'].values, np.less, order=5)[0]
        df.loc[df.index[highs], "WavePoint"] = df.loc[df.index[highs], "close"]
        df.loc[df.index[lows], "WavePoint"] = df.loc[df.index[lows], "close"]
        return df

    @staticmethod
    def detect_support_resistance(df, window=10):
        high = df['high'].rolling(window).max()
        low = df['low'].rolling(window).min()
        close = df['close'].rolling(window).mean()
        pivot = (high + low + close) / 3
        resistance = pivot + (high - low) * 0.382
        support = pivot - (high - low) * 0.382
        volume_profile = df['volume'].groupby(df['close'].round(2)).sum()
        vol_threshold = volume_profile.quantile(0.75)
        high_vol_levels = volume_profile[volume_profile > vol_threshold].index
        recent_highs = df['high'][(df['high'].shift(1) < df['high']) & (df['high'].shift(-1) < df['high'])].iloc[-window:]
        recent_lows = df['low'][(df['low'].shift(1) > df['low']) & (df['low'].shift(-1) > df['low'])].iloc[-window:]
        recent_resistance = recent_highs.max() if not recent_highs.empty else resistance
        recent_support = recent_lows.min() if not recent_lows.empty else support
        if 'support_levels' not in globals(): globals()['support_levels'] = []
        if 'resistance_levels' not in globals(): globals()['resistance_levels'] = []
        if recent_support not in support_levels: support_levels.append(recent_support)
        if recent_resistance not in resistance_levels: resistance_levels.append(recent_resistance)
        return recent_support, recent_resistance, high_vol_levels

    @staticmethod
    def detect_hammer(df):
        body = abs(df['close'] - df['open'])
        range_ = df['high'] - df['low']
        lower_wick = df['low'] - df[['close', 'open']].min(axis=1)
        return (body < 0.3 * range_) & (lower_wick > 2 * body) & (df['close'] > df['open'])

    @staticmethod
    def detect_rsi_divergence(df, lookback=5):
        rsi = IndicatorCalculator.compute_rsi(df)
        prices = df['close']
        recent_lows_price = argrelextrema(prices.values, np.less, order=lookback)[0]
        recent_highs_price = argrelextrema(prices.values, np.greater, order=lookback)[0]
        recent_lows_rsi = argrelextrema(rsi.values, np.less, order=lookback)[0]
        recent_highs_rsi = argrelextrema(rsi.values, np.greater, order=lookback)[0]
        # واگرایی صعودی (برای Long)
        if len(recent_lows_price) > 1 and len(recent_lows_rsi) > 1:
            last_price_low = prices.iloc[recent_lows_price[-1]]
            prev_price_low = prices.iloc[recent_lows_price[-2]]
            last_rsi_low = rsi.iloc[recent_lows_rsi[-1]]
            prev_rsi_low = rsi.iloc[recent_lows_rsi[-2]]
            bullish_divergence = last_price_low < prev_price_low and last_rsi_low > prev_rsi_low
        else:
            bullish_divergence = False
        # واگرایی نزولی (برای Short)
        if len(recent_highs_price) > 1 and len(recent_highs_rsi) > 1:
            last_price_high = prices.iloc[recent_highs_price[-1]]
            prev_price_high = prices.iloc[recent_highs_price[-2]]
            last_rsi_high = rsi.iloc[recent_highs_rsi[-1]]
            prev_rsi_high = rsi.iloc[recent_highs_rsi[-2]]
            bearish_divergence = last_price_high > prev_price_high and last_rsi_high < prev_rsi_high
        else:
            bearish_divergence = False
        return bullish_divergence, bearish_divergence

    @staticmethod
    def detect_macd_divergence(df, lookback=5):
        macd = df["MACD"]
        prices = df['close']
        recent_lows_price = argrelextrema(prices.values, np.less, order=lookback)[0]
        recent_highs_price = argrelextrema(prices.values, np.greater, order=lookback)[0]
        recent_lows_macd = argrelextrema(macd.values, np.less, order=lookback)[0]
        recent_highs_macd = argrelextrema(macd.values, np.greater, order=lookback)[0]
        # واگرایی صعودی (برای Long)
        if len(recent_lows_price) > 1 and len(recent_lows_macd) > 1:
            last_price_low = prices.iloc[recent_lows_price[-1]]
            prev_price_low = prices.iloc[recent_lows_price[-2]]
            last_macd_low = macd.iloc[recent_lows_macd[-1]]
            prev_macd_low = macd.iloc[recent_lows_macd[-2]]
            bullish_divergence = last_price_low < prev_price_low and last_macd_low > prev_macd_low
        else:
            bullish_divergence = False
        # واگرایی نزولی (برای Short)
        if len(recent_highs_price) > 1 and len(recent_highs_macd) > 1:
            last_price_high = prices.iloc[recent_highs_price[-1]]
            prev_price_high = prices.iloc[recent_highs_price[-2]]
            last_macd_high = macd.iloc[recent_highs_macd[-1]]
            prev_macd_high = macd.iloc[recent_highs_macd[-2]]
            bearish_divergence = last_price_high > prev_price_high and last_macd_high < prev_macd_high
        else:
            bearish_divergence = False
        return bullish_divergence, bearish_divergence

    @staticmethod
    def is_support_broken(df, support, lookback=2):
        recent_closes = df['close'].iloc[-lookback:]
        return all(recent_closes < support)

    @staticmethod
    def is_resistance_broken(df, resistance, lookback=2):
        recent_closes = df['close'].iloc[-lookback:]
        return all(recent_closes > resistance)

    @staticmethod
    def is_valid_breakout(df, level, direction="support", vol_threshold=2.0):
        last_vol = df['volume'].iloc[-1]
        vol_avg = df['volume'].rolling(VOLUME_WINDOW).mean().iloc[-1]
        if last_vol < vol_threshold * vol_avg:
            logging.warning(f"شکست رد شد: حجم ناکافی (current={last_vol}, threshold={vol_threshold * vol_avg})")
            return False
        if direction == "support" and not PatternDetector.is_support_broken(df, level):
            return False
        if direction == "resistance" and not PatternDetector.is_resistance_broken(df, level):
            return False
        last_candle = df.iloc[-1]
        body = abs(last_candle['close'] - last_candle['open'])
        wick_lower = min(last_candle['close'], last_candle['open']) - last_candle['low']
        wick_upper = last_candle['high'] - max(last_candle['close'], last_candle['open'])
        # فیلتر کندلی: کندل باید بدنه بزرگ و فتیله کوچک داشته باشه
        if body < 0.5 * (last_candle['high'] - last_candle['low']) or wick_lower > 0.3 * body or wick_upper > 0.3 * body:
            logging.warning(f"شکست رد شد: کندل ضعیف (body={body}, wick_lower={wick_lower}, wick_upper={wick_upper})")
            return False
        if len(df) > 2 and direction == "support" and df.iloc[-2]['close'] >= level:
            logging.warning(f"شکست رد شد: قیمت به بالای حمایت برگشته (close={df.iloc[-2]['close']}, support={level})")
            return False
        if len(df) > 2 and direction == "resistance" and df.iloc[-2]['close'] <= level:
            logging.warning(f"شکست رد شد: قیمت به زیر مقاومت برگشته (close={df.iloc[-2]['close']}, resistance={level})")
            return False
        return True

# فیلتر نهایی با Decision Tree
class SignalFilter:
    def __init__(self):
        self.model = DecisionTreeClassifier(max_depth=3)
        self.trained = False

    def train(self, X, y):
        if len(X) > 0 and len(y) > 0:
            self.model.fit(X, y)
            self.trained = True
            logging.info("Decision Tree آموزش داده شد.")

    def predict(self, features):
        if not self.trained:
            logging.warning("Decision Tree آموزش داده نشده است. سیگنال تأیید می‌شود.")
            return True
        return self.model.predict([features])[0]

# بررسی نقدینگی
async def check_liquidity(exchange, symbol):
    try:
        ticker = await exchange.fetch_ticker(symbol)
        bid = ticker['bid']
        ask = ticker['ask']
        spread = (ask - bid) / ((bid + ask) / 2)
        if spread >= LIQUIDITY_SPREAD_THRESHOLD:
            logging.warning(f"رد {symbol}: نقدینگی کافی نیست (spread={spread:.4f}, threshold={LIQUIDITY_SPREAD_THRESHOLD})")
            return False
        return True
    except Exception as e:
        logging.error(f"خطا در بررسی نقدینگی برای {symbol}: {e}")
        return False

# بررسی رویدادهای بازار
def check_market_events(symbol):
    url = f"https://api.coingecko.com/api/v3/coins/{symbol.lower()}?x_cg_api_key={COINGECKO_API_KEY}"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if 'market_data' in data and 'last_updated' in data:
            return "ندارد"
        return "رویداد مهم"
    except Exception as e:
        logging.error(f"خطا در دریافت از CoinGecko: {e}")
        return "ندارد"

# محاسبات اندیکاتورها
def compute_indicators(df):
    df["EMA12"] = df["close"].ewm(span=12).mean()
    df["EMA26"] = df["close"].ewm(span=26).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    df["RSI"] = IndicatorCalculator.compute_rsi(df)
    df["ATR"] = IndicatorCalculator.compute_atr(df)
    df["ADX"] = IndicatorCalculator.compute_adx(df)
    df["Stochastic"] = IndicatorCalculator.compute_stochastic(df)
    df["BB_upper"], df["BB_lower"] = IndicatorCalculator.compute_bollinger_bands(df)
    df["PinBar"] = PatternDetector.detect_pin_bar(df)
    df["Engulfing"] = PatternDetector.detect_engulfing(df)
    df = PatternDetector.detect_elliott_wave(df)
    return df

# تأیید اندیکاتورهای ترکیبی
def confirm_combined_indicators(df, trend_type):
    last = df.iloc[-1]
    rsi = last["RSI"]
    stochastic = last["Stochastic"]
    bullish_engulf = last["Engulfing"] and last["close"] > last["open"]
    bearish_engulf = last["Engulfing"] and last["close"] < last["open"]
    macd_cross_long = df["MACD"].iloc[-2] < df["Signal"].iloc[-2] and df["MACD"].iloc[-1] > df["Signal"].iloc[-1]
    macd_cross_short = df["MACD"].iloc[-2] > df["Signal"].iloc[-2] and df["MACD"].iloc[-1] < df["Signal"].iloc[-1]
    if trend_type == "Long":
        conditions = [rsi < 30, macd_cross_long, bullish_engulf]
        return sum(conditions) >= 1
    else:
        conditions = [rsi > 70, macd_cross_short, bearish_engulf]
        return sum(conditions) >= 1

# مدیریت ریسک و اندازه پوزیشن
def calculate_position_size(account_balance, risk_percentage, entry, stop_loss):
    risk_amount = account_balance * (risk_percentage / 100)
    distance = abs(entry - stop_loss)
    position_size = risk_amount / distance
    return round(position_size, 2)

# تأیید مولتی تایم‌فریم
async def multi_timeframe_confirmation(df, symbol, exchange):
    weights = {"1d": 0.4, "4h": 0.3, "1h": 0.2, "30m": 0.1}
    total_weight = 0
    trend_score = 0
    for tf, weight in weights.items():
        df_tf = await get_ohlcv_cached(exchange, symbol, tf)
        if df_tf is not None and len(df_tf) >= 50:
            long_trend = df_tf["EMA12"].iloc[-1] > df_tf["EMA26"].iloc[-1]
            trend_score += weight if long_trend else -weight
        total_weight += weight
    return abs(trend_score / total_weight) >= 0.5

# دریافت داده‌های کندل با استفاده بهتر از CACHE
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
async def get_ohlcv_cached(exchange, symbol, tf, limit=100):
    async with semaphore:
        await asyncio.sleep(WAIT_BETWEEN_REQUESTS)
        key = f"{symbol}_{tf}"
        now = time.time()
        if key in CACHE and now - CACHE[key]["time"] < CACHE_TTL:
            logging.info(f"استفاده از حافظه پنهان برای {symbol} @ {tf}")
            return CACHE[key]["data"]
        try:
            data = await exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            CACHE[key] = {"data": df.copy(), "time": now}
            return df
        except Exception as e:
            logging.error(f"خطا در دریافت داده برای {symbol}-{tf}: {e}")
            return None

# آموزش Decision Tree (برای تست اولیه)
signal_filter = SignalFilter()
# دیتای نمونه برای آموزش (در عمل باید داده واقعی بدی)
X_train = np.array([[30, 25, 2], [70, 20, 1], [50, 30, 1.5], [20, 40, 3]])  # [RSI, ADX, Volume Ratio]
y_train = np.array([1, 0, 0, 1])  # 1 = سیگنال خوب، 0 = سیگنال بد
signal_filter.train(X_train, y_train)

# تحلیل نماد با فیلترهای جدید
async def analyze_symbol(exchange, symbol, tf):
    logging.info(f"شروع تحلیل {symbol} @ {tf}")
    df = await get_ohlcv_cached(exchange, symbol, tf)
    if df is None or len(df) < 50:
        logging.warning(f"رد {symbol} @ {tf}: داده کافی نیست (<50)")
        return None

    vol_avg = df["volume"].rolling(VOLUME_WINDOW).mean().iloc[-1]
    scale_factor = VOLUME_SCALING.get(tf, 0.2)
    dynamic_threshold = max(VOLUME_THRESHOLD, vol_avg * scale_factor)
    if df["volume"].iloc[-1] < dynamic_threshold:
        if df["volume"].iloc[-1] < vol_avg * 0.1:
            logging.warning(f"رد {symbol} @ {tf}: حجم خیلی کم (current={df['volume'].iloc[-1]}, threshold={dynamic_threshold}, vol_avg={vol_avg})")
        else:
            logging.warning(f"رد {symbol} @ {tf}: حجم کم (current={df['volume'].iloc[-1]}, threshold={dynamic_threshold}, vol_avg={vol_avg})")
        return None

    df = compute_indicators(df)
    last = df.iloc[-1]
    long_trend = df["EMA12"].iloc[-1] > df["EMA26"].iloc[-1]
    short_trend = not long_trend

    if tf == "1h":
        df4 = await get_ohlcv_cached(exchange, symbol, "4h")
        df1d = await get_ohlcv_cached(exchange, symbol, "1d")
        if df4 is not None and len(df4) >= 50 and df1d is not None and len(df1d) >= 50:
            e12_4 = df4["close"].ewm(span=12).mean().iloc[-1]
            e26_4 = df4["close"].ewm(span=26).mean().iloc[-1]
            e12_1d = df1d["close"].ewm(span=12).mean().iloc[-1]
            e26_1d = df1d["close"].ewm(span=26).mean().iloc[-1]
            trend4 = e12_4 > e26_4
            trend1d = e12_1d > e26_1d
            if long_trend and (not trend4 or not trend1d):
                logging.warning(f"رد {symbol} @ {tf}: عدم تطابق روند چند تایم‌فریمی")
                return None
            if short_trend and (trend4 or trend1d):
                logging.warning(f"رد {symbol} @ {tf}: عدم تطابق روند چند تایم‌فریمی")
                return None
        else:
            logging.warning(f"رد {symbol} @ {tf}: داده چند تایم‌فریمی کافی نیست")
            return None

    volatility = df["ATR"].iloc[-1] / df["close"].iloc[-1]
    if volatility < VOLATILITY_THRESHOLD:
        logging.warning(f"رد {symbol} @ {tf}: نوسان خیلی کم (current={volatility:.4f})")
        return None

    support, resistance, vol_levels = PatternDetector.detect_support_resistance(df)
    if long_trend and abs(last["close"] - resistance) / last["close"] < S_R_BUFFER:
        logging.warning(f"رد {symbol} @ {tf}: خیلی نزدیک به مقاومت (distance={abs(last['close'] - resistance)/last['close']:.4f})")
        return None
    if short_trend and abs(last["close"] - support) / last["close"] < S_R_BUFFER:
        logging.warning(f"رد {symbol} @ {tf}: خیلی نزدیک به حمایت (distance={abs(last['close'] - support)/last['close']:.4f})")
        return None

    liquidity = await check_liquidity(exchange, symbol)
    if not liquidity:
        logging.warning(f"رد {symbol} @ {tf}: نقدینگی کافی نیست")
        return None

    fundamental = check_market_events(symbol.split('/')[0])
    if fundamental == "رویداد مهم":
        logging.warning(f"رد {symbol} @ {tf}: رویداد مهم بازار")
        return None

    # فیلترهای تأیید نهایی
    if last["ADX"] < ADX_THRESHOLD / 2:
        logging.warning(f"رد {symbol} @ {tf}: ADX خیلی پایین (current={last['ADX']:.2f})")
        return None
    if 40 <= last["RSI"] <= 60:
        logging.warning(f"رد {symbol} @ {tf}: RSI در محدوده خنثی (current={last['RSI']:.2f})")
        return None

    body = last["body"]
    bullish_pin = last["PinBar"] and last["lower"] > 2 * body
    bearish_pin = last["PinBar"] and last["upper"] > 2 * body
    bullish_engulf = last["Engulfing"] and last["close"] > last["open"]
    bearish_engulf = last["Engulfing"] and last["close"] < last["open"]

    rsi = last["RSI"]
    stochastic = last["Stochastic"]
    psych_long = "اشباع فروش" if rsi < 40 else "اشباع خرید" if rsi > 60 else "متعادل"
    psych_short = "اشباع خرید" if rsi > 60 else "اشباع فروش" if rsi < 40 else "متعادل"

    # تشخیص واگرایی
    bullish_rsi_div, bearish_rsi_div = PatternDetector.detect_rsi_divergence(df)
    bullish_macd_div, bearish_macd_div = PatternDetector.detect_macd_divergence(df)

    conds_long = {
        "PinBar": bullish_pin,
        "Engulfing": bullish_engulf,
        "EMA_Cross": df["EMA12"].iloc[-2] < df["EMA26"].iloc[-2] and long_trend,
        "MACD_Cross": df["MACD"].iloc[-2] < df["Signal"].iloc[-2] and df["MACD"].iloc[-1] > df["Signal"].iloc[-1],
        "RSI_Oversold": rsi < 30,
        "Stochastic_Oversold": stochastic < 20,
        "ADX_StrongTrend": last["ADX"] > ADX_THRESHOLD,
        "RSI_Divergence": bullish_rsi_div,
        "MACD_Divergence": bullish_macd_div,
    }
    conds_short = {
        "PinBar": bearish_pin,
        "Engulfing": bearish_engulf,
        "EMA_Cross": df["EMA12"].iloc[-2] > df["EMA26"].iloc[-2] and short_trend,
        "MACD_Cross": df["MACD"].iloc[-2] > df["Signal"].iloc[-2] and df["MACD"].iloc[-1] < df["Signal"].iloc[-1],
        "RSI_Overbought": rsi > 70,
        "Stochastic_Overbought": stochastic > 80,
        "ADX_StrongTrend": last["ADX"] > ADX_THRESHOLD,
        "RSI_Divergence": bearish_rsi_div,
        "MACD_Divergence": bearish_macd_div,
    }

    score_long = sum(conds_long.values())
    score_short = sum(conds_short.values())
    has_trend = last["ADX"] > ADX_TREND_THRESHOLD

    # فیلتر Decision Tree
    features = [rsi, last["ADX"], last["volume"] / vol_avg]

    if score_long >= 3 and psych_long != "اشباع خرید" and (long_trend or (psych_long == "اشباع فروش" and last["ADX"] < ADX_THRESHOLD)) and has_trend and confirm_combined_indicators(df, "Long") and await multi_timeframe_confirmation(df, symbol, exchange):
        # بررسی شکست مقاومت برای Long
        if not PatternDetector.is_valid_breakout(df, resistance, direction="resistance"):
            logging.warning(f"رد {symbol} @ {tf}: شکست مقاومت نامعتبر")
            return None
        # فیلتر نهایی با Decision Tree
        if not signal_filter.predict(features):
            logging.warning(f"رد {symbol} @ {tf}: فیلتر Decision Tree رد شد")
            return None
        entry = float(last["close"])
        atr_avg = df["ATR"].rolling(5).mean().iloc[-1]
        sl = entry - 2 * atr_avg
        tp = entry + 3 * atr_avg
        rr = round((tp - entry) / (entry - sl), 2)
        position_size = calculate_position_size(10000, 1, entry, sl)
        signal = {
            "نوع معامله": "Long",
            "نماد": symbol,
            "تایم‌فریم": tf,
            "قیمت ورود": entry,
            "حد ضرر": sl,
            "هدف سود": tp,
            "ریسک به ریوارد": rr,
            "حجم پوزیشن": position_size,
            "سطح اطمینان": min(score_long * 20, 100),
            "تحلیل": " | ".join([k for k, v in conds_long.items() if v]),
            "روانشناسی": psych_long,
            "روند بازار": "صعودی",
            "فاندامنتال": fundamental
        }
        logging.info(f"سیگنال تولید شد: {signal}")
        return signal

    if score_short >= 3 and psych_short != "اشباع فروش" and (short_trend or (psych_short == "اشباع خرید" and last["ADX"] < ADX_THRESHOLD)) and has_trend and confirm_combined_indicators(df, "Short") and await multi_timeframe_confirmation(df, symbol, exchange):
        # بررسی شکست حمایت برای Short
        if not PatternDetector.is_valid_breakout(df, support, direction="support"):
            logging.warning(f"رد {symbol} @ {tf}: شکست حمایت نامعتبر")
            return None
        if bearish_rsi_div or bearish_macd_div or PatternDetector.detect_hammer(df) or (last["Engulfing"] and last["close"] > last["open"]):
            logging.warning(f"رد {symbol} @ {tf}: واگرایی یا الگوی صعودی")
            return None
        # فیلتر نهایی با Decision Tree
        if not signal_filter.predict(features):
            logging.warning(f"رد {symbol} @ {tf}: فیلتر Decision Tree رد شد")
            return None
        entry = float(last["close"])
        atr_avg = df["ATR"].rolling(5).mean().iloc[-1]
        sl = entry + 2 * atr_avg
        tp = entry - 3 * atr_avg
        rr = round((entry - tp) / (sl - entry), 2)
        position_size = calculate_position_size(10000, 1, entry, sl)
        signal = {
            "نوع معامله": "Short",
            "نماد": symbol,
            "تایم‌فریم": tf,
            "قیمت ورود": entry,
            "حد ضرر": sl,
            "هدف سود": tp,
            "ریسک به ریوارد": rr,
            "حجم پوزیشن": position_size,
            "سطح اطمینان": min(score_short * 20, 100),
            "تحلیل": " | ".join([k for k, v in conds_short.items() if v]),
            "روانشناسی": psych_short,
            "روند بازار": "نزولی",
            "فاندامنتال": fundamental
        }
        logging.info(f"سیگنال تولید شد: {signal}")
        return signal

    logging.warning(f"رد {symbol} @ {tf}: امتیاز ناکافی (score_long={score_long}, score_short={score_short})، روانشناسی (long={psych_long}, short={psych_short})، ADX={last['ADX']:.2f}، مولتی تایم‌فریم={await multi_timeframe_confirmation(df, symbol, exchange)}")
    return None

# اسکن همه نمادها با مدیریت بهتر منابع
async def scan_all_crypto_symbols(on_signal=None):
    exchange = ccxt.kucoin({'enableRateLimit': True, 'rateLimit': 2000})
    try:
        await exchange.load_markets()
        top_coins = get_top_500_symbols_from_cmc()  # 500 نماد
        usdt_symbols = [s for s in exchange.symbols if any(s.startswith(f"{coin}/") and s.endswith("/USDT") for coin in top_coins)]
        chunk_size = 10
        total_chunks = (len(usdt_symbols) + chunk_size - 1) // chunk_size
        for idx in range(total_chunks):
            logging.info(f"اسکن دسته {idx+1}/{total_chunks}")
            chunk = usdt_symbols[idx*chunk_size:(idx+1)*chunk_size]
            tasks = [analyze_symbol(exchange, sym, tf) for sym in chunk for tf in TIMEFRAMES]
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
            async with semaphore:
                results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logging.error(f"خطا در تسک: {result}")
                    continue
                if result:
                    logging.info(f"سیگنال تولید شد: {result}")
                    if on_signal:
                        await on_signal(result)
            logging.info(f"اتمام دسته {idx+1}/{total_chunks}")
            await asyncio.sleep(WAIT_BETWEEN_CHUNKS)
    finally:
        await exchange.close()

# بک‌تست
def log_signal_result(signal, result):
    with open("signal_log.txt", "a", encoding="utf-8") as f:
        f.write(f"سیگنال: {signal}\nنتیجه: {result}\n{'-'*50}\n")

async def backtest_symbol(symbol, timeframe, start_date, end_date):
    exchange = ccxt.kucoin({'enableRateLimit': True})
    try:
        since = exchange.parse8601(start_date)
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        results = []
        for i in range(50, len(df)-1):
            window_df = df.iloc[:i+1].copy()
            sig = await analyze_symbol(exchange, symbol, timeframe)
            if not sig:
                continue
            entry, sl, tp = sig["قیمت ورود"], sig["حد ضرر"], sig["هدف سود"]
            future = df.iloc[i+1:]
            hit_tp = future['high'].ge(tp).idxmax() if any(future['high'] >= tp) else None
            hit_sl = future['low'].le(sl).idxmax() if any(future['low'] <= sl) else None
            if hit_tp and (not hit_sl or hit_tp <= hit_sl):
                results.append(True)
                log_signal_result(sig, "سود")
            else:
                results.append(False)
                log_signal_result(sig, "ضرر")
        win_rate = np.mean(results) if results else None
        logging.info(f"بک‌تست {symbol} {timeframe}: نرخ برد = {win_rate:.2%}")
    finally:
        await exchange.close()

# تست پیش‌رونده (Walkforward)
async def walkforward(symbol, timeframe, total_days=90, train_days=60, test_days=30):
    end = datetime.utcnow()
    start = end - timedelta(days=total_days)
    wf = []
    while start + timedelta(days=train_days+test_days) <= end:
        train_end = start + timedelta(days=train_days)
        test_end = train_end + timedelta(days=test_days)
        wr = await backtest_symbol(symbol, timeframe, train_end.isoformat(), test_end.isoformat())
        wf.append({"train_start": start, "train_end": train_end, "test_end": test_end, "win_rate": wr})
        start += timedelta(days=test_days)
    logging.info("نتایج تست پیش‌رونده:")
    logging.info(pd.DataFrame(wf).to_string())
    return wf

# اجرای اصلی
async def main():
    await scan_all_crypto_symbols()

if __name__ == "__main__":
    asyncio.run(main())
