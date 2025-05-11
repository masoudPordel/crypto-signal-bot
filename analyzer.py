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
from sklearn.tree import DecisionTreeClassifier

# تنظیمات logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# کلیدهای API
CMC_API_KEY = "7fc7dc4d-2d30-4c83-9836-875f9e0f74c7"
COINMARKETCAL_API_KEY = "iFrSo3PUBJ36P8ZnEIBMvakO5JutSIU1XJvG7ALa"
TIMEFRAMES = ["30m", "1h", "4h", "1d"]

# تابع دریافت آیدی ارز
def get_coin_id(symbol):
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/map"
    headers = {'Accepts': 'application/json', 'X-CMC_PRO_API_KEY': CMC_API_KEY}
    params = {'symbol': symbol}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        data = resp.json()
        if 'data' in data and len(data['data']) > 0:
            coin_id = data['data'][0]['id']
            logging.info(f"دریافت آیدی برای {symbol}: coin_id={coin_id}")
            return coin_id
        else:
            logging.warning(f"آیدی برای {symbol} یافت نشد")
            return None
    except Exception as e:
        logging.error(f"خطا در دریافت آیدی برای {symbol}: {e}")
        return None

# پارامترهای اصلی (شل کردن آستانه‌ها)
VOLUME_WINDOW = 20
S_R_BUFFER = 0.005  # افزایش به 0.005 برای شل کردن
ADX_THRESHOLD = 8  # کاهش به 8
ADX_TREND_THRESHOLD = 15  # کاهش به 15
CACHE = {}
CACHE_TTL = 600
MAX_CONCURRENT_REQUESTS = 15
WAIT_BETWEEN_REQUESTS = 0.5
WAIT_BETWEEN_CHUNKS = 3
LIQUIDITY_SPREAD_THRESHOLD = 0.02  # افزایش به 0.02 برای آلت‌کوین‌ها

# ضرایب مقیاس‌پذیری حجم
VOLUME_SCALING = {
    "30m": 0.005,
    "1h": 0.03,
    "4h": 0.10,
    "1d": 0.20
}

# متغیرهای شمارشگر رد شدن‌ها
LIQUIDITY_REJECTS = 0
VOLUME_REJECTS = 0
SR_REJECTS = 0

# تابع برای پویا کردن آستانه‌ها
def get_dynamic_thresholds(tf, df=None):
    volatility_base = {"30m": 0.0005, "1h": 0.001, "4h": 0.0015, "1d": 0.002}
    volume_base = {"30m": 5, "1h": 10, "4h": 25, "1d": 50}
    dynamic_volatility = volatility_base.get(tf, 0.001)
    dynamic_volume = volume_base.get(tf, 10)
    
    # تنظیم پویا با ATR
    if df is not None and len(df) > 0:
        atr = IndicatorCalculator.compute_atr(df).iloc[-1]
        volatility_factor = atr / df["close"].iloc[-1]
        dynamic_volatility = max(dynamic_volatility, volatility_factor * 0.5)  # حداقل 50% از ATR
        logging.debug(f"پویا کردن آستانه‌ها برای {tf}: volatility_factor={volatility_factor:.4f}, dynamic_volatility={dynamic_volatility:.4f}")
    
    return dynamic_volatility, dynamic_volume

# دریافت ۵۰۰ نماد برتر از CoinMarketCap
def get_top_500_symbols_from_cmc():
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    headers = {'Accepts': 'application/json', 'X-CMC_PRO_API_KEY': CMC_API_KEY}
    params = {'start': '1', 'limit': '500', 'convert': 'USD'}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        data = resp.json()
        logging.info(f"دریافت ۵۰۰ نماد برتر از CMC: تعداد نمادها={len(data['data'])}")
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
        rsi = 100 - (100 / (1 + rs))
        logging.debug(f"محاسبه RSI: آخرین مقدار={rsi.iloc[-1]:.2f}")
        return rsi

    @staticmethod
    def compute_atr(df, period=14):
        tr = pd.concat([df["high"] - df["low"], abs(df["high"] - df["close"].shift()), abs(df["low"] - df["close"].shift())], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        logging.debug(f"محاسبه ATR: آخرین مقدار={atr.iloc[-1]:.4f}")
        return atr

    @staticmethod
    def compute_bollinger_bands(df, period=20, std_dev=2):
        sma = df["close"].rolling(period).mean()
        std = df["close"].rolling(period).std()
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        logging.debug(f"محاسبه Bollinger Bands: upper={upper.iloc[-1]:.2f}, lower={lower.iloc[-1]:.2f}")
        return upper, lower

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
        adx = dx.rolling(window=period).mean()
        logging.debug(f"محاسبه ADX: آخرین مقدار={adx.iloc[-1]:.2f}")
        return adx

    @staticmethod
    def compute_stochastic(df, period=14):
        low_min = df["low"].rolling(window=period).min()
        high_max = df["high"].rolling(window=period).max()
        k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, 1e-10)
        logging.debug(f"محاسبه Stochastic: آخرین مقدار={k.iloc[-1]:.2f}")
        return k

    @staticmethod
    def compute_mfi(df, period=14):
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        raw_money_flow = typical_price * df['volume']
        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0).rolling(period).sum()
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0).rolling(period).sum()
        mfi = 100 - (100 / (1 + positive_flow / negative_flow.replace(0, 1e-10)))
        logging.debug(f"محاسبه MFI: آخرین مقدار={mfi.iloc[-1]:.2f}")
        return mfi

    @staticmethod
    def compute_linear_regression(df, period):
        if len(df) < period:
            logging.warning(f"داده کافی برای محاسبه رگرسیون خطی با دوره {period} نیست")
            return None, None, None, None
        
        data = df['close'].iloc[-period:].values
        x = np.arange(len(data))
        y = data

        coefficients = np.polyfit(x, y, 1)
        slope, intercept = coefficients
        regression_line = np.polyval(coefficients, x)

        floor = min(regression_line[0], regression_line[-1])
        ceiling = max(regression_line[0], regression_line[-1])
        current_price = df['close'].iloc[-1]

        logging.debug(f"رگرسیون خطی (دوره {period}): floor={floor:.2f}, ceiling={ceiling:.2f}, current_price={current_price:.2f}, slope={slope:.4f}")
        return floor, ceiling, slope, current_price

# تشخیص الگوها
class PatternDetector:
    @staticmethod
    def detect_pin_bar(df):
        df["body"] = abs(df["close"] - df["open"])
        df["range"] = df["high"] - df["low"]
        df["upper"] = df["high"] - df[["close", "open"]].max(axis=1)
        df["lower"] = df[["close", "open"]].min(axis=1) - df["low"]
        pin_bar = (df["body"] < 0.3 * df["range"]) & ((df["upper"] > 2 * df["body"]) | (df["lower"] > 2 * df["body"]))
        logging.debug(f"تشخیص PinBar: آخرین مقدار={pin_bar.iloc[-1]}")
        return pin_bar

    @staticmethod
    def detect_engulfing(df):
        prev_o = df["open"].shift(1)
        prev_c = df["close"].shift(1)
        engulfing = (((df["close"] > df["open"]) & (prev_c < prev_o) & (df["close"] > prev_o) & (df["open"] < prev_c)) |
                     ((df["close"] < df["open"]) & (prev_c > prev_o) & (df["close"] < prev_o) & (df["open"] > prev_c)))
        logging.debug(f"تشخیص Engulfing: آخرین مقدار={engulfing.iloc[-1]}")
        return engulfing

    @staticmethod
    def detect_elliott_wave(df):
        df["WavePoint"] = np.nan
        highs = argrelextrema(df['close'].values, np.greater, order=5)[0]
        lows = argrelextrema(df['close'].values, np.less, order=5)[0]
        df.loc[df.index[highs], "WavePoint"] = df.loc[df.index[highs], "close"]
        df.loc[df.index[lows], "WavePoint"] = df.loc[df.index[lows], "close"]
        logging.debug(f"تشخیص Elliott Wave: تعداد نقاط موج={len(highs) + len(lows)}")
        return df

    @staticmethod
    def detect_support_resistance(df, window=10):
        if len(df) < window:
            logging.warning("داده کافی برای محاسبه حمایت/مقاومت نیست")
            return 0.01, 0.01, []
        if df["close"].isnull().all() or (df["close"] == 0).all():
            logging.warning("داده‌های close نامعتبر یا صفر هستند، مقدار پیش‌فرض استفاده می‌شود")
            return 0.01, 0.01, []
        high = df['high'].rolling(window).max()
        low = df['low'].rolling(window).min()
        close = df['close'].rolling(window).mean()
        pivot = (high + low + close) / 3
        resistance = pivot + (high - low) * 0.382
        support = pivot - (high - low) * 0.382
        recent_highs = df['high'][(df['high'].shift(1) < df['high']) & (df['high'].shift(-1) < df['high'])].iloc[-window:]
        recent_lows = df['low'][(df['low'].shift(1) > df['low']) & (df['low'].shift(-1) > df['low'])].iloc[-window:]
        recent_resistance = recent_highs.max() if not recent_highs.empty else resistance.iloc[-1]
        recent_support = recent_lows.min() if not recent_lows.empty else support.iloc[-1]
        if recent_resistance == 0 or pd.isna(recent_resistance):
            recent_resistance = df['close'].mean()
            logging.warning("مقاومت صفر یا نامعتبر بود، میانگین قیمت جایگزین شد")
            if recent_resistance == 0 or pd.isna(recent_resistance):
                recent_resistance = 0.01
                logging.warning("میانگین قیمت نیز صفر بود، مقدار پیش‌فرض 0.01 استفاده شد")
        if recent_support == 0 or pd.isna(recent_support):
            recent_support = df['close'].mean()
            logging.warning("حمایت صفر یا نامعتبر بود، میانگین قیمت جایگزین شد")
            if recent_support == 0 or pd.isna(recent_support):
                recent_support = 0.01
                logging.warning("میانگین قیمت نیز صفر بود، مقدار پیش‌فرض 0.01 استفاده شد")
        volume_profile = df['volume'].groupby(df['close'].round(2)).sum()
        vol_threshold = volume_profile.quantile(0.75)
        high_vol_levels = volume_profile[volume_profile > vol_threshold].index
        if 'support_levels' not in globals(): globals()['support_levels'] = []
        if 'resistance_levels' not in globals(): globals()['resistance_levels'] = []
        if recent_support not in support_levels: support_levels.append(recent_support)
        if recent_resistance not in resistance_levels: resistance_levels.append(recent_resistance)
        logging.info(f"تشخیص حمایت/مقاومت: support={recent_support:.2f}, resistance={recent_resistance:.2f}")
        return recent_support, recent_resistance, high_vol_levels

    @staticmethod
    def detect_hammer(df):
        body = abs(df['close'] - df['open'])
        range_ = df['high'] - df['low']
        lower_wick = df['low'] - df[['close', 'open']].min(axis=1)
        hammer = (body < 0.3 * range_) & (lower_wick > 2 * body) & (df['close'] > df['open'])
        logging.debug(f"تشخیص Hammer: آخرین مقدار={hammer.iloc[-1]}")
        return hammer

    @staticmethod
    def detect_rsi_divergence(df, lookback=5):
        rsi = IndicatorCalculator.compute_rsi(df)
        prices = df['close']
        recent_lows_price = argrelextrema(prices.values, np.less, order=lookback)[0]
        recent_highs_price = argrelextrema(prices.values, np.greater, order=lookback)[0]
        recent_lows_rsi = argrelextrema(rsi.values, np.less, order=lookback)[0]
        recent_highs_rsi = argrelextrema(rsi.values, np.greater, order=lookback)[0]
        if len(recent_lows_price) > 1 and len(recent_lows_rsi) > 1:
            last_price_low = prices.iloc[recent_lows_price[-1]]
            prev_price_low = prices.iloc[recent_lows_price[-2]]
            last_rsi_low = rsi.iloc[recent_lows_rsi[-1]]
            prev_rsi_low = rsi.iloc[recent_lows_rsi[-2]]
            bullish_divergence = last_price_low < prev_price_low and last_rsi_low > prev_rsi_low
        else:
            bullish_divergence = False
        if len(recent_highs_price) > 1 and len(recent_highs_rsi) > 1:
            last_price_high = prices.iloc[recent_highs_price[-1]]
            prev_price_high = prices.iloc[recent_highs_price[-2]]
            last_rsi_high = rsi.iloc[recent_highs_rsi[-1]]
            prev_rsi_high = rsi.iloc[recent_highs_rsi[-2]]
            bearish_divergence = last_price_high > prev_price_high and last_rsi_high < prev_rsi_high
        else:
            bearish_divergence = False
        logging.debug(f"تشخیص واگرایی RSI: bullish={bullish_divergence}, bearish={bearish_divergence}")
        return bullish_divergence, bearish_divergence

    @staticmethod
    def detect_macd_divergence(df, lookback=5):
        macd = df["MACD"]
        prices = df['close']
        recent_lows_price = argrelextrema(prices.values, np.less, order=lookback)[0]
        recent_highs_price = argrelextrema(prices.values, np.greater, order=lookback)[0]
        recent_lows_macd = argrelextrema(macd.values, np.less, order=lookback)[0]
        recent_highs_macd = argrelextrema(macd.values, np.greater, order=lookback)[0]
        if len(recent_lows_price) > 1 and len(recent_lows_macd) > 1:
            last_price_low = prices.iloc[recent_lows_price[-1]]
            prev_price_low = prices.iloc[recent_lows_price[-2]]
            last_macd_low = macd.iloc[recent_lows_macd[-1]]
            prev_macd_low = macd.iloc[recent_lows_macd[-2]]
            bullish_divergence = last_price_low < prev_price_low and last_macd_low > prev_macd_low
        else:
            bullish_divergence = False
        if len(recent_highs_price) > 1 and len(recent_highs_macd) > 1:
            last_price_high = prices.iloc[recent_highs_price[-1]]
            prev_price_high = prices.iloc[recent_highs_price[-2]]
            last_macd_high = macd.iloc[recent_highs_macd[-1]]
            prev_macd_high = macd.iloc[recent_highs_macd[-2]]
            bearish_divergence = last_price_high > prev_price_high and last_macd_high < prev_macd_high
        else:
            bearish_divergence = False
        logging.debug(f"تشخیص واگرایی MACD: bullish={bullish_divergence}, bearish={bearish_divergence}")
        return bullish_divergence, bearish_divergence

    @staticmethod
    def is_support_broken(df, support, lookback=2):
        recent_closes = df['close'].iloc[-lookback:]
        broken = all(recent_closes < support)
        logging.debug(f"بررسی شکست حمایت: support={support:.2f}, recent_closes={recent_closes.tolist()}, broken={broken}")
        return broken

    @staticmethod
    def is_resistance_broken(df, resistance, lookback=2):
        recent_closes = df['close'].iloc[-lookback:]
        broken = all(recent_closes > resistance)
        logging.debug(f"بررسی شکست مقاومت: resistance={resistance:.2f}, recent_closes={recent_closes.tolist()}, broken={broken}")
        return broken

    @staticmethod
    def is_valid_breakout(df, level, direction="support", vol_threshold=0.1):
        last_vol = df['volume'].iloc[-1]
        vol_avg = df['volume'].rolling(VOLUME_WINDOW).mean().iloc[-1]
        vol_condition = last_vol >= vol_threshold * vol_avg
        logging.debug(f"بررسی حجم برای شکست: last_vol={last_vol:.2f}, vol_avg={vol_avg:.2f}, threshold={vol_threshold * vol_avg:.2f}, vol_condition={vol_condition}")
        if not vol_condition:
            logging.warning(f"حجم ناکافی (current={last_vol:.2f}, threshold={vol_threshold * vol_avg:.2f}) ولی به دلیل شل بودن شرط، نادیده گرفته شد")
            vol_condition = True
        if direction == "support":
            support_broken = df['close'].iloc[-1] < level * 1.10
            logging.debug(f"بررسی شکست حمایت: support_broken={support_broken}, level={level:.2f}, close={df['close'].iloc[-1]:.2f}")
            if not support_broken:
                logging.warning(f"حمایت شکسته نشده، ولی با بافر 10% نادیده گرفته شد")
                support_broken = True
        if direction == "resistance":
            resistance_broken = df['close'].iloc[-1] > level * 0.90
            logging.debug(f"بررسی شکست مقاومت: resistance_broken={resistance_broken}, level={level:.2f}, close={df['close'].iloc[-1]:.2f}")
            if not resistance_broken:
                logging.warning(f"مقاومت شکسته نشده، ولی با بافر 10% نادیده گرفته شد")
                resistance_broken = True
        logging.info(f"شکست معتبر (شل‌شده): direction={direction}, level={level:.2f}, close={df['close'].iloc[-1]:.2f}")
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
        prediction = self.model.predict([features])[0]
        logging.debug(f"پیش‌بینی Decision Tree: features={features}, result={prediction}")
        return prediction

# بررسی نقدینگی
async def check_liquidity(exchange, symbol):
    global LIQUIDITY_REJECTS
    try:
        ticker = await exchange.fetch_ticker(symbol)
        bid = ticker['bid']
        ask = ticker['ask']
        spread = (ask - bid) / ((bid + ask) / 2)
        logging.info(f"نماد {symbol}: اسپرد={spread:.4f}, آستانه={LIQUIDITY_SPREAD_THRESHOLD}")
        liquidity_passed = spread < LIQUIDITY_SPREAD_THRESHOLD
        logging.debug(f"بررسی نقدینگی: spread={spread:.4f}, threshold={LIQUIDITY_SPREAD_THRESHOLD}, passed={liquidity_passed}")
        if not liquidity_passed:
            LIQUIDITY_REJECTS += 1
            logging.warning(f"رد {symbol}: نقدینگی کافی نیست (spread={spread:.4f}, threshold={LIQUIDITY_SPREAD_THRESHOLD})")
        return liquidity_passed
    except Exception as e:
        logging.error(f"خطا در بررسی نقدینگی برای {symbol}: {e}")
        return False

# دریافت ID ارز و بررسی رویدادها
def check_market_events(symbol):
    coin_id = get_coin_id(symbol)
    if not coin_id:
        logging.warning(f"امتیاز فاندامنتال برای {symbol}: 0 (ارز یافت نشد)")
        return 0
    url = "https://developers.coinmarketcal.com/v1/events"
    headers = {
        "x-api-key": COINMARKETCAL_API_KEY,
        "Accept": "application/json",
        "Accept-Encoding": "deflate, gzip"
    }
    start_date = (datetime.utcnow() - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0).strftime("%Y-%m-%d")
    end_date = (datetime.utcnow() + timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0).strftime("%Y-%m-%d")
    params = {
        "coinId": str(coin_id),
        "max": 5,
        "dateRangeStart": start_date,
        "dateRangeEnd": end_date
    }
    try:
        time.sleep(0.5)
        logging.debug(f"درخواست برای رویدادها: URL={url}, Params={params}")
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        if resp.status_code != 200:
            logging.error(f"خطا در دریافت رویدادها: Status={resp.status_code}, Response={resp.text}")
            return 0
        events = resp.json()
        event_score = 0
        if not events or "body" not in events or not events["body"]:
            logging.info(f"فاندامنتال برای {symbol}: هیچ رویداد مهمی یافت نشد")
            return 0
        for event in events["body"]:
            title = event.get("title", "").lower() if isinstance(event.get("title"), str) else ""
            description = event.get("description", "").lower() if isinstance(event.get("description"), str) else ""
            if "burn" in title or "token burn" in description:
                event_score += 15
                logging.info(f"رویداد مثبت برای {symbol}: توکن‌سوزی (امتیاز +15)")
            elif "listing" in title or "exchange" in description:
                event_score += 10
                logging.info(f"رویداد مثبت برای {symbol}: لیست شدن (امتیاز +10)")
            elif "partnership" in title or "collaboration" in description:
                event_score += 5
                logging.info(f"رویداد مثبت برای {symbol}: همکاری (امتیاز +5)")
            elif "hack" in title or "security breach" in description:
                event_score -= 20
                logging.info(f"رویداد منفی برای {symbol}: هک (امتیاز -20)")
            elif "lawsuit" in title or "negative" in description:
                event_score -= 15
                logging.info(f"رویداد منفی برای {symbol}: اخبار منفی (امتیاز -15)")
        logging.info(f"فاندامنتال برای {symbol}: امتیاز کل رویداد={event_score}")
        return event_score
    except Exception as e:
        logging.error(f"خطا در دریافت رویدادها از CoinMarketCal برای {symbol}: {e}")
        return 0

# محاسبات اندیکاتورها
def compute_indicators(df):
    df["EMA12"] = df["close"].ewm(span=12).mean()
    df["EMA26"] = df["close"].ewm(span=26).mean()
    df["EMA20"] = df["close"].ewm(span=20).mean()
    df["EMA50"] = df["close"].ewm(span=50).mean()
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
    df["MFI"] = IndicatorCalculator.compute_mfi(df)
    logging.info(f"محاسبه اندیکاتورها: RSI={df['RSI'].iloc[-1]:.2f}, ADX={df['ADX'].iloc[-1]:.2f}, Stochastic={df['Stochastic'].iloc[-1]:.2f}, MFI={df['MFI'].iloc[-1]:.2f}")
    return df

# تأیید اندیکاتورهای ترکیبی
def confirm_combined_indicators(df, trend_type):
    global symbol, tf
    last = df.iloc[-1]
    rsi = last["RSI"]
    stochastic = last["Stochastic"]
    bullish_engulf = last["Engulfing"] and last["close"] > last["open"]
    bearish_engulf = last["Engulfing"] and last["close"] < last["open"]
    macd_cross_long = df["MACD"].iloc[-2] < df["Signal"].iloc[-2] and df["MACD"].iloc[-1] > df["Signal"].iloc[-1]
    macd_cross_short = df["MACD"].iloc[-2] > df["Signal"].iloc[-2] and df["MACD"].iloc[-1] < df["Signal"].iloc[-1]
    if rsi > 80:
        logging.warning(f"رد {symbol} @ {tf}: RSI خیلی بالا (RSI={rsi:.2f})")
        return False
    if rsi < 15:
        logging.warning(f"رد {symbol} @ {tf}: RSI خیلی پایین (RSI={rsi:.2f})")
        return False
    if trend_type == "Long":
        conditions = [rsi < 50, macd_cross_long, bullish_engulf, stochastic < 40]
        logging.debug(f"شرایط تأیید Long برای {symbol} @ {tf}: rsi<50={rsi < 50}, macd_cross_long={macd_cross_long}, bullish_engulf={bullish_engulf}, stochastic<40={stochastic < 40}, تعداد شرایط برقرار={sum(conditions)}")
        return sum(conditions) >= 1
    else:
        conditions = [rsi > 60, macd_cross_short, bearish_engulf, stochastic > 50]
        logging.debug(f"شرایط تأیید Short برای {symbol} @ {tf}: rsi>60={rsi > 60}, macd_cross_short={macd_cross_short}, bearish_engulf={bearish_engulf}, stochastic>50={stochastic > 50}, تعداد شرایط برقرار={sum(conditions)}")
        return sum(conditions) >= 1

# مدیریت ریسک و اندازه پوزیشن
def calculate_position_size(account_balance, risk_percentage, entry, stop_loss):
    risk_amount = account_balance * (risk_percentage / 100)
    distance = abs(entry - stop_loss)
    position_size = risk_amount / distance if distance != 0 else 0
    logging.info(f"محاسبه حجم پوزیشن: risk_amount={risk_amount:.2f}, distance={distance:.4f}, position_size={position_size:.2f}")
    return round(position_size, 2)

# تأیید مولتی تایم‌فریم
async def multi_timeframe_confirmation(df, symbol, exchange):
    weights = {"1d": 0.4, "4h": 0.3, "1h": 0.2, "30m": 0.1}
    total_weight = 0
    trend_score = 0
    for tf, weight in weights.items():
        df_tf = await get_ohlcv_cached(exchange, symbol, tf)
        if df_tf is not None and len(df_tf) >= 50:
            df_tf["EMA12"] = df_tf["close"].ewm(span=12).mean()
            df_tf["EMA26"] = df_tf["close"].ewm(span=26).mean()
            long_trend = df_tf["EMA12"].iloc[-1] > df_tf["EMA26"].iloc[-1]
            trend_score += weight if long_trend else -weight
            logging.debug(f"تأیید مولتی تایم‌فریم برای {symbol}: تایم‌فریم {tf}, long_trend={long_trend}, وزن={weight}, trend_score={trend_score:.2f}")
        else:
            logging.debug(f"داده برای تایم‌فریم {tf} کافی نیست، وزن {weight} نادیده گرفته شد")
        total_weight += weight
    final_score = trend_score / total_weight if total_weight > 0 else 0
    passed = abs(final_score) >= 0.1
    logging.info(f"نتیجه نهایی مولتی تایم‌فریم برای {symbol}: trend_score={trend_score:.2f}, total_weight={total_weight:.2f}, final_score={final_score:.2f}, passed={passed}")
    return passed

# دریافت داده‌های کندل با کش
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
async def get_ohlcv_cached(exchange, symbol, tf, limit=51):
    async with semaphore:
        await asyncio.sleep(WAIT_BETWEEN_REQUESTS)
        key = f"{symbol}_{tf}"
        now = time.time()
        if key in CACHE and now - CACHE[key]["time"] < CACHE_TTL:
            logging.debug(f"داده از کش برای {symbol} @ {tf} دریافت شد")
            return CACHE[key]["data"]
        try:
            data = await exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            current_time = pd.Timestamp.now(tz='UTC')
            if len(df) > 0 and (current_time - df.index[-1]).total_seconds() < 60:
                df = df.iloc[:-1].copy()
                logging.debug(f"کندل ناقص برای {symbol} @ {tf} حذف شد")
            CACHE[key] = {"data": df.copy(), "time": now}
            logging.info(f"داده جدید برای {symbol} @ {tf} دریافت شد: طول={len(df)}")
            return df
        except Exception as e:
            logging.error(f"خطا در دریافت داده برای {symbol}-{tf}: {e}")
            return None

# بک‌تست استراتژی
def backtest_strategy(df, symbol, initial_balance=10000, risk_percentage=1):
    balance = initial_balance
    position = 0
    trades = []
    for i in range(1, len(df)):
        current = df.iloc[i]
        previous = df.iloc[i-1]
        long_signal = (current["EMA12"] > current["EMA26"] and previous["EMA12"] <= previous["EMA26"] and current["RSI"] < 40)
        short_signal = (current["EMA12"] < current["EMA26"] and previous["EMA12"] >= previous["EMA26"] and current["RSI"] > 60)
        if long_signal and position == 0:
            entry = current["close"]
            atr = IndicatorCalculator.compute_atr(df.iloc[:i+1]).iloc[-1]
            sl = entry - 2 * atr
            tp = entry + 3 * atr
            position_size = calculate_position_size(balance, risk_percentage, entry, sl)
            position = position_size
            trades.append({"type": "Long", "entry": entry, "sl": sl, "tp": tp, "size": position_size})
            logging.info(f"بک‌تست - ورود Long: entry={entry:.2f}, sl={sl:.2f}, tp={tp:.2f}, size={position_size:.2f}")
        elif short_signal and position == 0:
            entry = current["close"]
            atr = IndicatorCalculator.compute_atr(df.iloc[:i+1]).iloc[-1]
            sl = entry + 2 * atr
            tp = entry - 3 * atr
            position_size = calculate_position_size(balance, risk_percentage, entry, sl)
            position = -position_size
            trades.append({"type": "Short", "entry": entry, "sl": sl, "tp": tp, "size": position_size})
            logging.info(f"بک‌تست - ورود Short: entry={entry:.2f}, sl={sl:.2f}, tp={tp:.2f}, size={position_size:.2f}")
        elif position > 0 and (current["close"] <= trades[-1]["sl"] or current["close"] >= trades[-1]["tp"]):
            balance += position * (current["close"] - trades[-1]["entry"])
            position = 0
            logging.info(f"بک‌تست - خروج Long: balance={balance:.2f}")
        elif position < 0 and (current["close"] >= trades[-1]["sl"] or current["close"] <= trades[-1]["tp"]):
            balance -= abs(position) * (trades[-1]["entry"] - current["close"])
            position = 0
            logging.info(f"بک‌تست - خروج Short: balance={balance:.2f}")
    final_balance = balance + (position * df["close"].iloc[-1] if position != 0 else 0)
    logging.info(f"بک‌تست برای {symbol} - موجودی نهایی: {final_balance:.2f}, تعداد معاملات: {len(trades)}")
    return final_balance, trades

# تست فوروارد
async def forward_test(exchange, symbol, tf, days=7):
    df = await get_ohlcv_cached(exchange, symbol, tf, limit=days * 24 * 2)
    if df is None or len(df) < 50:
        logging.warning(f"داده کافی برای تست فوروارد {symbol} @ {tf} نیست")
        return None
    df = compute_indicators(df)
    balance, trades = backtest_strategy(df, symbol)
    logging.info(f"تست فوروارد - {symbol} @ {tf}: موجودی اولیه=10000, موجودی نهایی={balance}, تعداد معاملات={len(trades)}")
    return balance, trades

# آموزش Decision Tree
signal_filter = SignalFilter()
X_train = np.array([[30, 25, 2], [70, 20, 1], [50, 30, 1.5], [20, 40, 3]])
y_train = np.array([1, 0, 0, 1])
signal_filter.train(X_train, y_train)

# تابع تنظیم پویای ضرایب
async def dynamic_volume_scaling(exchange, symbol, tf, df):
    atr = IndicatorCalculator.compute_atr(df).iloc[-1]
    volatility_factor = atr / df["close"].iloc[-1]
    base_volatility = 0.002
    volatility_weight = max(1.0, volatility_factor / base_volatility)
    try:
        order_book = await exchange.fetch_order_book(symbol, limit=20)
        bids = order_book['bids']
        asks = order_book['asks']
        bid_vol = sum([b[1] for b in bids[:5]])
        ask_vol = sum([a[1] for a in asks[:5]])
        total_depth = bid_vol + ask_vol
        spread = (asks[0][0] - bids[0][0]) / ((bids[0][0] + asks[0][0]) / 2)
        liquidity_factor = max(0.5, 1 / (1 + spread))
    except Exception as e:
        logging.warning(f"خطا در دریافت عمق مارکت برای {symbol}: {e}, استفاده از مقادیر پیش‌فرض")
        total_depth = 1000
        liquidity_factor = 1.0
    base_scaling = {
        "30m": 0.0025,
        "1h": 0.015,
        "4h": 0.05,
        "1d": 0.10
    }
    dynamic_factor = volatility_weight * liquidity_factor * (total_depth / 1000)
    dynamic_factor = min(dynamic_factor, 5)
    dynamic_scaling = {tf: base_scaling[tf] * dynamic_factor for tf in base_scaling}
    logging.info(f"تنظیم پویای VOLUME_SCALING برای {symbol} @ {tf}: volatility_factor={volatility_factor:.4f}, liquidity_factor={liquidity_factor:.4f}, dynamic_factor={dynamic_factor:.4f}, new_scaling={dynamic_scaling}")
    return dynamic_scaling

# تحلیل نماد (با وزن‌دهی و شل کردن فیلترها)
async def analyze_symbol(exchange, symbol, tf):
    global VOLUME_REJECTS, SR_REJECTS
    start_time = time.time()
    logging.info(f"شروع تحلیل {symbol} @ {tf}")
    df = await get_ohlcv_cached(exchange, symbol, tf)
    logging.info(f"دریافت داده برای {symbol} @ {tf} در {time.time() - start_time:.2f} ثانیه")
    
    # امتیازدهی اولیه
    score = 0
    if df is None or len(df) < 50:
        logging.warning(f"داده ناقص برای {symbol} @ {tf}, امتیاز -10")
        score -= 10
    else:
        score += 5  # امتیاز پایه برای داده کامل

    # پویا کردن آستانه‌ها
    dynamic_volatility, dynamic_volume = get_dynamic_thresholds(tf, df)
    dynamic_scaling = await dynamic_volume_scaling(exchange, symbol, tf, df)
    vol_avg = df["volume"].rolling(VOLUME_WINDOW).mean().iloc[-1]
    scale_factor = dynamic_scaling.get(tf, 0.2)
    dynamic_threshold = max(dynamic_volume, vol_avg * scale_factor * 0.005, 0.01)
    current_vol = df["volume"].iloc[-1]
    vol_condition = current_vol >= dynamic_threshold
    score += 10 if vol_condition else -5  # 10 امتیاز برای حجم کافی
    logging.info(f"نماد {symbol} @ {tf}: vol_avg={vol_avg:.2f}, scale_factor={scale_factor:.4f}, dynamic_threshold={dynamic_threshold:.2f}, current_vol={current_vol:.2f}, score={score}")
    
    if not vol_condition:
        VOLUME_REJECTS += 1
        logging.warning(f"حجم کم برای {symbol} @ {tf}: current={current_vol:.2f}, threshold={dynamic_threshold:.2f}")

    df = compute_indicators(df)
    last = df.iloc[-1]
    volatility = df["ATR"].iloc[-1] / df["close"].iloc[-1]
    volatility_condition = volatility > dynamic_volatility
    score += 10 if volatility_condition else -5  # 10 امتیاز برای نوسان کافی
    logging.info(f"اندیکاتورها برای {symbol} @ {tf}: RSI={last['RSI']:.2f}, ADX={last['ADX']:.2f}, volatility={volatility:.4f}, score={score}")

    adx_condition = last["ADX"] >= ADX_THRESHOLD
    trend_condition = last["ADX"] >= ADX_TREND_THRESHOLD
    score += 15 if adx_condition else 5  # 15 امتیاز برای ADX قوی
    score += 10 if trend_condition else 0  # 10 امتیاز برای روند قوی
    logging.info(f"فیلتر ADX: ADX={last['ADX']:.2f}, threshold={ADX_THRESHOLD}, trend_threshold={ADX_TREND_THRESHOLD}, score={score}")

    long_trend = df["EMA12"].iloc[-1] > df["EMA26"].iloc[-1]
    short_trend = not long_trend
    logging.info(f"روند بازار: long_trend={long_trend}, short_trend={short_trend}")

    if tf == "1h":
        df4 = await get_ohlcv_cached(exchange, symbol, "4h")
        if df4 is not None and len(df4) >= 50:
            e12_4 = df4["close"].ewm(span=12).mean().iloc[-1]
            e26_4 = df4["close"].ewm(span=26).mean().iloc[-1]
            trend4 = e12_4 > e26_4
            logging.info(f"روند چند تایم‌فریمی برای {symbol} @ {tf}: 1h={'صعودی' if long_trend else 'نزولی'}, 4h={'صعودی' if trend4 else 'نزولی'}")
            mtf_condition = (long_trend and trend4) or (short_trend and not trend4)
            score += 15 if mtf_condition else 5
            logging.info(f"تأیید مولتی تایم‌فریم برای {symbol} @ {tf}: passed={mtf_condition}, score={score}")
        else:
            logging.warning(f"داده چند تایم‌فریمی برای {symbol} @ {tf} کافی نیست")
            mtf_condition = True
            score += 5
            logging.info(f"تأیید مولتی تایم‌فریم پیش‌فرض برای {symbol} @ {tf}: passed={mtf_condition}, score={score}")

    support, resistance, vol_levels = PatternDetector.detect_support_resistance(df)
    logging.info(f"سطوح برای {symbol} @ {tf}: support={support:.2f}, resistance={resistance:.2f}, close={last['close']:.2f}")
    if support != 0 and resistance != 0:
        distance_to_resistance = abs(last["close"] - resistance) / last["close"]
        distance_to_support = abs(last["close"] - support) / last["close"]
        logging.info(f"فاصله تا سطوح: dist_to_resistance={distance_to_resistance:.4f}, dist_to_support={distance_to_support:.4f}")
        sr_condition_long = distance_to_resistance > S_R_BUFFER
        sr_condition_short = distance_to_support > S_R_BUFFER
        score += 10 if (long_trend and sr_condition_long) or (short_trend and sr_condition_short) else -5
        if long_trend and not sr_condition_long:
            SR_REJECTS += 1
            logging.warning(f"نزدیک به مقاومت برای {symbol} @ {tf}: distance={distance_to_resistance:.4f}, score={score}")
        elif short_trend and not sr_condition_short:
            SR_REJECTS += 1
            logging.warning(f"نزدیک به حمایت برای {symbol} @ {tf}: distance={distance_to_support:.4f}, score={score}")
    else:
        logging.warning(f"سطوح حمایت/مقاومت صفر هستند، امتیاز -5")
        score -= 5

    liquidity_passed = await check_liquidity(exchange, symbol)
    score += 15 if liquidity_passed else -5  # 15 امتیاز برای نقدینگی کافی

    fundamental_score = check_market_events(symbol.split('/')[0])
    score += fundamental_score  # امتیاز فاندامنتال مستقیماً اضافه می‌شه
    fundamental_condition = fundamental_score >= -10
    if not fundamental_condition:
        logging.warning(f"امتیاز فاندامنتال پایین برای {symbol} @ {tf}: {fundamental_score}")

    bullish_pin = last["PinBar"] and last["lower"] > 2 * last["body"]
    bearish_pin = last["PinBar"] and last["upper"] > 2 * last["body"]
    bullish_engulf = last["Engulfing"] and last["close"] > last["open"]
    bearish_engulf = last["Engulfing"] and last["close"] < last["open"]
    rsi = last["RSI"]
    stochastic = last["Stochastic"]
    mfi = last["MFI"]
    psych_long = "اشباع فروش" if rsi < 40 else "اشباع خرید" if rsi > 60 else "متعادل"
    psych_short = "اشباع خرید" if rsi > 60 else "اشباع فروش" if rsi < 40 else "متعادل"
    logging.info(f"روانشناسی بازار: psych_long={psych_long}, psych_short={psych_short}")

    bullish_rsi_div, bearish_rsi_div = PatternDetector.detect_rsi_divergence(df)
    bullish_macd_div, bearish_macd_div = PatternDetector.detect_macd_divergence(df)

    # امتیازدهی برای شرایط
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
        "BB_Breakout": df["close"].iloc[-1] > df["BB_upper"].iloc[-1],
        "MFI_Oversold": mfi < 20
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
        "BB_Breakout": df["close"].iloc[-1] < df["BB_lower"].iloc[-1],
        "MFI_Overbought": mfi > 80
    }

    # امتیازدهی برای هر شرط (هر شرط 10 امتیاز)
    score_long = sum([10 for v in conds_long.values() if v])
    score_short = sum([10 for v in conds_short.values() if v])
    score_long += score  # جمع امتیازهای اولیه
    score_short += score
    logging.info(f"نماد {symbol} @ {tf}: score_long={score_long}, score_short={score_short}, fundamental_score={fundamental_score}")

    has_trend = last["ADX"] > ADX_TREND_THRESHOLD
    features = [rsi, last["ADX"], last["volume"] / vol_avg]

    # بررسی سیگنال Long
    long_indicators_condition = confirm_combined_indicators(df, "Long")
    long_mtf_condition = await multi_timeframe_confirmation(df, symbol, exchange)
    long_breakout_condition = PatternDetector.is_valid_breakout(df, resistance, direction="resistance") if long_indicators_condition and long_mtf_condition else True
    long_filter_condition = signal_filter.predict(features) if long_breakout_condition else True

    score_long += 15 if long_indicators_condition else 5
    score_long += 15 if long_mtf_condition else 5
    score_long += 10 if long_breakout_condition else 0
    score_long += 10 if long_filter_condition else 0

    # بررسی سیگنال Short
    short_indicators_condition = confirm_combined_indicators(df, "Short")
    short_mtf_condition = await multi_timeframe_confirmation(df, symbol, exchange)
    short_divergence_condition = not (bearish_rsi_div or bearish_macd_div or PatternDetector.detect_hammer(df).iloc[-1] or (last["Engulfing"] and last["close"] > last["open"]))
    short_breakout_condition = PatternDetector.is_valid_breakout(df, support, direction="support") if short_indicators_condition and short_mtf_condition and short_divergence_condition else True
    short_filter_condition = signal_filter.predict(features) if short_breakout_condition else True

    score_short += 15 if short_indicators_condition else 5
    score_short += 15 if short_mtf_condition else 5
    score_short += 10 if short_divergence_condition else 0
    score_short += 10 if short_breakout_condition else 0
    score_short += 10 if short_filter_condition else 0

    # مقایسه score_long و score_short و انتخاب سیگنال با امتیاز بالاتر
    result = None
    logging.debug(f"مقایسه امتیازها: score_long={score_long}, score_short={score_short}")
    if score_long >= 60 and score_short >= 60:
        logging.info(f"هر دو شرط برقرار: score_long={score_long}, score_short={score_short}")
        if score_long > score_short:
            entry = float(last["close"])
            atr_avg = df["ATR"].rolling(5).mean().iloc[-1]
            sl = entry - 2 * atr_avg
            tp = entry + 3 * atr_avg
            rr = round((tp - entry) / (entry - sl), 2)
            position_size = calculate_position_size(10000, 1, entry, sl)
            signal_strength = "قوی" if score_long > 70 else "متوسط"
            result = {
                "نوع معامله": "Long",
                "نماد": symbol,
                "تایم‌فریم": tf,
                "قیمت ورود": entry,
                "حد ضرر": sl,
                "هدف سود": tp,
                "ریسک به ریوارد": rr,
                "حجم پوزیشن": position_size,
                "سطح اطمینان": min(score_long, 100),
                "امتیاز": score_long,
                "قدرت سیگنال": signal_strength,
                "تحلیل": " | ".join([k for k, v in conds_long.items() if v]),
                "روانشناسی": psych_long,
                "روند بازار": "صعودی",
                "فاندامنتال": f"امتیاز: {fundamental_score}"
            }
            logging.info(f"سیگنال Long انتخاب شد (امتیاز بالاتر): {result}")
        else:
            entry = float(last["close"])
            atr_avg = df["ATR"].rolling(5).mean().iloc[-1]
            sl = entry + 2 * atr_avg
            tp = entry - 3 * atr_avg
            rr = round((entry - tp) / (sl - entry), 2)
            position_size = calculate_position_size(10000, 1, entry, sl)
            signal_strength = "قوی" if score_short > 70 else "متوسط"
            result = {
                "نوع معامله": "Short",
                "نماد": symbol,
                "تایم‌فریم": tf,
                "قیمت ورود": entry,
                "حد ضرر": sl,
                "هدف سود": tp,
                "ریسک به ریوارد": rr,
                "حجم پوزیشن": position_size,
                "سطح اطمینان": min(score_short, 100),
                "امتیاز": score_short,
                "قدرت سیگنال": signal_strength,
                "تحلیل": " | ".join([k for k, v in conds_short.items() if v]),
                "روانشناسی": psych_short,
                "روند بازار": "نزولی",
                "فاندامنتال": f"امتیاز: {fundamental_score}"
            }
            logging.info(f"سیگنال Short انتخاب شد (امتیاز بالاتر): {result}")
    elif score_long >= 60:
        entry = float(last["close"])
        atr_avg = df["ATR"].rolling(5).mean().iloc[-1]
        sl = entry - 2 * atr_avg
        tp = entry + 3 * atr_avg
        rr = round((tp - entry) / (entry - sl), 2)
        position_size = calculate_position_size(10000, 1, entry, sl)
        signal_strength = "قوی" if score_long > 70 else "متوسط"
        result = {
            "نوع معامله": "Long",
            "نماد": symbol,
            "تایم‌فریم": tf,
            "قیمت ورود": entry,
            "حد ضرر": sl,
            "هدف سود": tp,
            "ریسک به ریوارد": rr,
            "حجم پوزیشن": position_size,
            "سطح اطمینان": min(score_long, 100),
            "امتیاز": score_long,
            "قدرت سیگنال": signal_strength,
            "تحلیل": " | ".join([k for k, v in conds_long.items() if v]),
            "روانشناسی": psych_long,
            "روند بازار": "صعودی",
            "فاندامنتال": f"امتیاز: {fundamental_score}"
        }
        logging.info(f"سیگنال Long برای {symbol} @ {tf}: {result}")
    elif score_short >= 60:
        entry = float(last["close"])
        atr_avg = df["ATR"].rolling(5).mean().iloc[-1]
        sl = entry + 2 * atr_avg
        tp = entry - 3 * atr_avg
        rr = round((entry - tp) / (sl - entry), 2)
        position_size = calculate_position_size(10000, 1, entry, sl)
        signal_strength = "قوی" if score_short > 70 else "متوسط"
        result = {
            "نوع معامله": "Short",
            "نماد": symbol,
            "تایم‌فریم": tf,
            "قیمت ورود": entry,
            "حد ضرر": sl,
            "هدف سود": tp,
            "ریسک به ریوارد": rr,
            "حجم پوزیشن": position_size,
            "سطح اطمینان": min(score_short, 100),
            "امتیاز": score_short,
            "قدرت سیگنال": signal_strength,
            "تحلیل": " | ".join([k for k, v in conds_short.items() if v]),
            "روانشناسی": psych_short,
            "روند بازار": "نزولی",
            "فاندامنتال": f"امتیاز: {fundamental_score}"
        }
        logging.info(f"سیگنال Short برای {symbol} @ {tf}: {result}")

    # استراتژی رگرسیون خطی (حفظ شده)
    logging.info(f"هیچ سیگنالی از استراتژی اصلی برای {symbol} @ {tf} تولید نشد. بررسی رگرسیون خطی...")
    if tf == "1d" and result is None:
        floor_145, ceiling_145, slope_145, current_price = IndicatorCalculator.compute_linear_regression(df, 145)
        floor_360, ceiling_360, slope_360, _ = IndicatorCalculator.compute_linear_regression(df, 360)

        if floor_145 is None or floor_360 is None:
            logging.warning(f"محاسبه رگرسیون خطی برای {symbol} @ {tf} ناموفق بود.")
            return None

        proximity_threshold = 0.02
        strength_threshold = 10

        floor_hit_145 = abs(current_price - floor_145) / floor_145 <= proximity_threshold
        floor_hit_360 = abs(current_price - floor_360) / floor_360 <= proximity_threshold
        strong_support_145 = any(abs(current_price - s) / current_price <= proximity_threshold for s in [support])
        strong_support_360 = any(abs(current_price - s) / current_price <= proximity_threshold for s in [support])

        if (floor_hit_145 or floor_hit_360) and (strong_support_145 or strong_support_360):
            entry = float(current_price)
            nearest_support = support * 0.99
            nearest_resistance = resistance * 0.99
            sl = nearest_support
            tp = nearest_resistance
            rr = round((tp - entry) / (entry - sl), 2) if (entry - sl) != 0 else 0
            position_size = calculate_position_size(10000, 1, entry, sl)
            result = {
                "نوع معامله": "Long (نوسان‌گیری رگرسیون)",
                "نماد": symbol,
                "تایم‌فریم": tf,
                "قیمت ورود": entry,
                "حد ضرر": sl,
                "هدف سود": tp,
                "ریسک به ریوارد": rr,
                "حجم پوزیشن": position_size,
                "سطح اطمینان": 70,
                "قدرت سیگنال": "متوسط",
                "تحلیل": f"قیمت به کف رگرسیون رسید (۱۴۵ روز: floor={floor_145:.2f}, ۳۶۰ روز: floor={floor_360:.2f}) | سطح حمایتی قوی وجود دارد",
                "روانشناسی": psych_long,
                "روند بازار": "صعودی (بر اساس رگرسیون)",
                "فاندامنتال": f"امتیاز: {fundamental_score}"
            }
            logging.info(f"سیگنال نوسان‌گیری Long (رگرسیون) برای {symbol} @ {tf}: {result}")
            return result

        ceiling_hit_145 = abs(current_price - ceiling_145) / ceiling_145 <= proximity_threshold
        ceiling_hit_360 = abs(current_price - ceiling_360) / ceiling_360 <= proximity_threshold
        strong_resistance_145 = any(abs(current_price - r) / current_price <= proximity_threshold for r in [resistance])
        strong_resistance_360 = any(abs(current_price - r) / current_price <= proximity_threshold for r in [resistance])

        if (ceiling_hit_145 or ceiling_hit_360) and (strong_resistance_145 or strong_resistance_360):
            entry = float(current_price)
            nearest_resistance = resistance * 1.01
            nearest_support = support * 1.01
            sl = nearest_resistance
            tp = nearest_support
            rr = round((entry - tp) / (sl - entry), 2) if (sl - entry) != 0 else 0
            position_size = calculate_position_size(10000, 1, entry, sl)
            result = {
                "نوع معامله": "Short (نوسان‌گیری رگرسیون)",
                "نماد": symbol,
                "تایم‌فریم": tf,
                "قیمت ورود": entry,
                "حد ضرر": sl,
                "هدف سود": tp,
                "ریسک به ریوارد": rr,
                "حجم پوزیشن": position_size,
                "سطح اطمینان": 70,
                "قدرت سیگنال": "متوسط",
                "تحلیل": f"قیمت به سقف رگرسیون رسید (۱۴۵ روز: ceiling={ceiling_145:.2f}, ۳۶۰ روز: ceiling={ceiling_360:.2f}) | سطح مقاومتی قوی وجود دارد",
                "روانشناسی": psych_short,
                "روند بازار": "نزولی (بر اساس رگرسیون)",
                "فاندامنتال": f"امتیاز: {fundamental_score}"
            }
            logging.info(f"سیگنال نوسان‌گیری Short (رگرسیون) برای {symbol} @ {tf}: {result}")
            return result

    if result is None:
        logging.info(f"نماد {symbol} @ {tf}: هیچ سیگنالی تولید نشد")
    return result

# اسکن همه نمادها
async def scan_all_crypto_symbols(on_signal=None):
    exchange = ccxt.kucoin({'enableRateLimit': True, 'rateLimit': 2000})
    try:
        await exchange.load_markets()
        top_coins = get_top_500_symbols_from_cmc()
        usdt_symbols = [s for s in exchange.symbols if any(s.startswith(f"{coin}/") and s.endswith("/USDT") for coin in top_coins)]
        logging.info(f"تعداد نمادهای USDT: {len(usdt_symbols)}")
        chunk_size = 10
        total_chunks = (len(usdt_symbols) + chunk_size - 1) // chunk_size
        for idx in range(total_chunks):
            chunk = usdt_symbols[idx*chunk_size:(idx+1)*chunk_size]
            logging.info(f"اسکن دسته {idx+1}/{total_chunks}: {chunk}")
            tasks = [analyze_symbol(exchange, sym, tf) for sym in chunk for tf in TIMEFRAMES]
            async with semaphore:
                results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logging.error(f"خطا در تسک: {result}")
                    continue
                if result and on_signal:
                    await on_signal(result)
            await asyncio.sleep(WAIT_BETWEEN_CHUNKS)
        logging.info(f"آمار رد شدن‌ها: نقدینگی={LIQUIDITY_REJECTS}, حجم={VOLUME_REJECTS}, حمایت/مقاومت={SR_REJECTS}")
    finally:
        await exchange.close()

# بک‌تست جدید
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
        return win_rate
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
    exchange = ccxt.kucoin({'enableRateLimit': True, 'rateLimit': 2000})
    await exchange.load_markets()
    result = await analyze_symbol(exchange, "BTC/USDT", "1h")
    if result:
        logging.info(f"سیگنال تولید شد: {result}")
    else:
        logging.info("هیچ سیگنالی تولید نشد.")
    await exchange.close()

if __name__ == "__main__":
    asyncio.run(main())