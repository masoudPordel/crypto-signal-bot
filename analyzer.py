import os
import requests
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import ccxt.async_support as ccxt
import asyncio
import time
import logging
import traceback
from datetime import datetime, timedelta
from sklearn.tree import DecisionTreeClassifier
from typing import Optional, Dict, Any, List
from tenacity import retry, stop_after_attempt, wait_exponential
import aiohttp

# تنظیم لاگ‌ها
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s - [File: %(filename)s | Line: %(lineno)d | Func: %(funcName)s]',
    handlers=[
        logging.FileHandler("debug_detailed.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# تنظیمات ثابت
CMC_API_KEY = "c1b589e6-5f67-46bd-9cfe-a34f925bc4cb"
COINMARKETCAL_API_KEY = "iFrSo3PUBJ36P8ZnEIBMvakO5JutSIU1XJvG7ALa"
TIMEFRAMES = ["15m", "1h", "4h", "1d"]
VOLUME_WINDOW = 20
CACHE = {}
CACHE_TTL = 600
MAX_CONCURRENT_REQUESTS = 10
WAIT_BETWEEN_REQUESTS = 0.5
WAIT_BETWEEN_CHUNKS = 3
LIQUIDITY_REJECTS = 0
VOLUME_REJECTS = 0
SR_REJECTS = 0

# سمافر برای کنترل درخواست‌ها
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# تابع دریافت آیدی کوین از CMC
def get_coin_id(symbol: str) -> Optional[int]:
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/map"
    headers = {'Accepts': 'application/json', 'X-CMC_PRO_API_KEY': CMC_API_KEY}
    params = {'symbol': symbol}
    try:
        logging.debug(f"شروع دریافت آیدی برای {symbol}")
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

# تابع دریافت 500 نماد برتر از CMC
def get_top_500_symbols_from_cmc() -> List[str]:
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    headers = {'Accepts': 'application/json', 'X-CMC_PRO_API_KEY': CMC_API_KEY}
    params = {'start': '1', 'limit': '500', 'convert': 'USD'}
    try:
        logging.debug(f"شروع دریافت ۵۰۰ نماد برتر از CMC")
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        data = resp.json()
        logging.info(f"دریافت ۵۰۰ نماد برتر از CMC: تعداد={len(data['data'])}")
        return [entry['symbol'] for entry in data['data']]
    except Exception as e:
        logging.error(f"خطا در دریافت از CMC: {e}")
        return []

# تابع بررسی رویدادهای بازار
def check_market_events(symbol: str) -> int:
    coin_id = get_coin_id(symbol.split('/')[0])
    if not coin_id:
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
        logging.debug(f"شروع دریافت رویدادها برای {symbol}, coin_id={coin_id}")
        time.sleep(0.5)
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        events = resp.json()
        event_score = 0
        if not events or "body" not in events or not events["body"]:
            return 0
        for event in events["body"]:
            title = event.get("title", "")
            description = event.get("description", "")
            if isinstance(title, dict):
                title = title.get("en", "")
            if isinstance(description, dict):
                description = description.get("en", "")
            title = title.lower() if isinstance(title, str) else ""
            description = description.lower() if isinstance(description, str) else ""
            if "burn" in title or "token burn" in description:
                event_score += 15
            elif "listing" in title or "exchange" in description:
                event_score += 10
            elif "partnership" in title or "collaboration" in description:
                event_score += 5
            elif "hack" in title or "security breach" in description:
                event_score -= 20
            elif "lawsuit" in title or "negative" in description:
                event_score -= 15
        return event_score
    except Exception as e:
        logging.error(f"خطا در دریافت رویدادها برای {symbol}: {e}")
        return 0

# تابع دریافت شاخص ترس و طمع
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def get_fear_and_greed_index() -> int:
    url = "https://api.alternative.me/fng/?limit=1"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=10) as resp:
            data = await resp.json()
            value = int(data["data"][0]["value"])
            logging.info(f"شاخص ترس و طمع دریافت شد: {value}")
            return value
    return 50

# تابع دریافت دامیننس تتر
async def get_usdt_dominance_data() -> List[float]:
    try:
        url = "https://api.coingecko.com/api/v3/coins/tether/market_chart?vs_currency=usd&days=7"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                data = await resp.json()
                dominance = [d[1] / 1e9 for d in data.get("market_caps", [])]
                logging.info(f"داده‌های دامیننس تتر دریافت شد: تعداد داده‌ها={len(dominance)}")
                return dominance
    except Exception as e:
        logging.error(f"خطا در دریافت دامیننس تتر: {e}")
        return [0.0] * 5

# کلاس‌های محاسبات اندیکاتورها و تشخیص الگوها
class IndicatorCalculator:
    @staticmethod
    def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        rs = gain / loss.replace(0, 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        tr = pd.concat([df["high"] - df["low"], abs(df["high"] - df["close"].shift()), abs(df["low"] - df["close"].shift())], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def compute_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> tuple:
        sma = df["close"].rolling(period).mean()
        std = df["close"].rolling(period).std()
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        return upper, lower

    @staticmethod
    def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
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
        return adx

    @staticmethod
    def compute_stochastic(df: pd.DataFrame, period: int = 14) -> pd.Series:
        low_min = df["low"].rolling(window=period).min()
        high_max = df["high"].rolling(window=period).max()
        k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, 1e-10)
        return k

    @staticmethod
    def compute_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        raw_money_flow = typical_price * df['volume']
        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0).rolling(period).sum()
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0).rolling(period).sum()
        mfi = 100 - (100 / (1 + positive_flow / negative_flow.replace(0, 1e-10)))
        return mfi

class PatternDetector:
    @staticmethod
    def detect_pin_bar(df: pd.DataFrame) -> pd.Series:
        df["body"] = abs(df["close"] - df["open"])
        df["range"] = df["high"] - df["low"]
        df["upper"] = df["high"] - df[["close", "open"]].max(axis=1)
        df["lower"] = df[["close", "open"]].min(axis=1) - df["low"]
        pin_bar = (df["body"] < 0.3 * df["range"]) & ((df["upper"] > 2 * df["body"]) | (df["lower"] > 2 * df["body"]))
        return pin_bar

    @staticmethod
    def detect_engulfing(df: pd.DataFrame) -> pd.Series:
        prev_o = df["open"].shift(1)
        prev_c = df["close"].shift(1)
        engulfing = (((df["close"] > df["open"]) & (prev_c < prev_o) & (df["close"] > prev_o) & (df["open"] < prev_c)) |
                     ((df["close"] < df["open"]) & (prev_c > prev_o) & (df["close"] < prev_o) & (df["open"] > prev_c)))
        return engulfing

    @staticmethod
    def detect_elliott_wave(df: pd.DataFrame) -> pd.DataFrame:
        df["WavePoint"] = np.nan
        highs = argrelextrema(df['close'].values, np.greater, order=5)[0]
        lows = argrelextrema(df['close'].values, np.less, order=5)[0]
        df.loc[df.index[highs], "WavePoint"] = df.loc[df.index[highs], "close"]
        df.loc[df.index[lows], "WavePoint"] = df.loc[df.index[lows], "close"]
        df["WaveTrend"] = np.nan
        df["WaveTrend"] = df["WaveTrend"].astype("object")
        wave_points = df["WavePoint"].dropna().index
        if len(wave_points) >= 5:
            recent_points = df.loc[wave_points[-5:], "close"]
            if recent_points.is_monotonic_increasing:
                df.loc[wave_points[-1], "WaveTrend"] = "Up"
            elif recent_points.is_monotonic_decreasing:
                df.loc[wave_points[-1], "WaveTrend"] = "Down"
        return df

    @staticmethod
    def detect_support_resistance(df: pd.DataFrame, window: int = 10) -> tuple:
        if len(df) < window:
            logging.warning(f"داده ناکافی برای تشخیص حمایت/مقاومت: {len(df)} کندل")
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
            recent_resistance = df['close'].iloc[-20:].mean() * 1.02
            logging.warning(f"مقاومت پیش‌فرض برای {len(df)} کندل تنظیم شد: {recent_resistance}")
        if recent_support == 0 or pd.isna(recent_support):
            recent_support = df['close'].iloc[-20:].mean() * 0.98
            logging.warning(f"حمایت پیش‌فرض برای {len(df)} کندل تنظیم شد: {recent_support}")
        volume_profile = df['volume'].groupby(df['close'].round(2)).sum()
        vol_threshold = volume_profile.quantile(0.5)
        high_vol_levels = volume_profile[volume_profile > vol_threshold].index.tolist()
        return recent_support, recent_resistance, high_vol_levels

    @staticmethod
    def detect_rsi_divergence(df: pd.DataFrame, lookback: int = 10) -> tuple:
        rsi = IndicatorCalculator.compute_rsi(df)
        prices = df['close']
        recent_lows_price = argrelextrema(prices.values, np.less, order=lookback)[0]
        recent_highs_price = argrelextrema(prices.values, np.greater, order=lookback)[0]
        recent_lows_rsi = argrelextrema(rsi.values, np.less, order=lookback)[0]
        recent_highs_rsi = argrelextrema(rsi.values, np.greater, order=lookback)[0]
        bullish_divergence = False
        bearish_divergence = False
        if len(recent_lows_price) > 1 and len(recent_lows_rsi) > 1:
            last_price_low = prices.iloc[recent_lows_price[-1]]
            prev_price_low = prices.iloc[recent_lows_price[-2]]
            last_rsi_low = rsi.iloc[recent_lows_rsi[-1]]
            prev_rsi_low = rsi.iloc[recent_lows_rsi[-2]]
            bullish_divergence = (last_price_low < prev_price_low * 1.02) and (last_rsi_low > prev_rsi_low * 0.98)
        if len(recent_highs_price) > 1 and len(recent_highs_rsi) > 1:
            last_price_high = prices.iloc[recent_highs_price[-1]]
            prev_price_high = prices.iloc[recent_highs_price[-2]]
            last_rsi_high = rsi.iloc[recent_highs_rsi[-1]]
            prev_rsi_high = rsi.iloc[recent_highs_rsi[-2]]
            bearish_divergence = (last_price_high > prev_price_high * 0.98) and (last_rsi_high < prev_rsi_high * 1.02)
        return bullish_divergence, bearish_divergence

    @staticmethod
    def detect_head_and_shoulders(df: pd.DataFrame, price_col: str = "close") -> int:
        data = df[price_col].values
        max_idx = argrelextrema(np.array(data), np.greater)[0]
        min_idx = argrelextrema(np.array(data), np.less)[0]
        if len(max_idx) >= 3:
            for i in range(1, len(max_idx) - 1):
                left = data[max_idx[i - 1]]
                head = data[max_idx[i]]
                right = data[max_idx[i + 1]]
                if head > left and head > right and abs(left - right) < 0.02 * head:
                    logging.info(f"الگوی سر و شانه نزولی شناسایی شد برای {df.index[-1]}")
                    return -5
        if len(min_idx) >= 3:
            for i in range(1, len(min_idx) - 1):
                left = data[min_idx[i - 1]]
                head = data[min_idx[i]]
                right = data[min_idx[i + 1]]
                if head < left and head < right and abs(left - right) < 0.02 * head:
                    logging.info(f"الگوی سر و شانه معکوس شناسایی شد برای {df.index[-1]}")
                    return 5
        return 0

    @staticmethod
    def detect_double_top(df: pd.DataFrame, price_col: str = "close") -> int:
        data = df[price_col].values
        max_idx = argrelextrema(np.array(data), np.greater)[0]
        if len(max_idx) < 2:
            return 0
        for i in range(len(max_idx) - 1):
            first = data[max_idx[i]]
            second = data[max_idx[i + 1]]
            if abs(first - second) < 0.02 * first:
                logging.info(f"الگوی دابل تاپ شناسایی شد برای {df.index[-1]}")
                return -3
        return 0

    @staticmethod
    def detect_double_bottom(df: pd.DataFrame, price_col: str = "close") -> int:
        data = df[price_col].values
        min_idx = argrelextrema(np.array(data), np.less)[0]
        if len(min_idx) < 2:
            return 0
        for i in range(len(min_idx) - 1):
            first = data[min_idx[i]]
            second = data[min_idx[i + 1]]
            if abs(first - second) < 0.02 * first:
                logging.info(f"الگوی دابل باتم شناسایی شد برای {df.index[-1]}")
                return 3
        return 0

class SignalFilter:
    def __init__(self):
        self.model = DecisionTreeClassifier(max_depth=3)
        self.trained = False

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(X) > 0 and len(y) > 0:
            self.model.fit(X, y)
            self.trained = True
            logging.info("Decision Tree آموزش داده شد.")
        else:
            logging.warning("داده‌های آموزشی برای Decision Tree کافی نیست.")

    def predict(self, features: list) -> float:
        if not self.trained:
            logging.warning("Decision Tree آموزش داده نشده است. پیش‌فرض=True")
            return 10
        try:
            prediction = self.model.predict_proba([features])[0][1]
            score = prediction * 20
            logging.debug(f"پیش‌بینی Decision Tree: features={features}, score={score:.2f}")
            return score
        except Exception as e:
            logging.error(f"خطا در پیش‌بینی Decision Tree: {e}")
            return 0

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def get_ohlcv_cached(exchange, symbol, tf, limit=50) -> Optional[pd.DataFrame]:
    try:
        key = f"{exchange.id}_{symbol}_{tf}"
        now = datetime.utcnow()
        if key in CACHE:
            cached_df, cached_time = CACHE[key]
            if (now - cached_time).total_seconds() < CACHE_TTL:
                return cached_df
        raw_data = await exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
        if not raw_data or len(raw_data) == 0:
            logging.warning(f"داده OHLCV برای {symbol} / {tf} خالی یا ناموجود است")
            return None
        df = pd.DataFrame(raw_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
        if df["timestamp"].isnull().all():
            logging.error(f"تمامی تایم‌استمپ‌ها برای {symbol} / {tf} نامعتبر هستند")
            return None
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df.set_index("timestamp", inplace=True)
        CACHE[key] = (df, now)
        logging.info(f"داده OHLCV برای {symbol} / {tf} با موفقیت دریافت و کش شد: تعداد کندل‌ها={len(df)}")
        return df
    except Exception as e:
        logging.error(f"خطا در گرفتن OHLCV برای {symbol} / {tf}: {e}")
        return None

async def get_live_price(exchange: ccxt.Exchange, symbol: str, max_attempts: int = 3) -> Optional[float]:
    attempt = 0
    while attempt < max_attempts:
        try:
            ticker = await exchange.fetch_ticker(symbol)
            bid = ticker.get('bid')
            ask = ticker.get('ask')
            last = ticker.get('last')
            if bid is None or ask is None or last is None or bid <= 0 or ask <= 0:
                logging.warning(f"داده قیمت واقعی برای {symbol} نامعتبر است: bid={bid}, ask={ask}, last={last}")
                attempt += 1
                await asyncio.sleep(0.3)
                continue
            live_price = (bid + ask) / 2 if bid and ask else last
            logging.info(f"قیمت واقعی بازار برای {symbol}: live_price={live_price}, bid={bid}, ask={ask}, last={last}")
            return live_price
        except Exception as e:
            logging.error(f"خطا در دریافت قیمت واقعی برای {symbol} در تلاش {attempt + 1}: {e}")
            attempt += 1
            await asyncio.sleep(0.3)
    try:
        df_1m = await get_ohlcv_cached(exchange, symbol, "1m")
        if df_1m is not None and len(df_1m) > 0:
            fallback_price = df_1m["close"].iloc[-1]
            logging.warning(f"قیمت واقعی برای {symbol} دریافت نشد، از قیمت کندل 1m استفاده شد: {fallback_price}")
            return fallback_price
    except Exception as e:
        logging.error(f"خطا در دریافت قیمت پیش‌فرض برای {symbol}: {e}")
    logging.error(f"ناتوانی در دریافت قیمت برای {symbol} پس از {max_attempts} تلاش")
    return None

async def check_liquidity(exchange: ccxt.Exchange, symbol: str, df: pd.DataFrame) -> tuple:
    global LIQUIDITY_REJECTS
    try:
        ticker = await exchange.fetch_ticker(symbol)
        bid = ticker.get('bid')
        ask = ticker.get('ask')
        if bid is None or ask is None or bid == 0 or ask == 0:
            logging.warning(f"داده نقدینگی برای {symbol} نامعتبر است: bid={bid}, ask={ask}")
            return float('inf'), 0
        spread = (ask - bid) / ((bid + ask) / 2)
        spread_history = []
        for _ in range(5):
            try:
                past_ticker = await exchange.fetch_ticker(symbol)
                past_bid = past_ticker.get('bid')
                past_ask = past_ticker.get('ask')
                if past_bid is None or past_ask is None or past_bid == 0 or past_ask == 0:
                    continue
                past_spread = (past_ask - past_bid) / ((past_bid + past_ask) / 2)
                spread_history.append(past_spread)
            except Exception as e:
                logging.warning(f"خطا در دریافت داده گذشته برای {symbol}: {e}")
                continue
        spread_mean = np.mean(spread_history) if spread_history else 0.02
        spread_std = np.std(spread_history) if spread_history else 0.005
        spread_threshold = spread_mean + spread_std
        if spread > spread_threshold:
            logging.warning(f"اسپرد برای {symbol} بیش از حد بالاست: spread={spread:.4f}")
            LIQUIDITY_REJECTS += 1
            return spread, -10
        score = 10 if spread < spread_threshold else -5
        logging.info(f"نقدینگی {symbol}: spread={spread:.4f}, threshold={spread_threshold:.4f}, score={score}")
        return spread, score
    except Exception as e:
        logging.error(f"خطا در بررسی نقدینگی برای {symbol}: {e}")
        return float('inf'), 0

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.ffill().bfill().fillna(0)
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
        df["MFI"] = IndicatorCalculator.compute_mfi(df)
        df["Hammer"] = ((df["close"] - df["low"]) / (df["high"] - df["low"]) > 0.66) & (df["close"] > df["open"])
        df["Doji"] = abs(df["close"] - df["open"]) / (df["high"] - df["low"]) < 0.1
        logging.debug(f"اندیکاتورها با موفقیت محاسبه شدند: {list(df.columns)}")
    except Exception as e:
        logging.error(f"خطا در محاسبه اندیکاتورها: {e}")
        df["EMA12"] = df["close"].mean()
        df["EMA26"] = df["close"].mean()
        df["MACD"] = 0
        df["Signal"] = 0
        df["RSI"] = 50
        df["ATR"] = 0
        df["ADX"] = 0
        df["Stochastic"] = 50
        df["BB_upper"] = df["close"].mean() * 1.1
        df["BB_lower"] = df["close"].mean() * 0.9
        df["PinBar"] = False
        df["Engulfing"] = False
        df["WavePoint"] = np.nan
        df["MFI"] = 50
        df["Hammer"] = False
        df["Doji"] = False
        logging.warning(f"اندیکاتورها با مقادیر پیش‌فرض پر شدند")
    return df

def calculate_fibonacci_levels(df: pd.DataFrame, high_col: str = "high", low_col: str = "low") -> Dict[str, float]:
    max_price = df[high_col].max()
    min_price = df[low_col].min()
    diff = max_price - min_price
    return {
        "0.236": max_price - 0.236 * diff,
        "0.382": max_price - 0.382 * diff,
        "0.5": max_price - 0.5 * diff,
        "0.618": max_price - 0.618 * diff,
        "0.786": max_price - 0.786 * diff,
    }

def get_moving_average_score(df: pd.DataFrame, price_col: str = "close") -> int:
    try:
        ma50 = df[price_col].rolling(window=50).mean()
        ma100 = df[price_col].rolling(window=100).mean()
        ma200 = df[price_col].rolling(window=200).mean()
        if len(df) < 200:
            logging.warning(f"داده ناکافی برای محاسبه MA200: تعداد کندل‌ها={len(df)}")
            return 0
        score = 0
        if df[price_col].iloc[-1] > ma200.iloc[-1]:
            score += 5
        else:
            score -= 5
        if ma50.iloc[-1] > ma100.iloc[-1] and ma100.iloc[-1] > ma200.iloc[-1]:
            score += 3
        logging.info(f"امتیاز میانگین‌های متحرک: score={score}")
        return score
    except Exception as e:
        logging.error(f"خطا در محاسبه امتیاز میانگین‌ها: {e}")
        return 0

def get_usdt_dominance_score(dominance_data: List[float]) -> int:
    try:
        if len(dominance_data) < 5:
            logging.warning("داده ناکافی برای محاسبه امتیاز دامیننس تتر")
            return 0
        recent = dominance_data[-1]
        previous = dominance_data[-5]
        if recent < previous:
            return 5
        elif recent > previous:
            return -5
        return 0
    except Exception as e:
        logging.error(f"خطا در محاسبه امتیاز دامیننس تتر: {e}")
        return 0

async def find_entry_point(exchange: ccxt.Exchange, symbol: str, signal_type: str, support: float, resistance: float) -> Optional[Dict]:
    try:
        logging.info(f"شروع پیدا کردن نقطه ورود برای {symbol} در 15m - نوع سیگنال: {signal_type} - {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}")
        df_15m = await get_ohlcv_cached(exchange, symbol, "15m", limit=50)
        df_1h = await get_ohlcv_cached(exchange, symbol, "1h", limit=50)
        if df_15m is None or len(df_15m) < 20 or df_1h is None or len(df_1h) < 20:
            logging.warning(f"نقطه ورود برای {symbol} در 15m پیدا نشد: دلیل=داده ناکافی")
            return None
        df_15m = compute_indicators(df_15m)
        df_1h = compute_indicators(df_1h)
        last_15m = df_15m.iloc[-1]
        last_1h = df_1h.iloc[-1]
        live_price = await get_live_price(exchange, symbol)
        if live_price is None:
            logging.warning(f"نقطه ورود برای {symbol} در 15m پیدا نشد: دلیل=عدم دریافت قیمت واقعی")
            return None
        price_diff = abs(live_price - last_15m["close"]) / live_price if live_price != 0 else float('inf')
        if price_diff > 0.03:
            logging.warning(f"نقطه ورود برای {symbol} در 15m پیدا نشد: دلیل=اختلاف قیمت زیاد, live_price={live_price:.6f}, candle_price={last_15m['close']:.6f}")
            return None
        volume_mean = df_15m["volume"].rolling(20).mean().iloc[-1]
        volume_condition = last_15m["volume"] > volume_mean * 0.5
        if not volume_condition:
            logging.info(f"نقطه ورود برای {symbol} در 15m پیدا نشد: دلیل=حجم ناکافی")
            return None
        price_action = last_15m.get("PinBar", False) or last_15m.get("Engulfing", False) or last_15m.get("Hammer", False) or last_15m.get("Doji", False)
        if not price_action:
            logging.info(f"نقطه ورود برای {symbol} در 15m پیدا نشد: دلیل=عدم وجود الگوی قیمتی")
            return None
        fib_levels = calculate_fibonacci_levels(df_1h)
        volatility = last_15m["ATR"] / last_15m["close"]
        adx_strength = last_1h["ADX"]
        fng_index = await get_fear_and_greed_index()
        atr_multiplier_sl = 0.6 if volatility > 0.02 or adx_strength > 30 else 0.75
        atr_multiplier_tp = 2.0 if volatility > 0.02 or adx_strength > 30 else 2.5
        if fng_index < 25:
            atr_multiplier_tp += 0.5
        elif fng_index > 75:
            atr_multiplier_sl += 0.25
        entry_price = live_price
        atr_15m = last_15m["ATR"]
        if signal_type == "Long":
            sl = entry_price - (atr_15m * atr_multiplier_sl)
            if support > 0 and sl < support * 0.98:
                sl = support * 0.98
            tp = entry_price + (atr_15m * atr_multiplier_tp)
            if resistance > 0 and tp > resistance * 1.02:
                tp = min(resistance * 1.02, fib_levels["0.618"])
            elif fib_levels["0.618"] < tp:
                tp = fib_levels["0.618"]
        elif signal_type == "Short":
            sl = entry_price + (atr_15m * atr_multiplier_sl)
            if resistance > 0 and sl > resistance * 1.02:
                sl = resistance * 1.02
            tp = entry_price - (atr_15m * atr_multiplier_tp)
            if support > 0 and tp < support * 0.98:
                tp = max(support * 0.98, fib_levels["0.382"])
            elif fib_levels["0.382"] > tp:
                tp = fib_levels["0.382"]
        else:
            logging.warning(f"نوع سیگنال نامعتبر برای {symbol}: {signal_type}")
            return None
        rr_ratio = (tp - entry_price) / (entry_price - sl) if signal_type == "Long" else (entry_price - tp) / (sl - entry_price)
        min_rr = 1.5 if adx_strength < 25 else 1.2
        if rr_ratio < min_rr:
            logging.warning(f"نقطه ورود برای {symbol} رد شد: دلیل=RR کمتر از {min_rr}, RR={rr_ratio:.2f}")
            return None
        order_book = await exchange.fetch_order_book(symbol, limit=10)
        bid_volume = sum([x[1] for x in order_book["bids"]])
        ask_volume = sum([x[1] for x in order_book["asks"]])
        if bid_volume + ask_volume < 1000:
            logging.warning(f"نقطه ورود برای {symbol} رد شد: دلیل=نقدینگی پایین")
            return None
        logging.info(f"نقطه ورود برای {symbol}: Entry={entry_price:.6f}, SL={sl:.6f}, TP={tp:.6f}, RR={rr_ratio:.2f}")
        return {
            "entry_price": entry_price,
            "sl": sl,
            "tp": tp,
            "rr": rr_ratio
        }
    except Exception as e:
        logging.error(f"خطا در پیدا کردن نقطه ورود برای {symbol}: {str(e)}")
        return None

async def manage_trailing_stop(exchange: ccxt.Exchange, symbol: str, entry_price: float, sl: float, tp: float, signal_type: str, trail_percentage: float = 0.5):
    logging.info(f"شروع Trailing Stop برای {symbol} ({signal_type}): ورود={entry_price}, SL اولیه={sl}, TP={tp}")
    while True:
        try:
            live_price = await get_live_price(exchange, symbol)
            if live_price is None:
                logging.warning(f"قیمت واقعی برای {symbol} دریافت نشد، 60 ثانیه صبر می‌کنم")
                await asyncio.sleep(60)
                continue
            if (signal_type == "Long" and (live_price <= sl or live_price >= tp)) or \
               (signal_type == "Short" and (live_price >= sl or live_price <= tp)):
                logging.info(f"Trailing Stop برای {symbol} متوقف شد: live_price={live_price}, SL={sl}, TP={tp}")
                break
            if (live_price > entry_price and signal_type == "Long") or (live_price < entry_price and signal_type == "Short"):
                trail_amount = live_price * (trail_percentage / 100)
                new_sl = live_price - trail_amount if signal_type == "Long" else live_price + trail_amount
                if (signal_type == "Long" and new_sl > sl) or (signal_type == "Short" and new_sl < sl):
                    sl = new_sl
                    logging.info(f"Trailing Stop برای {symbol} به‌روزرسانی شد: SL={sl}, Live Price={live_price}")
            await asyncio.sleep(300)
        except Exception as e:
            logging.error(f"خطا در مدیریت Trailing Stop برای {symbol}: {e}")
            await asyncio.sleep(30)

async def multi_timeframe_confirmation(exchange: ccxt.Exchange, symbol: str, base_tf: str) -> float:
    weights = {"1d": 0.4, "4h": 0.3, "1h": 0.2, "15m": 0.1}
    total_weight = 0
    score = 0
    for tf, weight in weights.items():
        if tf == base_tf:
            continue
        try:
            df_tf = await get_ohlcv_cached(exchange, symbol, tf)
            if df_tf is None or (len(df_tf) < 50 and tf != "1d") or (len(df_tf) < 30 and tf == "1d"):
                logging.warning(f"داده برای {symbol} @ {tf} در تأیید مولتی تایم‌فریم ناکافی است")
                continue
            df_tf = compute_indicators(df_tf)
            long_trend = df_tf["EMA12"].iloc[-1] > df_tf["EMA26"].iloc[-1]  # اصلاح: استفاده از df_tf
            score += (weight * 10) if long_trend else (-weight * 5)
            total_weight += weight
        except Exception as e:
            logging.error(f"خطا در پردازش تایم‌فریم {tf} برای {symbol}: {e}")
            continue
    final_score = score if total_weight > 0 else 0
    logging.info(f"مولتی تایم‌فریم برای {symbol}: score={final_score:.2f}")
    return final_score

def calculate_position_size(account_balance: float, risk_percentage: float, entry: float, stop_loss: float) -> float:
    if entry is None or stop_loss is None or entry == 0 or stop_loss == 0:
        logging.warning(f"مقادیر نامعتبر برای محاسبه حجم پوزیشن: entry={entry}, stop_loss={stop_loss}")
        return 0
    try:
        risk_amount = account_balance * (risk_percentage / 100)
        distance = abs(entry - stop_loss)
        position_size = risk_amount / distance if distance != 0 else 0
        return round(position_size, 2)
    except Exception as e:
        logging.error(f"خطا در محاسبه حجم پوزیشن: {e}")
        return 0

def ablation_test(symbol_results: List[Optional[Dict]], filter_name: str) -> int:
    total_signals = len([r for r in symbol_results if r is not None])
    logging.info(f"Ablation Test برای فیلتر {filter_name}: تعداد سیگنال‌های اولیه={total_signals}")
    return total_signals

async def analyze_market_structure(exchange: ccxt.Exchange, symbol: str) -> Dict:
    try:
        df_4h = await get_ohlcv_cached(exchange, symbol, "4h", limit=50)
        if df_4h is None or len(df_4h) < 20:
            logging.warning(f"داده ناکافی برای تحلیل ساختار بازار {symbol}")
            return {"trend": "Neutral", "score": 0, "support": 0, "resistance": 0, "fng_index": 50}
        df_4h = compute_indicators(df_4h)
        trend = "Up" if df_4h["EMA12"].iloc[-1] > df_4h["EMA26"].iloc[-1] else "Down"
        score = 10 if trend == "Up" else -10
        support, resistance, _ = PatternDetector.detect_support_resistance(df_4h)
        fng_index = await get_fear_and_greed_index()
        return {
            "trend": trend,
            "score": score,
            "support": support,
            "resistance": resistance,
            "fng_index": fng_index
        }
    except Exception as e:
        logging.error(f"خطا در تحلیل ساختار بازار برای {symbol}: {e}")
        return {"trend": "Neutral", "score": 0, "support": 0, "resistance": 0, "fng_index": 50}

async def analyze_symbol(exchange: ccxt.Exchange, symbol: str, tf: str) -> Optional[Dict]:
    global LIQUIDITY_REJECTS, VOLUME_REJECTS, SR_REJECTS
    start_time = time.time()
    logging.info(f"شروع تحلیل {symbol} @ {tf}, زمان شروع={datetime.now()}")

    try:
        market_structure = await analyze_market_structure(exchange, symbol)
        trend_4h = market_structure["trend"]
        trend_score_4h = market_structure["score"]
        support_4h = market_structure["support"]
        resistance_4h = market_structure["resistance"]
        fng_index = market_structure.get("fng_index", 50)

        if tf != "1h":
            logging.info(f"تحلیل برای {symbol} در تایم فقط 1h انجام می‌شود. تایم‌فریم فعلی: {tf}")
            return None

        df = await get_ohlcv_cached(exchange, symbol, tf, limit=50)
        if df is None or len(df) < 30:
            logging.warning(f"داده ناکافی برای {symbol} @ {tf}: تعداد کندل‌ها={len(df) if df is not None else 0}")
            return None
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            logging.error(f"ستون‌های مورد نیاز در دیتافریم {symbol} @ {tf} وجود ندارند")
            return None

        df = df.ffill().bfill().fillna(0)
        df = compute_indicators(df)
        last = df.iloc[-1]

        score_long = 0
        score_short = 0.0
        score_log = {"long": {}, "short": {}}

        df_1d = await get_ohlcv_cached(exchange, symbol, "1d", limit=50)
        trend_score_1d = 0
        if df_1d is not None and len(df_1d) > 0:
            df_1d = compute_indicators(df_1d)
            long_trend_1d = df_1d["EMA12"].iloc[-1] > df_1d["EMA26"].iloc[-1]
            trend_score_1d = 10 if long_trend_1d else -10
            logging.info(f"تأیید روند 1d برای {symbol}: score={trend_score_1d}")

        vol_mean = df["volume"].rolling(VOLUME_WINDOW).mean().iloc[-1]
        current_vol = df["volume"].iloc[-1]
        vol_std = df["volume"].rolling(20).std().iloc[-1]
        vol_threshold = vol_mean * 0.5
        vol_score = 10 if current_vol >= vol_threshold else -2
        score_long += vol_score
        score_short += vol_score
        score_log["long"]["volume"] = vol_score
        score_log["short"]["volume"] = vol_score
        if current_vol < vol_threshold:
            VOLUME_REJECTS += 1

        atr_1h = df["ATR"].iloc[-1]
        risk_buffer = atr_1h * 2
        dynamic_rr = 2.0
        if support_4h > 0 and resistance_4h > 0:
            dynamic_rr = max(dynamic_rr, (resistance_4h - support_4h) / risk_buffer)
        logging.info(f"نسبت RR پویا برای {symbol}: RR={dynamic_rr}")

        volatility = df["ATR"].iloc[-1] / last["close"]
        vola_mean = (df["ATR"] / df["close"]).rolling(20).mean().iloc[-1]
        vola_std = (df["ATR"] / df["close"]).rolling(20).std().iloc[-1]
        vola_threshold = vola_mean + vola_std
        vola_score = 10 if volatility > vola_threshold else -5
        score_long += vola_score
        score_short += vola_score
        score_log["long"]["volatility"] = vola_score
        score_log["short"]["volatility"] = vola_score

        adx_mean = df["ADX"].rolling(20).mean().iloc[-1]
        adx_std = df["ADX"].rolling(20).std().iloc[-1]
        adx_threshold = adx_mean + adx_std
        adx_score = 15 if last["ADX"] >= adx_threshold else -5
        trend_score = adx_score if adx_mean >= adx_threshold * 1.5 else 0
        score_long += adx_score + trend_score
        score_short += adx_score + trend_score
        score_log["long"]["adx"] = adx_score
        score_log["short"]["adx"] = adx_score
        score_log["long"]["trend"] = trend_score
        score_log["short"]["trend"] = trend_score

        long_trend = trend_score > 0
        trend_score_adjustment = 10 if long_trend else -5
        score_long += trend_score_adjustment
        score_short += -trend_score_adjustment
        score_log["long"]["trend"] = trend_score_adjustment
        score_log["short"]["trend"] = -trend_score_adjustment

        mtf_score = await multi_timeframe_confirmation(exchange, symbol, tf)
        score_long += mtf_score
        score_short += -mtf_score
        score_log["long"]["multi_timeframe"] = mtf_score
        score_log["short"]["multi_timeframe"] = -mtf_score

        support, resistance, vol_levels = PatternDetector.detect_support_resistance(df)
        s_r_buffer = (df["ATR"].iloc[-1] / last["close"]) * 2
        distance_to_resistance = abs(last["close"] - resistance) / last["close"]
        distance_to_support = abs(last["close"] - support) / last["close"]
        sr_score_long = 10 if distance_to_resistance > s_r_buffer else -5
        sr_score_short = 10 if distance_to_support > s_r_buffer else -5
        score_long += sr_score_long
        score_short += sr_score_short
        score_log["long"]["support_resistance"] = sr_score_long
        score_log["short"]["support_resistance"] = sr_score_short
        if distance_to_resistance <= s_r_buffer or distance_to_support <= s_r_buffer:
            SR_REJECTS += 1

        spread, liquidity_score = await check_liquidity(exchange, symbol, df)
        if spread == float('inf'):
            spread = 0.0
        score_long += liquidity_score
        score_short += liquidity_score
        score_log["long"]["liquidity"] = liquidity_score
        score_log["short"]["liquidity"] = liquidity_score
        if liquidity_score < 0:
            logging.warning(f"سیگنال برای {symbol} به دلیل نقدینگی ضعیف رد شد")
            return None

        fundamental_score = check_market_events(symbol)
        score_long += fundamental_score
        score_short += fundamental_score
        score_log["long"]["fundamental"] = fundamental_score
        score_log["short"]["fundamental"] = fundamental_score

        psych_long = "اشباع فروش" if last["RSI"] < 30 else "اشباع خرید" if last["RSI"] > 70 else "متعادل"
        psych_short = "اشباع خرید" if last["RSI"] > 70 else "اشباع فروش" if last["RSI"] < 30 else "متعادل"
        psych_score_long = 10 if psych_long == "اشباع فروش" else -10 if psych_long == "اشباع خرید" else 0
        psych_score_short = 10 if psych_short == "اشباع خرید" else -10 if psych_short == "اشباع فروش" else 0
        score_long += psych_score_long
        score_short += psych_score_short
        score_log["long"]["psychology"] = psych_score_long
        score_log["short"]["psychology"] = psych_score_short

        bullish_rsi_div, bearish_rsi_div = PatternDetector.detect_rsi_divergence(df)
        div_score_long = 10 if bullish_rsi_div else 0
        div_score_short = 10 if bearish_rsi_div else -0
        score_long += div_score_long
        score_short += div_score_short
        score_log["long"]["rsi_divergence"] = div_score_long
        score_log["short"]["rsi_divergence"] = div_score_short

        head_and_shoulders_score = PatternDetector.detect_head_and_shoulders(df)
        score_long += head_and_shoulders_score
        score_short += -head_and_shoulders_score
        score_log["long"]["head_and_shoulders"] = head_and_shoulders_score
        score_log["short"]["head_and_shoulders"] = -head_and_shoulders_score

        double_top_score = PatternDetector.detect_double_top(df)
        score_long += double_top_score
        score_short += -double_top_score
        score_log["long"]["double_top"] = double_top_score
        score_log["short"]["double_top"] = -double_top_score

        double_bottom_score = PatternDetector.detect_double_bottom(df)
        score_long += double_bottom_score
        score_short += -double_bottom_score
        score_log["long"]["double_bottom"] = double_bottom_score
        score_log["short"]["double_bottom"] = -double_bottom_score

        ma_score = get_moving_average_score(df)
        score_long += ma_score
        score_short += -ma_score
        score_log["long"]["moving_average"] = ma_score
        score_log["short"]["moving_average"] = -ma_score

        usdt_dominance_data = await get_usdt_dominance_data()
        usdt_score = get_usdt_dominance_score(usdt_dominance_data)
        score_long += usdt_score
        score_short += -usdt_score
        score_log["long"]["usdt_dominance"] = usdt_score
        score_log["short"]["usdt_score"] = -usdt_score

        support_buffer = (df["ATR"].iloc[-1] / last["close"]) * 1.5
        resistance_buffer = (df["ATR"].iloc[-1] / last["close"]) * 1.5
        min_conditions = 2
        conds_long = {
            "PinBar": last.get("PinBar", False),
            "Engulfing": last.get("Engulfing", False) and last["close"] > last["open"] and (df["volume"].iloc[-1] > df["volume"].rolling(20).mean().iloc[-1] * 1.5),
            "Elliott_Wave": last["WaveTrend"].astype(str).iloc[-1] == "Up",
            "EMA_Cross": df["EMA12"].iloc[-1] > df["EMA26"].iloc[-1] and (df["volume"].iloc[-1] > df["volume"].rolling(20).mean().iloc[-1] * 1.2),
            "MACD_Cross": df["MACD"].iloc[-2] < df["Signal"].iloc[-1] and df["MACD"].iloc[-1] > df["Signal"].iloc[-1] and (df["MACD"].iloc[-1] > 0),
            "RSI_Oversold": last["RSI"] < 30,
            "Stochastic_Oversold": last["Stochastic"] < 15,
            "BB_Breakout": last["close"] > last["BB_upper"] and (df["volume"].iloc[-1] > df["volume"].rolling(20).mean().iloc[-2] * 1.5),
            "MFI_Oversold": last["MFI"] < 15,
            "ADX_Strong": last["ADX"] > 25,
            "Support_Confirmation": abs(last["close"] - support) <= support_buffer and (last["PinBar"] or last["Engulfing"])
        }
        conds_short = {
            "PinBar": last["PinBar"],
            "Engulfing": last["Engulfing"] and last["close"] < last["open"] and (df["volume"].iloc[-1] > df["volume"].rolling(20).mean().iloc[-1] * 1.5),
            "Elliott_Wave": df["WaveTrend"].astype(str).iloc[-1] == "Down",
            "EMA_Cross": df["EMA12"].iloc[-1] < df["EMA26"].iloc[-1] and (df["volume"].iloc[-1] > df["volume"].rolling(20).mean().iloc[-2] * 1.2),
            "MACD_Cross": (df["MACD"].iloc[-2] > df["Signal"].iloc[-2] and df["MACD"].iloc[-1] < df["Signal"].iloc[-1]) and (df["MACD"].iloc[-1] < 0),
            "RSI_Overbought": last["RSI"] > 75,
            "Stochastic_Overbought": last["Stochastic"] > 85,
            "BB_Breakout": last["close"] < last["BB_lower"] and (df["volume"].iloc[-1] > df["volume"].rolling(20).mean().iloc-2] * 1.5),
            "MFI_Overbought": last["MFI"] > 85,
            "ADX_Strong": last["ADX"] > 25,
            "Resistance_Confirmation": abs(last["close"] - resistance) <= resistance_buffer and (last["PinBar"] or last["Engulfing"])
        }

        indicator_score_long = sum([
            10 if conds_long["PinBar"] else 0,
            10 if conds_long["Engulfing"] else 0,
            15 if conds_long["Elliott_Wave"] else 0,
            5 if conds_long["EMA_Cross"] else 0,
            5 if conds_long["MACD_Cross"] else 0,
            3 if conds_long["RSI_Oversold"] else 0,
            3 if conds_long["Stochastic_Oversold"] else 0,
            5 if conds_long["BB_Breakout"] else 0,
            3 if conds_long["MFI_Oversold"] else 0,
            5 if conds_long["ADX_Strong"] else 0,
            10 if conds_long["Support_Confirmation"] else 0
        ])
        indicator_score_short = sum([
            10 if conds_short["PinBar"] else 0,
            10 if conds_short["Engulfing"] else 0,
            15 if conds_short["Elliott_Wave"] else 0,
            5 if conds_short["EMA_Cross"] else 0,
            5 if conds_short["MACD_Cross"] else 0,
            3 if conds_short["RSI_Overbought"] else 0,
            3 if conds_short["Stochastic_Overbought"] else 0,
            5 if conds_short["BB_Breakout"] else 0,
            3 if conds_short["MFI_Overbought"] else 0,
            5 if conds_short["ADX_Strong"] else 0,
            10 if conds_short["Resistance_Confirmation"] else 0
        ])

        if sum(1 for v in conds_long.values() if v) < min_conditions:
            indicator_score_long = 0
        if sum(1 for v in conds_short.values()) < min_conditions:
            indicator_score_short = 0

        score_long += indicator_score_long
        score_short += indicator_score_short
        score_log["long"]["indicators"] = indicator_score_long
        score_log["short"]["indicators"] = indicator_score_short

        score_long += trend_score_4h
        score_short += -trend_score_4h
        score_log["long"]["market_structure"] = trend_score_4h
        score_log["short"]["market_structure"] = -trend_score_4h

        signal_filter = SignalFilter()
        X_train = np.array([
            [30, 25, 2, 0.01, 0.05, 0.05, 0.005, 10, -5],
            [70, 20, 1, 0.02, 0.03, 0.03, 0.02, -10, 5],
            [50, 30, 1.5, 0.01, 0.04, 0.04, 0.01, 0, 0],
        ])
        y_train = np.array([1, 0, 1])
        signal_filter.train(X_train, y_train)
        vol_avg = vol_mean  # تعریف vol_avg
        features = [
            last["RSI"],
            last["ADX"],
            current_vol / vol_avg if vol_avg != 0 else 1,
            volatility,
            distance_to_resistance,
            distance_to_support,
            spread,
            psych_score_long,
            psych_score_short
        ]
        dt_score = signal_filter.predict(features)
        score_long += dt_score
        score_short += dt_score
        score_log["long"]["decision_tree"] = dt_score
        score_log["short"]["decision_tree"] = dt_score

        logging.info(f"امتیاز نهایی برای {symbol} @ {tf}: score_long={score_long:.2f}, score_short={score_short:.2f}")
        logging.info(f"جزئیات امتیاز Long: {score_log['long']}")
        logging.info(f"جزئیات امتیاز Short: {score_log['short']}")

        THRESHOLD = 50
        if score_long >= THRESHOLD and trend_score_1d >= 0:
            signal_type = "Long"
            entry_data = await find_entry_point(exchange, symbol, signal_type, support_4h, resistance_4h)
            if entry_data is None:
                logging.info(f"نقطه ورود Long برای {symbol} پیدا نشد")
                return None

            entry = entry_data["entry_price"]
            sl = entry_data["sl"]
            tp = entry_data["tp"]
            rr_ratio = entry_data["rr"]

            live_price = await get_live_price(exchange, symbol)
            if live_price is None:
                logging.warning(f"قیمت واقعی برای {symbol} دریافت نشد")
                return None

            price_diff = abs(entry - live_price) / live_price if live_price != 0 else float('inf')
            if price_diff > 0.01 or sl >= entry or tp <= entry:
                logging.warning(f"سیگنال برای {symbol} رد شد: اختلاف قیمت یا SL/TP نامعتبر")
                return None

            position_size = calculate_position_size(10000, 1, entry, sl)
            signal_strength = "قوی" if score_long > 90 else "متوسط"
            result = {
                "نوع معامله": "Long",
                "نماد": symbol,
                "تایم‌فریم": tf,
                "قیمت ورود": entry,
                "حد ضرر": sl,
                "هدف سود": tp,
                "ریسک به ریوارد": rr_ratio,
                "حجم پوزیشن": position_size,
                "سطح اطمینان": min(score_long, 100),
                "امتیاز": score_long,
                "قدرت سیگنال": signal_strength,
                "تحلیل": " | ".join([k for k, v in conds_long.items() if v]),
                "روانشناسی": psych_long,
                "روند بازار": "صعودی",
                "فاندامنتال": f"امتیاز: {fundamental_score}",
                "شاخص ترس و طمع": fng_index,
                "روند 4h": trend_4h,
                "قیمت فعلی بازار": live_price
            }
            asyncio.create_task(manage_trailing_stop(exchange, symbol, entry, sl, tp, signal_type))
            logging.info(f"سیگنال Long تولید شد: {result}")
            return result

        elif score_short >= THRESHOLD and trend_score_1d <= 0:
            signal_type = "Short"
            entry_data = await find_entry_point(exchange, symbol, signal_type, support_4h, resistance_4h)
            if entry_data is None:
                logging.info(f"نقطه ورود Short برای {symbol} پیدا نشد")
                return None

            entry = entry_data["entry_price"]
            sl = entry_data["sl"]
            tp = entry_data["tp"]
            rr_ratio = entry_data["rr"]

            live_price = await get_live_price(exchange, symbol)
            if live_price is None:
                logging.warning(f"قیمت واقعی برای {symbol} دریافت نشد")
                return None

            price_diff = abs(entry - live_price) / live_price if live_price != 0 else float('inf')
            if price_diff > 0.01 or sl <= entry or tp >= entry:
                logging.warning(f"سیگنال برای {symbol} رد شد: اختلاف قیمت یا SL/TP نامعتبر")
                return None

            position_size = calculate_position_size(10000, 1, entry, sl)
            signal_strength = "قوی" if score_short > 90 else "متوسط"
            result = {
                "نوع معامله": "Short",
                "نماد": symbol,
                "تایم‌فریم": tf,
                "قیمت ورود": entry,
                "حد ضرر": sl,
                "هدف سود": tp,
                "ریسک به ریوارد": rr_ratio,
                "حجم پوزیشن": position_size,
                "سطح اطمینان": min(score_short, 100),
                "امتیاز": score_short,
                "قدرت سیگنال": signal_strength,
                "تحلیل": " | ".join([k for k, v in conds_short.items() if v]),
                "روانشناسی": psych_short,
                "روند بازار": "نزولی",
                "فاندامنتال": f"امتیاز: {fundamental_score}",
                "شاخص ترس و طمع": fng_index,
                "روند 4h": trend_4h,
                "قیمت فعلی بازار": live_price
            }
            asyncio.create_task(manage_trailing_stop(exchange, symbol, entry, sl, tp, signal_type))
            logging.info(f"سیگنال Short تولید شد: {result}")
            return result

        logging.info(f"سیگنال برای {symbol} @ {tf} رد شد")
        return None

    except Exception as e:
        logging.error(f"خطای کلی در تحلیل {symbol} @ {tf}: {str(e)}")
        return None

async def scan_all_crypto_symbols(on_signal=None) -> None:
    exchange = ccxt.async_support.mexc({
        'enableRateLimit': True,
        'rateLimit': 2000
    })
    try:
        logging.debug(f"شروع بارگذاری بازارها از MEXC")
        await exchange.load_markets()
        logging.info(f"بازارها بارگذاری شدند: تعداد نمادها={len(exchange.symbols)}")
        top_coins = get_top_500_symbols_from_cmc()
        usdt_symbols = [s for s in exchange.symbols if any(s.startswith(f"{coin}/") and s.endswith("/USDT") for coin in top_coins)]
        logging.info(f"نمادهای USDT: تعداد={len(usdt_symbols)}")
        chunk_size = 10
        total_chunks = (len(usdt_symbols) + chunk_size - 1) // chunk_size
        symbol_results = []
        for idx in range(total_chunks):
            chunk = usdt_symbols[idx*chunk_size:(idx+1)*chunk_size]
            logging.info(f"شروع اسکن دسته {idx+1}/{total_chunks}: {chunk}")
            tasks = []
            for sym in chunk:
                async with semaphore:
                    tasks.append(analyze_symbol(exchange, sym, "1h"))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for sym, result in zip(chunk, results):
                if isinstance(result, Exception):
                    logging.error(f"خطا در تحلیل {sym}: {str(result)}")
                    continue
                if result is not None:
                    symbol_results.append(result)
                    if on_signal:
                        try:
                            await on_signal(result)
                        except Exception as e:
                            logging.error(f"خطا در اجرای callback برای {sym}: {e}")
            logging.info(f"پایان اسکن دسته {idx+1}/{total_chunks}: سیگنال‌های یافت‌شده={len([r for r in results if r is not None])}")
            await asyncio.sleep(WAIT_BETWEEN_CHUNKS)
        logging.info(f"اسکن کامل شد: کل سیگنال‌ها={len(symbol_results)}, رد شده به دلیل نقدینگی={LIQUIDITY_REJECTS}, حجم={VOLUME_REJECTS}, حمایت/مقاومت={SR_REJECTS}")
        ablation_test(symbol_results, "all_filters")
    except Exception as e:
        logging.error(f"خطای کلی در اسکن نمادها: {e}")
    finally:
        await exchange.close()
