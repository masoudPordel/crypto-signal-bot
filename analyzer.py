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
import traceback
from datetime import datetime, timedelta
from sklearn.tree import DecisionTreeClassifier
from typing import Optional, Dict, Any, List

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
CMC_API_KEY = os.getenv("CMC_API_KEY", "c1b589e6-5f67-46bd-9cfe-a34f925bc4cb")
COINMARKETCAL_API_KEY = os.getenv("COINMARKETCAL_API_KEY", "iFrSo3PUBJ36P8ZnEIBMvakO5JutSIU1XJvG7ALa")
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
        logging.info(f"دریافت ۵۰۰ نماد برتر از CMC: تعداد نمادها={len(data['data'])}")
        return [entry['symbol'] for entry in data['data']]
    except Exception as e:
        logging.error(f"خطا در دریافت از CMC: {e}")
        return []

# تابع دریافت شاخص ترس و طمع
def get_fear_and_greed_index() -> int:
    url = "https://api.alternative.me/fng/?limit=1"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        value = int(data["data"][0]["value"])
        logging.info(f"شاخص ترس و طمع دریافت شد: {value}")
        return value
    except Exception as e:
        logging.error(f"خطا در دریافت شاخص ترس و طمع: {e}")
        return 50

# کلاس محاسبه اندیکاتورها
class IndicatorCalculator:
    @staticmethod
    def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        tr = pd.concat([df["high"] - df["low"], abs(df["high"] - df["close"].shift()), abs(df["low"] - df["close"].shift())], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr

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

# کلاس تشخیص الگوها
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

# کلاس فیلتر سیگنال
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
            logging.error(f"خطا در پیش‌بینی Decision Tree: {e}, traceback={str(traceback.format_exc())}")
            return 0

# تابع بررسی نقدینگی
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
        for i in range(-5, 0):
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
        spread_threshold = 0.005
        if spread > spread_threshold:
            logging.warning(f"اسپرد برای {symbol} بیش از حد بالاست: spread={spread:.4f}")
            LIQUIDITY_REJECTS += 1
            return spread, -10
        score = 15 if spread < spread_threshold else -5
        logging.info(f"نقدینگی {symbol}: spread={spread:.4f}, threshold={spread_threshold:.4f}, score={score}")
        if spread >= spread_threshold:
            LIQUIDITY_REJECTS += 1
        return spread, score
    except Exception as e:
        logging.error(f"خطا در بررسی نقدینگی برای {symbol}: {e}")
        return float('inf'), 0

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
    start_date = (datetime.utcnow() - timedelta(days=7)).replace(hour=20, minute=50, second=0, microsecond=0).strftime("%Y-%m-%d")
    end_date = (datetime.utcnow() + timedelta(days=7)).replace(hour=20, minute=50, second=0, microsecond=0).strftime("%Y-%m-%d")
    params = {
        "coinId": str(coin_id),
        "max": 5,
        "dateRangeStart": start_date,
        "dateRangeEnd": end_date
    }
    try:
        logging.debug(f"شروع دریافت رویدادها برای {symbol}, coin_id={coin_id}")
        time.sleep(0.5)
        resp = requests.get(url, headers=headers, params)
        events = resp.json()
        event_score = 0
        if not events or "body" not in events or not events["body"]:
            return 0
        for event in events["body"]:
            title = event.get("title", "")
            description = event.get('description', '')
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
        logging.error(f"Error fetching events for {symbol}: {e}")
        return 0

# تابع محاسبه شاخص‌ها
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.ffill().bfill().fillna(0)
        df["EMA12"] = df["close"].ewm(span=12).mean()
        df["EMA26"] = df["close"].ewm(span=26).mean()
        df["MACD"] = df["EMA12"] - df["EMA26"]
        df["Signal"] = df["MACD"].ewm(span=9).mean()
        df["RSI"] = IndicatorCalculator.compute_rsi(df)
        df["ATR"] = IndicatorCalculator.compute_rsi(df)
        df["ADX"] = IndicatorCalculator.compute_adx(df)
        df["Stochastic"] = IndicatorCalculator.compute_stochastic(df)
        df["BB_upper"], df["BB_lower"] = IndicatorCalculator.compute_bollinger_bands(df)
        df["PinBar"] = PatternDetector.detect_pin_bar(df)
        df["Engulfing"] = PatternDetector.detect_engulfing(df)
        df = PatternDetector.detect_elliott_wave(df)
        df["MFI"] = IndicatorCalculator.compute_mfi(df)

        # اضافه کردن Hammer و Doji
        df["Hammer"] = ((df["close"] - df["low"]) / (df["high"] - df["low"]) > 0.0.66) & (df["close"] > df["open"])  # Hammer صعودی
        df["Doji"] = abs(df["close"] - df["open"]) / (df["high"] - df["low"]) < 0.1  # Doji

        logging.debug(f"اندیکاتورها با موفقیت محاسبه شدند: {list(df.columns)}")
    except Exception as e:
        logging.error(f"Error in indicator calculations: {str(e)}")
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
        # مقادیر پیش‌فرض برای Hammer و Doji
        df["Hammer"] = False
        df["Doji"] = False
        logging.warning(f"اندیکاتور‌ها با مقادیر پیش‌فرض پر شدند")
    return df

# تابع تحلیل ساختار بازار
async def analyze_market_structure(exchange: ccxt.Exchange, symbol: str) -> Dict[str, Any]:
    try:
        logging.info(f"Starting market structure analysis for {symbol} @ 4h")
        df_4h = await get_ohlcv_cached(exchange, symbol, "4h", limit=50)
        if df_4h is None or len(df_4h) < 50:
            logging.warning(f"Insufficient data for market structure {symbol} @ 4h: candles={len(df_4h) if df_4h is not None else 0}")
            return {"trend": "Neutral", "score": 0, "support": 0, "resistance": 0, "fng_index": 50}

        df_4h = compute_indicators(df_4h)
        last_4h = df_4h.iloc[-1]

        trend_score = 0
        trend_direction = "Neutral"
        if last_4h["EMA12"] > last_4h["EMA26"]:
            trend_score += 10
            trend_direction = "Up"
        elif last_4h["EMA12"] < last_4h["EMA26"]:
            trend_score += -10
            trend_direction = "Down"

        support, resistance, _ = PatternDetector.detect_support_resistance(df_4h)
        logging.info(f"Key levels @ 4h for {symbol}: support={support:.2f}, resistance={resistance:.2f}")

        fng_index = get_fear_and_greed_index()
        if fng_index < 25:
            trend_score += 5
            logging.info(f"Fear & Greed @ 4h for {symbol}: {fng_index} (extreme fear) - Added 10 to bullish trend")
        elif fng_index > 75:
            trend_score += -5
            logging.info(f"Fear & Greed @ 4h for {symbol}: {fng_index} (extreme greed) - Added -10 to bearish trend")

        result = {
            "trend": trend_direction,
            "score": trend_score,
            "support": support,
            "resistance": resistance,
            "fng_index": fng_index
        }
        logging.info(f"Market structure analysis for {symbol} @ 4h completed: {result}")
        return result
    except Exception as e:
        logging.error(f"Error in market structure analysis for {symbol} @ 4h: {str(e)}")
        return {"trend": "Neutral", "score": 0, "support": 0, "resistance": 0, "fng_index": 50}

# تابع دریافت قیمت واقعی
async def get_live_price(exchange: ccxt.Exchange, symbol: str, max_attempts: int = 3) -> Optional[float]:
    attempt = 0
    last_ticker = None
    while attempt < max_attempts:
        try:
            ticker = await exchange.fetch_ticker(symbol)
            bid = ticker.get('bid')
            ask = ticker.get('ask')
            last = ticker.get('last')
            if bid is None or ask is None or last is None or bid <= 0 or ask <= 0:
                logging.warning(f"Invalid live price data for {symbol}: bid={bid}, ask={ask}, last={last}")
                attempt += 1
                await asyncio.sleep(0.3)
                continue
            live_price = (bid + ask) / 2 if bid and ask else last
            last_ticker = live_price
            logging.info(f"Market live price for {symbol}: live_price={live_price}, bid={bid}, ask={ask}, last={last}")
            return live_price
        except Exception as e:
            logging.error(f"Error fetching live price for {symbol} on attempt {attempt + 1}: {e}")
            attempt += 1
            await asyncio.sleep(0.3)
    try:
        df_1 = await get_ohlcv_cached(exchange, symbol, "1m")
        if df_1 is not None and len(df_1) > 0:
            fallback_price = df_1["close"].m.iloc[-1]
            logging.warning(f"Could not fetch live price for {symbol}, using 1m candle price: {fallback_price}")
            return float(fallback_price)
    except Exception as e:
        logging.error(f"Error fetching fallback price for {symbol}: {e}")
    logging.error(f"Failed to fetch price for {symbol} after {max_attempts} attempts")
    return None

# تابع پیدا کردن نقطه ورود
async def find_entry_point(exchange: ccxt.Exchange, symbol: str, signal_type: str, support: float, resistance: float) -> Optional[Dict]:
    """
    Find entry point for a Long or Short trade in the 15m timeframe.
    
    Args:
        exchange: Exchange object (e.g., ccxt)
        symbol: Trading pair (e.g., 'CYBER/USDT')
        signal_type: Signal type ('Long' or 'Short')
        support: Support level
        resistance: Resistance level
    
    Returns:
        dict: {"entry_price": float, "sl": float, "tp": float} or None if no entry point is found
    """
    try:
        logging.info(f"Starting to find entry point for {symbol} @ 15m - Signal type: {signal_type}")
        
        # دریافت داده‌های 15 دقیقه
        df_15m = await get_ohlcv_cached(exchange, symbol, "15m")
        if df_15m is None or len(df_15m) < 20:
            logging.warning(f"No entry point for {symbol} @ 15m: reason=Insufficient data, candles={len(df_15m) if df_15m is not None else 0}")
            return None

        # محاسبه اندیکاتورها
        df_15m = compute_indicators(df_15m)
        last_15m = df_15m.iloc[-1].to_dict()  # Last candle
        next_15m = df_15m.iloc[-2].to_dict() if len(df_15m) > 1 else None

        # دریافت قیمت واقعی
        live_price = await get_live_price(exchange, symbol)
        if live_price is None:
            logging.warning(f"No entry point for {symbol} @ 15m: reason=Failed to fetch live price")
            return None

        # Check price discrepancy
        price_diff = abs(live_price - last_15m["close"]) / live_price if live_price != 0 else float('inf')
        if price_diff > 0.02:  # ### تغییر: loosened from 0.01 to 0.2
            logging.warning(f"No entry price for {symbol} @ 15m: reason=Large price difference, live_price={live_price:.6f}, candle_price={last_15m['close']:.6f}, delta={price_diff:.4f}")
            return None

        # Check volume
        volume_mean = df_15m["volume"].rolling(20).mean().iloc[-1]
        volume_condition = last_15m["volume"] > volume_mean * 0.2  # ### تغییر: از 0.3 تا 0.2
        logging.info(f"Volume check for {symbol}: current_vol={last_15m['volume']:.2f}, mean={volume_mean:.2f}, condition={volume_condition}")
        if not volume_condition:
            logging.info(f"No entry point for {symbol} @ 15m: reason=Insufficient volume, current_vol={last_15m['volume']:.2f}, threshold={volume_mean * 0.2:.2f}")
            return None

        # Check PinBar pattern
        pin_bar_confirmed = False
        if last_15m.get("PinBar", ""): False):
            if next_15m and (
                (signal_type == "Long" and last_15m["close"] < next_15m["close"] * 1.10) or 
                (signal_type == "Short" and last_15m["close"] > next_15m["close"] * 0.90)
            ):
                pin_bar_confirmed = True
                logging.info(f"PinBar pattern for {symbol} confirmed with next candle (10% flexibility)")
            else:
                logging.info(f"No entry point for {symbol} @ 15m: reason=PinBar not confirmed")

        # Check price action
        price_action = (
            (last_15m["PinBar"].get("PinBar", False) and pin_bar_confirmed) or 
            last_15m["Engulfing"].get("Engulfing", False) or 
            last_15m["Hammer"].get("Hammer", False) or 
            last_15m["Doji"].get("Doji", False) or
            (last_15m["RSI"].get("RSI", 50) < 30 if signal_type == "Long" else last_15m.get("RSI", 50) > 70)  # ### تغییر: Added RSI
        )
        logging.info(f"{signal_type} details for {symbol}: close={last_15m']['close']:.6f}, resistance={resistance:.6f}, support={support:.6f}")
        logging.info(f"Pattern values: PinBar={last_15m.get('PinBar', False)}, Confirmed={pin_bar_confirmed}, Engulfing={last_15m.get('Engulfing', False)}, Hammer={last_15m.get('Hammer', False)}, Doji={last_15m.get('Doji', False)}, price_action={price_action}")

        # Fetch 1-hour data and check support
        df_1h = await get_ohlcv_cached(exchange, symbol, '1h')
        if df_1h is None or len(df_1h) == 0:
            logging.warning(f"No entry point for {symbol} @ 15m: reason=Failed to fetch 1h data")
            return None

        recent_low = df_1h["low"].iloc[-1]
        if recent_low < support * 0.98:  # ### تغییر: loosened from 0.95 to 0.98
            logging.warning(f"No entry point for {symbol} @ {15m}: reason=Broken support, recent_low={recent_low:.6f}, support={support:.6f}")
            return None

        close_price = last_15m["close"]: close_price
        fib_levels = calculate_fibonacci_levels(df_15m)

        # Entry conditions
        if signal_type == "Long":
            breakout_resistance = close_price > resistance and volume_condition
            near_support = abs(close_price - support) / close_price < close_price 0.1 and volume_condition and # ### تغییر: loosened from 0.05 to 0.1
            within_range = support < close_price < resistance and volume_condition
            entry_condition = (breakout_resistance or near_support or within_range) and price_action

            if entry_condition:
                entry_price = live_price
                atr_15m = df_15m["ATR'].iloc[-1]
                
                # Set SL and TP
                sl = entry_price - (atr_15m * 0.75)  # Closer SL
                tp = entry_price + atr_15m * (atr_15m * 2.5)  # Further TP
                
                # Restrict SL and TP with support/resistance
                if sl < support < * 0.95:  # ### تغییر: loosened from 0.98 to 0.95
                    sl = support * 0.95
                if tp > resistance > * 1.10:  # ### تغییر: loosened from lo1.05 to lo1.10
                    tp = resistance * 1.05

                # محاسبه RR
                rr_ratio = (tp["tp"] - tptp["entry_price"]) / (entry_price - tptp["sl"]]) / if tptp 0 else else 0
                logging.info(f"Calculated TP for {symbol}: Entry={entry_price:.6f}, TP={tp:.6f}, {sl:.6f}, f{ATR}={:.6f}, {RR}={:.2f}")

                # Filter RR
                if rr_ratio < 2:
                    logging.info(f"No entry point for {symbol} @ {15m}: reason=RR too low, RR={rr_ratio:.2f}")
                    return None

                return {"entry_price": float(entry_price), "sl": float(sl), "tp": float(tp)}

        elif signal_type == "Short":
            breakout_support = close_price < support and volume_condition
            near_resistance = abs(close_price - resistance) / close_price < close_price 0.1 and volume_conditioned # ### تغییر: loosened from 0.05
            within_range = support < close_price < resistance and volume_condition
            entry_condition = (breakout_support or near_resistance or within_range) and price_action

            if entry_condition:
                entry_price = live_price
                atr_15m = df["15m"].m.iloc[-1]
                
                # Set SL و TP
                sl = entry_price + (tp * atr_15m * 0.75)  # Closer SL
                # تنظیم فاصله بیشتر برای TP
                tp = entry_price - (atr_15m * 2.5)
                
                # محدود کردن SL و TP با حمایت/مقامت
                if sl > resistance * 1.10:  # ### تغییر: loosened from lo1.05 to lo1.10
                    sl = resistance * 0
                if ttp < support * 0.95:  # ### تغییر: loosened from lo0.98 to lo0
                    tp = support * 0.95

                # محاسبه نسبت بازگشت
                rr_ratio = (entry_price / - tptp) / (sltp - entry_price) if (sltp - entry_price) != 0 else 0
                logging.info(f"Calculated TP for {symbol} : Entry={entry_price:.6f}, TP={tp:.4f}, {sl:.{2f}, SL={:.2f}, ATR={atr:.2f}, RR={:.2}}")

                # فیلترا نسبت حداقل
                if rr_ratio < 2:
                    logging.info(f"No entry point for {symbol} @ 15m: reason=RR too small, RR={rr_ratio:.2f}")
                    return None

                return {"entry_price": float(entry_price), "sl": float(sl), "tp": float(tp)}

        logging.info(f"No entry point for {symbol} @ 15m: reason=Entry condition not met")
        return None

    except Exception as e:
        logging.error(f"Error finding entry point for {symbol} @ @ {15m}: {str(e)}")
        return None

# تابع مدیریت توقف متحرک
async def manage_trailing_stop(exchange: ccxt, symbol: str, entry_price: float, sl: float, signal_type: str, trail_percentage: float = 0.5):
    logging.info(f"Starting Trailing Stop for {symbol} with signal type {signal_type}, entry={entry_price}, initial={sl}")
    while True:
        live_price = await get_live_price(exchange, symbol)
        if live_price is None:
            logging.warning(f"Failed to fetch live price for {symbol}, retrying after in 60s")
            await asyncio.sleep(60)
            continue
        if (live_price > entry_price and signal_type == "Long") or (live_price < entry_price and signal_type == "Short"):
            trail_amount = live_price * (trail_percentage / / 100)
            new_sl = live_price - trail_amount if signal_type == "Long" else live_price + trail_amount
            if (signal_type == "Long' and new_sl > sl) or (signal_type == "Short" and new_sl < sl):
                sl = new_sl
                logging.info(f"Trailing Stop updated for {symbol}: {signal_type}: SL={sl:.2f}, Live Price={live_price:.2f}")
        await asyncio.sleep(300)  # Check every 5 minutes

# تابع تأیید چند تایم‌فریم
async def multi_timeframe_confirmation(exchange: ccxt, symbol: str, base_tf: str) -> float:
    weights = {"1d": 0.4, "4h": 0.3, "1h": 0.2, "15m": 0.1}
    total_weight = 0
    score = 0
    for tf, weight in weights.items():
        if tf == base_tf:
            continue
        try:
            df_tf = await get_ohlcv_cached(exchange, symbol, tf)
            if df_tf is None or (len(df_tf) < 50 and tf != "1d") or (len(df_tf) < 30 and tf == "1d"):
                logging.warning(f"Insufficient data for {symbol} @ {tf} in multi-timeframe: candles={len(df)} if df_tf is not None else 0}")
                continue
            df_tf["EMMA12"] = df_tf["close"].ewm["span=12].mean()
            df_tf["EMMA26"] = df_tf["close"].ewm(span=12).mean()
            long_trend = df_tf["EMA12"].iloc[-1] > df_tf["EMA26"].iloc[-1]
            score += (weight * 10) if long_trend else (-weight * 5)
            total_weight += weight
        except Exception as e:
            logging.error(f"Error processing timeframe {tf} @ {symbol}: for {str(e)}")
            continue
    final_score = score if total_weight > 0 else 0
    logging.info(f"Multi-timeframe for {symbol}: score={score:.2f}, total_weight={total_weight:.2f}")
    return float_score

# تنظیم سمافور برای کنترل درخواست‌ها
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# تابع دریافت داده کندل‌ها با کش
async def get_ohlcv_cached(exchange, symbol, tf, limit=50) -> Optional[pd.DataFrame]:
    try:
        key = f"f{exchange.id}_{symbol}_{tf}"
        now = datetime.utcnow()

        if key in CACHE:
            cached_df, cached_time = CACHE[key]
            if (now - cached_time).total_seconds() < CACHE_TTL:
                return cached_df

        # دریافت اطلاعات از صرافی
        raw_data = await exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)

        if not raw_data or len(raw_data) == 0:
            logging.warning(f"OHLCV data empty or unavailable for {symbol} / {tf}")
            return None

        df = pd.DataFrame(raw_data, columns=["timestamp", "open", "high", "low", "close", "volume"])

        # تبدیل تایم‌ستامپ به فرمت درست
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit=["ms"], errors="coerce")
        if df["timestamp"].isna().all():
            logging.error(f"All timestamps invalid for {symbol} / {tf}")
            return None

        # تبدیل ستون‌ها به نوع عددی
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # تنظیم ایندکس
        df.set_index("timestamp", inplace=True)

        # ذخیره در کش
        Cache[key] = (df, now)
        logging.info(f"OHLCV data fetched and cached for {symbol} / {tf}: candles={len(df)}")
        return df

    except Exception as e:
        logging.error(f"❌ Failed to fetch OHLCV for {symbol} / {tf}: {str(e)}")
        return None
        
# تابع محاسبه سطوح فیبوناچی
def calculate_fibonacci_levels(df: pd.DataFrame) -> Dict[str, float]:
    try:
        high = df["high"].iloc[-20:].max()
        low = df["low"].iloc[-20:].min()
        diff = high - low
        levels = {
            "0.0": low,
            "0.236": low + diff * 0.236,
            "0.382": low + diff * 0.382,
            "0.5": low + diff * 0.5,
            "0.618": low + diff * 0.618,
            "0.786": low + diff * 0.786,
            "1.0": high
        }
        logging.debug(f"Fibonacci levels calculated: {levels}")
        return levels
    except Exception as e:
        logging.error(f"Error calculating Fibonacci levels: {str(e)}")
        return {}

# تابع محاسبه حجم پوزیشن
def calculate_position_size(account_balance: float, risk_percentage: float, entry: float, stop_loss: float) -> float:
    if entry is None or stop_loss is None or entry == 0 or stop_loss == 0:
        logging.warning(f"Invalid inputs for position size: entry={entry}, stop_loss={stop_loss}")
        return 0
    risk_amount = account_balance * (risk_percentage / 100)
    distance = abs(entry - stop_loss)
    position_size = risk_amount / distance if distance != 0 else 0
    return round(position_size, 2)

# تابع تست ablation
def ablation_test(symbol_results: list, filter_name: str) -> int:
    total_signals = len([r for r in symbol_results if r is not None])
    logging.info(f"Ablution Test for filter {filter_name}: initial_signals={total_signals}")
    return total_signals

# تابع تحلیل نماد
async def analyze_symbol(exchange: ccxt.Exchange, symbol: str, tf: str) -> Optional[dict]:
    global LIQUIDITY_REJECTS, VOLUME_REJECTS, SR_REJECTS
    start_time = time.time()
    logging.info(f"Starting analysis for {symbol} @ {tf}, start_time={datetime.now()}")
    try:
        market_structure = await analyze_market_structure(exchange, symbol)
        trend_4h = market_structure["trend"]
        trend_score_4h = market_structure["score"]
        support_4h = market_structure["support"]
        resistance_4h = market_structure["resistance"]
        fng_index = market_structure.get("fng_index", 50)
        if tf != "1h":
            logging.info(f"Analysis for {symbol} only performed on 1h timeframe. Current: {tf}")
            return None
        df = await get_ohlcv_cached(exchange, symbol, tf, limit=50)
        if df is None or len(df) < 30:
            logging.warning(f"Insufficient data for {symbol} @ {tf}: candles={len(df) if df is not None else 0}")
            return None
        logging.info(f"Data fetched for {symbol} @ {tf} in {time.time() - start_time:.2f} seconds, rows={len(df)}")
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            logging.error(f"Required columns missing in dataframe for {symbol} @ {tf}")
            return None
        df = df.ffill().bfill().fillna(0)
        df = compute_indicators(df)
        last = df.iloc[-1]
        score_long = 0
        score_short = 0
        score_log = {"long": {}, "short": {}}
        # Higher timeframe trend confirmation (1d)
        df1d = await get_ohlcv_cached(exchange, symbol, "1d")
        trend_1d_score = 0
        if df1d is not None and len(df1d) > 0:
            df1d = compute_indicators(df1d)
            long_trend_1d = df1d["EMA12"].iloc[-1] > df1d["EMA26"].iloc[-1]
            trend_1d_score = 10 if long_trend_1d else -10
            logging.info(f"1d trend confirmation for {symbol}: trend_score={trend_1d_score}")
        vol_avg = df["volume"].rolling(VOLUME_WINDOW).mean().iloc[-1]
        current_vol = df["volume"].iloc[-1]
        vol_mean = df["volume"].rolling(20).mean().iloc[-1]
        vol_std = df["volume"].rolling(20).std().iloc[-1]
        vol_threshold = vol_mean * 0.2  # ### تغییر: از 0.3 تا 0.2
        vol_score = 10 if current_vol >= vol_threshold else -2
        score_long += vol_score
        score_short += vol_score
        score_log["long"]["volume"] = vol_score
        score_log["short"]["volume"] = vol_score
        logging.info(f"Volume for {symbol} @ {tf}: current_vol={current_vol:.2f}, threshold={vol_threshold:.2f}, score={vol_score}")
        if current_vol < vol_threshold:
            VOLUME_REJECTS += 1
        # مقدار پیش‌فرض برای dynamic_rr
        atr_1h = df["ATR"].iloc[-1]
        risk_buffer = atr_1h * 2
        dynamic_rr = 2.0  # مقدار پیش‌فرض
        logging.info(f"Dynamic RR for {symbol}: RR={dynamic_rr}")
        volatility = df["ATR"].iloc[-1] / last["close"]
        vola_mean = (df["ATR"] / df["close"]).rolling(20).mean().iloc[-1]
        vola_std = (df["ATR"] / df["close"]).rolling(20).std().iloc[-1]
        vola_threshold = vola_mean + vola_std
        vola_score = 10 if volatility > vola_threshold else -5
        score_long += vola_score
        score_short += vola_score
        score_log["long"]["volatility"] = vola_score
        score_log["short"]["volatility"] = vola_score
        adx_mean = df["ADX"].rolling(window=20).mean().iloc[-1]
        adx_std = df["ADX"].rolling(window=20).std().iloc[-1]
        adx_threshold = adx_mean + adx_std
        adx_score = 15 if last["ADX"] >= adx_threshold else -5
        trend_score = 10 if last["ADX"] >= adx_threshold * 1.5 else 0
        score_long += adx_score + trend_score
        score_short += adx_score + trend_score
        score_log["long"]["adx"] = adx_score
        score_log["short"]["adx"] = adx_score
        score_log["long"]["trend"] = trend_score
        score_log["short"]["trend"] = trend_score
        long_trend = df["EMA12"].iloc[-1] > df["EMA26"].iloc[-1]
        short_trend = not long_trend
        trend_score = 10 if long_trend else -5
        score_long += trend_score
        score_short += -trend_score
        score_log["long"]["trend_direction"] = trend_score
        score_log["short"]["trend_direction"] = -trend_score
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
        if distance_to_resistance <= s_r_buffer:
            SR_REJECTS += 1
        if distance_to_support <= s_r_buffer:
            SR_REJECTS += 1
        spread, liquidity_score = await check_liquidity(exchange, symbol, df)
        if spread == float('inf'):
            spread = 0.0
        score_long += liquidity_score
        score_short += liquidity_score
        score_log["long"]["liquidity"] = liquidity_score
        score_log["short"]["liquidity"] = liquidity_score
        if liquidity_score < 0:
            logging.warning(f"Signal rejected for {symbol} due to poor liquidity: liquidity_score={liquidity_score}")
            return None
        fundamental_score = check_market_events(symbol)
        score_long += fundamental_score
        score_short += fundamental_score
        score_log["long"]["fundamental"] = fundamental_score
        score_log["short"]["fundamental"] = fundamental_score
        psych_long = "اشباع فروش" if last["RSI"] < 40 else "اشباع خرید" if last["RSI"] > 60 else "متعادل"
        psych_short = "اشباع خرید" if last["RSI"] > 60 else "اشباع فروش" if last["RSI"] < 40 else "متعادل"
        psych_score_long = 10 if psych_long == "اشباع فروش" else -10 if psych_long == "اشباع خرید" else 0
        psych_score_short = 10 if psych_short == "اشباع خرید" else -10 if psych_short == "اشباع فروش" else 0
        score_long += psych_score_long
        score_short += psych_score_short
        score_log["long"]["psychology"] = psych_score_long
        score_log["short"]["psychology"] = psych_score_short
        bullish_rsi_div, bearish_rsi_div = PatternDetector.detect_rsi_divergence(df)
        div_score_long = 10 if bullish_rsi_div else 0
        div_score_short = 10 if bearish_rsi_div else 0
        score_long += div_score_long
        score_short += div_score_short
        score_log["long"]["rsi_divergence"] = div_score_long
        score_log["short"]["rsi_divergence"] = div_score_short
        support_buffer = (df["ATR"].iloc[-1] / last["close"]) * 1.5
        resistance_buffer = (df["ATR"].iloc[-1] / last["close"]) * 1.5
        min_conditions = 2
        conds_long = {
            "PinBar": last["PinBar"],
            "Engulfing": last["Engulfing"] and last["close"] > last["open"] and (df["volume"].iloc[-1] > df["volume"].rolling(window=20).mean().iloc[-1] * 1.5),
            "Elliott_Wave": df["WaveTrend"].iloc[-1] == "Up",
            "EMA_Cross": df["EMA12"].iloc[-1] > df["EMA26"].iloc[-1] and (df["volume"].iloc[-1] > df["volume"].rolling(window=20).mean().iloc[-1] * 1.2),
            "MACD_Cross": df["MACD"].iloc[-2] < df["Signal"].iloc[-2] and df["MACD"].iloc[-1] > df["Signal"].iloc[-1] and (df["MACD"].iloc[-1] > 0),
            "RSI_Oversold": last["RSI"] < 25,
            "Stochastic_Oversold": last["Stochastic"] < 15,
            "BB_Breakout": last["close"] > last["BB_upper"] and (df["volume"].iloc[-1] > df["volume"].rolling(window=20).mean().iloc[-1] * 1.5),
            "MFI_Oversold": last["MFI"] < 15,
            "ADX_Strong": last["ADX"] > 25,
            "Support_Confirmation": distance_to_support <= support_buffer and (last["PinBar"] or last["Engulfing"])
        }
        conds_short = {
            "PinBar": last["PinBar"],
            "Engulfing": last["Engulfing"] and last["close"] < last["open"] and (df["volume"].iloc[-1] > df["volume"].rolling(window=20).mean().iloc[-1] * 1.5),
            "Elliott_Wave": df["WaveTrend"].iloc[-1] == "Down",
            "EMA_Cross": df["EMA12"].iloc[-1] < df["EMA26"].iloc[-1] and (df["volume"].iloc[-1] > df["volume"].rolling(window=20).mean().iloc[-1] * 1.2),
            "MACD_Cross": df["MACD"].iloc[-2] > df["Signal"].iloc[-2] and df["MACD"].iloc[-1] < df["Signal"].iloc[-1] and (df["MACD"].iloc[-1] < 0),
            "RSI_Overbought": last["RSI"] > 75,
            "Stochastic_Overbought": last["Stochastic"] > 85,
            "BB_Breakout": last["close"] < last["BB_lower"] and (df["volume"].iloc[-1] > df["volume"].rolling(window=20).mean().iloc[-1] * 1.05),
            "MFI_Overbought": last["MFI"] > 85,
            "ADX": last["ADX"] > 25,
            "Resistance_Confirmation": distance_to_resistance <= resistance_buffer and (last["PinBar"] or last["Engulfing"])
        }
        indicator_score_long = (10 if conds_long["PinBar"] else 0) + \
                              (10 if conds_long["Engulfing"] else 0) + \
                              (15 if conds_long["Elliott_Wave"] else 0) + \
                              (5 if conds_long["EMA_Cross"] else 0) + \
                              (5 if conds_long["MACD_Cross"] else 0) + \
                              (3 if conds_long["RSI_Oversold"] else 0) + \
                              (3 if conds_long["Stochastic_Oversold"] else 0) + \
                              (5 if conds_long["BB_Breakout"] else 0) + \
                              (3 if conds_long["MFI_Oversold"] else 0) + \
                              (5 if conds_long["ADX_Strong"] else 0) + \
                              (10 if conds_long["Support_Confirmation"] else 0)
        indicator_score_short = (10 if conds_short["PinBar"] else 0) + \
                               (10 if conds_short["Engulfing"] else 0) + \
                               (15 if conds_short["Elliott_Wave"] else 0) + \
                               (5 if conds_short["EMA_Cross"] else 0) + \
                               (5 if conds_short["MACD_Cross"] else 0) + \
                               (3 if conds_short["RSI_Overbought"] else 0) + \
                               (3 if conds_short["Stochastic_Overbought"] else 0) + \
                               (5 if conds_short["BB_Breakout"] else 0) + \
                               (3 if conds_short["MFI_Overbought"] else 0) + \
                               (5 if conds_short["ADX_Strong"] else 0) + \
                               (10 if conds_short["Resistance_Confirmation"] else 0)
        if sum(1 for v in conds_long.values() if v) < min_conditions:
            indicator_score_long = 0
        if sum(1 for v in conds_short.values() if v) < min_conditions:
            indicator_score_short = 0
        score_long += indicator_score_long
        score_short += indicator_score_short
        score_log["long"]["indicators"] = indicator_score_long
        score_log["short"]["indicators"] = indicator_score_short
        logging.debug(f"Indicator conditions for {symbol} @ {tf}: long_score={indicator_score_long}, short_score={indicator_score_short}, conditions_long={conds_long}")
        score_long += trend_score_4h
        score_short += -trend_score_4h
        score_log["long"]["market_structure_4h"] = trend_score_4h
        score_log["short"]["market_structure_4h"] = -trend_score_4h
        logging.info(f"4h market structure score for {symbol}: Long={trend_score_4h}, Short={-trend_score_4h}")
        logging.debug(f"Starting Decision Tree filter for {symbol} @ {tf}")
        signal_filter = SignalFilter()
        X_train = np.array([
            [30, 25, 2, 0.01, 0.05, 0.05, 1],
            [70, 20, 1, 0.02, 0.03, 0.03, 0],
            [50, 30, 1.5, 0.01, 0.04, 0.04, 0]
        ])
        y_train = np.array([1, 0, 1])
        signal_filter.train(X_train, y_train)
        features = [
            last["RSI"],
            last["ADX"],
            current_vol / vol_avg if vol_avg != 0 else 0,
            volatility,
            distance_to_resistance,
            distance_to_support,
            spread
        ]
        dt_score = signal_filter.predict(features)
        score_long += dt_score
        score_short += dt_score
        score_log["long"]["decision_tree"] = dt_score
        score_log["short"]["decision_tree"] = dt_score
        logging.debug(f"Decision Tree filter for {symbol} @ {tf}: features={features}, score={dt_score:.2f}")
        logging.info(f"Final scores for {symbol} @ {tf}: score_long={score_long:.2f}, score_short={score_short:.2f}")
        logging.debug(f"Long score details: {score_log['long']}")
        logging.debug(f"Short score details: {score_log['short']}")
        THRESHOLD = 80  # ### تغییر: از 85 تا 80
        if score_long >= THRESHOLD and trend_1d_score >= 0:  # Mandatory 1d trend condition
            signal_type = "Long"
            # محاسبه RR dynamically after signal type determination
            if support_4h > 0:
                dynamic_rr = max(dynamic_rr, (resistance_4h - support_4h) / risk_buffer if risk_buffer != 0 else 0)
            logging.info(f"Dynamic RR for {symbol} (Long): RR={dynamic_rr}")
            entry_data = await find_entry_point(exchange, symbol, signal_type, support_4h, resistance_4h)
            if entry_data is None:
                logging.info(f"No Long entry point found for {symbol} @ 15m")
                return None
            entry = entry_data["entry_price"]
            sl = entry_data["sl"]
            tp = entry_data["tp"]
            # ### تغییر: محاسبه rr_ratio
            rr_ratio = (tp - entry) / (entry - sl) if (entry - sl) != 0 else 0
            if rr_ratio < 2:
                logging.info(f"Long signal rejected for {symbol}: RR={rr_ratio:.2f} < 2")
                return None
            live_price = await get_live_price(exchange, symbol)
            if live_price is None:
                logging.warning(f"Live price fetch failed for {symbol}, rejecting signal")
                return None
            price_diff = abs(entry - live_price) / live_price if live_price != 0 else float('inf')
            if price_diff > 0.01:
                logging.warning(f"Entry price discrepancy for {symbol} too high: entry={entry}, live_price={live_price}, diff={price_diff}")
                return None
            if sl >= entry or tp <= entry:
                logging.warning(f"Invalid SL or TP for {symbol}: entry={entry}, sl={sl}, stop={tp}")
                return None
            if abs(entry - live_price) / live_price > 0.01:
                logging.warning(f"Entry price for {symbol} too far from market: entry={entry}, live_price={live_price}")
                return None
            if abs(sl - live_price) / live_price > 0.1:
                logging.warning(f"SL too far for {symbol}: sl={sl}, live_price={live_price}")
                return None
            if abs(tp - live_price) / live_price > 1:
                logging.warning(f"TP too far for {symbol}: tp={tp}, live_price={live_price}")
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
                "ریسک به ریوارد": np.float64(rr_ratio),  # ### تغییر استفاده از rr_ratio محاسبه‌شده
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
                "قیمت فعلی": live_price
            }
            # Adding trailing stop task
            asyncio.create_task(manage_trailing_stop(exchange, symbol, entry, sl, signal_type))
            logging.info(f"Long signal generated: {result}")
            return result
        elif score_short >= THRESHOLD and trend_1d_score <= 0:  # Mandatory 1d trend condition
            signal_type = "Short"
            # محاسبه RR dynamically
            if resistance_4h > 0:
                dynamic_rr = max(dynamic_rr, (resistance_4h - support_4h) / risk_buffer if risk_buffer != 0 else 0)
            logging.info(f"Dynamic RR for {symbol} (Short): RR={dynamic_rr}")
            entry_data = await find_entry_point(exchange, symbol, signal_type, support_4h, resistance_4h)
            if entry_data is None:
                logging.info(f"No Short entry point found for {symbol} @ 15m")
                return None
            entry = entry_data["entry_price"]
            sl = entry_data["sl"]
            tp = entry_data["tp"]
            # ### تغییر: محاسبه کوتاه
            rr_ratio = (entry - tp) / (sl - entry) if (sl - entry) != 0 else 0
            if rr_ratio < 2:
                logging.info(f"Short signal rejected for {symbol}: RR={rr_ratio:.2f} < 2")
                return None
            live_price = await get_live_price(exchange, symbol)
            if live_price is None:
                logging.warning(f"Live price fetch failed for {symbol}, rejecting signal")
                return None
            price_diff = abs(entry - live_price) / live_price if live_price != 0 else float('inf')
            if price_diff > 0.01:
                logging.warning(f"Entry price discrepancy for {symbol} too high: entry={entry}, live_price={live_price}, diff={price_diff}")
                return None
            if sl <= entry or tp >= entry:
                logging.warning(f"Invalid SL or TP for {symbol}: entry={entry}, sl={sl}, tp={tp}")
                return None
            if abs(entry - live_price) / live_price > 0.01:
                logging.warning(f"Entry price too far from {symbol} market: entry={entry}, live_price={live_price}")
                return None
            if abs(sl - live_price) / live_price > 0.1:
                logging.warning(f"SL too far for {symbol}: sl={sl}, live_price={live_price}")
                return None
            if abs(tp - live_price) / live_price > 1:
                logging.warning(f"TP too far for {symbol}: tp={tp}, live_price={live_price}")
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
                "ریسک به ریوارد": np.float64(rr_ratio),  # ### تغییر: استفاده از محاسبه rr_ratio کوتاه
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
                "قیمت فعلی": live_price
            }
            # Adding trailing stop task
            asyncio.create_task(manage_trailing_stop(exchange, symbol), entry, sl, signal_type))
            logging.info(f"Short signal generated: {result}")
            return result
        logging.info(f"No signal generated for {symbol} @ {tf}")
        return None
    except Exception as e:
        logging.error(f"General error analyzing {symbol} @ {tf}: {str(e)}")
        return None

# تابع اسکن همه نمادها
async def scan_all_crypto_symbols(on_signal=None) -> None:
    exchange = ccxt.mexc({
        'enableRateLimit': True,
        'rateLimit': 2000
    })
    try:
        logging.info("Loading markets from MEXC...")
        await exchange.load_markets()
        logging.info(f"Markets loaded: {len(exchange.symbols)} symbols")
        top_coins = get_top_500_symbols_from_cmc()
        symbols = [s for s in exchange.symbols if any(s.startswith(f"{coin}/") and s.endswith("/USDT") for coin in top_coins])]
        logging.info(f"Filtered symbols: {len(symbols)} USDT symbols}")
        chunk_size = 10
        total_chunks = (len(symbols) + chunk_size - 1) // chunk_size
        symbol_results = []
        for idx in range(total_chunks):
            chunk = symbols[idx*chunk_size:(idx+1)*chunk_size]
            logging.info(f"Scanning chunk {idx+1}/{total_chunks}: {chunk}")
            tasks = []
            for sym in chunk:
                tasks.append(asyncio.create_task(analyze_symbol(exchange, sym, "1h")))
            async with semaphore:
                for task in asyncio.as_completed(tasks):
                    try:
                        result = await task
                        if isinstance(result, Exception):
                            logging.error(f"Task error: {result}")
                            continue
                        if result and on_signal:
                            await on_signal(result)
                        symbol_results.append(result)
                    except Exception as e:
                        logging.error(f"Error processing task for {sym}: {e}")
            await asyncio.sleep(WAIT_BETWEEN_CHUNKS)
        logging.info(f"Scan complete. Total signals: {len([r for r in result if r is not None])}")
        ablation_test(symbol_results, "final_scan")
    except Exception as e:
        logging.error(f"Error in main scan: {str(e)}")
    finally:
        await exchange.close()

if __name__ == "__main__":
    asyncio.run(scan_all_crypto_symbols())