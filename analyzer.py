import os
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["LC_ALL"] = "en_US.UTF-8"

import asyncio
import requests
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import ccxt.async_support as ccxt
import time
import logging
import traceback
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
import shap
from typing import Optional, Dict, Any, List
import matplotlib

# تنظیمات ثابت
CMC_API_KEY = os.getenv("CMC_API_KEY", "c1623ef8-7ef5-4ba4-8461-404a342b2e6b")
COINMARKETCAL_API_KEY = os.getenv("COINMARKETCAL_API_KEY", "iFrSo3PUBJ36P8ZnEIBMvakO5JutSIU1XJvG7ALa")
TIMEFRAMES = ["15m", "1h", "4h", "1d"]
MAX_CONCURRENT_REQUESTS = 5
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
WAIT_BETWEEN_REQUESTS = 1.0
WAIT_BETWEEN_CHUNKS = 10
VOLUME_WINDOW = 20
CACHE = {}
CACHE_TTL = 600
LIQUIDITY_REJECTS = 0
VOLUME_REJECTS = 0
SR_REJECTS = 0

# تنظیم لاگ‌ها
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s - [File: %(filename)s | Line: %(lineno)d | Func: %(funcName)s]',
    handlers=[
        logging.FileHandler("debug_detailed.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# تابع دریافت آیدی کوین از CMC
def get_coin_id(symbol: str) -> Optional[int]:
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/map"
    headers = {'Accepts': 'application/json', 'X-CMC_PRO_API_KEY': CMC_API_KEY}
    params = {'symbol': symbol}
    try:
        logging.debug(f"شروع دریافت آیدی برای {symbol} در ساعت 07:43 AM EEST, Saturday, May 17, 2025")
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
        logging.debug(f"شروع دریافت ۵۰۰ نماد برتر از CMC در ساعت 07:43 AM EEST, Saturday, May 17, 2025")
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
        self.model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        self.trained = False
        self.X_train = None
        self.y_train = None

    def analyze_training_data(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        logging.info("شروع تحلیل داده‌های آموزشی")
        unique, counts = np.unique(y, return_counts=True)
        imbalance = max(counts) / sum(counts) if sum(counts) > 0 else 0
        logging.info(f"عدم تعادل کلاس‌ها: {dict(zip(unique, counts))}, نسبت={imbalance:.2f}")
        
        if len(y) < 5 or min(counts) < 2:
            logging.warning("تعداد نمونه‌ها یا نمونه‌های کلاس اقلیت برای SMOTE کافی نیست، از داده‌های اصلی استفاده می‌شود")
            return {"imbalance": imbalance, "X_balanced": X, "y_balanced": y}
        
        minority_count = min(counts)
        k_neighbors = min(2, minority_count - 1) if minority_count > 1 else 1
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        logging.info(f"تعداد نمونه‌ها بعد از SMOTE: {len(y_balanced)}, کلاس‌ها: {np.unique(y_balanced, return_counts=True)[1]}")
        return {"imbalance": imbalance, "X_balanced": X_balanced, "y_balanced": y_balanced}

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        analysis = self.analyze_training_data(X, y)
        X_balanced = analysis["X_balanced"]
        y_balanced = analysis["y_balanced"]
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        for train_idx, val_idx in kf.split(X_balanced):
            X_train_fold, X_val_fold = X_balanced[train_idx], X_balanced[val_idx]
            y_train_fold, y_val_fold = y_balanced[train_idx], y_balanced[val_idx]
            self.model.fit(X_train_fold, y_train_fold)
            score = self.model.score(X_val_fold, y_val_fold)
            cv_scores.append(score)
        avg_cv_score = np.mean(cv_scores)
        logging.info(f"میانگین امتیاز Cross-validation: {avg_cv_score:.2f}")
        self.model.fit(X_balanced, y_balanced)
        self.trained = True
        self.X_train = X_balanced
        self.y_train = y_balanced
        try:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_balanced)
            shap.summary_plot(shap_values, X_balanced, feature_names=["RSI", "ADX", "Vol_Ratio", "Volatility", "Dist_Res", "Dist_Sup", "Spread", "Psych_Long", "Psych_Short"], show=False)
            logging.info("تحلیل SHAP تکمیل شد.")
        except Exception as e:
            logging.warning(f"خطا در ترسیم SHAP: {e}. ترسیم SHAP رد شد.")

    def predict(self, features: list) -> float:
        if not self.trained:
            logging.warning("مدل آموزش داده نشده است. پیش‌فرض=True")
            return 10
        try:
            prediction = self.model.predict_proba([features])[0][1]
            score = prediction * 20
            logging.debug(f"پیش‌بینی مدل: features={features}, score={score:.2f}")
            return score
        except Exception as e:
            logging.error(f"خطا در پیش‌بینی مدل: {e}")
            return 0

    def retrain_model(self, new_X: np.ndarray, new_y: np.ndarray) -> None:
        if self.X_train is not None and self.y_train is not None:
            X_combined = np.vstack((self.X_train, new_X))
            y_combined = np.hstack((self.y_train, new_y))
            self.train(X_combined, y_combined)
            logging.info("مدل با داده‌های جدید بازآموزی شد.")

# تابع دریافت داده کندل‌ها با کش
async def get_ohlcv_cached(exchange, symbol, tf, limit=50) -> Optional[pd.DataFrame]:
    key = f"{exchange.id}_{symbol}_{tf}"
    now = datetime.utcnow()

    if key in CACHE:
        cached_df, cached_time = CACHE[key]
        if now - cached_time < timedelta(minutes=5):
            return cached_df

    try:
        await exchange.load_markets()
        logging.info(f"📈 بررسی نماد {symbol} در لیست {exchange.id}: تعداد نمادها {len(exchange.symbols)}")
        if symbol not in exchange.symbols:
            logging.warning(f"❌ نماد {symbol} در لیست مارکت‌های صرافی {exchange.id} نیست.")
            return None
    except Exception as e:
        logging.error(f"❌ خطا در بارگذاری مارکت‌ها برای بررسی وجود {symbol}: {e}")
        return None

    for attempt in range(5):  # افزایش تلاش‌ها به 5
        try:
            async with asyncio.timeout(30):  # افزایش تایم‌اوت
                raw_data = await exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)

            if not raw_data or len(raw_data) == 0:
                logging.warning(f"داده ناکافی برای {symbol} @ {tf}: تعداد کندل‌ها=0")
                return None

            df = pd.DataFrame(raw_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            CACHE[key] = (df, now)
            return df

        except asyncio.TimeoutError:
            logging.error(f"❌ Timeout در گرفتن OHLCV برای {symbol} / {tf} - تلاش {attempt+1}")
            await asyncio.sleep(3 * (attempt + 1))  # افزایش فاصله بین تلاش‌ها
        except Exception as e:
            logging.error(f"❌ خطا در گرفتن OHLCV برای {symbol} / {tf}: {e}")
            return None

    return None

# تابع بررسی نقدینگی
async def check_liquidity(exchange: ccxt.Exchange, symbol: str, df: pd.DataFrame) -> tuple:
    global LIQUIDITY_REJECTS
    try:
        async with asyncio.timeout(5):
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
                async with asyncio.timeout(5):
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
        if spread > 0.002:
            logging.warning(f"اسپرد برای {symbol} بیش از حد بالاست: spread={spread:.4f}")
            LIQUIDITY_REJECTS += 1
            return spread, -10
        score = 15 if spread < spread_threshold else -5
        logging.info(f"نقدینگی {symbol}: spread={spread:.4f}, threshold={spread_threshold:.4f}, score={score}")
        if spread >= spread_threshold:
            LIQUIDITY_REJECTS += 1
        return spread, score
    except asyncio.TimeoutError:
        logging.error(f"Timeout در بررسی نقدینگی برای {symbol}")
        return float('inf'), 0
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
    start_date = (datetime.utcnow() - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0).strftime("%Y-%m-%d")
    end_date = (datetime.utcnow() + timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0).strftime("%Y-%m-%d")
    params = {
        "coinId": str(coin_id),
        "max": 5,
        "dateRangeStart": start_date,
        "dateRangeEnd": end_date
    }
    try:
        logging.debug(f"شروع دریافت رویدادها برای {symbol}, coin_id={coin_id} در ساعت 07:43 AM EEST, Saturday, May 17, 2025")
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

# تابع محاسبه اندیکاتورها
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
        logging.debug(f"اندیکاتورها با موفقیت محاسبه شدند: {list(df.columns)}")
    except Exception as e:
        logging.error(f"خطا در محاسبات اندیکاتورها: {str(e)}")
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
        logging.warning(f"اندیکاتورها با مقادیر پیش‌فرض پر شدند")
    return df

# تابع تحلیل ساختار بازار
async def analyze_market_structure(exchange: ccxt.Exchange, symbol: str) -> Dict[str, Any]:
    try:
        logging.info(f"شروع تحلیل ساختار بازار برای {symbol} در تایم‌فریم 4h در ساعت 07:43 AM EEST, Saturday, May 17, 2025")
        df_4h = await get_ohlcv_cached(exchange, symbol, "4h", limit=50)
        if df_4h is None or len(df_4h) < 50:
            logging.warning(f"داده ناکافی برای تحلیل ساختار بازار {symbol} @ 4h: تعداد کندل‌ها={len(df_4h) if df_4h is not None else 0}")
            return {"trend": "Neutral", "score": 0, "support": 0, "resistance": 0, "fng_index": 50, "market_regime": "Neutral"}

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
        logging.info(f"سطوح کلیدی در 4h برای {symbol}: حمایت={support:.2f}, مقاومت={resistance:.2f}")

        fng_index = get_fear_and_greed_index()
        if fng_index < 25:
            trend_score += 10
            logging.info(f"شاخص ترس و طمع در 4h برای {symbol}: {fng_index} (ترس شدید) - 10 امتیاز به روند صعودی اضافه شد")
        elif fng_index > 75:
            trend_score += -10
            logging.info(f"شاخص ترس و طمع در 4h برای {symbol}: {fng_index} (طمع شدید) - 10 امتیاز به روند نزولی اضافه شد")

        adx = last_4h["ADX"]
        rsi = last_4h["RSI"]
        atr = last_4h["ATR"]
        market_regime = "Neutral"
        if adx > 25 and rsi > 60 and atr > df_4h["ATR"].mean():
            market_regime = "Trending Up"
        elif adx > 25 and rsi < 40 and atr > df_4h["ATR"].mean():
            market_regime = "Trending Down"
        elif adx < 20 and atr < df_4h["ATR"].mean():
            market_regime = "Sideways"

        result = {
            "trend": trend_direction,
            "score": trend_score,
            "support": support,
            "resistance": resistance,
            "fng_index": fng_index,
            "market_regime": market_regime
        }
        logging.info(f"تحلیل ساختار بازار برای {symbol} @ 4h تکمیل شد: {result}")
        return result
    except Exception as e:
        logging.error(f"خطا در تحلیل ساختار بازار برای {symbol} @ 4h: {str(e)}")
        return {"trend": "Neutral", "score": 0, "support": 0, "resistance": 0, "fng_index": 50, "market_regime": "Neutral"}

# تابع دریافت قیمت واقعی
async def get_live_price(exchange: ccxt.Exchange, symbol: str, max_attempts: int = 3) -> Optional[float]:
    attempt = 0
    last_ticker = None
    while attempt < max_attempts:
        try:
            async with asyncio.timeout(5):
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
            last_ticker = live_price
            logging.info(f"قیمت واقعی بازار برای {symbol}: live_price={live_price}, bid={bid}, ask={ask}, last={last}")
            return live_price
        except asyncio.TimeoutError:
            logging.error(f"Timeout در دریافت قیمت واقعی برای {symbol} در تلاش {attempt + 1}")
            attempt += 1
            await asyncio.sleep(0.3)
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

# تابع پیدا کردن نقطه ورود
async def find_entry_point(exchange: ccxt.Exchange, symbol: str, signal_type: str, support: float, resistance: float) -> Optional[float]:
    try:
        logging.info(f"شروع پیدا کردن نقطه ورود برای {symbol} در تایم‌فریم 15m - نوع سیگنال: {signal_type}")
        df_15m = await get_ohlcv_cached(exchange, symbol, "15m")
        if df_15m is None or len(df_15m) < 20:
            logging.warning(f"داده ناکافی برای پیدا کردن نقطه ورود {symbol} @ 15m: تعداد کندل‌ها={len(df_15m) if df_15m is not None else 0}")
            return None

        df_15m = compute_indicators(df_15m)
        last_15m = df_15m.iloc[-1]

        live_price = await get_live_price(exchange, symbol)
        if live_price is None:
            logging.warning(f"قیمت واقعی برای {symbol} دریافت نشد، سیگنال رد می‌شود")
            return None

        price_diff = abs(live_price - last_15m["close"]) / live_price if live_price != 0 else float('inf')
        logging.debug(f"مقایسه قیمت برای {symbol}: live_price={live_price}, candle_price={last_15m['close']}, اختلاف={price_diff}")
        if price_diff > 0.01:
            logging.warning(f"اختلاف قیمت برای {symbol} بیش از حد است: live_price={live_price}, candle_price={last_15m['close']}, اختلاف={price_diff}")
            return None

        volume_mean = df_15m["volume"].rolling(20).mean().iloc[-1]
        volume_condition = last_15m["volume"] > volume_mean * 0.8
        price_action = last_15m["PinBar"] or last_15m["Engulfing"]
        logging.info(f"جزئیات {signal_type} برای {symbol}: close={last_15m['close']}, resistance={resistance}, support={support}")
        logging.info(f"حجم: current={last_15m['volume']:.2f}, mean={volume_mean:.2f}, condition={volume_condition}")
        logging.info(f"الگوهای قیمتی: PinBar={last_15m['PinBar']}, Engulfing={last_15m['Engulfing']}, price_action={price_action}")

        if signal_type == "Long":
            breakout_resistance = last_15m["close"] > resistance and volume_condition
            near_support = abs(last_15m["close"] - support) / last_15m["close"] < 0.03 and volume_condition
            within_range = support < last_15m["close"] < resistance and volume_condition
            if price_action:
                logging.info(f"الگوی قیمتی پیدا شد، امتیاز اضافی برای {symbol}")
            entry_condition = breakout_resistance or near_support or within_range
            logging.debug(f"شرایط Long برای {symbol}: breakout_resistance={breakout_resistance}, near_support={near_support}, within_range={within_range}, final_condition={entry_condition}")
            if entry_condition:
                entry_price = live_price
                logging.info(f"نقطه ورود Long برای {symbol} @ 15m پیدا شد: قیمت ورود={entry_price}")
                return entry_price
            else:
                logging.info(f"شرایط ورود Long برای {symbol} @ 15m برقرار نشد")
                return None

        elif signal_type == "Short":
            breakout_support = last_15m["close"] < support and volume_condition
            near_resistance = abs(last_15m["close"] - resistance) / last_15m["close"] < 0.03 and volume_condition
            within_range = support < last_15m["close"] < resistance and volume_condition
            if price_action:
                logging.info(f"الگوی قیمتی پیدا شد، امتیاز اضافی برای {symbol}")
            entry_condition = breakout_support or near_resistance or within_range
            logging.debug(f"شرایط Short برای {symbol}: breakout_support={breakout_support}, near_resistance={near_resistance}, within_range={within_range}, final_condition={entry_condition}")
            if entry_condition:
                entry_price = live_price
                logging.info(f"نقطه ورود Short برای {symbol} @ 15m پیدا شد: قیمت ورود={entry_price}")
                return entry_price
            else:
                logging.info(f"شرایط ورود Short برای {symbol} @ 15m برقرار نشد")
                return None

        return None
    except Exception as e:
        logging.error(f"خطا در پیدا کردن نقطه ورود برای {symbol} @ 15m: {str(e)}")
        return None
        
# تابع تأیید مولتی تایم‌فریم
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
                logging.warning(f"داده ناکافی برای {symbol} @ {tf} در تأیید مولتی تایم‌فریم: تعداد کندل‌ها={len(df_tf) if df_tf is not None else 0}")
                continue
            df_tf["EMA12"] = df_tf["close"].ewm(span=12).mean()
            df_tf["EMA26"] = df_tf["close"].ewm(span=26).mean()
            long_trend = df_tf["EMA12"].iloc[-1] > df_tf["EMA26"].iloc[-1]
            score += (weight * 10) if long_trend else (-weight * 5)
            total_weight += weight
        except Exception as e:
            logging.error(f"خطا در پردازش تایم‌فریم {tf} برای {symbol}: {e}")
            continue
    final_score = score if total_weight > 0 else 0
    logging.info(f"مولتی تایم‌فریم برای {symbol} تکمیل شد: score={final_score:.2f}, total_weight={total_weight}")
    return final_score

# تابع محاسبه حجم پوزیشن
def calculate_position_size(account_balance: float, risk_percentage: float, entry: float, stop_loss: float) -> float:
    if entry is None or stop_loss is None or entry == 0 or stop_loss == 0:
        logging.warning(f"مقادیر نامعتبر برای محاسبه حجم پوزیشن: entry={entry}, stop_loss={stop_loss}")
        return 0
    risk_amount = account_balance * (risk_percentage / 100)
    distance = abs(entry - stop_loss)
    position_size = risk_amount / distance if distance != 0 else 0
    return round(position_size, 2)

# تابع تست ablation
def ablation_test(symbol_results: list, filter_name: str) -> int:
    total_signals = len([r for r in symbol_results if r is not None])
    logging.info(f"Ablation Test برای فیلتر {filter_name}: تعداد سیگنال‌های اولیه={total_signals}")
    return total_signals

# تابع محاسبه سطوح فیبوناچی
def calculate_fibonacci_levels(df, high_col="high", low_col="low"):
    max_price = df[high_col].max()
    min_price = df[low_col].min()
    diff = max_price - min_price
    levels = {
        "0.236": max_price - 0.236 * diff,
        "0.382": max_price - 0.382 * diff,
        "0.5": max_price - 0.5 * diff,
        "0.618": max_price - 0.618 * diff,
        "0.786": max_price - 0.786 * diff,
    }
    return levels

# تابع تحلیل نماد
async def analyze_symbol(exchange: ccxt.Exchange, symbol: str, tf: str) -> Optional[dict]:
    global VOLUME_REJECTS, SR_REJECTS
    start_time = time.time()
    logging.info(f"شروع تحلیل {symbol} @ {tf} در ساعت 07:43 AM EEST, Saturday, May 17, 2025")

    try:
        market_structure = await analyze_market_structure(exchange, symbol)
        trend_4h = market_structure["trend"]
        trend_score_4h = market_structure["score"]
        support_4h = market_structure["support"]
        resistance_4h = market_structure["resistance"]
        fng_index = market_structure.get("fng_index", 50)
        market_regime = market_structure["market_regime"]

        if tf != "1h":
            logging.info(f"تحلیل برای {symbol} فقط در تایم‌فریم 1h انجام می‌شود. تایم‌فریم فعلی: {tf}")
            return None

        df = await get_ohlcv_cached(exchange, symbol, tf, limit=50)
        if df is None or len(df) < 30:
            logging.warning(f"داده ناکافی برای {symbol} @ {tf}: تعداد کندل‌ها={len(df) if df is not None else 0}")
            return None
        logging.info(f"داده دریافت شد برای {symbol} @ {tf} در {time.time() - start_time:.2f} ثانیه, تعداد ردیف‌ها={len(df)}")

        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            logging.error(f"ستون‌های مورد نیاز در دیتافریم {symbol} @ {tf} وجود ندارند")
            return None
        df = df.ffill().bfill().fillna(0)

        df = compute_indicators(df)
        last = df.iloc[-1]

        score_long = 0
        score_short = 0
        score_log = {"long": {}, "short": {}}

        vol_avg = df["volume"].rolling(VOLUME_WINDOW).mean().iloc[-1]
        current_vol = df["volume"].iloc[-1]
        vol_mean = df["volume"].rolling(20).mean().iloc[-1]
        vol_std = df["volume"].rolling(20).std().iloc[-1]
        logging.info(f"داده‌های حجم خام برای {symbol} @ {tf}: {df['volume'].tail(5).to_dict()}")
        vol_threshold = vol_mean * 0.5
        vol_score = 10 if current_vol >= vol_threshold else -2
        score_long += vol_score
        score_short += vol_score
        score_log["long"]["volume"] = vol_score
        score_log["short"]["volume"] = vol_score
        logging.info(f"حجم برای {symbol} @ {tf}: current_vol={current_vol:.2f}, threshold={vol_threshold:.2f}, score={vol_score}")
        if current_vol < vol_threshold:
            VOLUME_REJECTS += 1
            logging.info(f"حجم کم برای {symbol} @ {tf}: score={vol_score}")

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
            logging.warning(f"سیگنال برای {symbol} به دلیل نقدینگی ضعیف رد شد: liquidity_score={liquidity_score}")
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

        fib_levels = calculate_fibonacci_levels(df)
        fib_score_long = 5 if last["close"] > fib_levels["0.618"] else 0
        fib_score_short = 5 if last["close"] < fib_levels["0.382"] else 0
        score_long += fib_score_long
        score_short += fib_score_short
        score_log["long"]["fibonacci"] = fib_score_long
        score_log["short"]["fibonacci"] = fib_score_short

        support_buffer = (df["ATR"].iloc[-1] / last["close"]) * 1.5
        resistance_buffer = (df["ATR"].iloc[-1] / last["close"] * 1.5)
        min_conditions = 2
        conds_long = {
            "PinBar": last["PinBar"] and last["lower"] > 3 * last["body"],
            "Engulfing": last["Engulfing"] and last["close"] > last["open"] and (df["volume"].iloc[-1] > df["volume"].rolling(20).mean().iloc[-1] * 1.5),
            "Elliott_Wave": df["WaveTrend"].iloc[-1] == "Up",
            "EMA_Cross": df["EMA12"].iloc[-1] > df["EMA26"].iloc[-1] and (df["volume"].iloc[-1] > df["volume"].rolling(20).mean().iloc[-1] * 1.2),
            "MACD_Cross": df["MACD"].iloc[-2] < df["Signal"].iloc[-2] and df["MACD"].iloc[-1] > df["Signal"].iloc[-1] and (df["MACD"].iloc[-1] > 0),
            "RSI_Oversold": last["RSI"] < 25,
            "Stochastic_Oversold": last["Stochastic"] < 15,
            "BB_Breakout": last["close"] > last["BB_upper"] and (df["volume"].iloc[-1] > df["volume"].rolling(20).mean().iloc[-1] * 1.5),
            "MFI_Oversold": last["MFI"] < 15,
            "ADX_Strong": last["ADX"] > 25,
            "Support_Confirmation": distance_to_support <= support_buffer and (last["PinBar"] or last["Engulfing"])
        }
        conds_short = {
            "PinBar": last["PinBar"] and last["upper"] > 3 * last["body"],
            "Engulfing": last["Engulfing"] and last["close"] < last["open"] and (df["volume"].iloc[-1] > df["volume"].rolling(20).mean().iloc[-1] * 1.5),
            "Elliott_Wave": df["WaveTrend"].iloc[-1] == "Down",
            "EMA_Cross": df["EMA12"].iloc[-1] < df["EMA26"].iloc[-1] and (df["volume"].iloc[-1] > df["volume"].rolling(20).mean().iloc[-1] * 1.2),
            "MACD_Cross": df["MACD"].iloc[-2] > df["Signal"].iloc[-2] and df["MACD"].iloc[-1] < df["Signal"].iloc[-1] and (df["MACD"].iloc[-1] < 0),
            "RSI_Overbought": last["RSI"] > 75,
            "Stochastic_Overbought": last["Stochastic"] > 85,
            "BB_Breakout": last["close"] < last["BB_lower"] and (df["volume"].iloc[-1] > df["volume"].rolling(20).mean().iloc[-1] * 1.5),
            "MFI_Overbought": last["MFI"] > 85,
            "ADX_Strong": last["ADX"] > 25,
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
        logging.debug(f"شرایط اندیکاتورها برای {symbol} @ {tf}: long_score={indicator_score_long}, short_score={indicator_score_short}, conditions_long={conds_long}")

        score_long += trend_score_4h
        score_short += -trend_score_4h
        score_log["long"]["market_structure_4h"] = trend_score_4h
        score_log["short"]["market_structure_4h"] = -trend_score_4h
        logging.info(f"امتیاز ساختار بازار 4h برای {symbol}: Long={trend_score_4h}, Short={-trend_score_4h}")

        volume_condition = current_vol > vol_mean * 0.8

        if trend_4h == "Up" and distance_to_resistance < 0.01 and (last["RSI"] > 70 or last["Stochastic"] > 80):
            logging.info(f"سیگنال Long برای {symbol} به دلیل نزدیکی به سقف و اشباع خرید رد شد")
            score_long = -100

        if trend_4h == "Down" and last["close"] > support_4h and volume_condition and (last["PinBar"] or last["Engulfing"]):
            logging.info(f"سیگنال Long برای {symbol} به دلیل شکست روند نزولی و حجم بالا تولید شد")
            score_long += 20

        atr_1h = df["ATR"].iloc[-1]
        sl_distance = max(2 * atr_1h, last["close"] * 0.01)
        tp_distance = 2 * sl_distance

        logging.debug(f"شروع فیلتر مدل برای {symbol} @ {tf}")
        signal_filter = SignalFilter()
        X_train = np.array([
            [30, 25, 2, 0.01, 0.05, 0.05, 0.01, 10, -10],
            [70, 20, 1, 0.02, 0.03, 0.03, 0.02, -10, 10],
            [50, 30, 1.5, 0.01, 0.04, 0.04, 0.01, 0, 0],
            [40, 28, 1.8, 0.015, 0.06, 0.06, 0.015, 5, -5],
            [60, 22, 1.2, 0.025, 0.02, 0.02, 0.025, -5, 5],
            [45, 27, 1.7, 0.012, 0.05, 0.05, 0.012, 8, -8],
            [65, 18, 0.9, 0.03, 0.03, 0.03, 0.03, -8, 8],
            [55, 24, 1.4, 0.02, 0.04, 0.04, 0.02, 3, -3],
            [35, 26, 2.1, 0.01, 0.06, 0.06, 0.01, 7, -7],
            [75, 19, 0.8, 0.03, 0.02, 0.02, 0.03, -9, 9],
            [45, 23, 1.6, 0.015, 0.04, 0.04, 0.015, 6, -6],  # نمونه جدید
            [60, 21, 1.3, 0.02, 0.03, 0.03, 0.02, -7, 7],    # نمونه جدید
            [50, 27, 1.9, 0.01, 0.05, 0.05, 0.01, 4, -4],    # نمونه جدید
            [70, 17, 0.7, 0.03, 0.02, 0.02, 0.03, -6, 6],    # نمونه جدید
            [40, 25, 2.0, 0.012, 0.06, 0.06, 0.012, 9, -9],  # نمونه جدید
        ])
        y_train = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])  # تنظیم متناسب با X_train
        signal_filter.train(X_train, y_train)
        features = [
            last["RSI"],
            last["ADX"],
            current_vol / vol_avg if vol_avg != 0 else 0,
            volatility,
            distance_to_resistance,
            distance_to_support,
            spread if 'spread' in locals() else 0.0,
            psych_score_long,
            psych_score_short
        ]
        dt_score = signal_filter.predict(features)
        score_long += dt_score
        score_short += dt_score
        score_log["long"]["decision_tree"] = dt_score
        score_log["short"]["decision_tree"] = dt_score
        logging.debug(f"فیلتر مدل برای {symbol} @ {tf}: features={features}, score={dt_score:.2f}")

        logging.info(f"امتیاز نهایی برای {symbol} @ {tf}: score_long={score_long:.2f}, score_short={score_short:.2f}")
        logging.info(f"جزئیات امتیاز Long: {score_log['long']}")
        logging.info(f"جزئیات امتیاز Short: {score_log['short']}")

        THRESHOLD = 120
        if score_long >= THRESHOLD:
            entry = await find_entry_point(exchange, symbol, "Long", support_4h, resistance_4h)
            if entry is None:
                logging.info(f"نقطه ورود Long برای {symbol} در 15m پیدا نشد")
                return None

            live_price = await get_live_price(exchange, symbol)
            if live_price is None:
                logging.warning(f"قیمت واقعی برای {symbol} دریافت نشد، سیگنال رد می‌شود")
                return None

            price_diff = abs(entry - live_price) / live_price if live_price != 0 else float('inf')
            if price_diff > 0.01:
                logging.warning(f"اختلاف قیمت ورود با قیمت واقعی برای {symbol} بیش از حد است: entry={entry}, live_price={live_price}, اختلاف={price_diff}")
                return None

            sl = entry - sl_distance
            if sl < support_4h * 0.95:
                sl = support_4h * 0.98
                logging.info(f"حد ضرر برای {symbol} تنظیم شد تا خیلی زیر حمایت نباشه: sl={sl}")

            tp = entry + tp_distance
            if tp > resistance_4h * 1.05:
                tp = resistance_4h * 1.02
                logging.info(f"هدف سود برای {symbol} تنظیم شد تا خیلی بالای مقاومت نباشه: tp={tp}")

            if sl >= entry or tp <= entry:
                logging.warning(f"حد ضرر یا هدف سود برای {symbol} نامعتبر است: entry={entry}, sl={sl}, tp={tp}")
                return None

            rr = round((tp - entry) / (entry - sl), 2) if (entry - sl) != 0 else 0
            position_size = calculate_position_size(10000, 1, entry, sl)
            signal_strength = "قوی" if score_long > 90 else "متوسط"
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
                "فاندامنتال": f"امتیاز: {fundamental_score}",
                "شاخص ترس و طمع": fng_index,
                "روند 4h": trend_4h,
                "رژیم بازار": market_regime,
                "قیمت فعلی بازار": live_price
            }
            logging.info(f"سیگنال Long تولید شد: {result}")
            return result

        elif score_short >= THRESHOLD:
            entry = await find_entry_point(exchange, symbol, "Short", support_4h, resistance_4h)
            if entry is None:
                logging.info(f"نقطه ورود Short برای {symbol} در 15m پیدا نشد")
                return None

            live_price = await get_live_price(exchange, symbol)
            if live_price is None:
                logging.warning(f"قیمت واقعی برای {symbol} دریافت نشد، سیگنال رد می‌شود")
                return None

            price_diff = abs(entry - live_price) / live_price if live_price != 0 else float('inf')
            if price_diff > 0.01:
                logging.warning(f"اختلاف قیمت ورود با قیمت واقعی برای {symbol} بیش از حد است: entry={entry}, live_price={live_price}, اختلاف={price_diff}")
                return None

            sl = entry + sl_distance
            if sl > resistance_4h * 1.05:
                sl = resistance_4h * 1.02
                logging.info(f"حد ضرر برای {symbol} تنظیم شد تا خیلی بالای مقاومت نباشه: sl={sl}")

            tp = entry - tp_distance
            if tp < support_4h * 0.95:
                tp = support_4h * 0.98
                logging.info(f"هدف سود برای {symbol} تنظیم شد تا خیلی زیر حمایت نباشه: tp={tp}")

            if sl <= entry or tp >= entry:
                logging.warning(f"حد ضرر یا هدف سود برای {symbol} نامعتبر است: entry={entry}, sl={sl}, tp={tp}")
                return None

            rr = round((entry - tp) / (sl - entry), 2) if (sl - entry) != 0 else 0
            position_size = calculate_position_size(10000, 1, entry, sl)
            signal_strength = "قوی" if score_short > 90 else "متوسط"
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
                "فاندامنتال": f"امتیاز: {fundamental_score}",
                "شاخص ترس و طمع": fng_index,
                "روند 4h": trend_4h,
                "رژیم بازار": market_regime,
                "قیمت فعلی بازار": live_price
            }
            logging.info(f"سیگنال Short تولید شد: {result}")
            return result

        logging.info(f"سیگنال برای {symbol} @ {tf} رد شد")
        return None

    except Exception as e:
        logging.error(f"خطای کلی در تحلیل {symbol} @ {tf}: {str(e)}")
        return None

# تابع اسکن همه نمادها
async def scan_all_crypto_symbols(on_signal=None) -> None:
    exchange = ccxt.mexc({
        'enableRateLimit': True,
        'rateLimit': 2000
    })
    try:
        logging.debug(f"شروع بارگذاری بازارها از MEXC در ساعت 07:43 AM EEST, Saturday, May 17, 2025")
        await exchange.load_markets()
        logging.info(f"بازارها بارگذاری شد: تعداد نمادها={len(exchange.symbols)}")
        top_coins = get_top_500_symbols_from_cmc()
        usdt_symbols = [s for s in exchange.symbols if any(s.startswith(f"{coin}/") and s.endswith("/USDT") for coin in top_coins)]
        logging.debug(f"فیلتر نمادها: تعداد USDT symbols={len(usdt_symbols)}")
        chunk_size = 10
        total_chunks = (len(usdt_symbols) + chunk_size - 1) // chunk_size
        symbol_results = []
        for idx in range(total_chunks):
            chunk = usdt_symbols[idx*chunk_size:(idx+1)*chunk_size]
            logging.info(f"شروع اسکن دسته {idx+1}/{total_chunks}: {chunk}")
            tasks = []
            for sym in chunk:
                tasks.append(asyncio.create_task(analyze_symbol(exchange, sym, "1h")))
            async with semaphore:
                for task in asyncio.as_completed(tasks, timeout=300):
                    try:
                        result = await task
                        if isinstance(result, Exception):
                            logging.error(f"خطا در تسک: {result}")
                            continue
                        if result and on_signal:
                            await on_signal(result)
                        symbol_results.append(result)
                    except asyncio.TimeoutError:
                        logging.error(f"Timeout در انتظار تسک برای دسته {idx+1}")
                        continue
                    except Exception as e:
                        logging.error(f"خطا در انتظار تسک برای دسته {idx+1}: {e}")
                        continue
            await asyncio.sleep(WAIT_BETWEEN_CHUNKS)
        ablation_test(symbol_results, "volume")
        ablation_test(symbol_results, "liquidity")
        ablation_test(symbol_results, "support_resistance")
        logging.info(f"آمار رد شدن‌ها: نقدینگی={LIQUIDITY_REJECTS}, حجم={VOLUME_REJECTS}, حمایت/مقاومت={SR_REJECTS}")
    except Exception as e:
        logging.error(f"خطای کلی در اسکن نمادها: {str(e)}")
        raise
    finally:
        logging.debug(f"بستن اتصال به MEXC")
        await exchange.close()

# تابع اصلی برای تست
async def main():
    exchange = ccxt.mexc({
        'enableRateLimit': True,
        'rateLimit': 2000
    })
    try:
        logging.debug(f"شروع بارگذاری بازارها برای تست در ساعت 07:43 AM EEST, Saturday, May 17, 2025")
        await exchange.load_markets()
        logging.info(f"بازارها بارگذاری شد برای تست")
        test_symbols = ["ANIME/USDT", "BTC/USDT", "ETH/USDT"]
        for symbol in test_symbols:
            logging.info(f"شروع تست برای {symbol}")
            result = await analyze_symbol(exchange, symbol, "1h")
            if result:
                logging.info(f"سیگنال تولید شد برای {symbol}: {result}")
            else:
                logging.info(f"هیچ سیگنالی برای {symbol} تولید نشد.")
    except Exception as e:
        logging.error(f"خطا در اجرای تست: {str(e)}")
        raise
    finally:
        logging.debug(f"بستن اتصال به MEXC پس از تست")
        await exchange.close()

# اجرای برنامه
if __name__ == "__main__":
    logging.info(f"اجرای برنامه در ساعت 07:43 AM EEST, Saturday, May 17, 2025")
    asyncio.run(main())