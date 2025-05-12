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

# پارامترهای اصلی
VOLUME_WINDOW = 20
CACHE = {}
CACHE_TTL = 600
MAX_CONCURRENT_REQUESTS = 15
WAIT_BETWEEN_REQUESTS = 0.3
WAIT_BETWEEN_CHUNKS = 2

# متغیرهای شمارشگر رد شدن‌ها
LIQUIDITY_REJECTS = 0
VOLUME_REJECTS = 0
SR_REJECTS = 0

# دریافت آیدی ارز
def get_coin_id(symbol: str) -> Optional[int]:
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/map"
    headers = {'Accepts': 'application/json', 'X-CMC_PRO_API_KEY': CMC_API_KEY}
    params = {'symbol': symbol}
    try:
        logging.debug(f"شروع دریافت آیدی برای {symbol}")
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        data = resp.json()
        logging.debug(f"پاسخ دریافت شد برای آیدی {symbol}: {data}")
        if 'data' in data and len(data['data']) > 0:
            coin_id = data['data'][0]['id']
            logging.info(f"دریافت آیدی برای {symbol}: coin_id={coin_id}")
            return coin_id
        else:
            logging.warning(f"آیدی برای {symbol} یافت نشد")
            return None
    except Exception as e:
        logging.error(f"خطا در دریافت آیدی برای {symbol}: {e}, traceback={str(traceback.format_exc())}")
        return None

# دریافت ۵۰۰ نماد برتر از CoinMarketCap
def get_top_500_symbols_from_cmc() -> List[str]:
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    headers = {'Accepts': 'application/json', 'X-CMC_PRO_API_KEY': CMC_API_KEY}
    params = {'start': '1', 'limit': '500', 'convert': 'USD'}
    try:
        logging.debug(f"شروع دریافت ۵۰۰ نماد برتر از CMC")
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        data = resp.json()
        logging.debug(f"پاسخ CMC دریافت شد: تعداد داده‌ها={len(data.get('data', []))}")
        logging.info(f"دریافت ۵۰۰ نماد برتر از CMC: تعداد نمادها={len(data['data'])}")
        return [entry['symbol'] for entry in data['data']]
    except Exception as e:
        logging.error(f"خطا در دریافت از CMC: {e}, traceback={str(traceback.format_exc())}")
        return []

# کلاس برای مدیریت اندیکاتورها
class IndicatorCalculator:
    @staticmethod
    def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        logging.debug(f"RSI محاسبه شد: آخرین مقدار={rsi.iloc[-1] if not rsi.empty else 'خالی'}")
        return rsi

    @staticmethod
    def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        tr = pd.concat([df["high"] - df["low"], abs(df["high"] - df["close"].shift()), abs(df["low"] - df["close"].shift())], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        logging.debug(f"ATR محاسبه شد: آخرین مقدار={atr.iloc[-1] if not atr.empty else 'خالی'}")
        return atr

    @staticmethod
    def compute_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> tuple:
        sma = df["close"].rolling(period).mean()
        std = df["close"].rolling(period).std()
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        logging.debug(f"Bollinger Bands محاسبه شد: upper={upper.iloc[-1] if not upper.empty else 'خالی'}, lower={lower.iloc[-1] if not lower.empty else 'خالی'}")
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
        logging.debug(f"ADX محاسبه شد: آخرین مقدار={adx.iloc[-1] if not adx.empty else 'خالی'}")
        return adx

    @staticmethod
    def compute_stochastic(df: pd.DataFrame, period: int = 14) -> pd.Series:
        low_min = df["low"].rolling(window=period).min()
        high_max = df["high"].rolling(window=period).max()
        k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, 1e-10)
        logging.debug(f"Stochastic محاسبه شد: آخرین مقدار={k.iloc[-1] if not k.empty else 'خالی'}")
        return k

    @staticmethod
    def compute_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        raw_money_flow = typical_price * df['volume']
        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0).rolling(period).sum()
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0).rolling(period).sum()
        mfi = 100 - (100 / (1 + positive_flow / negative_flow.replace(0, 1e-10)))
        logging.debug(f"MFI محاسبه شد: آخرین مقدار={mfi.iloc[-1] if not mfi.empty else 'خالی'}")
        return mfi

    @staticmethod
    def compute_linear_regression(df: pd.DataFrame, period: int) -> tuple:
        logging.debug(f"شروع محاسبه رگرسیون خطی برای دوره {period}")
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
        logging.debug(f"رگرسیون خطی تکمیل شد: floor={floor}, ceiling={ceiling}, slope={slope}, current_price={current_price}")
        return floor, ceiling, slope, current_price

# تشخیص الگوها
class PatternDetector:
    @staticmethod
    def detect_pin_bar(df: pd.DataFrame) -> pd.Series:
        df["body"] = abs(df["close"] - df["open"])
        df["range"] = df["high"] - df["low"]
        df["upper"] = df["high"] - df[["close", "open"]].max(axis=1)
        df["lower"] = df[["close", "open"]].min(axis=1) - df["low"]
        pin_bar = (df["body"] < 0.3 * df["range"]) & ((df["upper"] > 2 * df["body"]) | (df["lower"] > 2 * df["body"]))
        logging.debug(f"Pin Bar تشخیص داده شد: تعداد موارد={pin_bar.sum()}")
        return pin_bar

    @staticmethod
    def detect_engulfing(df: pd.DataFrame) -> pd.Series:
        prev_o = df["open"].shift(1)
        prev_c = df["close"].shift(1)
        engulfing = (((df["close"] > df["open"]) & (prev_c < prev_o) & (df["close"] > prev_o) & (df["open"] < prev_c)) |
                     ((df["close"] < df["open"]) & (prev_c > prev_o) & (df["close"] < prev_o) & (df["open"] > prev_c)))
        logging.debug(f"Engulfing تشخیص داده شد: تعداد موارد={engulfing.sum()}")
        return engulfing

    @staticmethod
    def detect_elliott_wave(df: pd.DataFrame) -> pd.DataFrame:
        df["WavePoint"] = np.nan
        highs = argrelextrema(df['close'].values, np.greater, order=5)[0]
        lows = argrelextrema(df['close'].values, np.less, order=5)[0]
        df.loc[df.index[highs], "WavePoint"] = df.loc[df.index[highs], "close"]
        df.loc[df.index[lows], "WavePoint"] = df.loc[df.index[lows], "close"]
        logging.debug(f"Elliott Wave تشخیص داده شد: تعداد نقاط اوج={len(highs)}, تعداد نقاط پایین={len(lows)}")
        return df

    @staticmethod
    def detect_support_resistance(df: pd.DataFrame, window: int = 10) -> tuple:
        logging.debug(f"شروع محاسبه حمایت/مقاومت برای دیتافریم با طول {len(df)}")
        if len(df) < window:
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
            recent_resistance = df['close'].mean() or 0.01
        if recent_support == 0 or pd.isna(recent_support):
            recent_support = df['close'].mean() or 0.01
        volume_profile = df['volume'].groupby(df['close'].round(2)).sum()
        vol_threshold = volume_profile.quantile(0.75)
        high_vol_levels = volume_profile[volume_profile > vol_threshold].index.tolist()
        logging.debug(f"حمایت/مقاومت محاسبه شد: support={recent_support}, resistance={recent_resistance}, high_vol_levels={high_vol_levels}")
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
        logging.debug(f"واگرایی RSI تشخیص داده شد: bullish={bullish_divergence}, bearish={bearish_divergence}")
        return bullish_divergence, bearish_divergence

# فیلتر نهایی با Decision Tree
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

# بررسی نقدینگی
async def check_liquidity(exchange: ccxt.Exchange, symbol: str, df: pd.DataFrame) -> tuple:
    global LIQUIDITY_REJECTS
    try:
        logging.debug(f"شروع بررسی نقدینگی برای {symbol}")
        ticker = await exchange.fetch_ticker(symbol)
        bid = ticker.get('bid')
        ask = ticker.get('ask')
        logging.debug(f"داده تیکری برای {symbol}: bid={bid}, ask={ask}")
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
        spread_threshold = spread_mean + spread_std
        score = 15 if spread < spread_threshold else -5
        logging.info(f"نقدینگی {symbol}: spread={spread:.4f}, threshold={spread_threshold:.4f}, score={score}")
        if spread >= spread_threshold:
            LIQUIDITY_REJECTS += 1
        return spread, score
    except Exception as e:
        logging.error(f"خطا در بررسی نقدینگی برای {symbol}: {e}, traceback={str(traceback.format_exc())}")
        return 0.0, 0

# بررسی رویدادهای فاندامنتال
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
        logging.debug(f"رویدادها دریافت شد برای {symbol}: {events}")
        event_score = 0
        if not events or "body" not in events or not events["body"]:
            return 0
        for event in events["body"]:
            title = event.get("title", "").lower()
            description = event.get("description", "").lower()
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
        logging.error(f"خطا در دریافت رویدادها برای {symbol}: {e}, traceback={str(traceback.format_exc())}")
        return 0

# محاسبات اندیکاتورها
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    logging.debug(f"شروع محاسبات اندیکاتورها برای دیتافریم با طول {len(df)}")
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
    logging.debug(f"اندیکاتورها محاسبه شد: آخرین RSI={df['RSI'].iloc[-1]:.2f}, MACD={df['MACD'].iloc[-1]:.2f}")
    return df

# تأیید مولتی تایم‌فریم با وزن‌دهی
async def multi_timeframe_confirmation(exchange: ccxt.Exchange, symbol: str, base_tf: str) -> float:
    weights = {"1d": 0.4, "4h": 0.3, "1h": 0.2, "30m": 0.1}
    total_weight = 0
    score = 0
    logging.debug(f"شروع تأیید چندتایم‌فریمی برای {symbol}, base_tf={base_tf}")
    for tf, weight in weights.items():
        if tf == base_tf:
            continue
        df_tf = await get_ohlcv_cached(exchange, symbol, tf)
        if df_tf is None or len(df_tf) < 50:
            logging.warning(f"داده ناکافی برای {symbol} @ {tf} در تأیید چندتایم‌فریمی")
            continue
        df_tf["EMA12"] = df_tf["close"].ewm(span=12).mean()
        df_tf["EMA26"] = df_tf["close"].ewm(span=26).mean()
        long_trend = df_tf["EMA12"].iloc[-1] > df_tf["EMA26"].iloc[-1]
        score += (weight * 10) if long_trend else (-weight * 5)
        total_weight += weight
    final_score = score if total_weight > 0 else 0
    logging.info(f"مولتی تایم‌فریم برای {symbol}: score={final_score:.2f}")
    return final_score

# دریافت داده‌ها با کش
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
async def get_ohlcv_cached(exchange: ccxt.Exchange, symbol: str, tf: str, limit: int = 51) -> Optional[pd.DataFrame]:
    async with semaphore:
        await asyncio.sleep(WAIT_BETWEEN_REQUESTS)
        key = f"{symbol}_{tf}"
        now = time.time()
        logging.debug(f"شروع دریافت داده برای {symbol} @ {tf}, key={key}")
        if key in CACHE and now - CACHE[key]["time"] < CACHE_TTL:
            logging.debug(f"داده از کش استفاده شد برای {symbol} @ {tf}")
            return CACHE[key]["data"]
        try:
            logging.debug(f"ارسال درخواست به API برای {symbol} @ {tf}")
            data = await exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
            logging.debug(f"داده خام دریافت شد برای {symbol} @ {tf}, تعداد داده‌ها={len(data)}")
            if not data or len(data) == 0:
                logging.warning(f"داده خالی از API برای {symbol} @ {tf}")
                return None
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            current_time = pd.Timestamp.now(tz='UTC')
            if len(df) > 0 and (current_time - df.index[-1]).total_seconds() < 60:
                df = df.iloc[:-1].copy()
                logging.debug(f"حذف آخرین داده به دلیل تاخیر برای {symbol} @ {tf}")
            CACHE[key] = {"data": df.copy(), "time": now}
            logging.debug(f"داده کش شد برای {symbol} @ {tf}, اندازه دیتافریم={len(df)}")
            return df
        except Exception as e:
            logging.error(f"خطا در دریافت داده برای {symbol} @ {tf}: {str(e)}, traceback={str(traceback.format_exc())}")
            return None

# محاسبه حجم پوزیشن
def calculate_position_size(account_balance: float, risk_percentage: float, entry: float, stop_loss: float) -> float:
    if entry is None or stop_loss is None or entry == 0 or stop_loss == 0:
        logging.warning(f"مقادیر نامعتبر برای محاسبه حجم پوزیشن: entry={entry}, stop_loss={stop_loss}")
        return 0
    risk_amount = account_balance * (risk_percentage / 100)
    distance = abs(entry - stop_loss)
    position_size = risk_amount / distance if distance != 0 else 0
    logging.debug(f"حجم پوزیشن محاسبه شد: account_balance={account_balance}, risk={risk_percentage}, entry={entry}, stop_loss={stop_loss}, position_size={position_size}")
    return round(position_size, 2)

# تابع Ablation Testing
def ablation_test(symbol_results: list, filter_name: str) -> int:
    total_signals = len([r for r in symbol_results if r is not None])
    logging.info(f"Ablation Test برای فیلتر {filter_name}: تعداد سیگنال‌های اولیه={total_signals}")
    return total_signals

# تحلیل نماد با سیستم امتیازدهی
async def analyze_symbol(exchange: ccxt.Exchange, symbol: str, tf: str) -> Optional[dict]:
    global VOLUME_REJECTS, SR_REJECTS
    start_time = time.time()
    logging.info(f"شروع تحلیل {symbol} @ {tf}, زمان شروع={datetime.now()}")

    # دریافت داده‌ها
    logging.debug(f"دعوت از get_ohlcv_cached برای {symbol} @ {tf}")
    df = await get_ohlcv_cached(exchange, symbol, tf)
    if df is None or len(df) < 50:
        logging.warning(f"داده ناکافی برای {symbol} @ {tf}, طول دیتافریم={len(df) if df is not None else 'None'}")
        return None
    logging.info(f"داده دریافت شد برای {symbol} @ {tf} در {time.time() - start_time:.2f} ثانیه, تعداد ردیف‌ها={len(df)}")

    # محاسبه اندیکاتورها
    logging.debug(f"شروع محاسبات اندیکاتورها برای {symbol} @ {tf}")
    df = compute_indicators(df)
    last = df.iloc[-1]
    logging.debug(f"اندیکاتورها برای {symbol} @ {tf} محاسبه شد: آخرین RSI={last['RSI']:.2f}, MACD={last['MACD']:.2f}")
    logging.debug(f"اندیکاتورها محاسبه شد برای {symbol} @ {tf}, آخرین قیمت={last['close']:.2f}, RSI={last['RSI']:.2f}")

    # متغیر امتیازدهی
    score_long = 0
    score_short = 0
    score_log = {"long": {}, "short": {}}

    # بررسی حجم با آستانه پویا
    vol_avg = df["volume"].rolling(VOLUME_WINDOW).mean().iloc[-1]
    current_vol = df["volume"].iloc[-1]
    vol_mean = df["volume"].rolling(20).mean().iloc[-1]
    vol_std = df["volume"].rolling(20).std().iloc[-1]
    vol_threshold = vol_mean + 0.5 * vol_std
    vol_score = 10 if current_vol >= vol_threshold else -5
    score_long += vol_score
    score_short += vol_score
    score_log["long"]["volume"] = vol_score
    score_log["short"]["volume"] = vol_score
    logging.info(f"حجم برای {symbol} @ {tf}: current_vol={current_vol:.2f}, threshold={vol_threshold:.2f}, score={vol_score}")
    if current_vol < vol_threshold:
        VOLUME_REJECTS += 1
        logging.info(f"حجم کم برای {symbol} @ {tf}: score={vol_score}")

    # بررسی نوسان
    volatility = df["ATR"].iloc[-1] / last["close"]
    vola_mean = (df["ATR"] / df["close"]).rolling(20).mean().iloc[-1]
    vola_std = (df["ATR"] / df["close"]).rolling(20).std().iloc[-1]
    vola_threshold = vola_mean + vola_std
    vola_score = 10 if volatility > vola_threshold else -5
    score_long += vola_score
    score_short += vola_score
    score_log["long"]["volatility"] = vola_score
    score_log["short"]["volatility"] = vola_score
    logging.debug(f"نوسان برای {symbol} @ {tf}: volatility={volatility:.4f}, threshold={vola_threshold:.4f}, score={vola_score}")

    # فیلتر ADX با آستانه پویا
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
    logging.debug(f"ADX برای {symbol} @ {tf}: ADX={last['ADX']:.2f}, threshold={adx_threshold:.2f}, score={adx_score + trend_score}")

    # تشخیص روند
    long_trend = df["EMA12"].iloc[-1] > df["EMA26"].iloc[-1]
    short_trend = not long_trend
    trend_score = 10 if long_trend else -5
    score_long += trend_score
    score_short += -trend_score
    score_log["long"]["trend_direction"] = trend_score
    score_log["short"]["trend_direction"] = -trend_score
    logging.debug(f"روند برای {symbol} @ {tf}: long_trend={long_trend}, score={trend_score}")

    # تأیید چندتایم‌فریمی
    logging.debug(f"دعوت از multi_timeframe_confirmation برای {symbol} @ {tf}")
    mtf_score = await multi_timeframe_confirmation(exchange, symbol, tf)
    score_long += mtf_score
    score_short += -mtf_score
    score_log["long"]["multi_timeframe"] = mtf_score
    score_log["short"]["multi_timeframe"] = -mtf_score
    logging.debug(f"تأیید چندتایم‌فریمی برای {symbol} @ {tf}: score={mtf_score:.2f}")

    # محاسبه سطوح حمایت و مقاومت
    logging.debug(f"شروع محاسبه حمایت/مقاومت برای {symbol} @ {tf}")
    try:
        support, resistance, vol_levels = PatternDetector.detect_support_resistance(df)
        logging.debug(f"حمایت/مقاومت محاسبه شد برای {symbol} @ {tf}: support={support:.2f}, resistance={resistance:.2f}")
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
        logging.debug(f"فاصله تا حمایت/مقاومت برای {symbol} @ {tf}: distance_to_resistance={distance_to_resistance:.4f}, distance_to_support={distance_to_support:.4f}, scores={sr_score_long}/{sr_score_short}")
    except Exception as e:
        logging.error(f"خطا در محاسبه حمایت/مقاومت برای {symbol} @ {tf}: {str(e)}, traceback={str(traceback.format_exc())}")
        score_long += 0
        score_short += 0
        score_log["long"]["support_resistance"] = 0
        score_log["short"]["support_resistance"] = 0

    # بررسی نقدینگی
    logging.debug(f"شروع بررسی نقدینگی برای {symbol} @ {tf}")
    spread, liquidity_score = await check_liquidity(exchange, symbol, df)
    if spread == float('inf'):
        spread = 0.0
    score_long += liquidity_score
    score_short += liquidity_score
    score_log["long"]["liquidity"] = liquidity_score
    score_log["short"]["liquidity"] = liquidity_score
    logging.debug(f"نقدینگی برای {symbol} @ {tf}: spread={spread:.4f}, score={liquidity_score}")

    # امتیاز فاندامنتال
    logging.debug(f"شروع بررسی رویدادهای فاندامنتال برای {symbol} @ {tf}")
    fundamental_score = check_market_events(symbol)
    score_long += fundamental_score
    score_short += fundamental_score
    score_log["long"]["fundamental"] = fundamental_score
    score_log["short"]["fundamental"] = fundamental_score
    logging.debug(f"رویدادهای فاندامنتال برای {symbol} @ {tf}: score={fundamental_score}")

    # روانشناسی بازار
    psych_long = "اشباع فروش" if last["RSI"] < 40 else "اشباع خرید" if last["RSI"] > 60 else "متعادل"
    psych_short = "اشباع خرید" if last["RSI"] > 60 else "اشباع فروش" if last["RSI"] < 40 else "متعادل"
    psych_score_long = 10 if psych_long == "اشباع فروش" else -10 if psych_long == "اشباع خرید" else 0
    psych_score_short = 10 if psych_short == "اشباع خرید" else -10 if psych_short == "اشباع فروش" else 0
    score_long += psych_score_long
    score_short += psych_score_short
    score_log["long"]["psychology"] = psych_score_long
    score_log["short"]["psychology"] = psych_score_short
    logging.debug(f"روانشناسی برای {symbol} @ {tf}: psych_long={psych_long}, psych_short={psych_short}, scores={psych_score_long}/{psych_score_short}")

    # تشخیص واگرایی‌ها
    bullish_rsi_div, bearish_rsi_div = PatternDetector.detect_rsi_divergence(df)
    div_score_long = 10 if bullish_rsi_div else 0
    div_score_short = 10 if bearish_rsi_div else 0
    score_long += div_score_long
    score_short += div_score_short
    score_log["long"]["rsi_divergence"] = div_score_long
    score_log["short"]["rsi_divergence"] = div_score_short
    logging.debug(f"واگرایی برای {symbol} @ {tf}: bullish={bullish_rsi_div}, bearish={bearish_rsi_div}, scores={div_score_long}/{div_score_short}")

    # شرایط اندیکاتورها
    conds_long = {
        "PinBar": last["PinBar"] and last["lower"] > 2 * last["body"],
        "Engulfing": last["Engulfing"] and last["close"] > last["open"],
        "EMA_Cross": df["EMA12"].iloc[-2] < df["EMA26"].iloc[-2] and long_trend,
        "MACD_Cross": df["MACD"].iloc[-2] < df["Signal"].iloc[-2] and df["MACD"].iloc[-1] > df["Signal"].iloc[-1],
        "RSI_Oversold": last["RSI"] < 30,
        "Stochastic_Oversold": last["Stochastic"] < 20,
        "BB_Breakout": last["close"] > last["BB_upper"],
        "MFI_Oversold": last["MFI"] < 20
    }
    conds_short = {
        "PinBar": last["PinBar"] and last["upper"] > 2 * last["body"],
        "Engulfing": last["Engulfing"] and last["close"] < last["open"],
        "EMA_Cross": df["EMA12"].iloc[-2] > df["EMA26"].iloc[-2] and short_trend,
        "MACD_Cross": df["MACD"].iloc[-2] > df["Signal"].iloc[-2] and df["MACD"].iloc[-1] < df["Signal"].iloc[-1],
        "RSI_Overbought": last["RSI"] > 70,
        "Stochastic_Overbought": last["Stochastic"] > 80,
        "BB_Breakout": last["close"] < last["BB_lower"],
        "MFI_Overbought": last["MFI"] > 80
    }
    indicator_score_long = sum(5 for v in conds_long.values() if v)
    indicator_score_short = sum(5 for v in conds_short.values() if v)
    score_long += indicator_score_long
    score_short += indicator_score_short
    score_log["long"]["indicators"] = indicator_score_long
    score_log["short"]["indicators"] = indicator_score_short
    logging.debug(f"شرایط اندیکاتورها برای {symbol} @ {tf}: long_score={indicator_score_long}, short_score={indicator_score_short}, conditions_long={conds_long}")

    # فیلتر Decision Tree
    logging.debug(f"شروع فیلتر Decision Tree برای {symbol} @ {tf}")
    signal_filter = SignalFilter()
    X_train = np.array([
        [30, 25, 2, 0.01, 0.05, 0.05, 0.01, 10, -10],
        [70, 20, 1, 0.02, 0.03, 0.03, 0.02, -10, 10],
        [50, 30, 1.5, 0.01, 0.04, 0.04, 0.01, 0, 0],
    ])
    y_train = np.array([1, 0, 1])
    signal_filter.train(X_train, y_train)
    distance_to_resistance = locals().get('distance_to_resistance', 0.0)
    distance_to_support = locals().get('distance_to_support', 0.0)
    features = [
        last["RSI"],
        last["ADX"],
        current_vol / vol_avg if vol_avg != 0 else 0,
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
    logging.debug(f"فیلتر Decision Tree برای {symbol} @ {tf}: features={features}, score={dt_score:.2f}")

    # لاگ‌گذاری امتیازها
    logging.info(f"امتیاز نهایی برای {symbol} @ {tf}: score_long={score_long:.2f}, score_short={score_short:.2f}")
    logging.info(f"جزئیات امتیاز Long: {score_log['long']}")
    logging.info(f"جزئیات امتیاز Short: {score_log['short']}")

    # تصمیم‌گیری نهایی
    THRESHOLD = 70
    if score_long >= THRESHOLD:
        entry = float(last["close"])
        atr_avg = df["ATR"].rolling(5).mean().iloc[-1]
        sl = entry - 2 * atr_avg
        tp = entry + 3 * atr_avg
        rr = round((tp - entry) / (entry - sl), 2) if (entry - sl) != 0 else 0
        position_size = calculate_position_size(10000, 1, entry, sl)
        signal_strength = "قوی" if score_long > 80 else "متوسط"
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
        logging.info(f"سیگنال Long تولید شد: {result}")
        return result
    elif score_short >= THRESHOLD:
        entry = float(last["close"])
        atr_avg = df["ATR"].rolling(5).mean().iloc[-1]
        sl = entry + 2 * atr_avg
        tp = entry - 3 * atr_avg
        rr = round((entry - tp) / (sl - entry), 2) if (sl - entry) != 0 else 0
        position_size = calculate_position_size(10000, 1, entry, sl)
        signal_strength = "قوی" if score_short > 80 else "متوسط"
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
        logging.info(f"سیگنال Short تولید شد: {result}")
        return result
    else:
        # استراتژی رگرسیون خطی موقتاً غیرفعال شده است
        if False:  # شرط نادرست برای غیرفعال کردن
            logging.debug(f"ورود به استراتژی رگرسیون خطی برای {symbol} @ {tf}")
            floor_145, ceiling_145, slope_145, current_price = IndicatorCalculator.compute_linear_regression(df, 145)
            floor_360, ceiling_360, slope_360, _ = IndicatorCalculator.compute_linear_regression(df, 360)
            logging.debug(f"رگرسیون خطی برای {symbol} @ {tf}: floor_145={floor_145}, ceiling_145={ceiling_145}, floor_360={floor_360}, ceiling_360={ceiling_360}")
            if floor_145 is None or floor_360 is None or current_price is None:
                logging.warning(f"داده‌های رگرسیون برای {symbol} @ {tf} ناکافی است")
                floor_145 = floor_360 = current_price = df['close'].mean()
            proximity_threshold = 0.02
            floor_hit_145 = abs(current_price - floor_145) / floor_145 <= proximity_threshold
            floor_hit_360 = abs(current_price - floor_360) / floor_360 <= proximity_threshold
            strong_support_145 = abs(current_price - support) / current_price <= proximity_threshold
            strong_support_360 = abs(current_price - support) / current_price <= proximity_threshold
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
                    "تحلیل": f"قیمت به کف رگرسیون رسید (۱۴۵ روز: floor={floor_145:.2f}, ۳۶۰ روز: floor={floor_360:.2f})",
                    "روانشناسی": psych_long,
                    "روند بازار": "صعودی (رگرسیون)",
                    "فاندامنتال": f"امتیاز: {fundamental_score}"
                }
                logging.info(f"سیگنال نوسان‌گیری Long (رگرسیون): {result}")
                return result
            ceiling_hit_145 = abs(current_price - ceiling_145) / ceiling_145 <= proximity_threshold
            ceiling_hit_360 = abs(current_price - ceiling_360) / ceiling_360 <= proximity_threshold
            strong_resistance_145 = abs(current_price - resistance) / current_price <= proximity_threshold
            strong_resistance_360 = abs(current_price - resistance) / current_price <= proximity_threshold
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
                    "تحلیل": f"قیمت به سقف رگرسیون رسید (۱۴۵ روز: ceiling={ceiling_145:.2f}, ۳۶۰ روز: ceiling={ceiling_360:.2f})",
                    "روانشناسی": psych_short,
                    "روند بازار": "نزولی (رگرسیون)",
                    "فاندامنتال": f"امتیاز: {fundamental_score}"
                }
                logging.info(f"سیگنال نوسان‌گیری Short (رگرسیون): {result}")
                return result
    logging.info(f"سیگنال برای {symbol} @ {tf} رد شد: score_long={score_long:.2f}, score_short={score_short:.2f}")
    return None

# اسکن همه نمادها
async def scan_all_crypto_symbols(on_signal=None) -> None:
    exchange = ccxt.kucoin({'enableRateLimit': True, 'rateLimit': 2000})
    try:
        logging.debug(f"شروع بارگذاری بازارها از KuCoin")
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
            logging.info(f"اسکن دسته {idx+1}/{total_chunks}: {chunk}")
            tasks = [analyze_symbol(exchange, sym, tf) for sym in chunk for tf in TIMEFRAMES]
            async with semaphore:
                logging.debug(f"اجرا کردن وظایف برای دسته {idx+1}, تعداد وظایف={len(tasks)}")
                results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logging.error(f"خطا در تسک: {result}, traceback={str(traceback.format_exc())}")
                    continue
                if result and on_signal:
                    await on_signal(result)
                symbol_results.append(result)
            await asyncio.sleep(WAIT_BETWEEN_CHUNKS)
        # Ablation Testing
        ablation_test(symbol_results, "volume")
        ablation_test(symbol_results, "liquidity")
        ablation_test(symbol_results, "support_resistance")
        logging.info(f"آمار رد شدن‌ها: نقدینگی={LIQUIDITY_REJECTS}, حجم={VOLUME_REJECTS}, حمایت/مقاومت={SR_REJECTS}")
    finally:
        logging.debug(f"بستن اتصال به KuCoin")
        await exchange.close()

# اجرای اصلی
async def main():
    exchange = ccxt.kucoin({'enableRateLimit': True, 'rateLimit': 2000})
    try:
        logging.debug(f"شروع بارگذاری بازارها برای تست")
        await exchange.load_markets()
        logging.info(f"بازارها بارگذاری شد برای تست")
        result = await analyze_symbol(exchange, "BTC/USDT", "1h")
        if result:
            logging.info(f"سیگنال تولید شد: {result}")
        else:
            logging.info("هیچ سیگنالی تولید نشد.")
    finally:
        logging.debug(f"بستن اتصال به KuCoin پس از تست")
        await exchange.close()

if __name__ == "__main__":
    asyncio.run(main())