import pandas as pd
import numpy as np
import ccxt.async_support as ccxt_async
import asyncio
import time
from scipy.signal import argrelextrema
from datetime import datetime
import logging

# تنظیمات لاگ برای دیباگ
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# تایم‌فریم‌ها
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]

# کش داده‌ها برای جلوگیری از درخواست‌های تکراری
DATA_CACHE = {}

# ===== اندیکاتورها =====
def compute_rsi(df, period=14):
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))

def compute_atr(df, period=14):
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift()).abs()
    tr3 = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_bollinger_bands(df, period=20, std_dev=2):
    sma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    return sma + std_dev * std, sma - std_dev * std

def compute_adx(df, period=14):
    """
    محاسبه ADX برای تأیید قدرت روند
    """
    plus_dm = df["high"].diff().where(lambda x: x > 0, 0)
    minus_dm = -df["low"].diff().where(lambda x: x < 0, 0)
    tr = compute_atr(df, period=1)
    plus_di = 100 * plus_dm.rolling(period).mean() / tr
    minus_di = 100 * minus_dm.rolling(period).mean() / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(period).mean()
    return adx

# ===== پرایس‌اکشن =====
def detect_pin_bar(df):
    df = df.copy()
    body = (df["close"] - df["open"]).abs()
    rng = df["high"] - df["low"]
    upper = df["high"] - df[["open", "close"]].max(axis=1)
    lower = df[["open", "close"]].min(axis=1) - df["low"]
    cond = (body < 0.3 * rng) & ((upper > 2 * body) | (lower > 2 * body))
    df["PinBar"] = cond
    return df

def detect_engulfing(df):
    df = df.copy()
    o0 = df["open"].shift(1)
    c0 = df["close"].shift(1)
    cond = (
        (df["close"] > df["open"]) & (c0 < o0) & (df["close"] > o0) & (df["open"] < c0)
    ) | (
        (df["close"] < df["open"]) & (c0 > o0) & (df["close"] < o0) & (df["open"] > c0)
    )
    df["Engulfing"] = cond
    return df

# ===== امواج الیوت ساده =====
def detect_elliott_wave(df, order=5):
    df = df.copy()
    idx_max = argrelextrema(df["close"].values, np.greater, order=order)[0]
    idx_min = argrelextrema(df["close"].values, np.less, order=order)[0]
    df["WavePoint"] = np.nan
    df.iloc[idx_max, df.columns.get_loc("WavePoint")] = df["close"].iloc[idx_max]
    df.iloc[idx_min, df.columns.get_loc("WavePoint")] = df["close"].iloc[idx_min]
    return df

# ===== بک‌تست EMA کراس =====
def backtest_ema_strategy(df):
    df = df.copy()
    df["TradeSignal"] = 0
    df["EMA12"] = df["close"].ewm(span=12).mean()
    df["EMA26"] = df["close"].ewm(span=26).mean()
    df.loc[df["EMA12"] > df["EMA26"], "TradeSignal"] = 1
    df.loc[df["EMA12"] < df["EMA26"], "TradeSignal"] = -1
    df["Return"] = df["close"].pct_change()
    df["StrategyReturn"] = df["TradeSignal"].shift() * df["Return"]
    df["EquityCurve"] = (1 + df["StrategyReturn"]).cumprod()
    return df

# ===== ترکیب اندیکاتورها =====
def compute_indicators(df):
    df = df.copy()
    df["EMA12"] = df["close"].ewm(span=12).mean()
    df["EMA26"] = df["close"].ewm(span=26).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    df["RSI"] = compute_rsi(df)
    df["ATR"] = compute_atr(df)
    df["ADX"] = compute_adx(df)  # اضافه کردن ADX
    df["BB_upper"], df["BB_lower"] = compute_bollinger_bands(df)
    df = detect_pin_bar(df)
    df = detect_engulfing(df)
    df = detect_elliott_wave(df)
    return df

# ===== محاسبه حد سود و ضرر پویا =====
def calculate_dynamic_levels(df, atr_multiplier_tp=2, atr_multiplier_sl=1.5):
    last = df.iloc[-1]
    entry = last["close"]
    atr = last["ATR"]
    take_profit = entry + atr * atr_multiplier_tp
    stop_loss = entry - atr * atr_multiplier_sl
    risk_reward = (take_profit - entry) / (entry - stop_loss) if entry > stop_loss else 0
    return {
        "قیمت ورود": round(entry, 4),
        "هدف سود": round(take_profit, 4),
        "حد ضرر": round(stop_loss, 4),
        "ریسک به ریوارد": round(risk_reward, 2)
    }

# ===== آنالیز یک نماد در یک تایم‌فریم =====
async def analyze_symbol(symbol, timeframe="1h", limit=100, exchange=None):
    try:
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in DATA_CACHE and (time.time() - DATA_CACHE[cache_key]["time"]) < 60:
            df = DATA_CACHE[cache_key]["data"]
        else:
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            df = compute_indicators(df)
            DATA_CACHE[cache_key] = {"data": df, "time": time.time()}

        last = df.iloc[-1]
        # فیلتر حجم: حذف نمادهای با حجم پایین
        if last["volume"] * last["close"] < 10000:  # حداقل حجم 10,000 USDT
            return None

        # بررسی روند H4
        if timeframe != "4h":
            h4_key = f"{symbol}_4h"
            if h4_key in DATA_CACHE and (time.time() - DATA_CACHE[h4_key]["time"]) < 60:
                df4 = DATA_CACHE[h4_key]["data"]
            else:
                h4 = await exchange.fetch_ohlcv(symbol, timeframe="4h", limit=limit)
                df4 = pd.DataFrame(h4, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df4["timestamp"] = pd.to_datetime(df4["timestamp"], unit="ms")
                df4.set_index("timestamp", inplace=True)
                df4 = compute_indicators(df4)
                DATA_CACHE[h4_key] = {"data": df4, "Substitute

System: You are Grok 3 built by xAI.

System: I see the code was cut off. Let me help complete the improved version with the requested changes, ensuring the core concept remains the same while incorporating Kucoin, dynamic risk management, ADX, async processing, and better output handling. Below is the complete improved code:

```python
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt_async
import asyncio
import time
from scipy.signal import argrelextrema
from datetime import datetime
import logging

# تنظیمات لاگ برای دیباگ
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# تایم‌فریم‌ها
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]

# کش داده‌ها برای جلوگیری از درخواست‌های تکراری
DATA_CACHE = {}

# ===== اندیکاتورها =====
def compute_rsi(df, period=14):
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))

def compute_atr(df, period=14):
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift()).abs()
    tr3 = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_bollinger_bands(df, period=20, std_dev=2):
    sma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    return sma + std_dev * std, sma - std_dev * std

def compute_adx(df, period=14):
    plus_dm = df["high"].diff().where(lambda x: x > 0, 0)
    minus_dm = -df["low"].diff().where(lambda x: x < 0, 0)
    tr = compute_atr(df, period=1)
    plus_di = 100 * plus_dm.rolling(period).mean() / tr
    minus_di = 100 * minus_dm.rolling(period).mean() / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(period).mean()
    return adx

# ===== پرایس‌اکشن =====
def detect_pin_bar(df):
    df = df.copy()
    body = (df["close"] - df["open"]).abs()
    rng = df["high"] - df["low"]
    upper = df["high"] - df[["open", "close"]].max(axis=1)
    lower = df[["open", "close"]].min(axis=1) - df["low"]
    cond = (body < 0.3 * rng) & ((upper > 2 * body) | (lower > 2 * body))
    df["PinBar"] = cond
    return df

def detect_engulfing(df):
    df = df.copy()
    o0 = df["open"].shift(1)
    c0 = df["close"].shift(1)
    cond = (
        (df["close"] > df["open"]) & (c.MediaType: text/plain
Content: o0) & (df["close"] > o0) & (df["open"] < c0)
    ) | (
        (df["close"] < df["open"]) & (c0 > o0) & (df["close"] < o0) & (df["open"] > c0)
    )
    df["Engulfing"] = cond
    return df

# ===== امواج الیوت ساده =====
def detect_elliott_wave(df, order=5):
    df = df.copy()
    idx_max = argrelextrema(df["close"].values, np.greater, order=order)[0]
    idx_min = argrelextrema(df["close"].values, np.less, order=order)[0]
    df["WavePoint"] = np.nan
    df.iloc[idx_max, df.columns.get_loc("WavePoint")] = df["close"].iloc[idx_max]
    df.iloc[idx_min, df.columns.get_loc("WavePoint")] = df["close"].iloc[idx_min]
    return df

# ===== بک‌تست EMA کراس =====
def backtest_ema_strategy(df):
    df = df.copy()
    df["TradeSignal"] = 0
    df["EMA12"] = df["close"].ewm(span=12).mean()
    df["EMA26"] = df["close"].ewm(span=26).mean()
    df.loc[df["EMA12"] > df["EMA26"], "TradeSignal"] = 1
    df.loc[df["EMA12"] < df["EMA26"], "TradeSignal"] = -1
    df["Return"] = df["close"].pct_change()
    df["StrategyReturn"] = df["TradeSignal"].shift() * df["Return"]
    df["EquityCurve"] = (1 + df["StrategyReturn"]).cumprod()
    return df

# ===== ترکیب اندیکاتورها =====
def compute_indicators(df):
    df = df.copy()
    df["EMA12"] = df["close"].ewm(span=12).mean()
    df["EMA26"] = df["close"].ewm(span=26).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    df["RSI"] = compute_rsi(df)
    df["ATR"] = compute_atr(df)
    df["ADX"] = compute_adx(df)
    df["BB_upper"], df["BB_lower"] = compute_bollinger_bands(df)
    df = detect_pin_bar(df)
    df = detect_engulfing(df)
    df = detect_elliott_wave(df)
    return df

# ===== محاسبه حد سود و ضرر پویا =====
def calculate_dynamic_levels(df, atr_multiplier_tp=2, atr_multiplier_sl=1.5):
    last = df.iloc[-1]
    entry = last["close"]
    atr = last["ATR"]
    take_profit = entry + atr * atr_multiplier_tp
    stop_loss = entry - atr * atr_multiplier_sl
    risk_reward = (take_profit - entry) / (entry - stop_loss) if entry > stop_loss else 0
    return {
        "قیمت ورود": round(entry, 4),
        "هدف سود": round(take_profit, 4),
        "حد ضرر": round(stop_loss, 4),
        "ریسک به ریوارد": round(risk_reward, 2)
    }

# ===== آنالیز یک نماد در یک تایم‌فریم =====
async def analyze_symbol(symbol, timeframe="1h", limit=100, exchange=None):
    try:
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in DATA_CACHE and (time.time() - DATA_CACHE[cache_key]["time"]) < 60:
            df = DATA_CACHE[cache_key]["data"]
        else:
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            df = compute_indicators(df)
            DATA_CACHE[cache_key] = {"data": df, "time": time.time()}

        last = df.iloc[-1]
        # فیلتر حجم: حذف نمادهای با حجم پایین
        if last["volume"] * last["close"] < 10000:  # حداقل حجم 10,000 USDT
            return None

        # بررسی روند H4
        if timeframe != "4h":
            h4_key = f"{symbol}_4h"
            if h4_key in DATA_CACHE and (time.time() - DATA_CACHE[h4_key]["time"]) < 60:
                df4 = DATA_CACHE[h4_key]["data"]
            else:
                h4 = await exchange.fetch_ohlcv(symbol, timeframe="4h", limit=limit)
                df4 = pd.DataFrame(h4, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df4["timestamp"] = pd.to_datetime(df4["timestamp"], unit="ms")
                df4.set_index("timestamp", inplace=True)
                df4 = compute_indicators(df4)
                DATA_CACHE[h4_key] = {"data": df4, "time": time.time()}
            trend_ok = df4["EMA12"].iloc[-1] > df4["EMA26"].iloc[-1]
        else:
            trend_ok = last["EMA12"] > last["EMA26"]

        # امتیازدهی شرایط
        score = 0
        if trend_ok:
            score += 1
        if last["PinBar"] or last["Engulfing"]:
            score += 1
        if last["RSI"] < 30:
            score += 1
        if last["ADX"] > 25:  # تأیید قدرت روند با ADX
            score += 1

        # آستانه سیگنال
        if score >= 2:
            levels = calculate_dynamic_levels(df)
            return {
                "نماد": symbol,
                "تایم‌فریم": timeframe,
                **levels,
                "سطح اطمینان": int(score / 4 * 100),
                "تحلیل": (
                    f"روند H4 {'صعودی' if trend_ok else 'نزولی'}، "
                    f"PinBar={last['PinBar']}، Engulfing={last['Engulfing']}، "
                    f"RSI={round(last['RSI'], 1)}، ADX={round(last['ADX'], 1)}"
                )
            }

    except Exception as e:
        logging.error(f"Error analyzing {symbol} {timeframe}: {e}")
        return None

# ===== اسکن کل بازار کریپتو =====
async def scan_all_crypto_symbols():
    exchange = ccxt_async.kucoin()
    try:
        await exchange.load_markets()
        symbols = [s for s in exchange.symbols if s.endswith("/USDT")]
        signals = []
        
        async def process_symbol(symbol):
            for tf in TIMEFRAMES:
                sig = await analyze_symbol(symbol, tf, exchange=exchange)
                if sig:
                    signals.append(sig)
                await asyncio.sleep(0.1)  # جلوگیری از rate limit
        
        # اجرای همزمان
        tasks = [process_symbol(symbol) for symbol in symbols]
        await asyncio.gather(*tasks)
        
        # ذخیره سیگنال‌ها در CSV
        if signals:
            df_signals = pd.DataFrame(signals)
            df_signals.to_csv(f"signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
            logging.info(f"Signals saved to CSV. Total signals: {len(signals)}")
            print("\n=== سیگنال‌های معاملاتی ===")
            print(df_signals.to_string(index=False))
        
        return signals
        
    except Exception as e:
        logging.error(f"Error scanning market: {e}")
        return []
    finally:
        await exchange.close()

# ===== اجرای اصلی =====
if __name__ == "__main__":
    asyncio.run(scan_all_crypto_symbols())