import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import ccxt.async_support as ccxt
import asyncio
import time
import logging
import sqlite3
from datetime import datetime, timedelta

# --- تنظیمات ---
TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]
CACHE = {}
CACHE_TTL = 60  # ثانیه
VOLUME_THRESHOLD = 1000  # فیلتر حجم
DATABASE = "signals.db"

# --- لاگ ---
logging.basicConfig(
    level=logging.INFO,
    filename="signals.log",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- اندیکاتورها ---
def compute_rsi(df, period=14):
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))

def compute_atr(df, period=14):
    tr = pd.concat([
        df["high"] - df["low"],
        abs(df["high"] - df["close"].shift()),
        abs(df["low"] - df["close"].shift())
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_bollinger_bands(df, period=20, std_dev=2):
    sma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    return sma + std_dev * std, sma - std_dev * std

def compute_adx(df, period=14):
    df["up"] = df["high"].diff()
    df["down"] = -df["low"].diff()
    df["+DM"] = np.where((df["up"] > df["down"]) & (df["up"] > 0), df["up"], 0.0)
    df["-DM"] = np.where((df["down"] > df["up"]) & (df["down"] > 0), df["down"], 0.0)
    tr = pd.concat([
        df["high"] - df["low"],
        abs(df["high"] - df["close"].shift()),
        abs(df["low"] - df["close"].shift())
    ], axis=1).max(axis=1)
    tr_smooth = tr.rolling(window=period).sum()
    plus_di = 100 * (df["+DM"].rolling(window=period).sum() / tr_smooth)
    minus_di = 100 * (df["-DM"].rolling(window=period).sum() / tr_smooth)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(window=period).mean()

# --- الگوها و پرایس اکشن ---
def detect_pin_bar(df):
    df["body"] = abs(df["close"] - df["open"])
    df["range"] = df["high"] - df["low"]
    df["upper"] = df["high"] - df[["close", "open"]].max(axis=1)
    df["lower"] = df[["close", "open"]].min(axis=1) - df["low"]
    return (df["body"] < 0.3 * df["range"]) & ((df["upper"] > 2 * df["body"]) | (df["lower"] > 2 * df["body"]))

def detect_engulfing(df):
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    return (
        ((df["close"] > df["open"]) & (prev_close < prev_open) &
         (df["close"] > prev_open) & (df["open"] < prev_close)) |
        ((df["close"] < df["open"]) & (prev_close > prev_open) &
         (df["close"] < prev_open) & (df["open"] > prev_close))
    )

def detect_elliott_wave(df):
    df["WavePoint"] = np.nan
    highs = argrelextrema(df['close'].values, np.greater, order=5)[0]
    lows = argrelextrema(df['close'].values, np.less, order=5)[0]
    df.loc[df.index[highs], "WavePoint"] = df.loc[df.index[highs], "close"]
    df.loc[df.index[lows], "WavePoint"] = df.loc[df.index[lows], "close"]
    return df

# --- اندیکاتور نهایی ---
def compute_indicators(df):
    df["EMA12"] = df["close"].ewm(span=12).mean()
    df["EMA26"] = df["close"].ewm(span=26).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    df["RSI"] = compute_rsi(df)
    df["ATR"] = compute_atr(df)
    df["ADX"] = compute_adx(df)
    df["BB_upper"], df["BB_lower"] = compute_bollinger_bands(df)
    df["PinBar"] = detect_pin_bar(df)
    df["Engulfing"] = detect_engulfing(df)
    df = detect_elliott_wave(df)
    return df

# --- بررسی روند کلی بازار ---
async def check_market_trend(exchange):
    df = await get_ohlcv_cached(exchange, "BTC/USDT", "1h", limit=100)
    if df is None:
        return True  # اگر داده‌ای نبود، فرض می‌کنیم مشکلی نیست
    df = compute_indicators(df)
    return df["EMA12"].iloc[-1] > df["EMA26"].iloc[-1]  # روند صعودی

# --- کش داده‌ها ---
async def get_ohlcv_cached(exchange, symbol, tf, limit=100, max_retries=3):
    key = f"{symbol}_{tf}"
    now = time.time()
    if key in CACHE and now - CACHE[key]["time"] < CACHE_TTL:
        return CACHE[key]["data"]

    for attempt in range(max_retries):
        try:
            # تنظیم limit پویا بر اساس تایم‌فریم
            limit = 200 if tf in ["1h", "4h", "1d"] else 100
            data = await exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            if len(df) < 50:
                logging.warning(f"Insufficient data for {symbol}-{tf}")
                return None
            CACHE[key] = {"data": df.copy(), "time": now}
            return df
        except ccxt.RateLimitExceeded:
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
        except Exception as e:
            logging.error(f"Fetch error: {symbol}-{tf}: {e}")
            if attempt == max_retries - 1:
                return None
    return None

# --- پاک‌سازی کش ---
def clear_old_cache():
    now = time.time()
    for key in list(CACHE.keys()):
        if now - CACHE[key]["time"] > CACHE_TTL:
            del CACHE[key]

# --- تولید سیگنال ---
def generate_signal(df, index, is_market_trending):
    last = df.iloc[index]
    prev = df.iloc[index - 1]

    # شروط سیگنال با وزن‌دهی
    conds = {
        "PinBar": bool(last["PinBar"]),
        "Engulfing": bool(last["Engulfing"]),
        "EMA_Cross": prev["EMA12"] < prev["EMA26"] and last["EMA12"] > last["EMA26"],
        "MACD_Cross": prev["MACD"] < prev["Signal"] and last["MACD"] > last["Signal"],
        "RSI_Oversold": last["RSI"] < 30,
        "ADX_StrongTrend": last["ADX"] > 25,
    }
    cond_weights = {
        "PinBar": 0.5,
        "Engulfing": 0.7,
        "EMA_Cross": 1.0,
        "MACD_Cross": 1.0,
        "RSI_Oversold": 0.6,
        "ADX_StrongTrend": 0.8
    }
    score = sum(cond_weights[k] for k, v in conds.items() if v)

    if score >= 2 and is_market_trending:
        sl = last["close"] - 1.5 * last["ATR"]
        tp = last["close"] + 2 * last["ATR"]
        rr = round((tp - last["close"]) / (last["close"] - sl), 2)
        return {
            "نماد": df["symbol"].iloc[0] if "symbol" in df else "Unknown",
            "تایم_فریم": df["timeframe"].iloc[0] if "timeframe" in df else "Unknown",
            "قیمت_ورود": round(last["close"], 4),
            "هدف_سود": round(tp, 4),
            "حد_ضرر": round(sl, 4),
            "سطح_اطمینان": min(score * 20, 100),
            "تحلیل": " | ".join([k for k, v in conds.items() if v]),
            "ریسک_به_ریوارد": rr,
            "زمان": last.name.isoformat()
        }
    return None

# --- بررسی سیگنال (برای اسکن زنده) ---
async def analyze_symbol(exchange, symbol, tf):
    df = await get_ohlcv_cached(exchange, symbol, tf)
    if df is None or len(df) < 50 or df["volume"].iloc[-1] < VOLUME_THRESHOLD:
        return None

    df = compute_indicators(df)
    last = df.iloc[-1]

    # شروط سیگنال با وزن‌دهی
    conds = {
        "PinBar": bool(last["PinBar"]),
        "Engulfing": bool(last["Engulfing"]),
        "EMA_Cross": df["EMA12"].iloc[-2] < df["EMA26"].iloc[-2] and df["EMA12"].iloc[-1] > df["EMA26"].iloc[-1],
        "MACD_Cross": df["MACD"].iloc[-2] < df["Signal"].iloc[-2] and df["MACD"].iloc[-1] > df["Signal"].iloc[-1],
        "RSI_Oversold": last["RSI"] < 30,
        "ADX_StrongTrend": last["ADX"] > 25,
    }
    cond_weights = {
        "PinBar": 0.5,
        "Engulfing": 0.7,
        "EMA_Cross": 1.0,
        "MACD_Cross": 1.0,
        "RSI_Oversold": 0.6,
        "ADX_StrongTrend": 0.8
    }
    score = sum(cond_weights[k] for k, v in conds.items() if v)

    # فیلتر روند بازار
    is_market_trending = await check_market_trend(exchange)
    if score >= 2.5 and is_market_trending:
        sl = last["close"] - 1.5 * last["ATR"]
        tp = last["close"] + 2 * last["ATR"]
        rr = round((tp - last["close"]) / (last["close"] - sl), 2)
        signal = {
            "نماد": symbol,
            "تایم_فریم": tf,
            "قیمت_ورود": round(last["close"], 4),
            "هدف_سود": round(tp, 4),
            "حد_ضرر": round(sl, 4),
            "سطح_اطمینان": min(score * 20, 100),
            "تحلیل": " | ".join([k for k, v in conds.items() if v]),
            "ریسک_به_ریوارد": rr,
            "زمان": datetime.utcnow().isoformat()
        }
        save_signal_to_db(signal)
        logging.info(f"Signal generated: {symbol} - {tf} - Confidence: {signal['سطح_اطمینان']}")
        return signal
    return None

# --- ذخیره سیگنال در SQLite ---
def save_signal_to_db(signal, table="signals"):
    conn = sqlite3.connect(DATABASE)
    df = pd.DataFrame([signal])
    df.to_sql(table, conn, if_exists="append", index=False)
    conn.close()

# --- بک‌تست استراتژی ---
async def backtest_strategy(exchange, symbol, tf, limit=500):
    df = await get_ohlcv_cached(exchange, symbol, tf, limit=limit)
    if df is None or len(df) < 50:
        logging.error(f"Backtest failed: Insufficient data for {symbol}-{tf}")
        return None

    df = compute_indicators(df)
    df["symbol"] = symbol
    df["timeframe"] = tf

    # فرض می‌کنیم بازار همیشه در روند صعودی است برای بک‌تست (برای ساده‌سازی)
    is_market_trending = True  # می‌تونید این رو با check_market_trend جایگزین کنید

    trades = []
    equity = 10000  # سرمایه اولیه
    equity_curve = [equity]
    position = None

    for i in range(50, len(df) - 1):  # از 50 شروع می‌کنیم تا اندیکاتورها آماده باشن
        signal = generate_signal(df, i, is_market_trending)
        if signal and position is None:  # ورود به معامله
            position = {
                "entry_price": signal["قیمت_ورود"],
                "tp": signal["هدف_سود"],
                "sl": signal["حد_ضرر"],
                "rr": signal["ریسک_به_ریوارد"],
                "entry_time": signal["زمان"]
            }

        if position:
            next_candle = df.iloc[i + 1]
            # بررسی رسیدن به TP یا SL
            if next_candle["high"] >= position["tp"]:
                profit = (position["tp"] - position["entry_price"]) / position["entry_price"] * equity * 0.01  # 1% ریسک
                equity += profit
                trades.append({
                    "نماد": symbol,
                    "تایم_فریم": tf,
                    "زمان_ورود": position["entry_time"],
                    "قیمت_ورود": position["entry_price"],
                    "قیمت_خروج": position["tp"],
                    "سود_زیان": profit,
                    "نتیجه": "برد"
                })
                position = None
            elif next_candle["low"] <= position["sl"]:
                loss = (position["entry_price"] - position["sl"]) / position["entry_price"] * equity * 0.01
                equity -= loss
                trades.append({
                    "نماد": symbol,
                    "تایم_فریم": tf,
                    "زمان_ورود": position["entry_time"],
                    "قیمت_ورود": position["entry_price"],
                    "قیمت_خروج": position["sl"],
                    "سود_زیان": -loss,
                    "نتیجه": "باخت"
                })
                position = None
            equity_curve.append(equity)

    # محاسبه معیارهای عملکرد
    if not trades:
        return None

    trades_df = pd.DataFrame(trades)
    total_trades = len(trades)
    win_trades = len(trades_df[trades_df["نتیجه"] == "برد"])
    win_rate = win_trades / total_trades * 100 if total_trades > 0 else 0
    total_pnl = trades_df["سود_زیان"].sum()
    avg_rr = trades_df[trades_df["نتیجه"] == "برد"]["سود_زیان"].mean() / abs(
        trades_df[trades_df["نتیجه"] == "باخت"]["سود_زیان"].mean()
    ) if win_trades and total_trades - win_trades else 0
    max_drawdown = max([1 - min(equity_curve) / max(equity_curve), 0]) * 100

    result = {
        "نماد": symbol,
        "تایم_فریم": tf,
        "تعداد_معاملات": total_trades,
        "نرخ_برد": round(win_rate, 2),
        "سود_زیان_کلی": round(total_pnl, 2),
        "میانگین_ریسک_به_ریوارد": round(avg_rr, 2),
        "حداکثر_افت_سرمایه": round(max_drawdown, 2),
        "زمان_بک_تست": datetime.utcnow().isoformat()
    }

    # ذخیره نتایج بک‌تست
    save_signal_to_db(result, table="backtest_results")
    logging.info(f"Backtest completed: {symbol} - {tf} - Win Rate: {win_rate}% - PnL: {total_pnl}")
    return result

# --- اسکن کریپتو (زنده) ---
async def scan_all_crypto_symbols():
    exchange = ccxt.kucoin()
    try:
        await exchange.load_markets()
        symbols = [s for s in exchange.symbols if s.endswith("/USDT")]
        semaphore = asyncio.Semaphore(10)  # حداکثر 10 درخواست همزمان

        async def limited_analyze(symbol, tf):
            async with semaphore:
                return await analyze_symbol(exchange, symbol, tf)

        tasks = []
        for sym in symbols[:30]:  # برای تست محدود به 30 نماد
            for tf in TIMEFRAMES:
                tasks.append(limited_analyze(sym, tf))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        clear_old_cache()  # پاک‌سازی کش
        return [r for r in results if r and not isinstance(r, Exception)]
    finally:
        await exchange.close()

# --- اسکن فارکس (برای سازگاری با bot.py) ---
async def scan_all_forex_symbols():
    return []  # برای فارکس بعداً پیاده‌سازی کنید

# --- اجرای بک‌تست برای یک نماد ---
async def run_backtest(symbol="BTC/USDT", tf="1h", limit=500):
    exchange = ccxt.kucoin()
    try:
        result = await backtest_strategy(exchange, symbol, tf, limit)
        if result:
            print(f"Backtest Result: {result}")
        else:
            print(f"No trades generated for {symbol} - {tf}")
    finally:
        await exchange.close()

