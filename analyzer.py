import os
# Ensure UTF-8 encoding for console output
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
from datetime import datetime

# Logging setup with explicit UTF-8
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
    force=True,
    encoding="utf-8"
)

CMC_API_KEY = "7fc7dc4d-2d30-4c83-9836-875f9e0f74c7"
# تایم‌فریم‌های اصلی
TIMEFRAMES = ["1h", "4h", "1d", "15m", "30m", "5m"]

# فیلترهای شُل‌تر
VOLUME_WINDOW = 15        # میانگین حجم روی ۱۵ کندل
S_R_BUFFER    = 0.015     # فاصله‌ی ۱.۵٪ از حمایت/مقاومت
ADX_THRESHOLD = 30        # آستانه‌ی ADX

# پارامترهای قبلی
CACHE = {}
CACHE_TTL = 60
VOLUME_THRESHOLD = 1000
MAX_CONCURRENT_REQUESTS = 10
WAIT_BETWEEN_REQUESTS = 0.5
WAIT_BETWEEN_CHUNKS = 3

def get_top_500_symbols_from_cmc():
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    headers = {'Accepts': 'application/json', 'X-CMC_PRO_API_KEY': CMC_API_KEY}
    params = {'start': '1', 'limit': '500', 'convert': 'USD'}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        data = resp.json()
        return [entry['symbol'] for entry in data['data']]
    except Exception as e:
        logging.error(f"CMC error: {e}")
        return []

def compute_rsi(df, period=14):
    delta = df["close"].diff()
    gain  = delta.where(delta > 0,  0).rolling(period).mean()
    loss  = -delta.where(delta < 0, 0).rolling(period).mean()
    rs    = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))

def compute_atr(df, period=14):
    tr = pd.concat([
        df["high"] - df["low"],
        abs(df["high"] - df["close"].shift()),
        abs(df["low"]  - df["close"].shift())
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_bollinger_bands(df, period=20, std_dev=2):
    sma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    return sma + std_dev * std, sma - std_dev * std

def compute_adx(df, period=14):
    df["up"]   = df["high"].diff()
    df["down"] = -df["low"].diff()
    df["+DM"]  = np.where((df["up"] > df["down"]) & (df["up"] > 0), df["up"], 0.0)
    df["-DM"]  = np.where((df["down"] > df["up"]) & (df["down"] > 0), df["down"], 0.0)
    tr = pd.concat([
        df["high"] - df["low"],
        abs(df["high"] - df["close"].shift()),
        abs(df["low"]  - df["close"].shift())
    ], axis=1).max(axis=1)
    tr_smooth = tr.rolling(window=period).sum()
    plus_di  = 100 * (df["+DM"].rolling(window=period).sum() / tr_smooth)
    minus_di = 100 * (df["-DM"].rolling(window=period).sum() / tr_smooth)
    dx       = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(window=period).mean()

def detect_pin_bar(df):
    df["body"]  = abs(df["close"] - df["open"])
    df["range"] = df["high"] - df["low"]
    df["upper"] = df["high"] - df[["close","open"]].max(axis=1)
    df["lower"] = df[["close","open"]].min(axis=1) - df["low"]
    return (df["body"] < 0.3 * df["range"]) & (
        (df["upper"] > 2 * df["body"]) | (df["lower"] > 2 * df["body"])
    )

def detect_engulfing(df):
    prev_o = df["open"].shift(1)
    prev_c = df["close"].shift(1)
    return (
        ((df["close"] > df["open"]) & (prev_c < prev_o) &
         (df["close"] > prev_o)   & (df["open"] < prev_c))
        |
        ((df["close"] < df["open"]) & (prev_c > prev_o) &
         (df["close"] < prev_o)   & (df["open"] > prev_c))
    )

def detect_elliott_wave(df):
    df["WavePoint"] = np.nan
    highs = argrelextrema(df['close'].values, np.greater, order=5)[0]
    lows  = argrelextrema(df['close'].values, np.less,    order=5)[0]
    df.loc[df.index[highs], "WavePoint"] = df.loc[df.index[highs], "close"]
    df.loc[df.index[lows],  "WavePoint"] = df.loc[df.index[lows],  "close"]
    return df

def detect_support_resistance(df, window=10):
    highs      = df['high']
    lows       = df['low']
    resistance = highs[(highs.shift(1) < highs) & (highs.shift(-1) < highs)]
    support    = lows[(lows.shift(1) > lows)   & (lows.shift(-1) > lows)]
    recent_R   = resistance[-window:].max()
    recent_S   = support[-window:].min()
    return recent_S, recent_S, recent_R, recent_S

def compute_indicators(df):
    df["EMA12"]      = df["close"].ewm(span=12).mean()
    df["EMA26"]      = df["close"].ewm(span=26).mean()
    df["MACD"]       = df["EMA12"] - df["EMA26"]
    df["Signal"]     = df["MACD"].ewm(span=9).mean()
    df["RSI"]        = compute_rsi(df)
    df["ATR"]        = compute_atr(df)
    df["ADX"]        = compute_adx(df)
    df["BB_upper"], df["BB_lower"] = compute_bollinger_bands(df)
    df["PinBar"]     = detect_pin_bar(df)
    df["Engulfing"]  = detect_engulfing(df)
    df = detect_elliott_wave(df)
    return df

semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

async def get_ohlcv_cached(exchange, symbol, tf, limit=100):
    async with semaphore:
        await asyncio.sleep(WAIT_BETWEEN_REQUESTS)
        key = f"{symbol}_{tf}"
        now = time.time()
        if key in CACHE and now - CACHE[key]["time"] < CACHE_TTL:
            return CACHE[key]["data"]
        try:
            data = await exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
            df = pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            CACHE[key] = {"data": df.copy(), "time": now}
            return df
        except Exception as e:
            logging.error(f"Fetch error {symbol}-{tf}: {e}")
            return None

async def analyze_symbol(exchange, symbol, tf):
    logging.info(f"Start analyze {symbol} @ {tf}")
    df = await get_ohlcv_cached(exchange, symbol, tf)
    if df is None or len(df) < 50:
        return None

    # dynamic volume filter
    vol_avg = df["volume"].rolling(VOLUME_WINDOW).mean().iloc[-1]
    if df["volume"].iloc[-1] < max(VOLUME_THRESHOLD, vol_avg):
        return None

    df = compute_indicators(df)
    last = df.iloc[-1]

    long_trend  = df["EMA12"].iloc[-1] > df["EMA26"].iloc[-1]
    short_trend = not long_trend

    # multi-timeframe confirmation
    if tf == "1h":
        df4 = await get_ohlcv_cached(exchange, symbol, "4h")
        if df4 is not None and len(df4) >= 50:
            e12_4 = df4["close"].ewm(span=12).mean().iloc[-1]
            e26_4 = df4["close"].ewm(span=26).mean().iloc[-1]
            trend4 = e12_4 > e26_4
            if long_trend  and not trend4: return None
            if short_trend and  trend4: return None

    # support/resistance filter
    support, _, resistance, _ = detect_support_resistance(df)
     if long_trend and abs(last["close"]-resistance)/last["close"] < S_R_BUFFER: return None
    if short_trend and abs(last["close"]-support)/last["close"] < S_R_BUFFER: return None

    # candle patterns
    body           = last["body"]
    bullish_pin    = last["PinBar"]   and last["lower"] > 2*body
    bearish_pin    = last["PinBar"]   and last["upper"] > 2*body
    bullish_engulf = last["Engulfing"] and last["close"] > last["open"]
    bearish_engulf = last["Engulfing"] and last["close"] < last["open"]

    # RSI psychology
    rsi          = last["RSI"]
    psych_long   = "اشباع فروش" if rsi < 30 else "اشباع خرید" if rsi > 70 else "متعادل"
    psych_short  = "اشباع خرید" if rsi > 70 else "اشباع فروش" if rsi < 30 else "متعادل"

    conds_long = {
        "PinBar":        bullish_pin,
        "Engulfing":     bullish_engulf,
        "EMA_Cross":     df["EMA12"].iloc[-2] < df["EMA26"].iloc[-2] and long_trend,
        "MACD_Cross":    df["MACD"].iloc[-2] < df["Signal"].iloc[-2] and df["MACD"].iloc[-1] > df["Signal"].iloc[-1],
        "RSI_Oversold":  rsi < 30,
        "ADX_StrongTrend": last["ADX"] > ADX_THRESHOLD,
    }
    conds_short = {
        "PinBar":         bearish_pin,
        "Engulfing":      bearish_engulf,
        "EMA_Cross":      df["EMA12"].iloc[-2] > df["EMA26"].iloc[-2] and short_trend,
        "MACD_Cross":     df["MACD"].iloc[-2] > df["Signal"].iloc[-2] and df["MACD"].iloc[-1] < df["Signal"].iloc[-1],
        "RSI_Overbought": rsi > 70,
        "ADX_StrongTrend": last["ADX"] > ADX_THRESHOLD,
    }

    score_long  = sum(conds_long.values())
    score_short = sum(conds_short.values())

    # Long entry
    if score_long >= 3 and psych_long != "اشباع خرید" and (
        long_trend or (psych_long == "اشباع فروش" and last["ADX"] < ADX_THRESHOLD)
    ):
        entry = float(last["close"])
        sl    = entry - 1.5 * float(last["ATR"])
        tp    = entry + 2   * float(last["ATR"])
        rr    = round((tp - entry) / (entry - sl), 2)
        return {
            "نوع معامله":    "Long",
            "نماد":           symbol,
            "تایم‌فریم":      tf,
            "قیمت ورود":      entry,
            "حد ضرر":         sl,
            "هدف سود":        tp,
            "ریسک به ریوارد": rr,
            "سطح اطمینان":    min(score_long * 20, 100),
            "تحلیل":          " | ".join([k for k,v in conds_long.items() if v]),
            "روانشناسی":      psych_long,
            "روند بازار":     "صعودی",
            "فاندامنتال":     "ندارد"
        }

    # Short entry
    if score_short >= 3 and psych_short != "اشباع فروش" and (
        short_trend or (psych_short == "اشباع خرید" and last["ADX"] < ADX_THRESHOLD)
    ):
        entry = float(last["close"])
        sl    = entry + 1.5 * float(last["ATR"])
        tp    = entry - 2   * float(last["ATR"])
        rr    = round((entry - tp) / (sl - entry), 2)
        return {
            "نوع معامله":    "Short",
            "نماد":           symbol,
            "تایم‌فریم":      tf,
            "قیمت ورود":      entry,
            "حد ضرر":         sl,
            "هدف سود":        tp,
            "ریسک به ریوارد": rr,
            "سطح اطمینان":    min(score_short * 20, 100),
            "تحلیل":          " | ".join([k for k,v in conds_short.items() if v]),
            "روانشناسی":      psych_short,
            "روند بازار":     "نزولی",
            "فاندامنتال":     "ندارد"
        }

    return None

async def scan_all_crypto_symbols(on_signal=None):
    exchange = ccxt.kucoin({'enableRateLimit': True, 'rateLimit': 2000})
    await exchange.load_markets()
    top_coins = get_top_500_symbols_from_cmc()
    usdt_symbols = [
        s for s in exchange.symbols
        if any(s.startswith(f"{coin}/") and s.endswith("/USDT") for coin in top_coins)
    ]

    chunk_size = 10
    total_chunks = (len(usdt_symbols) + chunk_size - 1) // chunk_size
    for idx in range(total_chunks):
        logging.info(f"Scanning chunk {idx+1}/{total_chunks}")
        chunk = usdt_symbols[idx*chunk_size : (idx+1)*chunk_size]
        tasks = [
            asyncio.create_task(analyze_symbol(exchange, sym, tf))
            for sym in chunk for tf in TIMEFRAMES
        ]
        for task in asyncio.as_completed(tasks):
            sig = await task
            if sig:
                logging.info(f"Signal generated: {sig}")
                if on_signal:
                    await on_signal(sig)
        logging.info(f"Completed chunk {idx+1}/{total_chunks}")
        await asyncio.sleep(WAIT_BETWEEN_CHUNKS)

    await exchange.close()

# -------------------
# Backtest & Walkforward
# -------------------

def backtest_symbol(symbol, timeframe, start_date, end_date):
    import ccxt
    exchange = ccxt.kucoin({'enableRateLimit': True})
    since = exchange.parse8601(start_date)
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
    df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    results = []
    for i in range(50, len(df)-1):
        window_df = df.iloc[:i+1].copy()
        # You need a synchronous wrapper for analyze_symbol; assume analyze_sync exists
        try:
            sig = analyze_symbol_sync(window_df)  
        except:
            sig = None
        if not sig:
            continue

        entry, sl, tp = sig["قیمت ورود"], sig["حد ضرر"], sig["هدف سود"]
        future = df.iloc[i+1:]
        hit_tp = future['high'].ge(tp).idxmax() if any(future['high'] >= tp) else None
        hit_sl = future['low'].le(sl).idxmax() if any(future['low'] <= sl) else None

        if hit_tp and (not hit_sl or hit_tp <= hit_sl):
            results.append(True)
        else:
            results.append(False)

    win_rate = np.mean(results) if results else None
    logging.info(f"Backtest {symbol} {timeframe}: Win Rate = {win_rate:.2%}")
    return win_rate

def walkforward(symbol, timeframe, total_days=90, train_days=60, test_days=30):
    end = datetime.utcnow()
    start = end - timedelta(days=total_days)
    wf = []
    while start + timedelta(days=train_days+test_days) <= end:
        train_end = start + timedelta(days=train_days)
        test_end  = train_end + timedelta(days=test_days)
        wr = backtest_symbol(symbol, timeframe, train_end.isoformat(), test_end.isoformat())
        wf.append({
            "train_start": start,
            "train_end":   train_end,
            "test_end":    test_end,
            "win_rate":    wr
        })
        start += timedelta(days=test_days)

    logging.info("Walkforward Results:")
    logging.info(pd.DataFrame(wf).to_string())
    return wf

# Example usage:
# if __name__ == "__main__":
#     walkforward("BTC/USDT", "1h")
