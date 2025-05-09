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
from datetime import datetime, timedelta

# Logging setup with explicit UTF-8
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
    force=True,
    encoding="utf-8"
)

CMC_API_KEY = "7fc7dc4d-2d30-4c83-9836-875f9e0f74c7"
COINGECKO_API_KEY = "CG-cnXmskNzo7Bi2Lzj3j3QY6Gu"    
TIMEFRAMES = ["1h", "4h", "1d", "15m", "30m", "5m"]

# پارامترها
VOLUME_WINDOW = 15
S_R_BUFFER = 0.015
ADX_THRESHOLD = 30
CACHE = {}
CACHE_TTL = 60
VOLUME_THRESHOLD = 1000
MAX_CONCURRENT_REQUESTS = 10
WAIT_BETWEEN_REQUESTS = 0.5
WAIT_BETWEEN_CHUNKS = 3
VOLATILITY_THRESHOLD = 0.01  # حداقل نوسان مورد نیاز (1%)
LIQUIDITY_SPREAD_THRESHOLD = 0.001  # حداکثر اسپرد (0.1%)

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

def compute_stochastic(df, period=14):
    low_min = df["low"].rolling(window=period).min()
    high_max = df["high"].rolling(window=period).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, 1e-10)
    return k

def detect_pin_bar(df):
    df["body"] = abs(df["close"] - df["open"])
    df["range"] = df["high"] - df["low"]
    df["upper"] = df["high"] - df[["close", "open"]].max(axis=1)
    df["lower"] = df[["close", "open"]].min(axis=1) - df["low"]
    return (df["body"] < 0.3 * df["range"]) & (
        (df["upper"] > 2 * df["body"]) | (df["lower"] > 2 * df["body"])
    )

def detect_engulfing(df):
    prev_o = df["open"].shift(1)
    prev_c = df["close"].shift(1)
    return (
        ((df["close"] > df["open"]) & (prev_c < prev_o) &
         (df["close"] > prev_o) & (df["open"] < prev_c))
        |
        ((df["close"] < df["open"]) & (prev_c > prev_o) &
         (df["close"] < prev_o) & (df["open"] > prev_c))
    )

def detect_elliott_wave(df):
    df["WavePoint"] = np.nan
    highs = argrelextrema(df['close'].values, np.greater, order=5)[0]
    lows = argrelextrema(df['close'].values, np.less, order=5)[0]
    df.loc[df.index[highs], "WavePoint"] = df.loc[df.index[highs], "close"]
    df.loc[df.index[lows], "WavePoint"] = df.loc[df.index[lows], "close"]
    return df

def detect_support_resistance(df, window=10):
    # Pivot Points
    high = df['high'].rolling(window).max()
    low = df['low'].rolling(window).min()
    close = df['close'].rolling(window).mean()
    pivot = (high + low + close) / 3
    resistance = pivot + (high - low) * 0.382
    support = pivot - (high - low) * 0.382

    # Volume Profile
    volume_profile = df['volume'].groupby(df['close'].round(2)).sum()
    vol_threshold = volume_profile.quantile(0.75)
    high_vol_levels = volume_profile[volume_profile > vol_threshold].index

    # سطوح اخیر
    recent_highs = df['high'][(df['high'].shift(1) < df['high']) & (df['high'].shift(-1) < df['high'])].iloc[-window:]
    recent_lows = df['low'][(df['low'].shift(1) > df['low']) & (df['low'].shift(-1) > df['low'])].iloc[-window:]
    recent_resistance = recent_highs.max() if not recent_highs.empty else resistance
    recent_support = recent_lows.min() if not recent_lows.empty else support

    if 'support_levels' not in globals():
        globals()['support_levels'] = []
        globals()['resistance_levels'] = []
    if recent_support not in support_levels:
        support_levels.append(recent_support)
    if recent_resistance not in resistance_levels:
        resistance_levels.append(recent_resistance)

    return recent_support, recent_resistance, high_vol_levels

def detect_hammer(df):
    body = abs(df['close'] - df['open'])
    range_ = df['high'] - df['low']
    lower_wick = df['low'] - df[['close', 'open']].min(axis=1)
    return (body < 0.3 * range_) & (lower_wick > 2 * body) & (df['close'] > df['open'])

def detect_rsi_divergence(df, lookback=5):
    rsi = compute_rsi(df)
    prices = df['close']
    recent_lows_price = argrelextrema(prices.values, np.less, order=lookback)[0]
    recent_lows_rsi = argrelextrema(rsi.values, np.less, order=lookback)[0]
    if len(recent_lows_price) > 1 and len(recent_lows_rsi) > 1:
        last_price_low = prices.iloc[recent_lows_price[-1]]
        prev_price_low = prices.iloc[recent_lows_price[-2]]
        last_rsi_low = rsi.iloc[recent_lows_rsi[-1]]
        prev_rsi_low = rsi.iloc[recent_lows_rsi[-2]]
        return last_price_low < prev_price_low and last_rsi_low > prev_rsi_low
    return False

def is_support_broken(df, support, lookback=2):
    recent_closes = df['close'].iloc[-lookback:]
    return all(recent_closes < support)

def is_valid_breakout(df, support, vol_threshold=1.5):
    last_vol = df['volume'].iloc[-1]
    vol_avg = df['volume'].rolling(VOLUME_WINDOW).mean().iloc[-1]
    return last_vol > vol_threshold * vol_avg and is_support_broken(df, support)

def check_liquidity(exchange, symbol):
    try:
        ticker = exchange.fetch_ticker(symbol)
        bid = ticker['bid']
        ask = ticker['ask']
        spread = (ask - bid) / ((bid + ask) / 2)
        return spread < LIQUIDITY_SPREAD_THRESHOLD
    except Exception as e:
        logging.error(f"Liquidity check error for {symbol}: {e}")
        return False

def check_market_events(symbol):
    url = f"https://api.coingecko.com/api/v3/coins/{symbol.lower()}?x_cg_api_key={COINGECKO_API_KEY}"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if 'market_data' in data and 'last_updated' in data:
            return "ندارد"  # اگه API پیشرفته‌تری بخوای، می‌تونی اخبار رو پارس کنی
        return "رویداد مهم"
    except Exception as e:
        logging.error(f"Coingecko error: {e}")
        return "ندارد"

def compute_indicators(df):
    df["EMA12"] = df["close"].ewm(span=12).mean()
    df["EMA26"] = df["close"].ewm(span=26).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    df["RSI"] = compute_rsi(df)
    df["ATR"] = compute_atr(df)
    df["ADX"] = compute_adx(df)
    df["Stochastic"] = compute_stochastic(df)
    df["BB_upper"], df["BB_lower"] = compute_bollinger_bands(df)
    df["PinBar"] = detect_pin_bar(df)
    df["Engulfing"] = detect_engulfing(df)
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
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
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
        logging.warning(f"Reject {symbol} @ {tf}: No data or insufficient data length (<50)")
        return None

    # فیلتر حجم
    vol_avg = df["volume"].rolling(VOLUME_WINDOW).mean().iloc[-1]
    if df["volume"].iloc[-1] < max(VOLUME_THRESHOLD, vol_avg):
        logging.warning(f"Reject {symbol} @ {tf}: Volume too low (current={df['volume'].iloc[-1]}, threshold={max(VOLUME_THRESHOLD, vol_avg)})")
        return None

    df = compute_indicators(df)
    last = df.iloc[-1]

    long_trend = df["EMA12"].iloc[-1] > df["EMA26"].iloc[-1]
    short_trend = not long_trend

    # تأیید چندتایم‌فریمی
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
                logging.warning(f"Reject {symbol} @ {tf}: Multi-timeframe trend mismatch (long trend not confirmed)")
                return None
            if short_trend and (trend4 or trend1d):
                logging.warning(f"Reject {symbol} @ {tf}: Multi-timeframe trend mismatch (short trend not confirmed)")
                return None
        else:
            logging.warning(f"Reject {symbol} @ {tf}: Insufficient multi-timeframe data")
            return None

    # فیلتر نوسانات
    volatility = df["ATR"].iloc[-1] / df["close"].iloc[-1]
    if volatility < VOLATILITY_THRESHOLD:
        logging.warning(f"Reject {symbol} @ {tf}: Volatility too low (current={volatility:.4f}, threshold={VOLATILITY_THRESHOLD})")
        return None

    # حمایت و مقاومت
    support, resistance, vol_levels = detect_support_resistance(df)
    if long_trend and abs(last["close"] - resistance) / last["close"] < S_R_BUFFER:
        logging.warning(f"Reject {symbol} @ {tf}: Too close to resistance (distance={abs(last['close'] - resistance)/last['close']:.4f})")
        return None
    if short_trend and abs(last["close"] - support) / last["close"] < S_R_BUFFER:
        logging.warning(f"Reject {symbol} @ {tf}: Too close to support (distance={abs(last['close'] - support)/last['close']:.4f})")
        return None

    # نقدینگی
    if not check_liquidity(exchange, symbol):
        logging.warning(f"Reject {symbol} @ {tf}: Insufficient liquidity (spread too high)")
        return None

    # رویدادهای بازار
    fundamental = check_market_events(symbol.split('/')[0])
    if fundamental == "رویداد مهم":
        logging.warning(f"Reject {symbol} @ {tf}: Significant market event detected")
        return None

    # الگوهای کندلی
    body = last["body"]
    bullish_pin = last["PinBar"] and last["lower"] > 2 * body
    bearish_pin = last["PinBar"] and last["upper"] > 2 * body
    bullish_engulf = last["Engulfing"] and last["close"] > last["open"]
    bearish_engulf = last["Engulfing"] and last["close"] < last["open"]

    # روان‌شناسی
    rsi = last["RSI"]
    stochastic = last["Stochastic"]
    psych_long = "اشباع فروش" if rsi < 30 and stochastic < 20 else "اشباع خرید" if rsi > 70 and stochastic > 80 else "متعادل"
    psych_short = "اشباع خرید" if rsi > 70 and stochastic > 80 else "اشباع فروش" if rsi < 30 and stochastic < 20 else "متعادل"

    conds_long = {
        "PinBar": bullish_pin,
        "Engulfing": bullish_engulf,
        "EMA_Cross": df["EMA12"].iloc[-2] < df["EMA26"].iloc[-2] and long_trend,
        "MACD_Cross": df["MACD"].iloc[-2] < df["Signal"].iloc[-2] and df["MACD"].iloc[-1] > df["Signal"].iloc[-1],
        "RSI_Oversold": rsi < 30,
        "Stochastic_Oversold": stochastic < 20,
        "ADX_StrongTrend": last["ADX"] > ADX_THRESHOLD,
    }
    conds_short = {
        "PinBar": bearish_pin,
        "Engulfing": bearish_engulf,
        "EMA_Cross": df["EMA12"].iloc[-2] > df["EMA26"].iloc[-2] and short_trend,
        "MACD_Cross": df["MACD"].iloc[-2] > df["Signal"].iloc[-2] and df["MACD"].iloc[-1] < df["Signal"].iloc[-1],
        "RSI_Overbought": rsi > 70,
        "Stochastic_Overbought": stochastic > 80,
        "ADX_StrongTrend": last["ADX"] > ADX_THRESHOLD,
    }

    score_long = sum(conds_long.values())
    score_short = sum(conds_short.values())

    # Long entry
    if score_long >= 4 and psych_long != "اشباع خرید" and (
        long_trend or (psych_long == "اشباع فروش" and last["ADX"] < ADX_THRESHOLD)
    ):
        entry = float(last["close"])
        atr_avg = df["ATR"].rolling(5).mean().iloc[-1]
        sl = entry - 2 * atr_avg
        tp = entry + 3 * atr_avg
        rr = round((tp - entry) / (entry - sl), 2)
        return {
            "نوع معامله": "Long",
            "نماد": symbol,
            "تایم‌فریم": tf,
            "قیمت ورود": entry,
            "حد ضرر": sl,
            "هدف سود": tp,
            "ریسک به ریوارد": rr,
            "سطح اطمینان": min(score_long * 20, 100),
            "تحلیل": " | ".join([k for k, v in conds_long.items() if v]),
            "روانشناسی": psych_long,
            "روند بازار": "صعودی",
            "فاندامنتال": fundamental
        }

    # Short entry
    if score_short >= 4 and psych_short != "اشباع فروش" and (
        short_trend or (psych_short == "اشباع خرید" and last["ADX"] < ADX_THRESHOLD)
    ):
        if is_valid_breakout(df, support) and not detect_rsi_divergence(df) and not (
            detect_hammer(df) or (last["Engulfing"] and last["close"] > last["open"])
        ):
            entry = float(last["close"])
            atr_avg = df["ATR"].rolling(5).mean().iloc[-1]
            sl = entry + 2 * atr_avg
            tp = entry - 3 * atr_avg
            rr = round((entry - tp) / (sl - entry), 2)
            return {
                "نوع معامله": "Short",
                "نماد": symbol,
                "تایم‌فریم": tf,
                "قیمت ورود": entry,
                "حد ضرر": sl,
                "هدف سود": tp,
                "ریسک به ریوارد": rr,
                "سطح اطمینان": min(score_short * 20, 100),
                "تحلیل": " | ".join([k for k, v in conds_short.items() if v]),
                "روانشناسی": psych_short,
                "روند بازار": "نزولی",
                "فاندامنتال": fundamental
            }
        else:
            logging.warning(f"Reject {symbol} @ {tf}: Invalid breakout or bullish pattern detected")
            return None

    logging.warning(f"Reject {symbol} @ {tf}: Insufficient score or psychological condition")
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
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    results = []
    for i in range(50, len(df)-1):
        window_df = df.iloc[:i+1].copy()
        try:
            loop = asyncio.get_event_loop()
            sig = loop.run_until_complete(analyze_symbol(exchange, symbol, timeframe, window_df))
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
        test_end = train_end + timedelta(days=test_days)
        wr = backtest_symbol(symbol, timeframe, train_end.isoformat(), test_end.isoformat())
        wf.append({
            "train_start": start,
            "train_end": train_end,
            "test_end": test_end,
            "win_rate": wr
        })
        start += timedelta(days=test_days)

    logging.info("Walkforward Results:")
    logging.info(pd.DataFrame(wf).to_string())
    return wf

# Example usage:
# if __name__ == "__main__":
#     asyncio.run(scan_all_crypto_symbols())
#     # یا
#     backtest_symbol("BTC/USDT", "1h", "2024-01-01T00:00:00Z", "2024-02-01T00:00:00Z")