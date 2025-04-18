import requests
import pandas as pd

ALPHA_VANTAGE_API_KEY = "8VL54YT3N656MW5T"

# ==================== داده‌ها ====================

def get_all_symbols():
    url = "https://api.mexc.com/api/v3/exchangeInfo"
    response = requests.get(url)
    data = response.json()
    return [item["symbol"] for item in data["symbols"] if item["isSpotTradingAllowed"]]

def fetch_ohlcv(symbol, interval="5m", limit=100):
    url = f"https://api.mexc.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    df = pd.DataFrame(response.json(), columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ])
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df

def fetch_forex_ohlcv(from_symbol, to_symbol="USD", interval="5min", outputsize="compact"):
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=FX_INTRADAY"
        f"&from_symbol={from_symbol}"
        f"&to_symbol={to_symbol}"
        f"&interval={interval}"
        f"&outputsize={outputsize}"
        f"&apikey={ALPHA_VANTAGE_API_KEY}"
    )
    response = requests.get(url)
    data = response.json()
    if not any("Time Series" in k for k in data):
        return None
    ts_key = [k for k in data if "Time Series" in k][0]
    df = pd.DataFrame.from_dict(data[ts_key], orient="index").sort_index()
    df = df.rename(columns={
        "1. open": "open",
        "2. high": "high",
        "3. low": "low",
        "4. close": "close"
    }).astype(float)
    return df

# ==================== اندیکاتورها ====================

def compute_rsi(df, period=14):
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_adx(df, period=14):
    df['tr'] = df[['high', 'low', 'close']].max(axis=1) - df[['high', 'low', 'close']].min(axis=1)
    df['plus_dm'] = df['high'].diff()
    df['minus_dm'] = -df['low'].diff()
    df['plus_dm'][df['plus_dm'] < 0] = 0
    df['minus_dm'][df['minus_dm'] < 0] = 0
    tr_smooth = df['tr'].rolling(window=period).mean()
    plus_di = 100 * (df['plus_dm'].rolling(window=period).mean() / tr_smooth)
    minus_di = 100 * (df['minus_dm'].rolling(window=period).mean() / tr_smooth)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=period).mean()
    df["ADX"] = adx
    return df

def compute_atr(df, period=14):
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(window=period).mean()
    return df

def compute_indicators(df):
    df["EMA20"] = df["close"].ewm(span=20).mean()
    df["EMA50"] = df["close"].ewm(span=50).mean()
    df["RSI"] = compute_rsi(df)
    df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    df = compute_adx(df)
    df = compute_atr(df)
    return df

# ==================== پرایس‌اکشن ====================

def detect_price_action_advanced(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    body = abs(last["close"] - last["open"])
    upper_wick = last["high"] - max(last["close"], last["open"])
    lower_wick = min(last["close"], last["open"]) - last["low"]
    if last["close"] > last["open"] and prev["close"] < prev["open"] and last["open"] < prev["close"]:
        return "Bullish Engulfing"
    elif last["close"] < last["open"] and prev["close"] > prev["open"] and last["open"] > prev["close"]:
        return "Bearish Engulfing"
    elif upper_wick > body * 2 and lower_wick < body * 0.5:
        return "Shooting Star"
    elif lower_wick > body * 2 and upper_wick < body * 0.5:
        return "Hammer"
    elif last["high"] < prev["high"] and last["low"] > prev["low"]:
        return "Inside Bar"
    return None

# ==================== سیگنال‌دهی ====================

def generate_signal(symbol, df, interval="--"):
    if df is None or len(df) < 50:
        return None
    df = compute_indicators(df)

    rsi = df["RSI"].iloc[-1]
    macd = df["MACD"].iloc[-1]
    signal = df["Signal"].iloc[-1]
    ema_cross = df["EMA20"].iloc[-2] < df["EMA50"].iloc[-2] and df["EMA20"].iloc[-1] > df["EMA50"].iloc[-1]
    price_action = detect_price_action_advanced(df)
    adx = df["ADX"].iloc[-1]
    atr = df["ATR"].iloc[-1]
    close_price = df["close"].iloc[-1]

    score = 0
    if rsi < 35 or rsi > 70:
        score += 1
    if macd > signal:
        score += 1
    if ema_cross:
        score += 1
    if price_action:
        score += 1
    if adx > 20:
        score += 1

    confidence = int((score / 5) * 100)

    if score >= 3:
        return {
            "symbol": symbol,
            "entry": round(close_price, 5),
            "tp": round(close_price + atr * 2, 5),
            "sl": round(close_price - atr * 1.5, 5),
            "confidence": confidence,
            "volatility": round(atr / close_price * 100, 2),
            "price_action": price_action or "-",
            "trend": "Bullish" if ema_cross else "Ranging",
            "analysis": f"RSI: {round(rsi,1)} | ADX: {round(adx,1)} | EMA کراس: {ema_cross} | {price_action or '-'}",
            "tf": interval
        }
    return None

# ==================== اسکن بازار ====================

def scan_all_crypto_symbols():
    PRIORITY_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "LTCUSDT"]
    TIMEFRAMES = ["15m", "1h", "4h", "1d"]
    all_symbols = get_all_symbols()
    symbols = PRIORITY_SYMBOLS + [s for s in all_symbols if s not in PRIORITY_SYMBOLS and s.endswith("USDT")]
    results = []
    for symbol in symbols[:10]:  # محدودیت تستی
        for tf in TIMEFRAMES:
            try:
                df = fetch_ohlcv(symbol, interval=tf)
                signal = generate_signal(symbol, df, tf)
                if signal:
                    results.append(signal)
            except Exception as e:
                print(f"خطا در {symbol} - {tf}: {e}")
    return results

def scan_all_forex_symbols():
    pairs = [("EUR", "USD"), ("GBP", "USD"), ("USD", "JPY"), ("AUD", "USD"), ("USD", "CAD")]
    interval = "15min"
    results = []
    for base, quote in pairs:
        try:
            df = fetch_forex_ohlcv(base, quote, interval)
            symbol = base + quote
            if df is not None:
                signal = generate_signal(symbol, df, interval)
                if signal:
                    results.append(signal)
        except Exception as e:
            print(f"خطا در {base}/{quote}: {e}")
    return results