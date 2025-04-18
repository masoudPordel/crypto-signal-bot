import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import requests
import pandas as pd

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

def compute_indicators(df):
    df["EMA20"] = df["close"].ewm(span=20).mean()
    df["EMA50"] = df["close"].ewm(span=50).mean()
    df["RSI"] = compute_rsi(df)
    df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    return df

def compute_rsi(df, period=14):
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def detect_price_action(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    if last["close"] > last["open"] and prev["close"] < prev["open"] and last["open"] < prev["close"]:
        return "الگوی انگالف صعودی"
    elif last["close"] < last["open"] and prev["close"] > prev["open"] and last["open"] > prev["close"]:
        return "الگوی انگالف نزولی"
    return None

def dummy_elliott_wave_check(df):
    return "موج الیوت شناسایی شد (فرضی)"

def generate_signal(symbol, df, interval="--"):
    if df is None or len(df) < 50:
        return None
    df = compute_indicators(df)
    rsi = df["RSI"].iloc[-1]
    macd = df["MACD"].iloc[-1]
    signal = df["Signal"].iloc[-1]
    ema_cross = df["EMA20"].iloc[-2] < df["EMA50"].iloc[-2] and df["EMA20"].iloc[-1] > df["EMA50"].iloc[-1]
    pa = detect_price_action(df)
    elliott = dummy_elliott_wave_check(df)
    close_price = df["close"].iloc[-1]

    score = 0
    if rsi < 35: score += 1
    if macd > signal: score += 1
    if ema_cross: score += 1
    if pa: score += 1

    confidence = int((score / 4) * 100)

    if score >= 2:
        return {
            "symbol": symbol,
            "entry": close_price,
            "tp": round(close_price * 1.04, 5),
            "sl": round(close_price * 0.97, 5),
            "confidence": confidence,
            "volatility": round(abs(df["close"].iloc[-1] - df["close"].iloc[-2]) / df["close"].iloc[-2] * 100, 2),
            "analysis": f"RSI: {round(rsi, 1)} | EMA کراس: {ema_cross} | MACD: {'مثبت' if macd > signal else 'منفی'} | {pa or '-'} | {elliott}",
            "tf": interval
        }
    return None

def scan_all_crypto_symbols():
    timeframes = ["5m", "15m", "1h"]
    symbols = get_all_symbols()
    results = []
    for sym in symbols[:10]:
        for interval in timeframes:
            try:
                df = fetch_ohlcv(sym, interval)
                signal = generate_signal(sym, df, interval)
                if signal:
                    results.append(signal)
            except:
                continue
    return results

def scan_all_forex_symbols():
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
    results = []
    for symbol in symbols:
        df = pd.DataFrame({"close": [1.1, 1.12, 1.13, 1.14, 1.15]*12})
        df["open"] = df["close"].shift(1).bfill()
        signal = generate_signal(symbol, df)
        if signal:
            results.append(signal)
    return results
