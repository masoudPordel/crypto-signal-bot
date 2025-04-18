
import requests
import pandas as pd

ALPHA_VANTAGE_API_KEY = "8VL54YT3N656MW5T"

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
    raw_data = response.json()
    if len(raw_data) == 0 or len(raw_data[0]) < 6:
        return None
    df = pd.DataFrame(raw_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "close_time", "turnover"
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
        print(f"Ø¯Ø§Ø¯ÙâØ§Û Ø¨Ø±Ø§Û {from_symbol}/{to_symbol} Ø¯Ø±ÛØ§ÙØª ÙØ´Ø¯.")
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
        return "Ø§ÙÚ¯ÙÛ Ø§ÙÚ¯Ø§ÙÙ ØµØ¹ÙØ¯Û"
    elif last["close"] < last["open"] and prev["close"] > prev["open"] and last["open"] > prev["close"]:
        return "Ø§ÙÚ¯ÙÛ Ø§ÙÚ¯Ø§ÙÙ ÙØ²ÙÙÛ"
    return None

def dummy_elliott_wave_check(df):
    return "ÙÙØ¬ Ø§ÙÛÙØª Ø´ÙØ§Ø³Ø§ÛÛ Ø´Ø¯ (ÙØ±Ø¶Û)"

def simple_signal_strategy(df):
    if df is None or len(df) < 2:
        return None
    if df["close"].iloc[-1] > df["close"].iloc[-2]:
        return "buy"
    elif df["close"].iloc[-1] < df["close"].iloc[-2]:
        return "sell"
    return None

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
            "analysis": f"RSI: {round(rsi, 1)} | EMA Ú©Ø±Ø§Ø³: {ema_cross} | MACD: {'ÙØ«Ø¨Øª' if macd > signal else 'ÙÙÙÛ'} | {pa or '-'} | {elliott}",
            "tf": interval
        }
    return None

def scan_all_crypto_symbols():
    PRIORITY_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "LTCUSDT"]
    TIMEFRAMES = ["5m", "15m", "1h", "1d", "1w"]
    all_symbols = get_all_symbols()
    symbols = PRIORITY_SYMBOLS + [s for s in all_symbols if s not in PRIORITY_SYMBOLS and s.endswith("USDT")]
    signals = []
    for symbol in symbols[:10]:
        for tf in TIMEFRAMES:
            try:
                df = fetch_ohlcv(symbol, interval=tf)
                signal = generate_signal(symbol, df, tf)
                extra_check = simple_signal_strategy(df)
                if signal and extra_check:
                    signals.append(signal)
            except Exception as e:
                print(f"Ø®Ø·Ø§ Ø¯Ø± {symbol} - {tf}: {e}")
                continue
    return signals

def scan_all_forex_symbols():
    pairs = [("EUR", "USD"), ("GBP", "USD"), ("USD", "JPY"), ("AUD", "USD"), ("USD", "CAD")]
    interval = "5min"
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
            print(f"Ø®Ø·Ø§ Ø¯Ø± {base}/{quote}: {e}")
            continue
    return results
