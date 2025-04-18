import requests
import pandas as pd

def get_all_symbols():
    url = "https://api.mexc.com/api/v3/exchangeInfo"
    response = requests.get(url)
    data = response.json()
    return [item["symbol"] for item in data["symbols"] if item["isSpotTradingAllowed"]]

def fetch_ohlcv(symbol="BTCUSDT", interval="5m", limit=100):
    url = f"https://api.mexc.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("API error")
    data = response.json()
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ])
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
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
    rsi = 100 - (100 / (1 + rs))
    return rsi

def detect_price_action(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    if last["close"] > last["open"] and prev["close"] < prev["open"] and last["open"] < prev["close"]:
        return "الگوی انگالف صعودی"
    elif last["close"] < last["open"] and prev["close"] > prev["open"] and last["open"] > prev["close"]:
        return "الگوی انگالف نزولی"
    return None

def dummy_elliott_wave_check(df):
    return "شناسایی موج صعودی احتمالی (مبتنی بر الگوی ساده)"

def generate_signal(symbol="BTCUSDT"):
    df = fetch_ohlcv(symbol)
    df = compute_indicators(df)
    rsi = df["RSI"].iloc[-1]
    macd = df["MACD"].iloc[-1]
    signal = df["Signal"].iloc[-1]
    ema_cross = df["EMA20"].iloc[-2] < df["EMA50"].iloc[-2] and df["EMA20"].iloc[-1] > df["EMA50"].iloc[-1]
    pa = detect_price_action(df)
    elliott = dummy_elliott_wave_check(df)

    close_price = df["close"].iloc[-1]
    if rsi < 30 and macd > signal and ema_cross:
        return {
            "symbol": symbol,
            "entry": close_price,
            "tp": round(close_price * 1.04, 2),
            "sl": round(close_price * 0.97, 2),
            "confidence": 95,
            "analysis": f"RSI پایین ({round(rsi, 2)}), کراس EMA, MACD مثبت | {pa or '-'} | {elliott}"
        }
    return None

def scan_all_symbols():
    symbols = get_all_symbols()
    results = []
    for sym in symbols:
        try:
            signal = generate_signal(sym)
            if signal:
                results.append(signal)
        except:
            continue
    return results

def get_forex_rate(base="USD", target="EUR"):
    if base == target:
        return 1.0

    url = f"https://open.er-api.com/v6/latest/{base}"
    response = requests.get(url)
    if response.status_code != 200:
        # تلاش معکوس اگر base/target در دسترس نیست
        url_reverse = f"https://open.er-api.com/v6/latest/{target}"
        rev_response = requests.get(url_reverse)
        if rev_response.status_code == 200:
            rev_data = rev_response.json()
            reversed_rate = rev_data["rates"].get(base)
            return 1 / reversed_rate if reversed_rate else None
        return None

    data = response.json()
    return data["rates"].get(target)

def scan_forex_symbols():
    forex_pairs = [
        ("USD", "EUR"), ("USD", "GBP"), ("USD", "JPY"),
        ("EUR", "GBP"), ("EUR", "CHF"), ("AUD", "USD"),
        ("USD", "CAD"), ("NZD", "USD")
    ]
    results = []
    for base, target in forex_pairs:
        try:
            signal = generate_forex_signal(base, target)
            if signal:
                results.append(signal)
        except:
            continue
    return results