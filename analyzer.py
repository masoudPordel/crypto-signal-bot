import requests
import pandas as pd

# === کلید API برای فارکس (Alpha Vantage) ===
ALPHA_VANTAGE_API_KEY = "8VL54YT3N656MW5T"

# ---------- دریافت نمادهای کریپتو از KuCoin ----------
def get_all_symbols_kucoin():
    url = "https://api.kucoin.com/api/v1/symbols"
    response = requests.get(url)
    data = response.json()
    return [item["symbol"].replace("-", "") for item in data["data"] if item["symbol"].endswith("-USDT")]

# ---------- دریافت داده‌های OHLCV برای کریپتو ----------
def fetch_ohlcv_kucoin(symbol, interval="5min", limit=100):
    url = f"https://api.kucoin.com/api/v1/market/candles?type={interval}&symbol={symbol[:len(symbol)-4]}-USDT"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    raw_data = response.json()["data"]
    if not raw_data:
        return None
    df = pd.DataFrame(raw_data, columns=[
        "timestamp", "open", "close", "high", "low", "volume", "turnover"
    ])
    df = df.iloc[::-1]  # ترتیب زمانی
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    return df

# ---------- دریافت داده‌های فارکس ----------
def fetch_forex_ohlcv(from_symbol, to_symbol="USD", interval="5min"):
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=FX_INTRADAY"
        f"&from_symbol={from_symbol}"
        f"&to_symbol={to_symbol}"
        f"&interval={interval}"
        f"&outputsize=compact"
        f"&apikey={ALPHA_VANTAGE_API_KEY}"
    )
    response = requests.get(url)
    data = response.json()
    ts_key = [k for k in data if "Time Series" in k]
    if not ts_key:
        return None
    df = pd.DataFrame.from_dict(data[ts_key[0]], orient="index").sort_index()
    df = df.rename(columns={
        "1. open": "open",
        "2. high": "high",
        "3. low": "low",
        "4. close": "close"
    }).astype(float)
    return df

# ---------- اندیکاتورها ----------
def compute_indicators(df):
    df["EMA20"] = df["close"].ewm(span=20).mean()
    df["EMA50"] = df["close"].ewm(span=50).mean()
    df["RSI"] = compute_rsi(df)
    df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    return df

def compute_rsi(df, period=14):
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ---------- پرایس اکشن پیشرفته ----------
def detect_advanced_price_action(df):
    last = df.iloc[-1]
    body = abs(last["close"] - last["open"])
    wick = last["high"] - last["low"]
    upper_shadow = last["high"] - max(last["close"], last["open"])
    lower_shadow = min(last["close"], last["open"]) - last["low"]

    if body < wick * 0.2:
        return "دوجی"
    elif lower_shadow > body * 2 and upper_shadow < body:
        return "پین‌بار صعودی"
    elif upper_shadow > body * 2 and lower_shadow < body:
        return "پین‌بار نزولی"
    return None

def dummy_elliott_wave_check(df):
    return "موج الیوت شناسایی شد (فرضی)"

def detect_engulfing(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    if last["close"] > last["open"] and prev["close"] < prev["open"] and last["open"] < prev["close"] and last["close"] > prev["open"]:
        return "انگالف صعودی"
    elif last["close"] < last["open"] and prev["close"] > prev["open"] and last["open"] > prev["close"] and last["close"] < prev["open"]:
        return "انگالف نزولی"
    return None

def simple_signal_strategy(df):
    if df["close"].iloc[-1] > df["close"].iloc[-2]:
        return "buy"
    elif df["close"].iloc[-1] < df["close"].iloc[-2]:
        return "sell"
    return None

# ---------- تولید سیگنال ----------
def generate_signal(symbol, df, interval="5min"):
    if df is None or len(df) < 50:
        return None
    df = compute_indicators(df)

    rsi = df["RSI"].iloc[-1]
    macd = df["MACD"].iloc[-1]
    signal = df["Signal"].iloc[-1]
    ema_cross = df["EMA20"].iloc[-2] < df["EMA50"].iloc[-2] and df["EMA20"].iloc[-1] > df["EMA50"].iloc[-1]

    price_action = detect_engulfing(df)
    advanced_pa = detect_advanced_price_action(df)
    elliott = dummy_elliott_wave_check(df)

    score = 0
    if rsi < 35: score += 1
    if macd > signal: score += 1
    if ema_cross: score += 1
    if price_action or advanced_pa: score += 1

    confidence = int((score / 4) * 100)
    if confidence < 75:
        return None

    close_price = df["close"].iloc[-1]

    return {
        "symbol": symbol,
        "entry": close_price,
        "tp": round(close_price * 1.04, 5),
        "sl": round(close_price * 0.97, 5),
        "confidence": confidence,
        "analysis": f"RSI: {round(rsi,1)} | EMA کراس: {ema_cross} | MACD: {'مثبت' if macd > signal else 'منفی'} | {price_action or advanced_pa or '-'} | {elliott}",
        "tf": interval
    }

# ---------- اسکن کریپتو ----------
def scan_all_crypto_symbols():
    TIMEFRAMES = ["5min", "15min", "1hour"]
    all_symbols = get_all_symbols_kucoin()
    signals = []
    for symbol in all_symbols:
        for tf in TIMEFRAMES:
            try:
                df = fetch_ohlcv_kucoin(symbol, interval=tf)
                signal = generate_signal(symbol, df, tf)
                if signal:
                    signals.append(signal)
            except Exception as e:
                print(f"خطا در {symbol}-{tf}: {e}")
    return signals

# ---------- اسکن فارکس با نمادهای کامل ----------
def scan_all_forex_symbols():
    pairs = [
        ("EUR", "USD"), ("GBP", "USD"), ("USD", "JPY"), ("USD", "CAD"), ("USD", "CHF"),
        ("NZD", "USD"), ("AUD", "USD"), ("AUD", "NZD"), ("AUD", "CAD"), ("AUD", "CHF"), ("AUD", "JPY"),
        ("CAD", "CHF"), ("CAD", "JPY"), ("CHF", "JPY"), ("EUR", "AUD"), ("EUR", "CAD"),
        ("EUR", "CHF"), ("EUR", "GBP"), ("EUR", "JPY"), ("EUR", "NZD"), ("GBP", "CHF"),
        ("GBP", "JPY"), ("NZD", "JPY"), ("XAG", "USD"), ("XAU", "USD")
    ]
    results = []
    for base, quote in pairs:
        try:
            df = fetch_forex_ohlcv(base, quote)
            symbol = base + quote
            signal = generate_signal(symbol, df)
            if signal:
                results.append(signal)
        except Exception as e:
            print(f"خطا در {base}{quote}: {e}")
    return results

# ---------- اجرای کامل ----------
if __name__ == "__main__":
    crypto_signals = scan_all_crypto_symbols()
    forex_signals = scan_all_forex_symbols()
    all_signals = crypto_signals + forex_signals
    for s in all_signals:
        print(s)