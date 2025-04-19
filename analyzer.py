import requests
import pandas as pd

# ===== Alpha Vantage API Key =====
ALPHA_VANTAGE_API_KEY = "8VL54YT3N656MW5T"

# ======= دریافت نمادها از KuCoin =======
def get_all_symbols():
    url = "https://api.kucoin.com/api/v1/symbols"
    response = requests.get(url)
    data = response.json()
    return [s["symbol"].replace("-", "").upper() for s in data["data"] if s["enableTrading"] and s["symbol"].endswith("USDT")]

# ======= دریافت داده های کندل از KuCoin =======
def fetch_ohlcv(symbol, interval="5min", limit=100):
    symbol_formatted = symbol.replace("USDT", "-USDT")
    url = f"https://api.kucoin.com/api/v1/market/candles?type={interval}&symbol={symbol_formatted}"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    raw_data = response.json().get("data", [])
    if len(raw_data) == 0:
        return None
    raw_data.reverse()
    df = pd.DataFrame(raw_data, columns=["timestamp", "open", "close", "high", "low", "volume", "turnover"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df

# ======= فارکس =======
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

# ======= اندیکاتورها =======
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

# ======= پرایس اکشن =======
def detect_advanced_price_action(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    body = abs(last["close"] - last["open"])
    shadow = last["high"] - last["low"]

    patterns = []

    # پین بار
    if shadow > body * 3:
        patterns.append("پین‌بار")

    # دوجی
    if body / shadow < 0.1:
        patterns.append("دوجی")

    # انگالف صعودی / نزولی
    if last["close"] > last["open"] and prev["close"] < prev["open"] and last["open"] < prev["close"]:
        patterns.append("انگالف صعودی")
    elif last["close"] < last["open"] and prev["close"] > prev["open"] and last["open"] > prev["close"]:
        patterns.append("انگالف نزولی")

    return patterns if patterns else None

# ======= الیوت (فرضی) =======
def dummy_elliott_wave(df):
    recent = df["close"].iloc[-5:]
    if recent.is_monotonic_increasing:
        return "امواج صعودی (احتمال موج 3)"
    if recent.is_monotonic_decreasing:
        return "امواج نزولی (احتمال موج 5)"
    return None

# ======= استراتژی ساده =======
def simple_signal_strategy(df):
    if df is None or len(df) < 2:
        return None
    if df["close"].iloc[-1] > df["close"].iloc[-2]:
        return "buy"
    elif df["close"].iloc[-1] < df["close"].iloc[-2]:
        return "sell"
    return None

# ======= تولید سیگنال =======
def generate_signal(symbol, df, interval="--"):
    if df is None or len(df) < 50:
        return None
    df = compute_indicators(df)
    rsi = df["RSI"].iloc[-1]
    macd = df["MACD"].iloc[-1]
    signal = df["Signal"].iloc[-1]
    ema_cross = df["EMA20"].iloc[-2] < df["EMA50"].iloc[-2] and df["EMA20"].iloc[-1] > df["EMA50"].iloc[-1]
    price_actions = detect_advanced_price_action(df)
    elliott = dummy_elliott_wave(df)
    close_price = df["close"].iloc[-1]

    score = 0
    if rsi < 35: score += 1
    if macd > signal: score += 1
    if ema_cross: score += 1
    if price_actions: score += 1

    confidence = int((score / 4) * 100)

    if score >= 3 and confidence >= 75:
        return {
            "symbol": symbol,
            "entry": close_price,
            "tp": round(close_price * 1.04, 5),
            "sl": round(close_price * 0.97, 5),
            "confidence": confidence,
            "volatility": round(abs(df["close"].iloc[-1] - df["close"].iloc[-2]) / df["close"].iloc[-2] * 100, 2),
            "analysis": f"RSI: {round(rsi, 1)} | EMA کراس: {ema_cross} | MACD: {'مثبت' if macd > signal else 'منفی'} | PA: {','.join(price_actions) if price_actions else '-'} | {elliott or '-'}",
            "tf": interval
        }
    return None

# ======= اسکن کریپتو =======
def scan_all_crypto_symbols():
    TIMEFRAMES = ["5min", "15min", "1hour"]
    all_symbols = get_all_symbols()
    signals = []
    for symbol in all_symbols:
        for tf in TIMEFRAMES:
            try:
                df = fetch_ohlcv(symbol, interval=tf)
                signal = generate_signal(symbol, df, tf)
                extra_check = simple_signal_strategy(df)
                if signal and extra_check:
                    signals.append(signal)
            except Exception as e:
                print(f"خطا در {symbol} - {tf}: {e}")
    return signals

# ======= اسکن فارکس =======
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
            print(f"خطا در {base}/{quote}: {e}")
    return results

# ======= اجرای نهایی =======
if __name__ == "__main__":
    crypto_signals = scan_all_crypto_symbols()
    forex_signals = scan_all_forex_symbols()

    all_signals = crypto_signals + forex_signals
    for sig in all_signals:
        print(sig)