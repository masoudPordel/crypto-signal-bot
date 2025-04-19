import requests
import pandas as pd

# ===== کلید API =====
ALPHA_VANTAGE_API_KEY = "8VL54YT3N656MW5T"

# ===== دریافت داده کوکوین =====
def get_kucoin_symbols():
    url = "https://api.kucoin.com/api/v1/symbols"
    res = requests.get(url)
    data = res.json()
    return [i["symbol"] for i in data["data"] if i["quoteCurrency"] == "USDT" and i["enableTrading"]]

def fetch_kucoin_ohlcv(symbol, interval="5min", limit=100):
    url = f"https://api.kucoin.com/api/v1/market/candles?type={interval}&symbol={symbol}&limit={limit}"
    res = requests.get(url)
    data = res.json().get("data", [])
    if not data:
        return None
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "close", "high", "low", "volume", "turnover"
    ])
    df = df[::-1]  # صعودی مرتب شود
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df

# ===== دریافت داده فارکس =====
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
    key = [k for k in data if "Time Series" in k]
    if not key:
        return None
    df = pd.DataFrame.from_dict(data[key[0]], orient="index").sort_index()
    df = df.rename(columns={
        "1. open": "open",
        "2. high": "high",
        "3. low": "low",
        "4. close": "close"
    }).astype(float)
    return df

# ===== اندیکاتورها =====
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

# ===== پرایس اکشن پیشرفته =====
def detect_advanced_price_action(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    if last["close"] > last["open"] and prev["close"] < prev["open"] and last["open"] < prev["close"]:
        return "Engulfing صعودی"
    if last["close"] < last["open"] and prev["close"] > prev["open"] and last["open"] > prev["close"]:
        return "Engulfing نزولی"
    if last["high"] > prev["high"] and last["low"] > prev["low"] and last["close"] > last["open"]:
        return "Breakout صعودی"
    if last["low"] < prev["low"] and last["high"] < prev["high"] and last["close"] < last["open"]:
        return "Breakout نزولی"
    return None

# ===== موج الیوت (فرضی) =====
def dummy_elliott_wave_check(df):
    return "الیوت شناسایی شده"

# ===== استراتژی ساده =====
def simple_signal_strategy(df):
    if df["close"].iloc[-1] > df["close"].iloc[-2]:
        return "buy"
    if df["close"].iloc[-1] < df["close"].iloc[-2]:
        return "sell"
    return None

# ===== تولید سیگنال =====
def generate_signal(symbol, df, interval="--"):
    if df is None or len(df) < 50:
        return None
    df = compute_indicators(df)
    pa = detect_advanced_price_action(df)
    elliott = dummy_elliott_wave_check(df)
    rsi = df["RSI"].iloc[-1]
    macd = df["MACD"].iloc[-1]
    signal = df["Signal"].iloc[-1]
    ema_cross = df["EMA20"].iloc[-2] < df["EMA50"].iloc[-2] and df["EMA20"].iloc[-1] > df["EMA50"].iloc[-1]
    close_price = df["close"].iloc[-1]

    score = sum([
        rsi < 35,
        macd > signal,
        ema_cross,
        pa is not None
    ])
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

# ===== اسکن کوکوین =====
def scan_all_crypto_symbols():
    TIMEFRAMES = ["5min", "15min", "1hour"]
    PRIORITY = ["BTC-USDT", "ETH-USDT", "XRP-USDT"]
    all_symbols = get_kucoin_symbols()
    symbols = PRIORITY + [s for s in all_symbols if s not in PRIORITY]
    results = []
    for symbol in symbols[:10]:  # محدود به 10 برای سرعت
        for tf in TIMEFRAMES:
            try:
                df = fetch_kucoin_ohlcv(symbol, tf)
                signal = generate_signal(symbol, df, tf)
                if signal:
                    results.append(signal)
            except Exception as e:
                print(f"خطا در {symbol} | {tf} : {e}")
    return results

# ===== اسکن فارکس =====
def scan_all_forex_symbols():
    pairs = [("EUR", "USD"), ("GBP", "USD"), ("USD", "JPY")]
    results = []
    for base, quote in pairs:
        try:
            df = fetch_forex_ohlcv(base, quote)
            signal = generate_signal(f"{base}{quote}", df, "5min")
            if signal:
                results.append(signal)
        except Exception as e:
            print(f"خطا در {base}/{quote} : {e}")
    return results

# ===== اجرا =====
if __name__ == "__main__":
    print("در حال بررسی کریپتو...")
    crypto_signals = scan_all_crypto_symbols()
    print("در حال بررسی فارکس...")
    forex_signals = scan_all_forex_symbols()

    all_signals = crypto_signals + forex_signals
    if not all_signals:
        print("هیچ سیگنالی یافت نشد.")
    else:
        for s in all_signals:
            print(s)