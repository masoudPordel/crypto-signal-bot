import requests
import pandas as pd

# === کلید API برای Alpha Vantage ===
ALPHA_VANTAGE_API_KEY = "8VL54YT3N656MW5T"

# ---------- دریافت نمادهای کریپتو از کوکوین ----------
def get_all_kucoin_symbols():
    url = "https://api.kucoin.com/api/v1/symbols"
    response = requests.get(url)
    data = response.json()
    return [item["symbol"].replace("-", "") for item in data["data"] if item["quoteCurrency"] == "USDT"]

# ---------- دریافت داده OHLCV از کوکوین ----------
def fetch_ohlcv(symbol, interval="5m", limit=100):
    symbol = symbol.upper().replace("USDT", "-USDT")
    url = f"https://api.kucoin.com/api/v1/market/candles?type={interval}&symbol={symbol}"
    response = requests.get(url)
    raw_data = response.json().get("data", [])[::-1][:limit]
    if not raw_data or len(raw_data[0]) < 7:
        return None
    df = pd.DataFrame(raw_data, columns=[
        "timestamp", "open", "close", "high", "low", "volume", "turnover"
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df

# ---------- دریافت داده فارکس ----------
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
def detect_price_action(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    if last["close"] > last["open"] and prev["close"] < prev["open"] and last["open"] < prev["close"]:
        return "انگالف صعودی"
    elif last["close"] < last["open"] and prev["close"] > prev["open"] and last["open"] > prev["close"]:
        return "انگالف نزولی"
    return None

# ---------- موج الیوت فرضی ----------
def dummy_elliott_wave_check(df):
    return "موج الیوت (فرضی)"

# ---------- استراتژی ساده ----------
def simple_signal_strategy(df):
    if df is None or len(df) < 2:
        return None
    if df["close"].iloc[-1] > df["close"].iloc[-2]:
        return "buy"
    elif df["close"].iloc[-1] < df["close"].iloc[-2]:
        return "sell"
    return None

# ---------- تولید سیگنال با منطق دقیق ----------
def generate_signal(symbol, df, interval="--"):
    if df is None or len(df) < 50:
        return None
    df = compute_indicators(df)

    rsi = df["RSI"].iloc[-1]
    macd = df["MACD"].iloc[-1]
    signal_line = df["Signal"].iloc[-1]
    ema_cross = df["EMA20"].iloc[-2] < df["EMA50"].iloc[-2] and df["EMA20"].iloc[-1] > df["EMA50"].iloc[-1]
    pa = detect_price_action(df)
    elliott = dummy_elliott_wave_check(df)
    close_price = df["close"].iloc[-1]

    # شروط ترکیبی دقیق
    conditions_met = {
        "RSI": rsi < 35,
        "EMA_Cross": ema_cross,
        "MACD_Positive": macd > signal_line,
        "Price_Action": bool(pa),
        "Elliott": bool(elliott)
    }

    # اگر سه شرط کلیدی برقرار باشند
    if conditions_met["RSI"] and conditions_met["EMA_Cross"] and conditions_met["Price_Action"]:
        return {
            "symbol": symbol,
            "entry": close_price,
            "tp": round(close_price * 1.04, 5),
            "sl": round(close_price * 0.97, 5),
            "confidence": int((sum(conditions_met.values()) / len(conditions_met)) * 100),
            "volatility": round(abs(df["close"].iloc[-1] - df["close"].iloc[-2]) / df["close"].iloc[-2] * 100, 2),
            "analysis": f"RSI: {round(rsi, 1)} | EMA کراس: {ema_cross} | MACD: {'مثبت' if macd > signal_line else 'منفی'} | {pa or '-'} | {elliott}",
            "tf": interval
        }
    return None

# ---------- اسکن کریپتو ----------
def scan_all_crypto_symbols():
    PRIORITY_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
    TIMEFRAMES = ["5min", "15min", "1hour"]
    all_symbols = get_all_kucoin_symbols()
    symbols = PRIORITY_SYMBOLS + [s for s in all_symbols if s not in PRIORITY_SYMBOLS]
    signals = []
    for symbol in symbols[:10]:
        for tf in TIMEFRAMES:
            try:
                df = fetch_ohlcv(symbol, tf)
                signal = generate_signal(symbol, df, tf)
                extra_check = simple_signal_strategy(df)
                if signal and extra_check:
                    signals.append(signal)
            except Exception as e:
                print(f"خطا در {symbol} - {tf}: {e}")
                continue
    return signals

# ---------- اسکن فارکس ----------
def scan_all_forex_symbols():
    pairs = [("EUR", "USD"), ("GBP", "USD"), ("USD", "JPY"), ("AUD", "USD")]
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
            continue
    return results