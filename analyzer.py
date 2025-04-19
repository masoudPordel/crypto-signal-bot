import requests
import pandas as pd

# === Alpha Vantage API Key ===
ALPHA_VANTAGE_API_KEY = "8VL54YT3N656MW5T"

# ---------- کریپتو از CoinEx ----------
def get_all_coinex_symbols():
    url = "https://api.coinex.com/v1/market/list"
    response = requests.get(url)
    data = response.json()
    return [symbol.replace('/', '').upper() for symbol in data["data"] if symbol.endswith("USDT")]

def fetch_coinex_ohlcv(symbol, interval="5min", limit=100):
    market = symbol.replace("USDT", "USDT").lower()
    url = f"https://api.coinex.com/v1/market/kline?market={market}&type={interval}&limit={limit}"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    data = response.json().get("data", {}).get("kline", [])
    if not data or len(data[0]) < 6:
        return None
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume"
    ])
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df

# ---------- فارکس ----------
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
        print(f"داده‌ای برای {from_symbol}/{to_symbol} دریافت نشد.")
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
def compute_rsi(df, period=14):
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = -delta.clip(upper=0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_indicators(df):
    df["EMA20"] = df["close"].ewm(span=20).mean()
    df["EMA50"] = df["close"].ewm(span=50).mean()
    df["RSI"] = compute_rsi(df)
    df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    return df

# ---------- پرایس اکشن و موج ----------
def detect_price_action(df):
    last, prev = df.iloc[-1], df.iloc[-2]
    if last["close"] > last["open"] and prev["close"] < prev["open"] and last["open"] < prev["close"]:
        return "الگوی انگالف صعودی"
    elif last["close"] < last["open"] and prev["close"] > prev["open"] and last["open"] > prev["close"]:
        return "الگوی انگالف نزولی"
    return None

def dummy_elliott_wave_check(df):
    return "موج الیوت شناسایی شد (فرضی)"

# ---------- استراتژی ساده ----------
def simple_signal_strategy(df):
    if df is None or len(df) < 2:
        return None
    if df["close"].iloc[-1] > df["close"].iloc[-2]:
        return "buy"
    elif df["close"].iloc[-1] < df["close"].iloc[-2]:
        return "sell"
    return None

# ---------- تولید سیگنال نهایی ----------
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

# ---------- اسکن کریپتو (CoinEx) ----------
def scan_coinex():
    PRIORITY = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
    TIMEFRAMES = ["5min", "15min", "30min"]
    all_symbols = get_all_coinex_symbols()
    symbols = PRIORITY + [s for s in all_symbols if s not in PRIORITY]
    signals = []
    for symbol in symbols[:10]:
        for tf in TIMEFRAMES:
            try:
                df = fetch_coinex_ohlcv(symbol, interval=tf)
                signal = generate_signal(symbol, df, tf)
                if signal and simple_signal_strategy(df):
                    signals.append(signal)
            except Exception as e:
                print(f"خطا در {symbol} - {tf}: {e}")
                continue
    return signals

# ---------- اسکن فارکس ----------
def scan_forex():
    pairs = [("EUR", "USD"), ("GBP", "USD"), ("USD", "JPY")]
    interval = "5min"
    signals = []
    for base, quote in pairs:
        try:
            df = fetch_forex_ohlcv(base, quote, interval)
            symbol = base + quote
            if df is not None:
                signal = generate_signal(symbol, df, interval)
                if signal:
                    signals.append(signal)
        except Exception as e:
            print(f"خطا در {base}/{quote}: {e}")
            continue
    return signals

# ---------- اجرای یکجا ----------
def run_all():
    crypto_signals = scan_coinex()
    forex_signals = scan_forex()
    return {
        "crypto": crypto_signals,
        "forex": forex_signals
    }

# تست سریع
if __name__ == "__main__":
    result = run_all()
    for s in result["crypto"] + result["forex"]:
        print(s)