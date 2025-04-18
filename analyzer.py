import requests
import pandas as pd
from time import sleep

# === Alpha Vantage API Key ===
ALPHA_VANTAGE_API_KEY = "8VL54YT3N656MW5T"

# ---------- کریپتو ----------
def get_all_symbols():
    url = "https://api.mexc.com/api/v3/exchangeInfo"
    response = requests.get(url)
    data = response.json()
    return [item["symbol"] for item in data["symbols"] if item["isSpotTradingAllowed"]]

def fetch_ohlcv(symbol, interval="5m", limit=200):
    sleep(0.5)
    url = f"https://api.mexc.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"خطا در دریافت داده برای {symbol}: {response.status_code}")
        return None
    data = response.json()
    if not data or len(data[0]) < 6:
        print(f"داده ناقص برای {symbol}: {len(data[0])} ستون دریافت شد")
        return None
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ][:len(data[0])])
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df

# ---------- فارکس ----------
request_count = 0
def fetch_forex_ohlcv(from_symbol, to_symbol="USD", interval="5min", outputsize="full"):
    global request_count
    if request_count >= 5:
        print("محدودیت Alpha Vantage: 60 ثانیه صبر کنید")
        sleep(60)
        request_count = 0
    request_count += 1
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

# ---------- تحلیل ----------
def compute_indicators(df):
    # EMA
    df["EMA20"] = df["close"].ewm(span=20, min_periods=20).mean()
    df["EMA50"] = df["close"].ewm(span=50, min_periods=50).mean()
    # RSI
    df["RSI"] = compute_rsi(df)
    # MACD
    df["MACD"] = df["close"].ewm(span=12, min_periods=26).mean() - df["close"].ewm(span=26, min_periods=26).mean()
    df["Signal"] = df["MACD"].ewm(span=9, min_periods=9).mean()
    # Bollinger Bands
    df["BB_Mid"] = df["close"].rolling(window=20).mean()
    df["BB_Std"] = df["close"].rolling(window=20).std()
    df["BB_Upper"] = df["BB_Mid"] + 2 * df["BB_Std"]
    df["BB_Lower"] = df["BB_Mid"] - 2 * df["BB_Std"]
    # ADX
    df["ADX"] = compute_adx(df)
    # Volume Trend
    df["Volume_MA"] = df["volume"].rolling(window=20).mean()
    df = df.fillna(method="bfill")
    return df

def compute_rsi(df, period=14):
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_adx(df, period=14):
    tr = pd.DataFrame(index=df.index)
    tr["h-l"] = df["high"] - df["low"]
    tr["h-pc"] = (df["high"] - df["close"].shift(1)).abs()
    tr["l-pc"] = (df["low"] - df["close"].shift(1)).abs()
    tr["TR"] = tr[["h-l", "h-pc", "l-pc"]].max(axis=1)
    tr["DM_plus"] = (df["high"] - df["high"].shift(1)).where((df["high"] - df["high"].shift(1)) > (df["low"].shift(1) - df["low"]), 0)
    tr["DM_minus"] = (df["low"].shift(1) - df["low"]).where((df["low"].shift(1) - df["low"]) > (df["high"] - df["high"].shift(1)), 0)
    atr = tr["TR"].rolling(window=period).mean()
    di_plus = 100 * (tr["DM_plus"].rolling(window=period).mean() / atr)
    di_minus = 100 * (tr["DM_minus"].rolling(window=period).mean() / atr)
    dx = 100 * ((di_plus - di_minus).abs() / (di_plus + di_minus))
    adx = dx.rolling(window=period).mean()
    return adx

def detect_price_action(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    if last["close"] > last["open"] and prev["close"] < prev["open"] and last["open"] < prev["close"]:
        return "الگوی انگالف صعودی"
    elif last["close"] < last["open"] and prev["close"] > prev["open"] and last["open"] > prev["close"]:
        return "الگوی انگالف نزولی"
    return None

def generate_signal(symbol, df, interval="--"):
    if df is None or len(df) < 50:
        return None
    df = compute_indicators(df)
    rsi = df["RSI"].iloc[-1]
    macd = df["MACD"].iloc[-1]
    signal = df["Signal"].iloc[-1]
    macd_cross = macd > signal and df["MACD"].iloc[-2] <= df["Signal"].iloc[-2]
    ema_cross = df["EMA20"].iloc[-1] > df["EMA50"].iloc[-1] and df["EMA20"].iloc[-2] <= df["EMA50"].iloc[-2]
    ema_distance = (df["EMA20"].iloc[-1] - df["EMA50"].iloc[-1]) / df["EMA50"].iloc[-1] * 100
    bb_breakout = df["close"].iloc[-1] > df["BB_Upper"].iloc[-1]
    adx = df["ADX"].iloc[-1]
    volume_trend = df["volume"].iloc[-1] > df["Volume_MA"].iloc[-1]
    pa = detect_price_action(df)
    close_price = df["close"].iloc[-1]

    score = 0
    weights = {"rsi": 0.2, "macd": 0.3, "ema": 0.2, "adx": 0.2, "volume": 0.1}
    if rsi < 30:
        score += weights["rsi"]
    if macd_cross:
        score += weights["macd"]
    if ema_cross and ema_distance > 0.5:
        score += weights["ema"]
    if adx > 25:
        score += weights["adx"]
    if volume_trend:
        score += weights["volume"]
    if bb_breakout:
        score += 0.1
    if pa == "الگوی انگالف صعودی":
        score += 0.1

    confidence = int(score * 100)
    if confidence >= 70:
        return {
            "symbol": symbol,
            "entry": close_price,
            "tp": round(close_price * 1.06, 5),
            "sl": round(close_price * 0.97, 5),
            "confidence": confidence,
            "volatility": round(abs(df["close"].iloc[-1] - df["close"].iloc[-2]) / df["close"].iloc[-2] * 100, 2),
            "analysis": f"RSI: {round(rsi, 1)} | EMA Cross: {ema_cross} | MACD Cross: {macd_cross} | ADX: {round(adx, 1)} | Volume Trend: {volume_trend} | BB Breakout: {bb_breakout} | {pa or '-'}",
            "tf": interval
        }
    return None

# ---------- اسکن ----------
def scan_all_crypto_symbols():
    PRIORITY_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
    TIMEFRAMES = ["1h", "4h", "1d"]
    all_symbols = get_all_symbols()
    symbols = PRIORITY_SYMBOLS + [s for s in all_symbols if s.endswith("USDT")]
    signals = []
    for symbol in symbols[:10]:
        for tf in TIMEFRAMES:
            try:
                df = fetch_ohlcv(symbol, interval=tf)
                signal = generate_signal(symbol, df, tf)
                if signal:
                    signals.append(signal)
            except Exception as e:
                print(f"خطا در {symbol} - {tf}: {e}")
                continue
    return signals

def scan_all_forex_symbols():
    pairs = [("EUR", "USD"), ("GBP", "USD"), ("USD", "JPY"), ("AUD", "USD"), ("USD", "CAD")]
    interval = "1hour"
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

# تست
if __name__ == "__main__":
    crypto_signals = scan_all_crypto_symbols()
    forex_signals = scan_all_forex_symbols()
    print("سیگنال‌های کریپتو:", crypto_signals)
    print("سیگنال‌های فارکس:", forex_signals)