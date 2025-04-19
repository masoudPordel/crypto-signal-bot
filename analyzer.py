import requests
import pandas as pd
import numpy as np

ALPHA_VANTAGE_API_KEY = "8VL54YT3N656MW5T"

# -------- KuCoin ----------
def get_all_kucoin_symbols():
    url = "https://api.kucoin.com/api/v1/symbols"
    response = requests.get(url)
    data = response.json()
    symbols = [item['symbol'].replace("-", "") for item in data['data'] if item['quoteCurrency'] == "USDT"]
    return list(set(symbols))

def fetch_ohlcv_kucoin(symbol, interval="5min", limit=100):
    url = f"https://api.kucoin.com/api/v1/market/candles?type={interval}&symbol={symbol.replace('', '-')}&limit={limit}"
    response = requests.get(url)
    if response.status_code != 200: return None
    raw = response.json().get("data", [])
    if not raw or len(raw[0]) < 7: return None
    df = pd.DataFrame(raw[::-1], columns=[
        "timestamp", "open", "close", "high", "low", "volume", "turnover"
    ])
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df

# -------- Forex ----------
def fetch_forex_ohlcv(from_symbol, to_symbol="USD", interval="5min", outputsize="compact"):
    url = (
        f"https://www.alphavantage.co/query?function=FX_INTRADAY"
        f"&from_symbol={from_symbol}&to_symbol={to_symbol}"
        f"&interval={interval}&outputsize={outputsize}&apikey={ALPHA_VANTAGE_API_KEY}"
    )
    response = requests.get(url)
    data = response.json()
    ts_key = [k for k in data if "Time Series" in k]
    if not ts_key: return None
    df = pd.DataFrame.from_dict(data[ts_key[0]], orient="index").sort_index()
    df = df.rename(columns={
        "1. open": "open", "2. high": "high", "3. low": "low", "4. close": "close"
    }).astype(float)
    return df

# -------- اندیکاتورها و تحلیل --------
def compute_rsi(df, period=14):
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_indicators(df):
    df["EMA20"] = df["close"].ewm(span=20).mean()
    df["EMA50"] = df["close"].ewm(span=50).mean()
    df["RSI"] = compute_rsi(df)
    df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    df["ATR"] = (df["high"] - df["low"]).rolling(window=14).mean()
    return df

def detect_price_action(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]

    if last["close"] > last["open"] and prev["close"] < prev["open"] and last["open"] < prev["close"]:
        return "Engulfing Bullish"
    if last["close"] < last["open"] and prev["close"] > prev["open"] and last["open"] > prev["close"]:
        return "Engulfing Bearish"
    body = abs(last["close"] - last["open"])
    wick = last["high"] - last["low"]
    if body < wick * 0.3:
        return "Doji / Pin Bar"
    return None

# -------- تولید سیگنال نهایی --------
def generate_signal(symbol, df, interval="--"):
    if df is None or len(df) < 50:
        return None
    df = compute_indicators(df)

    rsi = df["RSI"].iloc[-1]
    macd = df["MACD"].iloc[-1]
    signal_line = df["Signal"].iloc[-1]
    ema_cross = df["EMA20"].iloc[-2] < df["EMA50"].iloc[-2] and df["EMA20"].iloc[-1] > df["EMA50"].iloc[-1]
    pa = detect_price_action(df)
    atr = df["ATR"].iloc[-1]
    close = df["close"].iloc[-1]

    score = 0
    if rsi < 35: score += 1
    if macd > signal_line: score += 1
    if ema_cross: score += 1
    if pa: score += 1

    confidence = int((score / 4) * 100)
    if confidence >= 75:
        return {
            "symbol": symbol,
            "entry": close,
            "tp": round(close + 2 * atr, 5),
            "sl": round(close - 1.5 * atr, 5),
            "confidence": confidence,
            "analysis": f"RSI: {round(rsi,1)} | EMA Cross: {ema_cross} | MACD: {'Pos' if macd > signal_line else 'Neg'} | PA: {pa or '-'}",
            "interval": interval
        }
    return None

# -------- اسکن کریپتو --------
def scan_kucoin_all():
    symbols = get_all_kucoin_symbols()
    timeframes = ["5min", "15min", "1hour", "1day"]
    results = []
    for symbol in symbols:
        for tf in timeframes:
            try:
                df = fetch_ohlcv_kucoin(symbol, tf)
                sig = generate_signal(symbol, df, tf)
                if sig: results.append(sig)
            except Exception as e:
                continue
    return results

# -------- اسکن فارکس --------
def scan_forex_all():
    pairs = [("EUR", "USD"), ("GBP", "USD"), ("USD", "JPY"), ("AUD", "USD")]
    interval = "5min"
    results = []
    for base, quote in pairs:
        try:
            df = fetch_forex_ohlcv(base, quote, interval)
            symbol = f"{base}{quote}"
            sig = generate_signal(symbol, df, interval)
            if sig: results.append(sig)
        except Exception:
            continue
    return results

# -------- اجرای نهایی --------
if __name__ == "__main__":
    crypto_signals = scan_kucoin_all()
    forex_signals = scan_forex_all()
    all_signals = crypto_signals + forex_signals
    print(f"\n{'='*30}\nسیگنال‌های نهایی ({len(all_signals)}):\n")
    for s in all_signals:
        print(f"{s['symbol']} | TF: {s['interval']} | Entry: {s['entry']} | TP: {s['tp']} | SL: {s['sl']} | Conf: {s['confidence']}%\n{ s['analysis'] }\n")