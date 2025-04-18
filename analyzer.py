import pandas as pd
import requests
from time import sleep

# دریافت داده از API (مثلاً MEXC)
def fetch_ohlcv(symbol, interval="4h", limit=300):
    sleep(0.5)
    url = f"https://api.mexc.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    data = response.json()
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "trades", "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"])
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df

# محاسبه اندیکاتورها
def compute_indicators(df):
    df["EMA20"] = df["close"].ewm(span=20).mean()
    df["EMA50"] = df["close"].ewm(span=50).mean()
    df["EMA200"] = df["close"].ewm(span=200).mean()
    df["RSI"] = compute_rsi(df)
    df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    df["ATR"] = compute_atr(df)
    df["Pivot"] = (df["high"] + df["low"] + df["close"]) / 3
    df["Support1"] = 2 * df["Pivot"] - df["high"]
    df["Resistance1"] = 2 * df["Pivot"] - df["low"]
    return df.fillna(method="ffill").fillna(method="bfill")

def compute_rsi(df, period=14):
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_atr(df, period=14):
    df["TR"] = df[["high", "low", "close"]].apply(lambda x: max(x["high"] - x["low"], abs(x["high"] - x["close"].shift()), abs(x["low"] - x["close"].shift())), axis=1)
    return df["TR"].rolling(window=period).mean()

# تشخیص الگوهای پرایس اکشن
def detect_price_action(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    if last["close"] > last["open"] and prev["close"] < prev["open"] and last["open"] < prev["close"] and last["close"] > prev["open"]:
        return "Bullish Engulfing"
    elif last["close"] < last["open"] and prev["close"] > prev["open"] and last["open"] > prev["close"] and last["close"] < prev["open"]:
        return "Bearish Engulfing"
    elif (last["high"] - last["close"]) / (last["high"] - last["low"]) < 0.2 and (last["close"] - last["low"]) / (last["high"] - last["low"]) > 0.6:
        return "Hammer"
    elif (last["close"] - last["low"]) / (last["high"] - last["low"]) < 0.2 and (last["high"] - last["close"]) / (last["high"] - last["low"]) > 0.6:
        return "Shooting Star"
    elif abs(last["open"] - last["close"]) / (last["high"] - last["low"]) < 0.1:
        return "Doji"
    return None

# تولید سیگنال
def generate_signal(symbol, df, interval="4h"):
    if df is None or len(df) < 200:
        return None
    df = compute_indicators(df)
    
    last = df.iloc[-1]
    pa = detect_price_action(df)
    volume_avg = df["volume"].rolling(window=20).mean().iloc[-1]
    volume_current = last["volume"]
    close_price = last["close"]
    atr = last["ATR"]

    # فیلتر حجم
    if volume_current < 3 * volume_avg:
        return None

    # امتیازدهی
    score = 0
    if pa == "Bullish Engulfing" and last["RSI"] < 30 and last["MACD"] > last["Signal"]:
        score += 4
    elif pa == "Hammer" and last["RSI"] < 20 and close_price < last["Support1"]:
        score += 3
    elif pa == "Doji" and abs(last["MACD"] - last["Signal"]) < 0.001 * close_price:
        score += 2
    if last["EMA20"] > last["EMA50"] > last["EMA200"] and last["close"] > last["EMA20"]:
        score += 3

    confidence = min(100, int((score / 12) * 100))
    if score >= 10 and confidence >= 90:
        sl = round(close_price - 3 * atr, 5)
        tp = round(close_price + 12 * atr, 5)
        return {
            "symbol": symbol,
            "entry": close_price,
            "tp": tp,
            "sl": sl,
            "confidence": confidence,
            "analysis": f"Price Action: {pa} | RSI: {round(last['RSI'], 1)}",
            "tf": interval
        }
    return None

# اسکن نمادها
def scan_symbols():
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    timeframes = ["4h", "1d"]
    signals = []
    for symbol in symbols:
        for tf in timeframes:
            df = fetch_ohlcv(symbol, interval=tf)
            signal = generate_signal(symbol, df, tf)
            if signal:
                signals.append(signal)
    return signals

# اجرا
if __name__ == "__main__":
    signals = scan_symbols()
    print("سیگنال‌ها:", signals)