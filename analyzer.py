import pandas as pd
import requests
from time import sleep

# دریافت داده از API
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
    df["MACD_Hist"] = df["MACD"] - df["Signal"]
    df["ATR"] = compute_atr(df)
    df["Pivot"] = (df["high"] + df["low"] + df["close"]) / 3
    df["R1"] = 2 * df["Pivot"] - df["low"]
    df["S1"] = 2 * df["Pivot"] - df["high"]
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
    prev1 = df.iloc[-2]
    prev2 = df.iloc[-3]
    # Bullish Engulfing
    if last["close"] > last["open"] and prev1["close"] < prev1["open"] and last["open"] < prev1["close"] and last["close"] > prev1["open"]:
        return "Bullish Engulfing"
    # Doji
    if abs(last["open"] - last["close"]) / (last["high"] - last["low"]) < 0.1:
        return "Doji"
    # Marubozu (کندل معنادار)
    if (last["close"] - last["open"]) / (last["high"] - last["low"]) > 0.7 and last["close"] > last["open"]:
        return "Bullish Marubozu"
    # Morning Star
    if prev2["close"] < prev2["open"] and abs(prev1["open"] - prev1["close"]) / (prev1["high"] - prev1["low"]) < 0.3 and last["close"] > last["open"] and last["close"] > (prev2["open"] + prev2["close"]) / 2:
        return "Morning Star"
    return None

# تشخیص Breakout فیک
def detect_fake_breakout(df, resistance, close_price):
    if len(df) < 3:
        return False
    last_two = df.iloc[-2:]
    if close_price > resistance and all(last_two["close"] < resistance):
        return True  # Breakout فیک تشخیص داده شد
    return False

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

    # فیلتر حجم معاملات
    if volume_current < 5 * volume_avg:
        return None

    # فیلتر بازار رنج (روانشناسی بازار)
    if atr < 0.015 * close_price:  # بازار کم‌نوسان
        return None

    # تشخیص Breakout فیک
    if detect_fake_breakout(df, last["R1"], close_price):
        return None

    # امتیازدهی
    score = 0
    if pa in ["Bullish Engulfing", "Morning Star", "Bullish Marubozu"] and last["RSI"] < 20:
        score += 5
    if last["MACD"] > last["Signal"] and last["MACD_Hist"] > 0.002 * close_price:
        score += 3
    if last["EMA20"] > last["EMA50"] > last["EMA200"]:
        score += 3
    if close_price > last["R1"] and volume_current > 5 * volume_avg:
        score += 3

    # وزن‌دهی بر اساس تایم‌فریم
    tf_weight = {"5m": 0.5, "15m": 0.7, "1h": 1.0, "1d": 1.5}
    score *= tf_weight.get(interval, 1.0)

    # اعتمادسنجی هوشمند
    confidence = min(100, int((score / 16) * 100))
    if score >= 12 and confidence >= 95:
        sl = round(close_price - atr, 5)
        tp = round(close_price + 5 * atr, 5)
        return {
            "symbol": symbol,
            "entry": close_price,
            "tp": tp,
            "sl": sl,
            "confidence": confidence,
            "analysis": f"Price Action: {pa} | RSI: {round(last['RSI'], 1)} | MACD: {'صعودی' if last['MACD'] > last['Signal'] else 'خنثی'} | Market: {'Trend' if atr > 0.015 * close_price else 'Range'}",
            "tf": interval
        }
    return None

# اسکن نمادها در تایم‌فریم‌های چندگانه
def scan_symbols():
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    timeframes = ["5m", "15m", "1h", "1d"]
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