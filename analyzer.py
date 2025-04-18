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

def fetch_ohlcv(symbol, interval="1h", limit=200):
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
def fetch_forex_ohlcv(from_symbol, to_symbol="USD", interval="1h", outputsize="full"):
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
    print(f"تعداد کندل‌ها برای {from_symbol}/{to_symbol}: {len(df)}")
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
    # ATR
    df["ATR"] = compute_atr(df)
    df = df.fillna(method="ffill").fillna(method="bfill")  # رفع اخطار FutureWarning
    return df

def compute_rsi(df, period=14):
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_adx(df, period=14):
    df["UpMove"] = df["high"].diff()
    df["DownMove"] = -df["low"].diff()
    df["+DM"] = ((df["UpMove"] > df["DownMove"]) & (df["UpMove"] > 0)).astype(int) * df["UpMove"]
    df["-DM"] = ((df["DownMove"] > df["UpMove"]) & (df["DownMove"] > 0)).astype(int) * df["DownMove"]
    df["TR"] = df[["high", "low", "close"]].apply(lambda x: max(x["high"] - x["low"], abs(x["high"] - x["close"].shift()), abs(x["low"] - x["close"].shift())), axis=1)
    df["+DI"] = 100 * df["+DM"].rolling(window=period).sum() / df["TR"].rolling(window=period).sum()
    df["-DI"] = 100 * df["-DM"].rolling(window=period).sum() / df["TR"].rolling(window=period).sum()
    df["DX"] = 100 * abs(df["+DI"] - df["-DI"]) / (df["+DI"] + df["-DI"])
    df["ADX"] = df["DX"].rolling(window=period).mean()
    return df["ADX"]

def compute_atr(df, period=14):
    df["TR"] = df[["high", "low", "close"]].apply(lambda x: max(x["high"] - x["low"], abs(x["high"] - x["close"].shift()), abs(x["low"] - x["close"].shift())), axis=1)
    return df["TR"].rolling(window=period).mean()

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
        print(f"داده ناکافی برای {symbol} در {interval}")
        return None
    df = compute_indicators(df)
    
    # مقادیر اندیکاتورها
    rsi = df["RSI"].iloc[-1]
    macd = df["MACD"].iloc[-1]
    signal_line = df["Signal"].iloc[-1]
    macd_cross = (df["MACD"].iloc[-2] < df["Signal"].iloc[-2]) and (macd > signal_line) and (macd - signal_line > 0.001 * df["close"].iloc[-1])  # اختلاف قابل توجه
    ema_cross = (df["EMA20"].iloc[-2] < df["EMA50"].iloc[-2]) and (df["EMA20"].iloc[-1] > df["EMA50"].iloc[-1])
    ema_distance = (df["EMA20"].iloc[-1] - df["EMA50"].iloc[-1]) / df["EMA50"].iloc[-1] * 100
    adx = df["ADX"].iloc[-1]
    bb_upper = df["BB_Upper"].iloc[-1]
    bb_lower = df["BB_Lower"].iloc[-1]
    close_price = df["close"].iloc[-1]
    atr = df["ATR"].iloc[-1]
    volume_avg = df["volume"].rolling(window=20).mean().iloc[-1]
    volume_current = df["volume"].iloc[-1]
    pa = detect_price_action(df)

    # فیلتر نقدینگی (حجم معاملات بالا)
    quote_volume = float(df["quote_asset_volume"].iloc[-1]) if "quote_asset_volume" in df.columns else 0
    if quote_volume < 1_000_000:  # حداقل 1M USDT
        print(f"{symbol}: حجم معاملات پایین ({quote_volume})")
        return None

    # امتیازدهی
    score = 0
    if rsi < 25:  # اشباع فروش خیلی قوی
        score += 2
        print(f"{symbol}: RSI < 25 ({rsi})")
    if macd_cross:
        score += 3
        print(f"{symbol}: MACD Cross Bullish (Diff: {(macd - signal_line):.6f})")
    if ema_cross and ema_distance > 1.0:  # کراس EMA با فاصله قوی
        score += 2
        print(f"{symbol}: EMA Cross Bullish with Distance {ema_distance:.2f}%")
    if close_price < bb_lower and volume_current > 2 * volume_avg:  # شکست باند پایینی با حجم بالا
        score += 1
        print(f"{symbol}: Price Below BB Lower with High Volume")
    if adx > 30:  # روند خیلی قوی
        score += 2
        print(f"{symbol}: ADX > 30 ({adx})")
    if volume_current > 2 * volume_avg:  # حجم خیلی بالا
        score += 1
        print(f"{symbol}: Very High Volume")
    if pa == "الگوی انگالف صعودی":
        score += 1
        print(f"{symbol}: Bullish Engulfing")

    # محاسبه Confidence
    confidence = min(100, int(score * 10))
    if score >= 8 and confidence >= 80:  # حداقل امتیاز 8 و Confidence 80%
        # تنظیم پویا SL و TP بر اساس ATR
        sl = round(close_price - 2 * atr, 5)  # استاپ لاس: 2 برابر ATR
        tp = round(close_price + 6 * atr, 5)  # تیک پروفیت: 6 برابر ATR (نسبت 3:1)
        return {
            "symbol": symbol,
            "entry": close_price,
            "tp": tp,
            "sl": sl,
            "confidence": confidence,
            "volatility": round(abs(df["close"].iloc[-1] - df["close"].iloc[-2]) / df["close"].iloc[-2] * 100, 2),
            "analysis": f"RSI: {round(rsi, 1)} | EMA Cross: {ema_cross} (Dist: {ema_distance:.2f}%) | MACD: {'صعودی' if macd_cross else 'خنثی'} | ADX: {round(adx, 1)} | Volume: {'خیلی بالا' if volume_current > 2 * volume_avg else 'عادی'} | ATR: {round(atr, 5)} | {pa or '-'}",
            "tf": interval
        }
    else:
        print(f"{symbol}: سیگنال رد شد (Score: {score}, Confidence: {confidence}%)")
    return None

# ---------- اسکن ----------
def scan_all_crypto_symbols():
    PRIORITY_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
    TIMEFRAMES = ["4h", "1d"]
    all_symbols = get_all_symbols()
    symbols = PRIORITY_SYMBOLS + [s for s in all_symbols if s.endswith("USDT")]
    signals = []
    for symbol in symbols[:10]:
        for tf in TIMEFRAMES:
            try:
                df = fetch_ohlcv(symbol, interval=tf, limit=200)
                signal = generate_signal(symbol, df, tf)
                if signal:
                    signals.append(signal)
            except Exception as e:
                print(f"خطا در {symbol} - {tf}: {e}")
                continue
    return signals

def scan_all_forex_symbols():
    pairs = [("EUR", "USD"), ("GBP", "USD"), ("USD", "JPY"), ("AUD", "USD"), ("USD", "CAD")]
    interval = "1h"
    results = []
    for base, quote in pairs:
        try:
            df = fetch_forex_ohlcv(base, quote, interval, outputsize="full")
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