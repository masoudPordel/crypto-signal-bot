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

def fetch_ohlcv(symbol, interval="15m", limit=100):
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

def fetch_volume(symbol):
    url = f"https://api.mexc.com/api/v3/ticker/24hr?symbol={symbol}"
    response = requests.get(url)
    if response.status_code != 200:
        return 0
    data = response.json()
    return float(data.get("quoteVolume", 0))

# ---------- فارکس ----------
request_count = 0
def fetch_forex_ohlcv(from_symbol, to_symbol="USD", interval="15min", outputsize="full"):
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
    # ATR for dynamic SL/TP
    df["ATR"] = compute_atr(df)
    df = df.fillna(method="bfill")
    print(f"آخرین مقادیر: RSI={df['RSI'].iloc[-1]}, MACD={df['MACD'].iloc[-1]}, Signal={df['Signal'].iloc[-1]}, ADX={df['ADX'].iloc[-1]}")
    return df

def compute_rsi(df, period=14):
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_adx(df, period=14):
    tr = pd.DataFrame(index=df.index)
    tr["TR"] = pd.concat([
        (df["high"] - df["low"]),
        (df["high"] - df["close"].shift()),
        (df["low"] - df["close"].shift())
    ], axis=1).abs().max(axis=1)
    atr = tr["TR"].rolling(window=period).mean()
    plus_dm = (df["high"] - df["high"].shift()).where((df["high"] - df["high"].shift()) > (df["low"].shift() - df["low"]), 0)
    minus_dm = (df["low"].shift() - df["low"]).where((df["low"].shift() - df["low"]) > (df["high"] - df["high"].shift()), 0)
    plus_di = 100 * plus_dm.rolling(window=period).mean() / atr
    minus_di = 100 * minus_dm.rolling(window=period).mean() / atr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    return adx

def compute_atr(df, period=14):
    tr = pd.DataFrame(index=df.index)
    tr["TR"] = pd.concat([
        (df["high"] - df["low"]),
        (df["high"] - df["close"].shift()),
        (df["low"] - df["close"].shift())
    ], axis=1).abs().max(axis=1)
    return tr["TR"].rolling(window=period).mean()

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
    rsi = df["RSI"].iloc[-1]
    macd = df["MACD"].iloc[-1]
    signal = df["Signal"].iloc[-1]
    ema_trend = df["EMA20"].iloc[-1] > df["EMA50"].iloc[-1]
    adx = df["ADX"].iloc[-1]
    bb_upper = df["BB_Upper"].iloc[-1]
    bb_lower = df["BB_Lower"].iloc[-1]
    close_price = df["close"].iloc[-1]
    atr = df["ATR"].iloc[-1]
    pa = detect_price_action(df)
    volume = df["volume"].iloc[-1]
    volume_spike = volume > df["volume"].rolling(window=20).mean().iloc[-1] * 1.5

    # فیلتر حجم معاملات
    quote_volume = fetch_volume(symbol)
    if quote_volume < 1_000_000:
        print(f"{symbol}: حجم معاملات پایین ({quote_volume})")
        return None

    # فیلتر نوسان
    volatility = abs(df["close"].iloc[-1] - df["close"].iloc[-2]) / df["close"].iloc[-2] * 100
    if volatility < 0.5 or volatility > 5:
        print(f"{symbol}: نوسان نامناسب ({volatility}%)")
        return None

    score = 0
    direction = "buy"  # پیش‌فرض خرید
    analysis = []

    # شرط‌های خرید
    if rsi < 40:  # کاهش آستانه برای دقت بیشتر
        score += 1
        analysis.append(f"RSI < 40 ({rsi})")
    if macd > signal:
        score += 1
        analysis.append("MACD > Signal")
    if ema_trend:
        score += 1
        analysis.append("EMA Trend Bullish")
    if adx > 25:  # روند قوی
        score += 1
        analysis.append(f"ADX > 25 ({adx})")
    if close_price < bb_lower:  # شکست بولینگر پایین
        score += 1
        analysis.append("Bollinger Breakout Lower")
    if pa == "الگوی انگالف صعودی":
        score += 1
        analysis.append(f"Price Action: {pa}")
    if volume_spike:
        score += 1
        analysis.append("Volume Spike")

    # شرط‌های فروش
    if rsi > 60:  # بیش‌خرید
        score -= 1
        direction = "sell"
        analysis.append(f"RSI > 60 ({rsi})")
    if macd < signal:
        score -= 1
        direction = "sell"
        analysis.append("MACD < Signal")
    if not ema_trend:
        score -= 1
        direction = "sell"
        analysis.append("EMA Trend Bearish")
    if close_price > bb_upper:  # شکست بولینگر بالا
        score -= 1
        direction = "sell"
        analysis.append("Bollinger Breakout Upper")
    if pa == "الگوی انگالف نزولی":
        score -= 1
        direction = "sell"
        analysis.append(f"Price Action: {pa}")

    # هماهنگی جهت
    if (direction == "buy" and score < 0) or (direction == "sell" and score > 0):
        print(f"{symbol}: جهت سیگنال متناقض")
        return None

    confidence = int((abs(score) / 7) * 100)
    print(f"{symbol}: Score = {score}, Direction = {direction}")
    if abs(score) >= 2:  # حداقل 2 شرط برای تولید سیگنال
        sl_multiplier = 1.5 * atr  # SL پویا
        tp_multiplier = 2.5 * atr  # TP پویا
        if direction == "buy":
            return {
                "symbol": symbol,
                "direction": "buy",
                "entry": close_price,
                "tp": round(close_price + tp_multiplier, 5),
                "sl": round(close_price - sl_multiplier, 5),
                "confidence": confidence,
                "volatility": round(volatility, 2),
                "analysis": " | ".join(analysis),
                "tf": interval
            }
        else:
            return {
                "symbol": symbol,
                "direction": "sell",
                "entry": close_price,
                "tp": round(close_price - tp_multiplier, 5),
                "sl": round(close_price + sl_multiplier, 5),
                "confidence": confidence,
                "volatility": round(volatility, 2),
                "analysis": " | ".join(analysis),
                "tf": interval
            }
    return None

# ---------- اسکن ----------
def scan_all_crypto_symbols():
    PRIORITY_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
    TIMEFRAMES = ["15m", "1h", "4h"]
    all_symbols = get_all_symbols()
    symbols = PRIORITY_SYMBOLS + [s for s in all_symbols if s.endswith("USDT")]
    signals = []
    for symbol in symbols[:20]:
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
    interval = "15min"
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

# ---------- ارسال به تلگرام ----------
def send_to_telegram(signal):
    # این بخش باید با توکن و آیدی چت تلگرام شما تنظیم بشه
    TELEGRAM_TOKEN = "YOUR_TELEGRAM_TOKEN"
    CHAT_ID = "YOUR_CHAT_ID"
    message = (
        f"📡 {signal['symbol']} | {signal['tf']}\n"
        f"جهت: {signal['direction'].upper()}\n"
        f"ورود: {signal['entry']}\n"
        f"حد سود: {signal['tp']}\n"
        f"حد ضرر: {signal['sl']}\n"
        f"اعتماد: {signal['confidence']}%\n"
        f"نوسان: {signal['volatility']}%\n"
        f"تحلیل: {signal['analysis']}"
    )
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage?chat_id={CHAT_ID}&text={message}"
    requests.get(url)

# تست
if __name__ == "__main__":
    crypto_signals = scan_all_crypto_symbols()
    forex_signals = scan_all_forex_symbols()
    all_signals = crypto_signals + forex_signals
    for signal in all_signals:
        send_to_telegram(signal)
    print("سیگنال‌های کریپتو:", crypto_signals)
    print("سیگنال‌های فارکس:", forex_signals)