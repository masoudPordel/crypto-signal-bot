import requests

import pandas as pd



# === Alpha Vantage API Key ===

ALPHA_VANTAGE_API_KEY = "8VL54YT3N656MW5T"



# ---------- کریپتو ----------

def get_all_symbols():

    url = "https://api.mexc.com/api/v3/exchangeInfo"

    response = requests.get(url)

    data = response.json()

    return [item["symbol"] for item in data["symbols"] if item["isSpotTradingAllowed"]]



def fetch_ohlcv(symbol, interval="5m", limit=100):

    url = f"https://api.mexc.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"

    response = requests.get(url)

    if response.status_code != 200:

        return None

    df = pd.DataFrame(response.json(), columns=[

        "timestamp", "open", "high", "low", "close", "volume",

        "close_time", "quote_asset_volume", "trades",

        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"

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



# ---------- تحلیل ----------

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



def detect_price_action(df):

    last = df.iloc[-1]

    prev = df.iloc[-2]

    if last["close"] > last["open"] and prev["close"] < prev["open"] and last["open"] < prev["close"]:

        return "الگوی انگالف صعودی"

    elif last["close"] < last["open"] and prev["close"] > prev["open"] and last["open"] > prev["close"]:

        return "الگوی انگالف نزولی"

    return None



def dummy_elliott_wave_check(df):

    return "موج الیوت شناسایی شد (فرضی)"



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



# ---------- اسکن ----------

def scan_all_crypto_symbols():

    PRIORITY_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "LTCUSDT"]

    TIMEFRAMES = ["5m", "15m", "1h", "1d", "1w"]



    all_symbols = get_all_symbols()

    symbols = PRIORITY_SYMBOLS + [s for s in all_symbols if s not in PRIORITY_SYMBOLS and s.endswith("USDT")]



    signals = []

    for symbol in symbols:

        for tf in TIMEFRAMES:

            df = fetch_ohlcv(symbol, interval=tf)

            signal = simple_signal_strategy(df)

            if signal:

                entry = df["close"].iloc[-1]

                tp = entry * 1.01

                sl = entry * 0.99

                signals.append({

                    "symbol": symbol,

                    "entry": round(entry, 4),

                    "tp": round(tp, 4),

                    "sl": round(sl, 4),

                    "tf": tf,

                    "signal": signal

                })

    return signals





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

            continue

    return results

