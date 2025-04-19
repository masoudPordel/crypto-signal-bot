
from strategy_engine import compute_indicators, detect_elliott_wave, detect_price_action
from logger import logger
import requests

TELEGRAM_TOKEN = "8111192844:AAHuVZYs6RolBhdqPpTWW9g7ksGRaq3p0WA"
TELEGRAM_CHAT_ID = "632886964"

def send_to_telegram(message):
    url = "https://api.telegram.org/bot{}/sendMessage".format(TELEGRAM_TOKEN)
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }
    try:
        requests.post(url, data=payload)
    except Exception as e:
        logger.error("Telegram Error: {}".format(e))

def analyze(df, symbol, interval="5min"):
    df = compute_indicators(df)

    rsi = df["RSI"].iloc[-1]
    macd = df["MACD"].iloc[-1]
    signal_line = df["Signal"].iloc[-1]
    ema_cross = df["EMA12"].iloc[-2] < df["EMA26"].iloc[-2] and df["EMA12"].iloc[-1] > df["EMA26"].iloc[-1]
    volume = df["volume"].iloc[-1]
    avg_volume = df["volume"].rolling(20).mean().iloc[-1]
    volume_spike = volume > avg_volume * 1.2

    elliott = detect_elliott_wave(df)
    price_action = detect_price_action(df)

    score = sum([
        rsi < 40,
        macd > signal_line,
        ema_cross,
        bool(price_action),
        volume_spike,
        bool(elliott)
    ])
    confidence = int((score / 6) * 100)

    msg = (
        "نماد: {}
"
        "RSI: {:.1f}
"
        "MACD: {}
"
        "EMA کراس: {}
"
        "الگو: {}
"
        "موج الیوت: {}
"
        "حجم: {}
"
        "سطح اطمینان: {}%"
    ).format(
        symbol,
        rsi,
        "مثبت" if macd > signal_line else "منفی",
        "بلی" if ema_cross else "خیر",
        price_action or "-",
        elliott or "-",
        "بالا" if volume_spike else "نرمال",
        confidence
    )

    logger.info(msg)
    if confidence >= 50:
        send_to_telegram("سیگنال جدید از KuCoin:
" + msg)
        return "BUY" if rsi < 40 else "SELL"
    return "HOLD"
