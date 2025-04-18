from ai_model import get_ai_confidence
import random

def generate_signals():
    symbols = ["BTC/USDT", "ETH/USDT", "EUR/USD", "GBP/USD"]
    signals = []

    for symbol in symbols:
        entry = round(random.uniform(100, 50000), 2)
        tp = round(entry * 1.03, 2)
        sl = round(entry * 0.97, 2)
        confidence = get_ai_confidence(symbol)
        analysis = "ØªØ±Ú©ÛØ¨ Ø§ÙØ¯ÛÚ©Ø§ØªÙØ±ÙØ§Û Ù¾ÛØ´Ø±ÙØªÙØ Ø§ÙÙØ§Ø¬ Ø§ÙÛÙØª Ù Ù¾Ø±Ø§ÛØ³ Ø§Ú©Ø´Ù"

        if confidence > 75:
            signals.append({
                "symbol": symbol,
                "entry": entry,
                "tp": tp,
                "sl": sl,
                "confidence": confidence,
                "analysis": analysis
            })

    return signals