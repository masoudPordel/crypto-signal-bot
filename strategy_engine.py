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

        analysis = "ÊÑ˜íÈ ÇäÏí˜ÇÊæÑåÇí íÔÑÝÊå¡ ÇãæÇÌ ÇáíæÊ æ ÑÇíÓ Ç˜Ôä"



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

