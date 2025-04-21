import pandas as pd
import numpy as np
from analyzer import compute_indicators
from scipy.signal import argrelextrema

def analyze(df):
    df = compute_indicators(df)
    last = df.iloc[-1]
    entry = float(last["close"])
    sl = float(entry - 1.5 * float(last["ATR"]))
    tp = float(entry + 2 * float(last["ATR"]))
    rr = round((tp - entry) / (entry - sl), 2)

    conds = {
        "PinBar": bool(last.get("PinBar", False)),
        "Engulfing": bool(last.get("Engulfing", False)),
        "EMA_Cross": df["EMA12"].iloc[-2] < df["EMA26"].iloc[-2] and df["EMA12"].iloc[-1] > df["EMA26"].iloc[-1],
        "MACD_Cross": df["MACD"].iloc[-2] < df["Signal"].iloc[-2] and df["MACD"].iloc[-1] > df["Signal"].iloc[-1],
        "RSI_Oversold": last["RSI"] < 30,
    }

    score = sum(conds.values())
    if score >= 2:
        return {
            "نماد": "SYMBOL",
            "تایم‌فریم": "1h",
            "قیمت ورود": entry,
            "هدف سود": tp,
            "حد ضرر": sl,
            "سطح اطمینان": int(min(score * 20, 100)),
            "تحلیل": " | ".join([k for k,v in conds.items() if v]),
            "ریسک به ریوارد": rr
        }
    return None