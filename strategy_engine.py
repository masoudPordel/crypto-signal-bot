import numpy as np

# فرض: compute_indicators و detect_* توابع از قبل تعریف شده‌اند
def generate_signal(symbol, df, interval="5min", is_crypto=True, min_confidence=40):
    import logging

    if df is None:
        logging.warning(f"[{symbol}] دیتا None است.")
        return None
    if len(df) < 50:
        logging.warning(f"[{symbol}] دیتا کمتر از 50 کندل دارد ({len(df)} کندل).")
        return None

    df = compute_indicators(df)

    rsi = df["RSI"].iloc[-1]
    macd, signal_line = df["MACD"].iloc[-1], df["Signal"].iloc[-1]
    ema_cross = df["EMA12"].iloc[-2] < df["EMA26"].iloc[-2] and df["EMA12"].iloc[-1] > df["EMA26"].iloc[-1]
    volume_spike = df["volume"].iloc[-1] > df["volume"].rolling(20).mean().iloc[-1] * 1.2
    atr = df["ATR"].iloc[-1]
    close = df["close"].iloc[-1]

    # بررسی وضعیت شاخص‌ها
    logging.info(f"[{symbol}] RSI: {rsi:.2f}, MACD: {macd:.2f}, Signal: {signal_line:.2f}, EMA کراس: {ema_cross}, Volume spike: {volume_spike}")

    # بررسی الگوها و استراتژی‌ها
    pattern = detect_engulfing(df) or detect_advanced_price_action(df)
    trend = detect_trend(df)
    breakout = breakout_strategy(df)
    bollinger = bollinger_strategy(df)

    logging.info(f"[{symbol}] الگو: {pattern}, روند: {trend}, Breakout: {breakout}, Bollinger: {bollinger}")

    score = sum([
        rsi < 45,
        macd > signal_line,
        ema_cross,
        bool(pattern),
        volume_spike,
        bool(breakout),
        bool(bollinger)
    ])

    confidence = int((score / 7) * 100)
    logging.info(f"[{symbol}] امتیاز سیگنال: {score}/7 => اطمینان: {confidence}%")

    if confidence < min_confidence:
        logging.warning(f"[{symbol}] سطح اطمینان ({confidence}%) کمتر از حد ({min_confidence}%) است.")
        return None

    return {
        "نماد": symbol,
        "قیمت ورود": round(close, 5),
        "هدف سود": round(close + 2 * atr, 5),
        "حد ضرر": round(close - 1.5 * atr, 5),
        "سطح اطمینان": confidence,
        "تحلیل": f"RSI={round(rsi,1)}, EMA کراس={ema_cross}, MACD={'مثبت' if macd > signal_line else 'منفی'}, "
                 f"الگو={pattern}, {trend}, {breakout or '-'}, {bollinger or '-'}, "
                 f"حجم={'بالا' if volume_spike else 'نرمال'}",
        "تایم‌فریم": interval
    }