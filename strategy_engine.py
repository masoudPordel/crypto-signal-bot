import pandas as pd

def calculate_indicators(df):
    df = df.copy()
    
    # Moving Averages
    df['MA_fast'] = df['close'].rolling(window=5).mean()
    df['MA_slow'] = df['close'].rolling(window=20).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def check_for_signal(df):
    df = calculate_indicators(df)

    # فقط اگر دیتافریم کافی باشه
    if df.shape[0] < 20:
        print("داده کافی نیست.")
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # شرط خرید: کراس MA + RSI
    if prev['MA_fast'] < prev['MA_slow'] and last['MA_fast'] > last['MA_slow'] and last['RSI'] < 40:
        print(f"سیگنال خرید تأیید شد - قیمت: {last['close']}, RSI: {last['RSI']}")
        return 'BUY'

    # شرط فروش: کراس معکوس MA + RSI بالا
    elif prev['MA_fast'] > prev['MA_slow'] and last['MA_fast'] < last['MA_slow'] and last['RSI'] > 60:
        print(f"سیگنال فروش تأیید شد - قیمت: {last['close']}, RSI: {last['RSI']}")
        return 'SELL'

    print("شرایط سیگنال فراهم نیست.")
    return None