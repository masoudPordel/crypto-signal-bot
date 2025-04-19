from analyzer import scan_all_crypto_symbols, scan_all_forex_symbols

def generate_crypto_signals():
    crypto_signals = scan_all_crypto_symbols()
    signals = []
    for signal in crypto_signals:
        signals.append(f"Symbol: {signal['symbol']}\nEntry: {signal['entry']}\nTP: {signal['tp']}\nSL: {signal['sl']}\nTF: {signal['tf']}")
    return signals

def generate_forex_signals():
    forex_signals = scan_all_forex_symbols()
    signals = []
    for signal in forex_signals:
        signals.append(f"Symbol: {signal['symbol']}\nEntry: {signal['entry']}\nTP: {signal['tp']}\nSL: {signal['sl']}\nTF: {signal['tf']}")
    return signals