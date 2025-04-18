from analyzer import scan_all_symbols, scan_forex_symbols

def generate_signals():
    crypto_signals = scan_all_symbols()
    forex_signals = scan_forex_symbols()
    return crypto_signals + forex_signals