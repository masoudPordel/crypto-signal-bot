import requests

def get_crypto_price(symbol="BTCUSDT"):
    url = f"https://api.mexc.com/api/v3/ticker/price?symbol={symbol.upper()}"
    response = requests.get(url)
    data = response.json()
    return float(data["price"]) if "price" in data else None

def get_forex_rate(base="USD", target="EUR"):
    url = f"https://open.er-api.com/v6/latest/{base.upper()}"
    response = requests.get(url)
    data = response.json()
    return float(data["rates"].get(target.upper())) if data.get("rates") else None

def generate_signals():
    crypto_price = get_crypto_price("BTCUSDT")
    forex_rate = get_forex_rate("USD", "EUR")
    
    message = f"سیگنال خودکار:\n"
    if crypto_price:
        message += f"قیمت BTC/USDT: {crypto_price}\n"
    else:
        message += "قیمت BTC/USDT در دسترس نیست\n"
    
    if forex_rate:
        message += f"نرخ USD به EUR: {forex_rate}\n"
    else:
        message += "نرخ USD/EUR در دسترس نیست\n"
    
    return message