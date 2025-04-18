
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
