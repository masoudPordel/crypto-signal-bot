# -*- coding: utf-8 -*-

import requests

def get_crypto_price(symbol: str) -> float:
    """دریافت قیمت لحظه‌ای از صرافی MEXC"""
    url = f'https://api.mexc.com/api/v3/ticker/price?symbol={symbol}'
    response = requests.get(url)
    data = response.json()
    return float(data['price'])

def get_forex_rate(base: str, quote: str) -> float:
    """دریافت نرخ فارکس از ExchangeRate-API"""
    url = f'https://open.er-api.com/v6/latest/{base}'
    response = requests.get(url)
    data = response.json()
    return float(data['rates'][quote])

def generate_signals():
    signals = []

    # سیگنال کریپتو - BTC/USDT
    try:
        price = get_crypto_price("BTCUSDT")
        signals.append({
            'pair': 'BTC/USDT',
            'price': price,
            'target1': round(price * 1.03, 2),
            'target2': round(price * 1.05, 2),
            'stop_loss': round(price * 0.97, 2),
            'confidence': 88
        })
    except Exception:
        pass

    # سیگنال فارکس - EUR/USD
    try:
        price = get_forex_rate("USD", "EUR")
        signals.append({
            'pair': 'EUR/USD',
            'price': price,
            'target1': round(price * 1.01, 4),
            'target2': round(price * 1.02, 4),
            'stop_loss': round(price * 0.99, 4),
            'confidence': 81
        })
    except Exception:
        pass

    return signals
