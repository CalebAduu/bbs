import streamlit as st
import requests
import time

st.set_page_config(page_title="Crypto Arbitrage Tracker", layout="wide")

# Binance special cases if needed
BINANCE_SPECIAL_CASES = {
    "DOGE": "DOGEUSDT",
    "SHIB": "SHIBUSDT",
    # Add more if necessary
}

def get_binance_price(symbol):
    try:
        symbol = BINANCE_SPECIAL_CASES.get(symbol, symbol.upper() + "USDT")
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        data = requests.get(url, timeout=5).json()
        if 'code' in data:  # error from Binance API
            return None
        return float(data['price']) if 'price' in data else None
    except:
        return None

def get_coinbase_price(symbol):
    try:
        url = f"https://api.coinbase.com/v2/prices/{symbol}-USD/spot"
        data = requests.get(url, timeout=5).json()
        return float(data['data']['amount']) if 'data' in data else None
    except:
        return None

def get_kraken_price(symbol):
    try:
        special_cases = {"BTC": "XBT", "DOGE": "XDG", "USDT": "USDTZ"}
        kraken_symbol = special_cases.get(symbol, symbol)
        url = f"https://api.kraken.com/0/public/Ticker?pair={kraken_symbol}USD"
        data = requests.get(url, timeout=5).json()
        if 'result' in data and len(data['result']) > 0:
            pair = list(data['result'].keys())[0]
            return float(data['result'][pair]['c'][0])
        return None
    except:
        return None

def get_top_cryptos(limit=20):
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": limit,
            "page": 1,
            "sparkline": "false"
        }
        response = requests.get(url, timeout=5, params=params)
        return response.json()
    except:
        return []

def calculate_spread(prices):
    values = [v for v in prices.values() if v is not None]
    if not values:
        return 0
    return max(values) - min(values)

st.title("📊 Crypto Prices & Exchange Spreads")

cryptos = get_top_cryptos(limit=20)

table_data = []

for crypto in cryptos:
    name = crypto['name']
    symbol = crypto['symbol'].upper()
    
    binance = get_binance_price(symbol)
    coinbase = get_coinbase_price(symbol)
    kraken = get_kraken_price(symbol)
    
    prices = {"Binance": binance, "Coinbase": coinbase, "Kraken": kraken}
    spread = calculate_spread(prices)
    
    table_data.append({
        "Crypto": name,
        "Binance": binance,
        "Coinbase": coinbase,
        "Kraken": kraken,
        "Spread": spread
    })

st.table(table_data)

st.markdown("Data updates each time the app reruns. Refresh the page for latest prices.")
