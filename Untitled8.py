import streamlit as st
import pandas as pd
from pycoingecko import CoinGeckoAPI

# Initialize CoinGecko API
cg = CoinGeckoAPI()

# List of cryptos to track
cryptos = ['bitcoin', 'ethereum', 'solana']

# List of exchanges to track
exchanges = ['binance', 'kraken', 'coinbase-pro']

# Fetch prices
data = []
for crypto in cryptos:
    crypto_prices = {}
    for exchange in exchanges:
        try:
            price_info = cg.get_price(ids=crypto, vs_currencies='usd', include_market_cap=False,
                                      include_24hr_vol=False, include_24hr_change=True, include_last_updated_at=False,
                                      include_exchanges=[exchange])
            # pycoingecko returns a nested dict
            crypto_prices[exchange] = price_info[crypto]['usd']
        except:
            crypto_prices[exchange] = None
    crypto_prices['Crypto'] = crypto.capitalize()
    crypto_prices['Spread'] = max([p for p in crypto_prices.values() if p is not None]) - min([p for p in crypto_prices.values() if p is not None])
    data.append(crypto_prices)

# Convert to DataFrame
df = pd.DataFrame(data)
df = df[['Crypto'] + exchanges + ['Spread']]  # reorder columns

# Streamlit display
st.title("Crypto Prices & Exchange Spreads")
st.table(df)
