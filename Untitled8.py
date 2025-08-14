import streamlit as st
import pandas as pd
from pycoingecko import CoinGeckoAPI

cg = CoinGeckoAPI()

cryptos = ['bitcoin', 'ethereum', 'solana']
exchanges = ['binance', 'kraken', 'coinbase-pro']

data = []

for crypto in cryptos:
    crypto_prices = {}
    for exchange in exchanges:
        try:
            price_info = cg.get_price(ids=crypto, vs_currencies='usd', include_market_cap=False,
                                      include_24hr_vol=False, include_24hr_change=True)
            crypto_prices[exchange] = price_info[crypto]['usd']
        except:
            crypto_prices[exchange] = None

    crypto_prices['Crypto'] = crypto.capitalize()

    # Compute spread correctly
    numeric_prices = [p for k, p in crypto_prices.items() if isinstance(p, (int, float))]
    if numeric_prices:
        crypto_prices['Spread'] = max(numeric_prices) - min(numeric_prices)
    else:
        crypto_prices['Spread'] = None

    data.append(crypto_prices)

df = pd.DataFrame(data)
df = df[['Crypto'] + exchanges + ['Spread']]

st.title("Crypto Prices & Exchange Spreads")
st.table(df)

