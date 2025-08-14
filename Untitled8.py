import streamlit as st
from pycoingecko import CoinGeckoAPI
import pandas as pd
from streamlit_autorefresh import st_autorefresh

# --- Page Configuration ---
st.set_page_config(
    page_title="Crypto Spread Dashboard",
    page_icon="💸",
    layout="wide"
)

# --- Auto-Refresh ---
# Auto-refresh the page every 30 seconds to fetch the latest data.
st_autorefresh(interval=30000, key="crypto_data_refresh")

# --- API Initialization ---
cg = CoinGeckoAPI()

# --- Main Application ---
st.title("📈 Real-Time Crypto Dashboard")
st.caption("Data is refreshed automatically every 30 seconds.")

# List of popular crypto IDs from CoinGecko for the selection dropdown.
# You can find more IDs on the CoinGecko API documentation.
crypto_options = [
    'bitcoin', 'ethereum', 'ripple', 'litecoin', 'cardano', 
    'solana', 'dogecoin', 'polkadot', 'chainlink', 'tether'
]

# --- Sidebar for Selections ---
st.sidebar.header("Select Cryptocurrencies")
selected_cryptos = st.sidebar.multiselect(
    "Choose cryptocurrencies to display:", 
    crypto_options, 
    default=['bitcoin', 'ethereum', 'solana']
)

if not selected_cryptos:
    st.warning("Please select at least one cryptocurrency from the sidebar.")
    st.stop()

# --- Data Fetching and Display ---
try:
    # Fetch real-time data using the CoinGecko API
    # 'usd' is the currency, but you can change it to 'eur', 'gbp', etc.
    price_data = cg.get_price(
        ids=selected_cryptos, 
        vs_currencies='usd', 
        include_market_cap='true', 
        include_24hr_vol='true', 
        include_24hr_change='true'
    )
    
    # --- Data Processing with Pandas ---
    # Convert the dictionary from the API into a Pandas DataFrame
    df = pd.DataFrame(price_data).T
    df = df.reset_index()
    
    # Rename columns for clarity
    df.columns = [
        'Crypto', 'Price (USD)', 'Market Cap (USD)', 
        '24h Volume (USD)', '24h Change (%)'
    ]
    
    # Format the numeric columns for better readability
    df['Price (USD)'] = df['Price (USD)'].map('${:,.2f}'.format)
    df['Market Cap (USD)'] = df['Market Cap (USD)'].map('${:,.0f}'.format)
    df['24h Volume (USD)'] = df['24h Volume (USD)'].map('${:,.0f}'.format)
    df['24h Change (%)'] = df['24h Change (%)'].map('{:.2f}%'.format)

    # --- Display Data Table ---
    st.dataframe(df, use_container_width=True, hide_index=True)

    # --- Display Metrics and Spread ---
    if len(selected_cryptos) > 1:
        # Re-fetch raw prices for calculation without formatting
        raw_prices = [price_data[crypto]['usd'] for crypto in selected_cryptos]
        max_price = max(raw_prices)
        min_price = min(raw_prices)
        spread = max_price - min_price
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Highest Price", f"${max_price:,.2f}")
        col2.metric("Lowest Price", f"${min_price:,.2f}")
        col3.metric("Price Spread", f"${spread:,.2f}")

except Exception as e:
    st.error(f"An error occurred while fetching data from the CoinGecko API: {e}")
    st.info("The API might be temporarily unavailable or the selected cryptocurrency ID is incorrect.")
