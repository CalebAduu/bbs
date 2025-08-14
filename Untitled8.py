{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "888c6308-9d63-4c62-8804-97a871a39b40",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-08-14 08:03:13.442 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\Caleb\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "# streamlit_app.py\n",
    "import streamlit as st\n",
    "from pycoingecko import CoinGeckoAPI\n",
    "import pandas as pd\n",
    "from streamlit_autorefresh import st_autorefresh\n",
    "\n",
    "# Auto-refresh every 10 seconds\n",
    "st_autorefresh(interval=10000, key=\"crypto_refresh\")\n",
    "\n",
    "# Initialize CoinGecko API\n",
    "cg = CoinGeckoAPI()\n",
    "\n",
    "st.title(\"Real-Time Crypto Spread Dashboard\")\n",
    "\n",
    "# List of popular crypto IDs from CoinGecko\n",
    "crypto_options = ['bitcoin', 'ethereum', 'ripple', 'litecoin', 'cardano', 'dogecoin']\n",
    "selected_cryptos = st.multiselect(\"Select Cryptocurrencies\", crypto_options, default=['bitcoin','ethereum'])\n",
    "\n",
    "if not selected_cryptos:\n",
    "    st.warning(\"Please select at least one cryptocurrency.\")\n",
    "    st.stop()\n",
    "\n",
    "# Fetch real-time data\n",
    "try:\n",
    "    data = cg.get_price(ids=selected_cryptos, vs_currencies='usd', include_24hr_change='true', include_market_cap='true')\n",
    "    # Convert to DataFrame\n",
    "    df = pd.DataFrame(data).T.reset_index()\n",
    "    df.columns = ['Crypto', 'Price (USD)', '24h Change (%)', 'Market Cap (USD)']\n",
    "    st.dataframe(df)\n",
    "except Exception as e:\n",
    "    st.error(f\"Error fetching data: {e}\")\n",
    "\n",
    "# Optional: display spread between highest and lowest price among selected cryptos\n",
    "if len(selected_cryptos) > 1:\n",
    "    max_price = df['Price (USD)'].max()\n",
    "    min_price = df['Price (USD)'].min()\n",
    "    spread = max_price - min_price\n",
    "    st.metric(\"Price Spread\", f\"${spread:,.2f}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1f48e92b-9817-49fc-abb8-2a0f0ac4087f",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
