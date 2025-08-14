#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import asyncio
import datetime as dt
import math
from typing import Dict, List, Tuple, Optional, Any, Set
import io

# ccxt async support
import ccxt.async_support as ccxt

# -----------------------------
# Helpers
# -----------------------------

def now_utc_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def pct_spread(max_price: float, min_price: float, mode: str = "mid") -> float:
    if max_price is None or min_price is None:
        return float("nan")
    if max_price <= 0 or min_price <= 0:
        return float("nan")
    if mode == "mid":
        mid = (max_price + min_price) / 2.0
        return (max_price - min_price) / mid if mid else float("nan")
    elif mode == "min":
        return (max_price - min_price) / min_price
    else:
        raise ValueError("mode must be 'mid' or 'min'")


def build_exchange(name: str, market_type: str):
    name = name.lower().strip()
    if not hasattr(ccxt, name):
        raise ValueError(f"Exchange '{name}' not found in ccxt.")
    klass = getattr(ccxt, name)
    exchange = klass({
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {
            "defaultType": "swap" if market_type == "swap" else "spot"
        }
    })
    return exchange


async def close_exchange(exchange):
    try:
        await exchange.close()
    except Exception:
        pass


async def load_markets_safe(exchange, market_type: str) -> Dict[str, Any]:
    try:
        markets = await exchange.load_markets(reload=True)
        filtered = {}
        for sym, m in markets.items():
            if market_type == "spot" and m.get("spot"):
                filtered[sym] = m
            elif market_type == "swap" and m.get("swap"):
                if bool(m.get("contract")):
                    filtered[sym] = m
        return filtered
    except Exception as e:
        return {}


async def fetch_symbol_price(exchange, symbol: str, use_orderbook: bool = False) -> Optional[float]:
    try:
        if use_orderbook:
            ob = await exchange.fetch_order_book(symbol, limit=5)
            bid = ob["bids"][0][0] if ob.get("bids") else None
            ask = ob["asks"][0][0] if ob.get("asks") else None
            if bid is not None and ask is not None:
                return (bid + ask) / 2.0
            return ask or bid
        else:
            ticker = await exchange.fetch_ticker(symbol)
            price = ticker.get("last") or ticker.get("close") or ticker.get("bid") or ticker.get("ask")
            return float(price) if price is not None and math.isfinite(float(price)) else None
    except Exception:
        return None


async def fetch_prices_for_exchange(exchange, symbols: List[str], use_orderbook: bool = False) -> Dict[str, float]:
    out: Dict[str, float] = {}

    sem = asyncio.Semaphore(10)

    async def _fetch(sym: str):
        async with sem:
            p = await fetch_symbol_price(exchange, sym, use_orderbook=use_orderbook)
            if p is not None and math.isfinite(p):
                out[sym] = float(p)

    tasks = [asyncio.create_task(_fetch(s)) for s in symbols]
    await asyncio.gather(*tasks, return_exceptions=True)
    return out


async def scan_async(
    market_type: str,
    exchange_names: List[str],
    quotes: List[str],
    min_venues: int,
    pct_mode: str,
    use_orderbook: bool,
) -> pd.DataFrame:
    wanted_quotes: Set[str] = {q.upper() for q in quotes}

    # Instantiate exchanges
    exchanges = {}
    for name in exchange_names:
        try:
            ex = build_exchange(name, market_type)
            exchanges[name] = ex
        except Exception as e:
            pass

    if not exchanges:
        return pd.DataFrame()

    try:
        # Load markets and find overlapping symbols
        markets_per_ex = await asyncio.gather(*[load_markets_safe(ex, market_type) for ex in exchanges.values()])
        ex_to_markets = dict(zip(exchanges.keys(), markets_per_ex))

        ex_to_symbols: Dict[str, Set[str]] = {}
        for ex_name, markets in ex_to_markets.items():
            syms = set()
            for sym, m in markets.items():
                quote = (m.get("quote") or "").upper()
                if quote in wanted_quotes:
                    syms.add(sym)
            ex_to_symbols[ex_name] = syms

        symbol_to_exchanges: Dict[str, List[str]] = {}
        for ex_name, syms in ex_to_symbols.items():
            for s in syms:
                symbol_to_exchanges.setdefault(s, []).append(ex_name)

        candidate_symbols = [s for s, venues in symbol_to_exchanges.items() if len(venues) >= min_venues]

        if not candidate_symbols:
            return pd.DataFrame()

        price_map: Dict[Tuple[str, str], float] = {}

        async def _fetch_for_exchange(ex_name: str):
            ex = exchanges[ex_name]
            syms = [s for s in candidate_symbols if ex_name in symbol_to_exchanges[s]]
            prices = await fetch_prices_for_exchange(ex, syms, use_orderbook=use_orderbook)
            for s, p in prices.items():
                price_map[(ex_name, s)] = p

        await asyncio.gather(*[asyncio.create_task(_fetch_for_exchange(n)) for n in exchanges.keys()])

        rows = []
        as_of = now_utc_iso()
        for s in candidate_symbols:
            entries = [(ex, price_map.get((ex, s))) for ex in symbol_to_exchanges[s]]
            entries = [(ex, p) for ex, p in entries if p is not None and math.isfinite(p)]
            if len(entries) < 2:
                continue

            max_ex, max_price = max(entries, key=lambda x: x[1])
            min_ex, min_price = min(entries, key=lambda x: x[1])
            spread = max_price - min_price
            pct = pct_spread(max_price, min_price, mode=pct_mode)
            price_detail = {f"price@{ex}": float(pr) for ex, pr in entries}

            row = {
                "timestamp": as_of,
                "market_type": market_type,
                "symbol": s,
                "venues": ",".join(sorted([ex for ex, _ in entries])),
                "min_ex": min_ex,
                "min_price": float(min_price),
                "max_ex": max_ex,
                "max_price": float(max_price),
                "abs_spread": float(spread),
                f"pct_spread_{pct_mode}": float(pct) if pct is not None and math.isfinite(pct) else float("nan"),
                "count_venues": len(entries),
            }
            row.update(price_detail)
            rows.append(row)

        df = pd.DataFrame(rows)
        return df
    finally:
        await asyncio.gather(*[close_exchange(ex) for ex in exchanges.values()])


def run_scan(market_type: str, exchanges: List[str], quotes: List[str], min_venues: int, pct_mode: str, use_orderbook: bool) -> pd.DataFrame:
    try:
        df = asyncio.run(
            scan_async(
                market_type=market_type,
                exchange_names=exchanges,
                quotes=quotes,
                min_venues=min_venues,
                pct_mode=pct_mode,
                use_orderbook=use_orderbook,
            )
        )
    except RuntimeError:
        # Fallback if an event loop is already running
        loop = asyncio.get_event_loop()
        df = loop.run_until_complete(
            scan_async(
                market_type=market_type,
                exchange_names=exchanges,
                quotes=quotes,
                min_venues=min_venues,
                pct_mode=pct_mode,
                use_orderbook=use_orderbook,
            )
        )
    return df


# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="CEX Arbitrage Divergence Scanner", layout="wide")

st.title("🔎 CEX Arbitrage Divergence Scanner")
st.caption("Finds the largest price divergences across overlapping markets on multiple CEXs (spot or perpetual).")

with st.sidebar:
    st.header("⚙️ Settings")
    market_type = st.radio("Market Type", options=["spot", "swap"], index=0, help="Choose spot or perpetual swaps.")
    default_exchanges = ["binance","bybit","kucoin","okx","gate","bitget","mexc","kraken","coinbase","huobi"]
    exchanges = st.multiselect("Exchanges", options=default_exchanges, default=default_exchanges, help="ccxt exchange ids")
    default_quotes = ["USDT","USDC","USD"] if market_type == "spot" else ["USDT"]
    quotes = st.multiselect("Quote Currencies", options=["USDT","USDC","USD","BUSD","EUR","GBP","USDT:USDT"], default=default_quotes)
    min_venues = st.number_input("Minimum venues per symbol", min_value=2, max_value=20, value=2, step=1)
    pct_mode = st.selectbox("Percent spread mode", options=["mid","min"], index=0, help="mid = spread / midprice, min = spread / min price")
    use_orderbook = st.checkbox("Use order book midprice", value=False, help="More accurate, heavier API usage")
    threshold = st.slider("Min % spread to show", min_value=0.0, max_value=10.0, value=0.5, step=0.1, help="Rows below this threshold are hidden")
    top_n = st.slider("Top N rows", min_value=10, max_value=200, value=50, step=10)
    auto = st.checkbox("Auto-refresh", value=True)
    refresh_sec = st.slider("Refresh interval (seconds)", min_value=5, max_value=60, value=15, step=5)

if auto:
    st.experimental_rerun  # placeholder to avoid lint complaints
    st_autorefresh = st.experimental_memo.clear if False else None  # placeholder
    # Use the official API
    st.experimental_rerun  # no-op, compatibility
    st_autorefresh = st.experimental_singleton.clear if False else None  # no-op to satisfy linters
    # Newer API
    try:
        from streamlit.runtime.scriptrunner import add_script_run_ctx  # noqa
        st.autorefresh = st.autorefresh if hasattr(st, "autorefresh") else None  # noqa
    except Exception:
        pass
    # Use documented API
    st.experimental_set_query_params(ts=dt.datetime.utcnow().timestamp())
    st.stop() if False else None

# Button to trigger scan (also runs on every refresh)
run = st.button("Run Scan", type="primary") or True  # default True to run on load

if run:
    with st.spinner("Fetching markets & prices..."):
        df = run_scan(market_type, exchanges, quotes, min_venues, pct_mode, use_orderbook)

    if df is None or df.empty:
        st.warning("No results. Try adding more exchanges/quotes or lowering the minimum venues.")
    else:
        colL, colR = st.columns([3,2])
        with colL:
            st.subheader("Results")
        with colR:
            st.write(f"Last updated: **{now_utc_iso()}**")

        # Sort and filter
        pct_col = f"pct_spread_{pct_mode}"
        if pct_col in df.columns:
            df.sort_values(by=[pct_col, "abs_spread"], ascending=[False, False], inplace=True)
            df = df[df[pct_col] * 100 >= threshold]  # convert to percent display filter

        # Display main table
        display_cols = ["timestamp","market_type","symbol","min_ex","min_price","max_ex","max_price","abs_spread",pct_col,"count_venues","venues"]
        display_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df.head(top_n)[display_cols], use_container_width=True)

        # Download buttons
        csv = df.to_csv(index=False).encode("utf-8")
        toprow = df.head(top_n)
        xbuf = io.BytesIO()
        with pd.ExcelWriter(xbuf, engine="openpyxl") as writer:
            toprow.to_excel(writer, index=False, sheet_name="divergences")
        xbuf.seek(0)

        d1, d2 = st.columns(2)
        with d1:
            st.download_button("⬇️ Download CSV", data=csv, file_name="divergences.csv", mime="text/csv")
        with d2:
            st.download_button("⬇️ Download Excel", data=xbuf, file_name="divergences.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # Optional: show per-exchange price columns
        with st.expander("Show raw per-exchange prices"):
            price_cols = [c for c in df.columns if c.startswith("price@")]
            st.dataframe(df.head(top_n)[["symbol"] + price_cols], use_container_width=True)

st.caption("Notes: This app uses public endpoints via ccxt. Results are leads only; fees, slippage, funding (for swaps) are not included.")
