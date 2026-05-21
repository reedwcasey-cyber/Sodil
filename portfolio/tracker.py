"""Fetches and structures live portfolio data from Robinhood."""
import time
from functools import lru_cache
from typing import Any

import pandas as pd
import robin_stocks.robinhood as rh

from config import CACHE_TTL

_portfolio_cache: dict[str, Any] = {}
_cache_ts: float = 0.0


def _is_stale() -> bool:
    return (time.time() - _cache_ts) > CACHE_TTL


def get_portfolio() -> dict:
    """Return cached or freshly-fetched portfolio snapshot."""
    global _portfolio_cache, _cache_ts
    if not _is_stale() and _portfolio_cache:
        return _portfolio_cache

    profile = rh.profiles.load_portfolio_profile() or {}
    account = rh.profiles.load_account_profile() or {}

    _portfolio_cache = {
        "equity": float(profile.get("equity") or 0),
        "extended_hours_equity": float(profile.get("extended_hours_equity") or profile.get("equity") or 0),
        "last_core_equity": float(profile.get("last_core_equity") or 0),
        "cash": float(account.get("cash") or 0),
        "buying_power": float(account.get("buying_power") or 0),
        "total_return_pct": _total_return_pct(profile),
    }
    _cache_ts = time.time()
    return _portfolio_cache


def _total_return_pct(profile: dict) -> float:
    equity = float(profile.get("equity") or 0)
    deposits = float(profile.get("net_moving_average") or equity or 1)
    return ((equity - deposits) / deposits * 100) if deposits else 0.0


def get_positions() -> pd.DataFrame:
    """Return current open positions as a DataFrame."""
    raw = rh.account.get_open_stock_positions() or []
    rows = []
    symbols = []

    for pos in raw:
        qty = float(pos.get("quantity") or 0)
        if qty == 0:
            continue
        instrument_url = pos.get("instrument", "")
        instrument = rh.stocks.get_instrument_by_url(instrument_url) or {}
        symbol = instrument.get("symbol", "?")
        symbols.append(symbol)
        rows.append({
            "symbol": symbol,
            "quantity": qty,
            "avg_cost": float(pos.get("average_buy_price") or 0),
            "instrument_url": instrument_url,
        })

    if not rows:
        return pd.DataFrame(columns=["symbol", "quantity", "avg_cost", "current_price",
                                     "market_value", "total_return", "return_pct", "day_return_pct"])

    # Batch price fetch
    quotes = rh.stocks.get_latest_price(symbols, priceType="bid_price", includeExtendedHours=True)
    for i, row in enumerate(rows):
        price = float(quotes[i]) if quotes and i < len(quotes) and quotes[i] else 0.0
        row["current_price"] = price
        row["market_value"] = price * row["quantity"]
        cost_basis = row["avg_cost"] * row["quantity"]
        row["total_return"] = row["market_value"] - cost_basis
        row["return_pct"] = ((price - row["avg_cost"]) / row["avg_cost"] * 100) if row["avg_cost"] else 0.0

    df = pd.DataFrame(rows)

    # Day return: compare to previous close
    try:
        fundamentals = rh.stocks.get_fundamentals(symbols) or []
        prev_closes = {symbols[i]: float(f.get("low") or 0) for i, f in enumerate(fundamentals)}
        quotes_full = rh.stocks.get_quotes(symbols) or []
        prev_map = {}
        for i, q in enumerate(quotes_full):
            if q:
                prev_map[symbols[i]] = float(q.get("adjusted_previous_close") or 0)
        df["day_return_pct"] = df.apply(
            lambda r: ((r["current_price"] - prev_map.get(r["symbol"], r["current_price"])) /
                       prev_map.get(r["symbol"], r["current_price"]) * 100)
            if prev_map.get(r["symbol"], 0) > 0 else 0.0,
            axis=1,
        )
    except Exception:
        df["day_return_pct"] = 0.0

    return df.sort_values("market_value", ascending=False).reset_index(drop=True)


def get_order_history(limit: int = 500) -> pd.DataFrame:
    """Return completed buy/sell orders with enriched fields."""
    orders = rh.orders.get_all_stock_orders(info=None) or []
    rows = []
    for o in orders[:limit]:
        if o.get("state") != "filled":
            continue
        symbol = (o.get("symbol") or
                  (rh.stocks.get_instrument_by_url(o.get("instrument", "")) or {}).get("symbol", "?"))
        side = o.get("side", "")
        qty = float(o.get("quantity") or 0)
        avg_price = float(o.get("average_price") or o.get("price") or 0)
        rows.append({
            "symbol": symbol,
            "side": side,
            "quantity": qty,
            "avg_price": avg_price,
            "notional": qty * avg_price,
            "created_at": pd.to_datetime(o.get("created_at")),
            "updated_at": pd.to_datetime(o.get("last_transaction_at") or o.get("updated_at")),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("created_at").reset_index(drop=True)
    return df


def get_option_positions() -> pd.DataFrame:
    """Return open options positions."""
    raw = rh.options.get_open_option_positions() or []
    rows = []
    for pos in raw:
        qty = float(pos.get("quantity") or 0)
        if qty == 0:
            continue
        rows.append({
            "symbol": pos.get("chain_symbol", "?"),
            "type": pos.get("type", ""),
            "quantity": qty,
            "avg_cost": float(pos.get("average_price") or 0),
            "trade_value_multiplier": float(pos.get("trade_value_multiplier") or 100),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()
