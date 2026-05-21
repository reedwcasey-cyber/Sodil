"""Stock universe management and filtering."""

from __future__ import annotations
import logging
from typing import Optional

import pandas as pd
import numpy as np

from .fetcher import get_sp500_tickers, fetch_batch_fundamentals, fetch_batch_prices

logger = logging.getLogger(__name__)


def build_universe(
    tickers: Optional[list[str]] = None,
    min_market_cap: float = 2e9,     # $2B minimum
    min_avg_volume: float = 500_000,  # 500k shares/day
    max_tickers: int = 100,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build a screened investment universe.
    Returns DataFrame with fundamental data for all passing stocks.
    """
    if tickers is None:
        tickers = get_sp500_tickers()

    if verbose:
        print(f"[Universe] Starting with {len(tickers)} tickers...")

    # Limit for speed in demo
    if len(tickers) > max_tickers:
        tickers = tickers[:max_tickers]

    fundamentals = fetch_batch_fundamentals(tickers, max_workers=10)

    rows = []
    for ticker, data in fundamentals.items():
        mc = data.get("market_cap")
        vol = data.get("avg_volume")
        if mc and mc >= min_market_cap and vol and vol >= min_avg_volume:
            rows.append(data)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.set_index("ticker")
    if verbose:
        print(f"[Universe] {len(df)} stocks passed liquidity/size filters.")
    return df


def enrich_with_prices(
    universe: pd.DataFrame,
    period: str = "1y",
) -> dict[str, pd.DataFrame]:
    """Fetch price histories for all tickers in universe."""
    tickers = universe.index.tolist()
    return fetch_batch_prices(tickers, period=period)
