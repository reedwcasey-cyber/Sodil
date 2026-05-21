"""
Stock screener — generates a candidate universe based on your edge profile.
Uses a curated list of liquid, well-known tickers across sectors as the base universe.
"""
from __future__ import annotations

import warnings
from typing import Optional

import pandas as pd

from config import MIN_AVG_VOLUME, MIN_MARKET_CAP, MIN_PRICE, MAX_PRICE
from market.data import get_ticker_fundamentals, get_price_history, compute_technicals

warnings.filterwarnings("ignore")

# Broad universe across sectors (liquid, well-known names)
UNIVERSE: dict[str, list[str]] = {
    "Technology": [
        "AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMZN", "AMD", "INTC", "QCOM", "TSM",
        "AVGO", "TXN", "MU", "AMAT", "LRCX", "KLAC", "SNPS", "CDNS", "ADBE", "CRM",
        "NOW", "SNOW", "PLTR", "DDOG", "ZS", "NET", "CRWD", "OKTA", "PANW", "FTNT",
        "INTU", "VEEV", "WDAY", "TEAM", "ZM", "DOCN", "HUBS", "MDB", "COUP",
    ],
    "Healthcare": [
        "UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY", "AMGN",
        "ISRG", "SYK", "GILD", "REGN", "VRTX", "BIIB", "MRNA", "BNTX", "RVMD",
        "DXCM", "IDXX", "IQV", "CRL", "CTLT", "ZBH", "BSX", "EW", "HOLX",
    ],
    "Consumer Cyclical": [
        "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TGT", "BKNG", "MAR", "HLT",
        "ABNB", "UBER", "LYFT", "DASH", "ETSY", "EBAY", "W", "RH", "WSM",
        "ANF", "AEO", "GPS", "PVH", "LULU", "EL",
    ],
    "Financial Services": [
        "BRK-B", "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SCHW",
        "CB", "PGR", "TRV", "MMC", "AON", "MET", "PRU", "AFL", "ALL", "HIG",
        "COIN", "SQ", "PYPL", "V", "MA",
    ],
    "Communication Services": [
        "GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "CHTR", "TMUS",
        "SNAP", "PINS", "RDDT", "SPOT", "WBD", "PARA",
    ],
    "Industrials": [
        "CAT", "DE", "HON", "GE", "BA", "LMT", "RTX", "NOC", "GD", "LHX",
        "FDX", "UPS", "XPO", "JBHT", "ODFL", "SAIA", "EMR", "ETN", "PH",
        "AME", "ROK", "XYL", "CARR", "OTIS", "TT", "IR",
    ],
    "Energy": [
        "XOM", "CVX", "COP", "EOG", "PXD", "MPC", "PSX", "VLO", "HES", "DVN",
        "OXY", "APA", "HAL", "SLB", "BKR", "FANG", "AR", "EQT",
    ],
    "Basic Materials": [
        "LIN", "APD", "DD", "ECL", "SHW", "PPG", "NEM", "FCX", "NUE", "STLD",
        "AA", "ALB", "LTHM", "MP", "CF", "MOS", "FMC",
    ],
    "Real Estate": [
        "AMT", "PLD", "CCI", "EQIX", "PSA", "EXR", "WELL", "ARE", "VTR",
        "SPG", "O", "VICI", "MPW", "NNN", "WPC",
    ],
    "Consumer Defensive": [
        "PG", "KO", "PEP", "WMT", "COST", "PM", "MO", "MDLZ", "GIS", "HSY",
        "CLX", "CHD", "CL", "KMB", "SYY", "ADM", "BG",
    ],
}


def build_candidate_universe(
    target_sectors: Optional[list[str]] = None,
    target_mcap: Optional[str] = None,
    max_per_sector: int = 15,
) -> list[str]:
    """Return a filtered list of candidate tickers matching the edge profile."""
    sectors = target_sectors or list(UNIVERSE.keys())
    candidates = []
    for sector in sectors:
        tickers = UNIVERSE.get(sector, [])[:max_per_sector]
        candidates.extend(tickers)
    return list(dict.fromkeys(candidates))  # dedup, preserve order


def screen_candidates(
    symbols: list[str],
    edge_profile: dict,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Screen a list of symbols and return a DataFrame with fundamentals + technicals,
    filtered by the user's edge profile.
    """
    rows = []
    total = len(symbols)

    for i, sym in enumerate(symbols):
        if progress_callback:
            progress_callback(i, total, sym)

        try:
            fund = get_ticker_fundamentals(sym)
            price = fund.get("current_price", 0) or 0

            if price < MIN_PRICE or price > MAX_PRICE:
                continue
            if (fund.get("market_cap", 0) or 0) < MIN_MARKET_CAP:
                continue
            if (fund.get("avg_volume", 0) or 0) < MIN_AVG_VOLUME:
                continue

            hist = get_price_history(sym, period="6mo")
            if hist.empty:
                continue

            tech = compute_technicals(hist)
            if not tech:
                continue

            row = {**fund, **tech}
            rows.append(row)
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)
