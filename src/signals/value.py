"""Value signals: fundamental valuation metrics."""

import numpy as np
import pandas as pd
from typing import Optional


def composite_value_score(fundamentals: dict, sector_medians: Optional[dict] = None) -> float:
    """
    Composite value score from multiple valuation metrics.
    Lower multiples → higher value score.
    Returns z-score style value composite (higher = cheaper).
    """
    scores = []

    # P/E ratio: lower is cheaper
    pe = fundamentals.get("pe_ratio")
    if pe and 0 < pe < 200:
        scores.append(-np.log(pe))  # negative log: lower PE = higher score

    # Forward P/E
    fpe = fundamentals.get("forward_pe")
    if fpe and 0 < fpe < 200:
        scores.append(-np.log(fpe))

    # P/B ratio
    pb = fundamentals.get("pb_ratio")
    if pb and 0 < pb < 50:
        scores.append(-np.log(pb))

    # P/S ratio
    ps = fundamentals.get("ps_ratio")
    if ps and 0 < ps < 100:
        scores.append(-np.log(ps))

    # EV/EBITDA
    ev_ebitda = fundamentals.get("ev_ebitda")
    if ev_ebitda and 0 < ev_ebitda < 100:
        scores.append(-np.log(ev_ebitda))

    # PEG ratio: ideally < 1
    peg = fundamentals.get("peg_ratio")
    if peg and 0 < peg < 10:
        scores.append(-peg)

    # Analyst upside
    target = fundamentals.get("analyst_target")
    price = fundamentals.get("current_price")
    if target and price and price > 0:
        upside = (target - price) / price
        scores.append(upside)

    if not scores:
        return np.nan

    return float(np.mean(scores))


def price_to_52w_low_ratio(fundamentals: dict) -> float:
    """How far above 52w low is current price? Lower = more oversold/value."""
    low = fundamentals.get("52w_low")
    price = fundamentals.get("current_price")
    if low and price and low > 0:
        return float((price - low) / low)
    return np.nan


def analyst_upside(fundamentals: dict) -> float:
    """Percentage upside to analyst mean price target."""
    target = fundamentals.get("analyst_target")
    price = fundamentals.get("current_price")
    if target and price and price > 0:
        return float((target - price) / price)
    return np.nan


def valuation_summary(fundamentals: dict) -> dict:
    """Return structured valuation metrics for display."""
    price = fundamentals.get("current_price", 0) or 0
    target = fundamentals.get("analyst_target", 0) or 0
    return {
        "P/E": fundamentals.get("pe_ratio"),
        "Fwd P/E": fundamentals.get("forward_pe"),
        "P/B": fundamentals.get("pb_ratio"),
        "P/S": fundamentals.get("ps_ratio"),
        "EV/EBITDA": fundamentals.get("ev_ebitda"),
        "PEG": fundamentals.get("peg_ratio"),
        "Analyst Target": f"${target:.2f}" if target else "N/A",
        "Upside to Target": f"{analyst_upside(fundamentals)*100:.1f}%" if analyst_upside(fundamentals) is not np.nan else "N/A",
    }
