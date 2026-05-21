"""Composite factor scoring and opportunity ranking."""

import numpy as np
import pandas as pd
from typing import Optional

from .momentum import momentum_score, rsi, macd_signal, bollinger_position, volume_trend
from .value import composite_value_score, analyst_upside
from .quality import quality_score


FACTOR_WEIGHTS = {
    "momentum": 0.35,
    "value": 0.30,
    "quality": 0.25,
    "technical": 0.10,
}


def zscore_series(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score normalization."""
    mean, std = s.mean(), s.std()
    if std == 0:
        return s - mean
    return (s - mean) / std


def compute_composite_score(
    ticker: str,
    fundamentals: dict,
    price_df: pd.DataFrame,
    weights: Optional[dict] = None,
) -> dict:
    """
    Compute all factor scores for a single stock.
    Returns dict with individual scores and composite.
    """
    w = weights or FACTOR_WEIGHTS

    result = {"ticker": ticker}

    # Momentum factor
    mom = momentum_score(price_df) if not price_df.empty else np.nan
    result["momentum_raw"] = mom

    # Technical signals
    empty = price_df.empty
    rsi_val = rsi(price_df) if not empty else np.nan
    macd_val = macd_signal(price_df) if not empty else np.nan
    boll_val = bollinger_position(price_df) if not empty else np.nan
    vol_trend = volume_trend(price_df) if not empty else np.nan

    # RSI: ~50 is neutral, <30 oversold (buy), >70 overbought (sell)
    # We want stocks with RSI 40-65: recovering or strong but not overextended
    rsi_score = np.nan
    if rsi_val is not None and not np.isnan(rsi_val):
        if 40 <= rsi_val <= 65:
            rsi_score = 0.5
        elif rsi_val < 40:
            rsi_score = float(rsi_val / 40 - 0.5)
        else:
            rsi_score = float(-(rsi_val - 65) / 35)

    tech_components = [
        x for x in [macd_val, boll_val, rsi_score, vol_trend]
        if x is not None and not np.isnan(x)
    ]
    technical = float(np.mean(tech_components)) if tech_components else np.nan

    result["technical_raw"] = technical
    result["rsi"] = rsi_val
    result["macd"] = macd_val

    # Value factor
    value = composite_value_score(fundamentals)
    result["value_raw"] = value

    # Quality factor
    quality = quality_score(fundamentals)
    result["quality_raw"] = quality

    # Analyst upside as bonus
    upside = analyst_upside(fundamentals)
    result["analyst_upside"] = upside

    # Composite (pre-normalization)
    components = {}
    if not np.isnan(mom):
        components["momentum"] = mom
    if not np.isnan(technical):
        components["technical"] = technical
    if not np.isnan(value):
        components["value"] = value
    if not np.isnan(quality):
        components["quality"] = quality

    if not components:
        result["composite_raw"] = np.nan
    else:
        total_w = sum(w[k] for k in components)
        composite = sum(components[k] * w[k] for k in components) / total_w
        result["composite_raw"] = composite

    result["factor_breakdown"] = components
    return result


def rank_opportunities(
    scores: list[dict],
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Cross-sectionally normalize and rank all stocks.
    Returns sorted DataFrame of top opportunities.
    """
    df = pd.DataFrame(scores).set_index("ticker")

    # Z-score each factor across the universe for fair comparison
    for col in ["momentum_raw", "value_raw", "quality_raw", "technical_raw"]:
        if col in df.columns:
            valid = df[col].dropna()
            if len(valid) > 1:
                df[col.replace("_raw", "_z")] = zscore_series(df[col])
            else:
                df[col.replace("_raw", "_z")] = 0.0

    # Composite z-score
    z_cols = [c for c in df.columns if c.endswith("_z")]
    w = FACTOR_WEIGHTS
    factor_map = {
        "momentum_z": "momentum",
        "value_z": "value",
        "quality_z": "quality",
        "technical_z": "technical",
    }

    if z_cols:
        weighted_sum = pd.Series(0.0, index=df.index)
        total_w = 0.0
        for col in z_cols:
            factor = factor_map.get(col, col)
            weight = w.get(factor, 0.1)
            weighted_sum += df[col].fillna(0) * weight
            total_w += weight
        df["composite_score"] = weighted_sum / total_w

    df = df.sort_values("composite_score", ascending=False)

    # Add rank
    df["rank"] = range(1, len(df) + 1)

    return df.head(top_n)
