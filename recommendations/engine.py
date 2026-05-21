"""
Recommendation engine — scores screened candidates against your historical edge.
Uses a multi-factor scoring model calibrated from your trade process stats.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import SIMILARITY_TOP_N

warnings.filterwarnings("ignore")


def score_candidates(
    candidates: pd.DataFrame,
    process_stats: dict,
    current_positions: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Score each candidate against the user's edge profile.
    Returns top N sorted by composite score.
    """
    if candidates.empty or not process_stats:
        return pd.DataFrame()

    df = candidates.copy()
    edge = process_stats.get("edge_profile", {})

    # ── Remove already-owned positions ───────────────────────────────────────
    if current_positions is not None and not current_positions.empty:
        owned = set(current_positions["symbol"].str.upper())
        df = df[~df["symbol"].str.upper().isin(owned)]

    if df.empty:
        return pd.DataFrame()

    df = df.reset_index(drop=True)

    # ── Factor 1: Sector alignment ────────────────────────────────────────────
    best_sector = edge.get("best_sector", "")
    by_sector = process_stats.get("by_sector", pd.DataFrame())
    sector_score_map = _build_sector_score_map(by_sector)
    df["f_sector"] = df.get("sector", pd.Series(["Unknown"] * len(df))).map(
        lambda s: sector_score_map.get(s, 0.5)
    )

    # ── Factor 2: RSI alignment ───────────────────────────────────────────────
    best_rsi_band = edge.get("best_rsi_band", "Neutral(45-55)")
    df["f_rsi"] = df.get("rsi", pd.Series([50.0] * len(df))).apply(
        lambda r: _rsi_score(r, best_rsi_band)
    )

    # ── Factor 3: Trend alignment ─────────────────────────────────────────────
    best_regime = (
        process_stats.get("by_entry_regime", pd.DataFrame())
    )
    df["f_trend"] = df.get("trend", pd.Series(["Sideways"] * len(df))).apply(
        lambda t: _trend_score(t, best_regime)
    )

    # ── Factor 4: MACD momentum ───────────────────────────────────────────────
    df["f_macd"] = df.get("macd_bullish", pd.Series([False] * len(df))).astype(float)

    # ── Factor 5: Fundamental quality ─────────────────────────────────────────
    df["f_fundamental"] = _score_fundamentals(df, process_stats)

    # ── Factor 6: Momentum alignment ─────────────────────────────────────────
    df["f_momentum"] = _score_momentum(df, process_stats)

    # ── Composite score (weighted sum) ────────────────────────────────────────
    weights = {
        "f_sector": 0.20,
        "f_rsi": 0.15,
        "f_trend": 0.20,
        "f_macd": 0.15,
        "f_fundamental": 0.15,
        "f_momentum": 0.15,
    }
    df["score"] = sum(df[f] * w for f, w in weights.items())

    # Normalize score to 0-100
    scaler = MinMaxScaler(feature_range=(0, 100))
    if len(df) > 1:
        df["score"] = scaler.fit_transform(df[["score"]]).flatten()
    else:
        df["score"] = 75.0

    df["score"] = df["score"].round(1)

    # ── Build rationale strings ───────────────────────────────────────────────
    df["rationale"] = df.apply(lambda r: _build_rationale(r, edge, process_stats), axis=1)

    # ── Sort and return top N ─────────────────────────────────────────────────
    result_cols = [
        "symbol", "name", "sector", "score", "rationale",
        "current_price", "market_cap", "rsi", "trend", "macd_bullish",
        "momentum_20d", "momentum_5d", "volatility_20d",
        "pe_ratio", "fwd_pe", "revenue_growth", "beta",
        "52w_high", "52w_low",
    ]
    available = [c for c in result_cols if c in df.columns]
    return df[available].sort_values("score", ascending=False).head(SIMILARITY_TOP_N).reset_index(drop=True)


def score_options_candidates(
    candidates: pd.DataFrame,
    process_stats: dict,
    strategy: str = "auto",
) -> pd.DataFrame:
    """
    Score stocks specifically for options plays.
    strategy: "auto", "calls", "puts", "covered_call", "cash_secured_put"
    """
    if candidates.empty:
        return pd.DataFrame()

    df = score_candidates(candidates, process_stats)
    if df.empty:
        return df

    # Add options-specific scoring
    df["options_strategy"] = df.apply(
        lambda r: _recommend_options_strategy(r, strategy, process_stats),
        axis=1,
    )
    df["options_rationale"] = df.apply(_options_rationale, axis=1)
    return df


def _build_sector_score_map(by_sector: pd.DataFrame) -> dict[str, float]:
    if by_sector is None or by_sector.empty:
        return {}
    col = by_sector.columns[0]  # sector column
    pnl_col = "avg_pnl_pct" if "avg_pnl_pct" in by_sector.columns else None
    if not pnl_col:
        return {}
    pnl = by_sector[pnl_col]
    if pnl.max() == pnl.min():
        return {row[col]: 0.5 for _, row in by_sector.iterrows()}
    norm = (pnl - pnl.min()) / (pnl.max() - pnl.min())
    return dict(zip(by_sector[col].astype(str), norm.round(3)))


def _rsi_score(rsi: float, best_band: str) -> float:
    """Score how well this RSI matches the user's historically profitable entry band."""
    band_centers = {
        "Oversold(<30)": 20,
        "Low(30-45)": 37,
        "Neutral(45-55)": 50,
        "High(55-70)": 62,
        "Overbought(>70)": 80,
    }
    target = band_centers.get(best_band, 50)
    distance = abs(rsi - target)
    return max(0.0, 1.0 - distance / 50.0)


def _trend_score(trend: str, best_regime_df: pd.DataFrame) -> float:
    if best_regime_df is None or best_regime_df.empty:
        return 0.5
    col = best_regime_df.columns[0]
    pnl_col = "avg_pnl_pct" if "avg_pnl_pct" in best_regime_df.columns else None
    if not pnl_col:
        return 0.5

    row = best_regime_df[best_regime_df[col].astype(str) == trend]
    if row.empty:
        return 0.3
    raw = float(row.iloc[0][pnl_col])
    return min(1.0, max(0.0, 0.5 + raw / 20.0))


def _score_fundamentals(df: pd.DataFrame, stats: dict) -> pd.Series:
    """Score fundamental quality based on profitability and growth."""
    scores = pd.Series([0.5] * len(df), index=df.index)

    if "pe_ratio" in df.columns:
        pe = pd.to_numeric(df["pe_ratio"], errors="coerce").fillna(30)
        scores += (pe < 25).astype(float) * 0.2
        scores -= (pe > 60).astype(float) * 0.2

    if "revenue_growth" in df.columns:
        rg = pd.to_numeric(df["revenue_growth"], errors="coerce").fillna(0)
        scores += (rg > 0.15).astype(float) * 0.15
        scores += (rg > 0.30).astype(float) * 0.10

    if "profit_margin" in df.columns:
        pm = pd.to_numeric(df["profit_margin"], errors="coerce").fillna(0)
        scores += (pm > 0.10).astype(float) * 0.15

    return scores.clip(0, 1)


def _score_momentum(df: pd.DataFrame, stats: dict) -> pd.Series:
    """Score momentum based on what kind of momentum worked for the user."""
    avg_pnl = stats.get("avg_pnl_pct", 0)
    scores = pd.Series([0.5] * len(df), index=df.index)

    if "momentum_20d" in df.columns:
        m20 = pd.to_numeric(df["momentum_20d"], errors="coerce").fillna(0)
        if avg_pnl > 5:  # user profits from momentum
            scores += (m20 > 5).astype(float) * 0.3
            scores += (m20 > 15).astype(float) * 0.2
        else:  # user profits from reversals
            scores += (m20 < -5).astype(float) * 0.2
            scores += ((m20 > -15) & (m20 < -3)).astype(float) * 0.1

    return scores.clip(0, 1)


def _recommend_options_strategy(row: pd.Series, strategy: str, stats: dict) -> str:
    rsi = row.get("rsi", 50)
    trend = str(row.get("trend", ""))
    vol = row.get("volatility_20d", 20)

    if strategy != "auto":
        return strategy

    if "Uptrend" in trend and rsi < 60:
        return "Long Call" if vol < 40 else "Bull Call Spread"
    elif "Downtrend" in trend and rsi > 60:
        return "Long Put" if vol < 40 else "Bear Put Spread"
    elif rsi < 35:
        return "Cash-Secured Put"
    elif rsi > 70 and vol > 35:
        return "Covered Call"
    else:
        return "Bull Call Spread"


def _options_rationale(row: pd.Series) -> str:
    strategy = row.get("options_strategy", "")
    rsi = row.get("rsi", 50)
    trend = row.get("trend", "")
    vol = row.get("volatility_20d", 0)

    parts = [f"{strategy}:"]
    if "Call" in strategy:
        parts.append(f"uptrend ({trend}), RSI={rsi}")
    elif "Put" in strategy:
        parts.append(f"entry on weakness, RSI={rsi}")
    if vol:
        parts.append(f"vol={vol:.1f}%")
    return " ".join(parts)


def _build_rationale(row: pd.Series, edge: dict, stats: dict) -> str:
    parts = []

    # Sector match
    best_s = edge.get("best_sector", "")
    if best_s and str(row.get("sector", "")) == best_s:
        wr = edge.get("best_sector_win_rate", 0)
        parts.append(f"Top sector ({best_s}, {wr:.0f}% win rate)")

    # Trend match
    best_hold = edge.get("best_hold_duration", "")
    trend = str(row.get("trend", ""))
    if "Uptrend" in trend:
        parts.append(f"In uptrend ({trend})")

    # RSI
    rsi = row.get("rsi", 50)
    best_band = edge.get("best_rsi_band", "")
    if "Oversold" in best_band and rsi < 35:
        parts.append(f"RSI oversold ({rsi})")
    elif "Low" in best_band and 30 <= rsi <= 45:
        parts.append(f"RSI in sweet spot ({rsi})")
    elif rsi < 55:
        parts.append(f"RSI={rsi}")

    # MACD
    if row.get("macd_bullish", False):
        parts.append("MACD bullish")

    # Growth
    rev_growth = row.get("revenue_growth", None)
    if rev_growth and rev_growth > 0.20:
        parts.append(f"Rev growth {rev_growth*100:.0f}%")

    # Momentum
    mom = row.get("momentum_20d", None)
    if mom and mom > 5:
        parts.append(f"20d momentum +{mom:.1f}%")

    return " | ".join(parts) if parts else "Matches your historical edge"
