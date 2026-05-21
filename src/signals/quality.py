"""Quality signals: profitability, efficiency, balance sheet health."""

import numpy as np
import pandas as pd


def quality_score(fundamentals: dict) -> float:
    """
    Composite quality score. Higher = better quality business.
    Combines profitability, efficiency, and financial health metrics.
    """
    scores = []

    # Profitability
    roe = fundamentals.get("roe")
    if roe is not None and not np.isnan(roe):
        # ROE > 15% is excellent
        scores.append(np.tanh(roe / 0.15))

    roa = fundamentals.get("roa")
    if roa is not None and not np.isnan(roa):
        scores.append(np.tanh(roa / 0.08))

    profit_margin = fundamentals.get("profit_margin")
    if profit_margin is not None and not np.isnan(profit_margin):
        scores.append(np.tanh(profit_margin / 0.15))

    operating_margin = fundamentals.get("operating_margin")
    if operating_margin is not None and not np.isnan(operating_margin):
        scores.append(np.tanh(operating_margin / 0.20))

    # Growth
    rev_growth = fundamentals.get("revenue_growth")
    if rev_growth is not None and not np.isnan(rev_growth):
        scores.append(np.tanh(rev_growth / 0.15))

    earnings_growth = fundamentals.get("earnings_growth")
    if earnings_growth is not None and not np.isnan(earnings_growth):
        scores.append(np.tanh(earnings_growth / 0.20))

    # Balance sheet health: lower D/E is better
    de = fundamentals.get("debt_to_equity")
    if de is not None and not np.isnan(de) and de >= 0:
        scores.append(np.tanh(-de / 100))  # penalize high leverage

    # Liquidity
    current_ratio = fundamentals.get("current_ratio")
    if current_ratio is not None and not np.isnan(current_ratio):
        scores.append(min(current_ratio / 2.0, 1.0) - 0.5)  # ideal ~2x

    # Free cash flow (positive is good)
    fcf = fundamentals.get("free_cash_flow")
    if fcf is not None:
        scores.append(1.0 if fcf > 0 else -0.5)

    if not scores:
        return np.nan

    return float(np.mean(scores))


def financial_health_grade(fundamentals: dict) -> str:
    """Letter grade for financial health."""
    score = quality_score(fundamentals)
    if np.isnan(score):
        return "N/A"
    if score >= 0.6:
        return "A"
    elif score >= 0.35:
        return "B"
    elif score >= 0.1:
        return "C"
    elif score >= -0.15:
        return "D"
    else:
        return "F"


def short_squeeze_potential(fundamentals: dict) -> float:
    """
    Short squeeze potential score based on short ratio.
    Higher short ratio = higher squeeze potential (contrarian signal).
    """
    short_ratio = fundamentals.get("short_ratio")
    if short_ratio is None or np.isnan(short_ratio):
        return 0.0
    # Days to cover > 10 is high short interest
    return float(min(short_ratio / 10.0, 1.0))


def quality_summary(fundamentals: dict) -> dict:
    """Return structured quality metrics for display."""
    return {
        "ROE": f"{fundamentals.get('roe', 0)*100:.1f}%" if fundamentals.get('roe') else "N/A",
        "ROA": f"{fundamentals.get('roa', 0)*100:.1f}%" if fundamentals.get('roa') else "N/A",
        "Profit Margin": f"{fundamentals.get('profit_margin', 0)*100:.1f}%" if fundamentals.get('profit_margin') else "N/A",
        "Revenue Growth": f"{fundamentals.get('revenue_growth', 0)*100:.1f}%" if fundamentals.get('revenue_growth') else "N/A",
        "Earnings Growth": f"{fundamentals.get('earnings_growth', 0)*100:.1f}%" if fundamentals.get('earnings_growth') else "N/A",
        "D/E Ratio": f"{fundamentals.get('debt_to_equity', 0):.1f}" if fundamentals.get('debt_to_equity') is not None else "N/A",
        "Current Ratio": f"{fundamentals.get('current_ratio', 0):.2f}" if fundamentals.get('current_ratio') else "N/A",
        "Quality Grade": financial_health_grade(fundamentals),
    }
