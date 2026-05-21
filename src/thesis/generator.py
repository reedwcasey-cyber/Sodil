"""Investment thesis generation from quantitative signals."""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

from ..signals.value import valuation_summary
from ..signals.quality import quality_summary, financial_health_grade


@dataclass
class InvestmentThesis:
    ticker: str
    name: str
    sector: str
    composite_score: float
    conviction: str          # HIGH / MEDIUM / LOW
    expected_return_1y: float
    expected_return_bear: float
    expected_return_bull: float
    primary_catalysts: list[str] = field(default_factory=list)
    key_risks: list[str] = field(default_factory=list)
    valuation: dict = field(default_factory=dict)
    quality_metrics: dict = field(default_factory=dict)
    narrative: str = ""
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    position_size_pct: float = 0.05
    time_horizon_months: int = 12


def _conviction_level(composite_score: float, z_score: float) -> str:
    if composite_score > 0.5 and z_score > 1.5:
        return "HIGH"
    elif composite_score > 0.2 and z_score > 0.5:
        return "MEDIUM"
    else:
        return "LOW"


def _estimate_returns(
    fundamentals: dict,
    momentum_z: float,
    value_z: float,
    quality_z: float,
    beta: Optional[float],
) -> tuple[float, float, float]:
    """
    Estimate base/bear/bull 1-year expected returns using multi-factor model.
    """
    beta = beta or 1.0

    # Base: market return + factor alpha
    market_return = 0.10  # S&P 500 long-run average
    factor_alpha = (
        0.03 * np.tanh(momentum_z) +
        0.025 * np.tanh(value_z) +
        0.02 * np.tanh(quality_z)
    )

    # Analyst upside bonus
    analyst_target = fundamentals.get("analyst_target")
    current_price = fundamentals.get("current_price")
    if analyst_target and current_price and current_price > 0:
        analyst_upside = (analyst_target - current_price) / current_price
        factor_alpha += min(analyst_upside * 0.5, 0.15)  # cap contribution

    base_return = market_return * beta + factor_alpha

    # Bear/bull scenarios: apply volatility based on beta
    vol_premium = 0.15 * beta
    bear_return = base_return - 1.5 * vol_premium
    bull_return = base_return + 2.0 * vol_premium

    return base_return, bear_return, bull_return


def _generate_catalysts(
    fundamentals: dict,
    momentum_z: float,
    value_z: float,
    quality_z: float,
    rsi: float,
    analyst_upside: float,
) -> list[str]:
    catalysts = []

    if momentum_z > 1.0:
        catalysts.append(f"Strong price momentum (top-tier vs. peers, z={momentum_z:.1f})")

    rev_growth = fundamentals.get("revenue_growth")
    if rev_growth and rev_growth > 0.15:
        catalysts.append(f"Accelerating revenue growth of {rev_growth*100:.1f}% YoY")

    earnings_growth = fundamentals.get("earnings_growth")
    if earnings_growth and earnings_growth > 0.20:
        catalysts.append(f"Strong earnings growth of {earnings_growth*100:.1f}% YoY")

    if value_z > 1.0:
        catalysts.append(f"Attractive valuation vs. sector peers (value z={value_z:.1f})")

    fwd_pe = fundamentals.get("forward_pe")
    pe = fundamentals.get("pe_ratio")
    if fwd_pe and pe and fwd_pe < pe * 0.85:
        catalysts.append(f"Earnings expansion expected: trailing P/E {pe:.1f}x → fwd {fwd_pe:.1f}x")

    if quality_z > 1.0:
        catalysts.append(f"Above-average business quality (ROE, margins, cash flow)")

    roe = fundamentals.get("roe")
    if roe and roe > 0.20:
        catalysts.append(f"High-quality compounding business with {roe*100:.1f}% ROE")

    if analyst_upside and analyst_upside > 0.15:
        catalysts.append(f"Consensus analyst target implies {analyst_upside*100:.1f}% upside")

    if rsi and rsi < 40:
        catalysts.append(f"Technically oversold (RSI={rsi:.0f}), mean-reversion opportunity")

    short_ratio = fundamentals.get("short_ratio")
    if short_ratio and short_ratio > 8:
        catalysts.append(f"High short interest ({short_ratio:.1f} days to cover) — squeeze potential")

    fcf = fundamentals.get("free_cash_flow")
    if fcf and fcf > 0:
        catalysts.append("Positive free cash flow generation")

    return catalysts[:6]


def _generate_risks(
    fundamentals: dict,
    momentum_z: float,
    value_z: float,
    quality_z: float,
    rsi: float,
) -> list[str]:
    risks = []

    beta = fundamentals.get("beta")
    if beta and beta > 1.5:
        risks.append(f"High beta ({beta:.2f}) amplifies market drawdowns")

    de = fundamentals.get("debt_to_equity")
    if de and de > 150:
        risks.append(f"Elevated leverage (D/E={de:.0f}%) increases rate-sensitivity")

    pe = fundamentals.get("pe_ratio")
    if pe and pe > 40:
        risks.append(f"Demanding valuation (P/E={pe:.0f}x) leaves limited margin of safety")

    if momentum_z < 0:
        risks.append("Negative price momentum — potential continued underperformance")

    rev_growth = fundamentals.get("revenue_growth")
    if rev_growth and rev_growth < 0:
        risks.append(f"Revenue contraction of {rev_growth*100:.1f}% signals demand weakness")

    profit_margin = fundamentals.get("profit_margin")
    if profit_margin and profit_margin < 0.05:
        risks.append(f"Thin profit margins ({profit_margin*100:.1f}%) leave little room for error")

    current_ratio = fundamentals.get("current_ratio")
    if current_ratio and current_ratio < 1.0:
        risks.append(f"Current ratio below 1.0x ({current_ratio:.2f}x) — near-term liquidity risk")

    if rsi and rsi > 75:
        risks.append(f"Technically overbought (RSI={rsi:.0f}), short-term pullback risk")

    risks.append("General market risk and macro uncertainty")
    risks.append("Execution risk — thesis may take longer than expected to materialize")

    return risks[:6]


def _generate_narrative(
    ticker: str,
    fundamentals: dict,
    base_return: float,
    catalysts: list[str],
    conviction: str,
) -> str:
    name = fundamentals.get("name", ticker)
    sector = fundamentals.get("sector", "Unknown")
    pe = fundamentals.get("pe_ratio")
    roe = fundamentals.get("roe")

    pe_str = f" at {pe:.1f}x earnings" if pe else ""
    roe_str = f" with {roe*100:.1f}% ROE" if roe else ""

    top_catalyst = catalysts[0] if catalysts else "improving fundamentals"

    return (
        f"{name} ({ticker}) is a {sector} company trading{pe_str}{roe_str}. "
        f"Our quantitative model assigns {conviction} conviction with an estimated "
        f"{base_return*100:.1f}% 1-year expected return. "
        f"The primary driver is: {top_catalyst}. "
        f"The position is sized conservatively to manage downside risk while "
        f"maintaining meaningful exposure to the upside thesis."
    )


def generate_thesis(
    ticker: str,
    fundamentals: dict,
    score_row: pd.Series,
) -> InvestmentThesis:
    """
    Generate a complete investment thesis from factor scores and fundamentals.
    """
    composite = score_row.get("composite_score", 0) or 0
    mom_z = score_row.get("momentum_z", 0) or 0
    val_z = score_row.get("value_z", 0) or 0
    qual_z = score_row.get("quality_z", 0) or 0
    rsi_val = score_row.get("rsi", 50) or 50

    conviction = _conviction_level(composite, mom_z + val_z + qual_z)

    base_ret, bear_ret, bull_ret = _estimate_returns(
        fundamentals, mom_z, val_z, qual_z,
        fundamentals.get("beta")
    )

    upside = score_row.get("analyst_upside", 0) or 0
    catalysts = _generate_catalysts(fundamentals, mom_z, val_z, qual_z, rsi_val, upside)
    risks = _generate_risks(fundamentals, mom_z, val_z, qual_z, rsi_val)
    narrative = _generate_narrative(ticker, fundamentals, base_ret, catalysts, conviction)

    current_price = fundamentals.get("current_price") or 0
    target_price = current_price * (1 + base_ret) if current_price else None

    # Conviction-based position sizing
    position_map = {"HIGH": 0.08, "MEDIUM": 0.05, "LOW": 0.03}
    position_size = position_map[conviction]

    # Stop loss: 1x ATR proxy (use 8% below entry for simplicity)
    stop_loss = current_price * 0.92 if current_price else None

    return InvestmentThesis(
        ticker=ticker,
        name=fundamentals.get("name", ticker),
        sector=fundamentals.get("sector", "Unknown"),
        composite_score=composite,
        conviction=conviction,
        expected_return_1y=base_ret,
        expected_return_bear=bear_ret,
        expected_return_bull=bull_ret,
        primary_catalysts=catalysts,
        key_risks=risks,
        valuation=valuation_summary(fundamentals),
        quality_metrics=quality_summary(fundamentals),
        narrative=narrative,
        entry_price=current_price,
        target_price=target_price,
        stop_loss=stop_loss,
        position_size_pct=position_size,
    )
