"""
Thesis challenger: systematic devil's advocate analysis.
Stress-tests assumptions, finds counterarguments, flags crowding.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from .generator import InvestmentThesis


@dataclass
class ChallengeResult:
    ticker: str
    overall_risk_score: float        # 0 (safe) to 1 (very risky)
    risk_rating: str                  # LOW / MEDIUM / HIGH / VERY HIGH
    bear_case_return: float
    bull_case_return: float
    challenges: list[str] = field(default_factory=list)
    red_flags: list[str] = field(default_factory=list)
    mitigants: list[str] = field(default_factory=list)
    revised_conviction: str = ""
    adjustment_factor: float = 1.0   # multiply expected return by this


def _check_valuation_traps(fundamentals: dict, thesis: InvestmentThesis) -> list[str]:
    """Identify value traps and valuation landmines."""
    traps = []

    pe = fundamentals.get("pe_ratio")
    fwd_pe = fundamentals.get("forward_pe")
    if pe and fwd_pe and fwd_pe > pe:
        traps.append(
            f"EARNINGS DECELERATION RISK: Forward P/E ({fwd_pe:.1f}x) > trailing ({pe:.1f}x) "
            f"implies analysts expect earnings to decline."
        )

    pb = fundamentals.get("pb_ratio")
    roe = fundamentals.get("roe")
    if pb and roe and pb > 1 and roe < 0.08:
        traps.append(
            f"VALUE TRAP RISK: Paying {pb:.1f}x book for a business generating only "
            f"{roe*100:.1f}% ROE — excess returns being destroyed."
        )

    de = fundamentals.get("debt_to_equity")
    ev_ebitda = fundamentals.get("ev_ebitda")
    if de and de > 200 and ev_ebitda and ev_ebitda > 15:
        traps.append(
            f"LEVERAGED VALUATION: EV/EBITDA of {ev_ebitda:.1f}x with D/E={de:.0f}% "
            f"means equity is a levered bet on operations."
        )

    return traps


def _check_momentum_reversal_risk(score_row: pd.Series) -> list[str]:
    """Check for momentum exhaustion and reversal signals."""
    risks = []

    rsi = score_row.get("rsi", 50) or 50
    if rsi > 70:
        risks.append(
            f"MOMENTUM EXHAUSTION: RSI={rsi:.0f} is in overbought territory. "
            f"Historical hit rate for pullbacks within 4-6 weeks is >60%."
        )

    mom_z = score_row.get("momentum_z", 0) or 0
    if mom_z > 2.5:
        risks.append(
            f"EXTREME MOMENTUM: z-score of {mom_z:.1f} puts this in top 1% of momentum — "
            f"mean-reversion risk is elevated. Crowded momentum trades unwind violently."
        )

    return risks


def _check_crowding_risk(fundamentals: dict) -> list[str]:
    """Detect potential crowded positioning."""
    risks = []

    short_ratio = fundamentals.get("short_ratio")
    if short_ratio and short_ratio < 1.0:
        risks.append(
            f"LOW SHORT INTEREST: {short_ratio:.1f} days to cover suggests high long crowding. "
            f"Crowded longs can unwind rapidly on disappointing news."
        )

    # Very high market cap in a hot sector
    mc = fundamentals.get("market_cap", 0) or 0
    sector = fundamentals.get("sector", "")
    if mc > 1e12 and sector in ["Technology", "Consumer Cyclical"]:
        risks.append(
            f"MEGA-CAP CONCENTRATION: ${mc/1e12:.1f}T market cap means institutional "
            f"ownership is near saturation — limited new buyer universe."
        )

    return risks


def _check_macro_sensitivity(fundamentals: dict) -> list[str]:
    """Flag macro-sensitive exposures."""
    risks = []

    beta = fundamentals.get("beta")
    if beta and beta > 1.8:
        risks.append(
            f"HIGH BETA RISK: Beta={beta:.2f}x means a 20% market drawdown "
            f"implies a ~{20*beta:.0f}% loss in this position. Tail risk is substantial."
        )

    de = fundamentals.get("debt_to_equity")
    if de and de > 200:
        risks.append(
            f"RATE SENSITIVITY: High leverage (D/E={de:.0f}%) means rising interest rates "
            f"directly compress earnings and equity value."
        )

    sector = fundamentals.get("sector", "")
    if sector in ["Energy", "Materials", "Industrials"]:
        risks.append(
            f"CYCLICAL EXPOSURE: {sector} stocks are highly sensitive to economic slowdowns. "
            f"In a recession, earnings can drop 30-50%."
        )

    return risks


def _check_execution_risks(fundamentals: dict, thesis: InvestmentThesis) -> list[str]:
    """Identify risks to thesis execution."""
    risks = []

    rev_growth = fundamentals.get("revenue_growth")
    earnings_growth = fundamentals.get("earnings_growth")
    if rev_growth and earnings_growth:
        if earnings_growth > rev_growth * 2:
            risks.append(
                f"MARGIN EXPANSION PRICED IN: Earnings growing {earnings_growth*100:.1f}% "
                f"vs. revenue {rev_growth*100:.1f}%. Thesis relies on sustained margin expansion "
                f"which is hard to maintain."
            )

    if thesis.time_horizon_months < 6:
        risks.append(
            "SHORT TIME HORIZON: Fundamental thesis may not materialize within the holding period. "
            "Catalysts often take 12-24 months to be recognized by market."
        )

    pe = fundamentals.get("pe_ratio")
    if pe and pe > 50:
        risks.append(
            f"MULTIPLE CONTRACTION RISK: At {pe:.0f}x P/E, any guidance miss or "
            f"macro shock can compress multiples 20-40%, overwhelming fundamental improvement."
        )

    return risks


def _generate_mitigants(
    thesis: InvestmentThesis,
    fundamentals: dict,
    red_flags: list[str],
) -> list[str]:
    mitigants = []

    if thesis.conviction == "HIGH":
        mitigants.append(
            "Multi-factor confluence: momentum, value, and quality signals all agree, "
            "reducing single-factor reliance."
        )

    quality_grade = fundamentals.get("quality_grade", "")
    roe = fundamentals.get("roe")
    if roe and roe > 0.15:
        mitigants.append(
            f"High-quality business with {roe*100:.1f}% ROE provides downside cushion "
            f"— earnings power supports the stock during drawdowns."
        )

    de = fundamentals.get("debt_to_equity")
    if not de or de < 50:
        mitigants.append("Clean balance sheet limits bankruptcy/distress risk even in downturns.")

    fcf = fundamentals.get("free_cash_flow")
    if fcf and fcf > 0:
        mitigants.append(
            "Positive free cash flow provides optionality: buybacks, dividends, or acquisitions "
            "can act as valuation floor."
        )

    if thesis.position_size_pct <= 0.05:
        mitigants.append(
            f"Conservative position sizing ({thesis.position_size_pct*100:.0f}% of portfolio) "
            f"limits maximum drawdown impact."
        )

    mitigants.append(
        f"Stop-loss discipline: exit at ${thesis.stop_loss:.2f} caps downside at ~8% from entry."
        if thesis.stop_loss else
        "Hard stop-loss rule enforced at portfolio level to limit drawdown."
    )

    return mitigants


def _compute_risk_score(challenges: list[str], red_flags: list[str]) -> float:
    """Quantify overall risk based on number and severity of challenges."""
    base = len(challenges) * 0.08 + len(red_flags) * 0.15
    return min(base, 1.0)


def _risk_rating(score: float) -> str:
    if score < 0.2:
        return "LOW"
    elif score < 0.4:
        return "MEDIUM"
    elif score < 0.65:
        return "HIGH"
    else:
        return "VERY HIGH"


def challenge_thesis(
    thesis: InvestmentThesis,
    fundamentals: dict,
    score_row: pd.Series,
) -> ChallengeResult:
    """
    Run systematic devil's advocate analysis on an investment thesis.
    Returns ChallengeResult with challenges, red flags, and revised conviction.
    """
    challenges = []
    red_flags = []

    # Run all challenge modules
    valuation_traps = _check_valuation_traps(fundamentals, thesis)
    momentum_risks = _check_momentum_reversal_risk(score_row)
    crowding_risks = _check_crowding_risk(fundamentals)
    macro_risks = _check_macro_sensitivity(fundamentals)
    execution_risks = _check_execution_risks(fundamentals, thesis)

    challenges.extend(valuation_traps + momentum_risks + execution_risks)
    red_flags.extend(crowding_risks + macro_risks)

    # Add any outright red flags for very bad metrics
    current_ratio = fundamentals.get("current_ratio")
    if current_ratio and current_ratio < 0.8:
        red_flags.append(
            f"RED FLAG: Current ratio of {current_ratio:.2f}x — potential liquidity crisis."
        )

    profit_margin = fundamentals.get("profit_margin")
    if profit_margin and profit_margin < 0:
        red_flags.append(
            f"RED FLAG: Negative profit margins ({profit_margin*100:.1f}%) — "
            f"company is destroying shareholder value."
        )

    rev_growth = fundamentals.get("revenue_growth")
    if rev_growth and rev_growth < -0.10:
        red_flags.append(
            f"RED FLAG: Revenue shrinking {abs(rev_growth)*100:.1f}% — "
            f"fundamental demand destruction in progress."
        )

    risk_score = _compute_risk_score(challenges, red_flags)
    risk_rating = _risk_rating(risk_score)

    # Adjust conviction downward if many red flags
    conviction_map = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    rev_map = {3: "HIGH", 2: "MEDIUM", 1: "LOW"}
    base_conviction = conviction_map.get(thesis.conviction, 2)
    revised_conviction_num = max(1, base_conviction - len(red_flags))
    revised_conviction = rev_map[revised_conviction_num]

    # Adjustment factor for expected return
    adjustment = 1.0 - (risk_score * 0.3)  # up to 30% haircut on expected return

    mitigants = _generate_mitigants(thesis, fundamentals, red_flags)

    return ChallengeResult(
        ticker=thesis.ticker,
        overall_risk_score=risk_score,
        risk_rating=risk_rating,
        bear_case_return=thesis.expected_return_bear,
        bull_case_return=thesis.expected_return_bull,
        challenges=challenges,
        red_flags=red_flags,
        mitigants=mitigants,
        revised_conviction=revised_conviction,
        adjustment_factor=adjustment,
    )
