"""Stress testing: historical scenario analysis and Monte Carlo simulation."""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# Historical crisis scenarios: [start, end, market_drop]
HISTORICAL_CRISES = {
    "2008 Financial Crisis": {"market_drop": -0.57, "duration_days": 365, "recovery_days": 730},
    "2020 COVID Crash": {"market_drop": -0.34, "duration_days": 33, "recovery_days": 120},
    "2022 Rate Hike Bear": {"market_drop": -0.25, "duration_days": 282, "recovery_days": 400},
    "2000 Dot-Com Bust": {"market_drop": -0.49, "duration_days": 700, "recovery_days": 1800},
    "2011 Euro Crisis": {"market_drop": -0.19, "duration_days": 90, "recovery_days": 180},
    "2015 China Shock": {"market_drop": -0.12, "duration_days": 60, "recovery_days": 100},
}


@dataclass
class StressScenario:
    name: str
    market_drop: float
    position_drop: float    # beta-adjusted drop
    recovery_time_days: int
    max_loss_dollars: float


@dataclass
class MonteCarloResult:
    mean_return_1y: float
    median_return_1y: float
    percentile_5: float
    percentile_25: float
    percentile_75: float
    percentile_95: float
    prob_loss: float
    prob_gain_20pct: float
    prob_gain_50pct: float
    simulated_paths: int
    distribution: list[float] = field(default_factory=list)


@dataclass
class StressTestResult:
    ticker: str
    scenarios: list[StressScenario] = field(default_factory=list)
    monte_carlo: Optional[MonteCarloResult] = None
    worst_case_loss_pct: float = 0.0
    expected_loss_in_crash: float = 0.0


def run_historical_scenarios(
    ticker: str,
    beta: float,
    position_value: float,
) -> list[StressScenario]:
    """Apply historical crisis drops (beta-adjusted) to current position."""
    scenarios = []
    for name, params in HISTORICAL_CRISES.items():
        market_drop = params["market_drop"]
        # Beta-adjusted: high-beta stocks fall more in crashes
        position_drop = market_drop * beta
        # Liquidity discount in severe crashes
        if market_drop < -0.3:
            position_drop *= 1.1  # additional 10% for liquidity crunch

        loss_dollars = position_value * abs(position_drop)
        scenarios.append(StressScenario(
            name=name,
            market_drop=market_drop,
            position_drop=position_drop,
            recovery_time_days=params["recovery_days"],
            max_loss_dollars=loss_dollars,
        ))

    return sorted(scenarios, key=lambda s: s.position_drop)


def monte_carlo_simulation(
    annual_return: float,
    annual_vol: float,
    n_paths: int = 10_000,
    horizon_days: int = 252,
    initial_value: float = 100.0,
) -> MonteCarloResult:
    """
    Geometric Brownian Motion Monte Carlo for return distribution.
    Uses daily steps with fat tails (t-distribution).
    """
    dt = 1 / 252
    mu = annual_return - 0.5 * annual_vol ** 2  # drift (Ito correction)
    sigma = annual_vol

    # Use t-distribution with 5 degrees of freedom for fat tails
    df_t = 5
    t_scale = sigma * np.sqrt(dt * (df_t - 2) / df_t)

    np.random.seed(42)
    # Draw from t-distribution for fat tails
    shocks = np.random.standard_t(df_t, size=(n_paths, horizon_days)) * t_scale
    shocks += mu * dt  # add drift

    log_returns = np.cumsum(shocks, axis=1)
    paths = initial_value * np.exp(log_returns)
    final_values = paths[:, -1]
    final_returns = (final_values / initial_value) - 1

    return MonteCarloResult(
        mean_return_1y=float(np.mean(final_returns)),
        median_return_1y=float(np.median(final_returns)),
        percentile_5=float(np.percentile(final_returns, 5)),
        percentile_25=float(np.percentile(final_returns, 25)),
        percentile_75=float(np.percentile(final_returns, 75)),
        percentile_95=float(np.percentile(final_returns, 95)),
        prob_loss=float(np.mean(final_returns < 0)),
        prob_gain_20pct=float(np.mean(final_returns > 0.20)),
        prob_gain_50pct=float(np.mean(final_returns > 0.50)),
        simulated_paths=n_paths,
        distribution=final_returns.tolist(),
    )


def run_stress_test(
    ticker: str,
    fundamentals: dict,
    expected_return: float,
    annual_vol: float,
    position_value: float = 10_000.0,
) -> StressTestResult:
    """Full stress test: historical scenarios + Monte Carlo."""
    beta = fundamentals.get("beta") or 1.0
    beta = float(beta)

    scenarios = run_historical_scenarios(ticker, beta, position_value)
    mc = monte_carlo_simulation(expected_return, annual_vol)

    worst_case = min(s.position_drop for s in scenarios)

    # Expected loss in a market downturn (weighted by scenario severity)
    avg_drop = np.mean([s.position_drop for s in scenarios])

    return StressTestResult(
        ticker=ticker,
        scenarios=scenarios,
        monte_carlo=mc,
        worst_case_loss_pct=worst_case,
        expected_loss_in_crash=avg_drop,
    )
