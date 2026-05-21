"""Tests for risk metrics and stress testing."""

import numpy as np
import pandas as pd
import pytest


def make_returns(n: int = 252, mean: float = 0.0005, vol: float = 0.012, seed: int = 0) -> pd.Series:
    np.random.seed(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.Series(np.random.normal(mean, vol, n), index=idx)


class TestRiskMetrics:
    def test_returns_risk_metrics_object(self):
        from src.risk.metrics import compute_risk_metrics
        r = make_returns()
        metrics = compute_risk_metrics(r)
        assert metrics is not None

    def test_var_is_negative(self):
        from src.risk.metrics import compute_risk_metrics
        r = make_returns()
        metrics = compute_risk_metrics(r)
        assert metrics.var_95 < 0, "VaR should be a negative number (a loss)"

    def test_cvar_less_than_var(self):
        from src.risk.metrics import compute_risk_metrics
        r = make_returns()
        metrics = compute_risk_metrics(r)
        assert metrics.cvar_95 <= metrics.var_95, "CVaR must be <= VaR (worse tail)"

    def test_max_drawdown_non_positive(self):
        from src.risk.metrics import compute_risk_metrics
        r = make_returns()
        metrics = compute_risk_metrics(r)
        assert metrics.max_drawdown <= 0

    def test_positive_returns_positive_sharpe(self):
        from src.risk.metrics import compute_risk_metrics
        r = make_returns(n=252, mean=0.002, vol=0.005)
        metrics = compute_risk_metrics(r, rf=0.0)
        assert metrics.sharpe_ratio > 0, "Strongly positive returns should have positive Sharpe"

    def test_volatility_annualized(self):
        from src.risk.metrics import compute_risk_metrics
        vol_daily = 0.01
        r = make_returns(n=1000, mean=0, vol=vol_daily)
        metrics = compute_risk_metrics(r)
        # Annualized vol should be approximately daily_vol * sqrt(252)
        expected = vol_daily * np.sqrt(252)
        assert abs(metrics.volatility_ann - expected) < 0.01, (
            f"Annualized vol {metrics.volatility_ann:.4f} != expected {expected:.4f}"
        )

    def test_positive_months_pct_range(self):
        from src.risk.metrics import compute_risk_metrics
        r = make_returns(n=504)  # 2 years
        metrics = compute_risk_metrics(r)
        assert 0 <= metrics.positive_months_pct <= 1

    def test_with_benchmark(self):
        from src.risk.metrics import compute_risk_metrics
        r = make_returns(252)
        bench = make_returns(252, mean=0.0003, seed=1)
        metrics = compute_risk_metrics(r, benchmark_returns=bench)
        assert isinstance(metrics.beta, float)
        assert isinstance(metrics.alpha_ann, float)

    def test_empty_returns(self):
        from src.risk.metrics import compute_risk_metrics
        metrics = compute_risk_metrics(pd.Series(dtype=float))
        assert metrics.sharpe_ratio == 0


class TestStressTest:
    def test_historical_scenarios_count(self):
        from src.risk.stress_test import run_historical_scenarios
        scenarios = run_historical_scenarios("TEST", beta=1.0, position_value=10_000)
        assert len(scenarios) > 0
        assert len(scenarios) == 6, "Should have 6 historical crisis scenarios"

    def test_high_beta_loses_more(self):
        from src.risk.stress_test import run_historical_scenarios
        low_beta = run_historical_scenarios("TEST", beta=0.5, position_value=10_000)
        high_beta = run_historical_scenarios("TEST", beta=2.0, position_value=10_000)
        worst_low = min(s.position_drop for s in low_beta)
        worst_high = min(s.position_drop for s in high_beta)
        assert worst_high < worst_low, "High beta should lose more in crashes"

    def test_monte_carlo_paths(self):
        from src.risk.stress_test import monte_carlo_simulation
        mc = monte_carlo_simulation(0.10, 0.20, n_paths=1000)
        assert mc.simulated_paths == 1000
        assert -1 < mc.percentile_5 < mc.percentile_95
        assert mc.prob_loss >= 0
        assert mc.prob_loss <= 1

    def test_monte_carlo_positive_expected_return(self):
        from src.risk.stress_test import monte_carlo_simulation
        mc = monte_carlo_simulation(annual_return=0.20, annual_vol=0.15, n_paths=5000)
        # With 20% expected return and 15% vol, mean should be positive
        assert mc.mean_return_1y > 0, f"Mean return should be positive, got {mc.mean_return_1y:.3f}"

    def test_full_stress_test(self):
        from src.risk.stress_test import run_stress_test
        fundamentals = {"beta": 1.2, "sector": "Technology"}
        result = run_stress_test("AAPL", fundamentals, 0.15, 0.22, 10_000)
        assert result.ticker == "AAPL"
        assert len(result.scenarios) > 0
        assert result.monte_carlo is not None
        assert result.worst_case_loss_pct < 0


class TestPositionVaR:
    def test_position_var_positive(self):
        from src.risk.metrics import position_var
        r = make_returns(252)
        result = position_var(10_000, r, confidence=0.95)
        assert result["var_1d"] >= 0, "Dollar VaR should be non-negative"
        assert result["cvar_1d"] >= 0

    def test_var_10d_larger_than_1d(self):
        from src.risk.metrics import position_var
        r = make_returns(252)
        result = position_var(10_000, r)
        assert result["var_10d"] > result["var_1d"], "10-day VaR should exceed 1-day"

    def test_larger_position_larger_var(self):
        from src.risk.metrics import position_var
        r = make_returns(252)
        small = position_var(1_000, r)
        large = position_var(100_000, r)
        assert large["var_1d"] > small["var_1d"]
