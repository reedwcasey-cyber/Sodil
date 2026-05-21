"""Tests for backtesting engine."""

import numpy as np
import pandas as pd
import pytest
from datetime import date


def make_price_df(n: int = 500, trend: float = 0.0003, vol: float = 0.015, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.date_range(end=date.today(), periods=n, freq="B")
    returns = np.random.normal(trend, vol, n)
    close = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame({
        "Open": close, "High": close * 1.01, "Low": close * 0.99,
        "Close": close, "Volume": np.ones(n) * 1_000_000,
    }, index=dates)


class TestMomentumBacktest:
    def test_returns_backtest_result(self):
        from src.thesis.backtest import backtest_momentum
        prices = make_price_df(500)
        result = backtest_momentum("TEST", prices)
        assert result is not None
        assert hasattr(result, "total_return")
        assert hasattr(result, "sharpe_ratio")
        assert hasattr(result, "max_drawdown")

    def test_max_drawdown_non_positive(self):
        from src.thesis.backtest import backtest_momentum
        prices = make_price_df(500)
        result = backtest_momentum("TEST", prices)
        assert result.max_drawdown <= 0, "Max drawdown must be non-positive"

    def test_insufficient_data(self):
        from src.thesis.backtest import backtest_momentum
        prices = make_price_df(n=30)
        result = backtest_momentum("TEST", prices)
        assert not result.passed, "Should fail with insufficient data"
        assert result.failure_reason != ""

    def test_trending_market_profitable(self):
        from src.thesis.backtest import backtest_momentum
        prices = make_price_df(500, trend=0.001, vol=0.005)
        result = backtest_momentum("TREND", prices)
        assert result.annualized_return > 0, "Momentum should profit in strongly trending market"

    def test_sharpe_ratio_type(self):
        from src.thesis.backtest import backtest_momentum
        prices = make_price_df(500)
        result = backtest_momentum("TEST", prices)
        assert isinstance(result.sharpe_ratio, float)

    def test_equity_curve_length(self):
        from src.thesis.backtest import backtest_momentum
        prices = make_price_df(300)
        result = backtest_momentum("TEST", prices)
        if result.equity_curve:
            assert len(result.equity_curve) > 0


class TestBuyAndHoldBacktest:
    def test_matches_price_return(self):
        from src.thesis.backtest import backtest_buy_and_hold
        prices = make_price_df(500, trend=0.001, vol=0.005)
        result = backtest_buy_and_hold("TEST", prices)
        close = prices["Close"].squeeze()
        expected = float(close.iloc[-1] / close.iloc[0] - 1)
        assert abs(result.total_return - expected) < 0.01, "B&H return should match price return"

    def test_empty_prices(self):
        from src.thesis.backtest import backtest_buy_and_hold
        result = backtest_buy_and_hold("EMPTY", pd.DataFrame())
        assert not result.passed

    def test_positive_in_bull_market(self):
        from src.thesis.backtest import backtest_buy_and_hold
        prices = make_price_df(500, trend=0.002, vol=0.005)
        result = backtest_buy_and_hold("BULL", prices)
        assert result.total_return > 0


class TestFullBacktest:
    def test_returns_dict(self):
        from src.thesis.backtest import run_full_backtest
        prices = make_price_df(500)
        results = run_full_backtest("TEST", prices)
        assert isinstance(results, dict)
        assert "momentum" in results
        assert "buy_and_hold" in results

    def test_all_strategies_have_valid_metrics(self):
        from src.thesis.backtest import run_full_backtest
        prices = make_price_df(500)
        results = run_full_backtest("TEST", prices)
        for name, r in results.items():
            assert isinstance(r.sharpe_ratio, float), f"{name}: Sharpe must be float"
            assert r.max_drawdown <= 0, f"{name}: Max DD must be non-positive"
            assert 0 <= r.win_rate <= 1, f"{name}: Win rate must be in [0,1]"
