"""Tests for signal computation modules."""

import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta


def make_price_df(n: int = 300, trend: float = 0.0003, vol: float = 0.015, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    np.random.seed(seed)
    dates = pd.date_range(end=date.today(), periods=n, freq="B")
    returns = np.random.normal(trend, vol, n)
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + abs(np.random.normal(0, vol / 2, n)))
    low = close * (1 - abs(np.random.normal(0, vol / 2, n)))
    volume = np.random.randint(500_000, 5_000_000, n).astype(float)
    return pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close, "Volume": volume}, index=dates)


def make_trending_prices(n: int = 300, trend: float = 0.001) -> pd.DataFrame:
    return make_price_df(n=n, trend=trend, vol=0.01)


def make_declining_prices(n: int = 300) -> pd.DataFrame:
    return make_price_df(n=n, trend=-0.001, vol=0.01)


class TestMomentumSignals:
    def test_momentum_score_trending_positive(self):
        from src.signals.momentum import momentum_score
        prices = make_trending_prices(300)
        score = momentum_score(prices)
        assert not np.isnan(score), "Momentum score should not be NaN for valid data"
        assert score > 0, "Trending stock should have positive momentum score"

    def test_momentum_score_declining_negative(self):
        from src.signals.momentum import momentum_score
        prices = make_declining_prices(300)
        score = momentum_score(prices)
        assert not np.isnan(score)
        assert score < 0, "Declining stock should have negative momentum score"

    def test_momentum_score_insufficient_data(self):
        from src.signals.momentum import momentum_score
        prices = make_price_df(n=20)
        score = momentum_score(prices)
        assert np.isnan(score), "Should return NaN with insufficient data"

    def test_rsi_range(self):
        from src.signals.momentum import rsi
        prices = make_price_df(100)
        val = rsi(prices)
        assert not np.isnan(val)
        assert 0 <= val <= 100, f"RSI must be in [0, 100], got {val}"

    def test_rsi_overbought_trending(self):
        from src.signals.momentum import rsi
        prices = make_trending_prices(100, trend=0.005)
        val = rsi(prices)
        assert val > 50, "Strongly trending stock should have RSI > 50"

    def test_macd_signal_type(self):
        from src.signals.momentum import macd_signal
        prices = make_price_df(100)
        val = macd_signal(prices)
        assert isinstance(val, float) or np.isnan(val)

    def test_bollinger_position_range(self):
        from src.signals.momentum import bollinger_position
        prices = make_price_df(100)
        val = bollinger_position(prices)
        assert not np.isnan(val)
        # Should be within expanded range but not extreme
        assert -5 < val < 5

    def test_volume_trend_range(self):
        from src.signals.momentum import volume_trend
        prices = make_price_df(100)
        val = volume_trend(prices)
        if not np.isnan(val):
            assert -1 <= val <= 1, f"Volume trend must be in [-1, 1], got {val}"


class TestValueSignals:
    def setup_method(self):
        self.good_fundamentals = {
            "pe_ratio": 15.0,
            "forward_pe": 12.0,
            "pb_ratio": 2.0,
            "ps_ratio": 2.5,
            "ev_ebitda": 10.0,
            "peg_ratio": 0.8,
            "analyst_target": 150.0,
            "current_price": 120.0,
        }
        self.expensive_fundamentals = {
            "pe_ratio": 100.0,
            "forward_pe": 90.0,
            "pb_ratio": 20.0,
            "ps_ratio": 15.0,
            "ev_ebitda": 50.0,
            "peg_ratio": 5.0,
            "analyst_target": 110.0,
            "current_price": 120.0,
        }

    def test_cheap_scores_higher_than_expensive(self):
        from src.signals.value import composite_value_score
        cheap_score = composite_value_score(self.good_fundamentals)
        expensive_score = composite_value_score(self.expensive_fundamentals)
        assert cheap_score > expensive_score, "Cheap stock should score higher than expensive"

    def test_analyst_upside_positive(self):
        from src.signals.value import analyst_upside
        upside = analyst_upside(self.good_fundamentals)
        assert abs(upside - 0.25) < 0.01, f"Expected 25% upside, got {upside}"

    def test_analyst_downside_negative(self):
        from src.signals.value import analyst_upside
        downside = analyst_upside(self.expensive_fundamentals)
        assert downside < 0, "Stock trading above target should have negative upside"

    def test_empty_fundamentals(self):
        from src.signals.value import composite_value_score
        score = composite_value_score({})
        assert np.isnan(score), "Empty fundamentals should return NaN"


class TestQualitySignals:
    def setup_method(self):
        self.high_quality = {
            "roe": 0.25,
            "roa": 0.12,
            "profit_margin": 0.20,
            "operating_margin": 0.25,
            "revenue_growth": 0.20,
            "earnings_growth": 0.25,
            "debt_to_equity": 30.0,
            "current_ratio": 2.5,
            "free_cash_flow": 5_000_000_000,
        }
        self.low_quality = {
            "roe": -0.05,
            "roa": -0.02,
            "profit_margin": -0.03,
            "operating_margin": 0.01,
            "revenue_growth": -0.10,
            "earnings_growth": -0.20,
            "debt_to_equity": 300.0,
            "current_ratio": 0.7,
            "free_cash_flow": -500_000_000,
        }

    def test_high_quality_scores_higher(self):
        from src.signals.quality import quality_score
        high = quality_score(self.high_quality)
        low = quality_score(self.low_quality)
        assert high > low, f"High-quality score ({high:.3f}) should exceed low-quality ({low:.3f})"

    def test_quality_score_range(self):
        from src.signals.quality import quality_score
        high = quality_score(self.high_quality)
        low = quality_score(self.low_quality)
        assert -1.5 < high < 1.5, "Quality score should be within reasonable bounds"
        assert -1.5 < low < 1.5

    def test_grade_a_for_high_quality(self):
        from src.signals.quality import financial_health_grade
        grade = financial_health_grade(self.high_quality)
        assert grade in ("A", "B"), f"High-quality business should get A or B, got {grade}"

    def test_grade_f_for_low_quality(self):
        from src.signals.quality import financial_health_grade
        grade = financial_health_grade(self.low_quality)
        assert grade in ("D", "F"), f"Low-quality business should get D or F, got {grade}"


class TestCompositeScoring:
    def test_composite_score_returns_dict(self):
        from src.signals.composite import compute_composite_score
        fundamentals = {"pe_ratio": 20, "roe": 0.15, "revenue_growth": 0.10}
        prices = make_price_df(300)
        result = compute_composite_score("TEST", fundamentals, prices)
        assert isinstance(result, dict)
        assert "ticker" in result
        assert "composite_raw" in result

    def test_rank_opportunities_sorted(self):
        from src.signals.composite import compute_composite_score, rank_opportunities

        tickers = ["GOOD", "BAD", "MID"]
        scores = []
        for t, trend in zip(tickers, [0.002, -0.002, 0.0]):
            prices = make_price_df(300, trend=trend)
            fundamentals = {"pe_ratio": 20 if t == "GOOD" else 50, "roe": 0.20 if t == "GOOD" else 0.05}
            scores.append(compute_composite_score(t, fundamentals, prices))

        ranked = rank_opportunities(scores, top_n=3)
        assert len(ranked) <= 3
        assert "composite_score" in ranked.columns

    def test_handles_empty_prices(self):
        from src.signals.composite import compute_composite_score
        result = compute_composite_score("EMPTY", {}, pd.DataFrame())
        assert isinstance(result, dict)
        assert np.isnan(result["composite_raw"])
