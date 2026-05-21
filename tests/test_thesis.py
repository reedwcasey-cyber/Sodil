"""Tests for thesis generation and challenger."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch


def make_score_row() -> pd.Series:
    return pd.Series({
        "composite_score": 0.8,
        "momentum_z": 1.5,
        "value_z": 1.2,
        "quality_z": 0.9,
        "technical_z": 0.6,
        "rsi": 55.0,
        "macd": 0.002,
        "analyst_upside": 0.18,
        "momentum_raw": 0.15,
        "value_raw": -1.5,
        "quality_raw": 0.45,
        "technical_raw": 0.12,
    })


def make_good_fundamentals() -> dict:
    return {
        "ticker": "GOOD",
        "name": "Good Company Inc",
        "sector": "Technology",
        "industry": "Software",
        "market_cap": 50_000_000_000,
        "pe_ratio": 22.0,
        "forward_pe": 18.0,
        "pb_ratio": 5.0,
        "ps_ratio": 4.0,
        "ev_ebitda": 15.0,
        "peg_ratio": 1.2,
        "debt_to_equity": 20.0,
        "roe": 0.25,
        "roa": 0.12,
        "profit_margin": 0.18,
        "operating_margin": 0.22,
        "revenue_growth": 0.20,
        "earnings_growth": 0.28,
        "free_cash_flow": 3_000_000_000,
        "current_ratio": 2.0,
        "beta": 1.2,
        "52w_high": 185.0,
        "52w_low": 120.0,
        "avg_volume": 5_000_000,
        "short_ratio": 2.5,
        "analyst_target": 175.0,
        "current_price": 150.0,
        "dividend_yield": 0.005,
    }


def make_risky_fundamentals() -> dict:
    return {
        "ticker": "RISKY",
        "name": "Risky Corp",
        "sector": "Energy",
        "pe_ratio": 80.0,
        "forward_pe": 95.0,  # earnings declining!
        "pb_ratio": 15.0,
        "roe": 0.04,
        "roa": 0.01,
        "profit_margin": -0.02,
        "revenue_growth": -0.08,
        "debt_to_equity": 250.0,
        "current_ratio": 0.75,
        "free_cash_flow": -500_000_000,
        "beta": 2.1,
        "analyst_target": 45.0,
        "current_price": 50.0,
        "short_ratio": 0.5,  # low short = crowded long
    }


class TestThesisGenerator:
    def test_generates_thesis(self):
        from src.thesis.generator import generate_thesis
        fundamentals = make_good_fundamentals()
        score_row = make_score_row()
        thesis = generate_thesis("GOOD", fundamentals, score_row)
        assert thesis is not None
        assert thesis.ticker == "GOOD"

    def test_conviction_high_for_good_stock(self):
        from src.thesis.generator import generate_thesis
        fundamentals = make_good_fundamentals()
        score_row = make_score_row()
        thesis = generate_thesis("GOOD", fundamentals, score_row)
        assert thesis.conviction in ("HIGH", "MEDIUM")

    def test_expected_returns_ordered(self):
        from src.thesis.generator import generate_thesis
        fundamentals = make_good_fundamentals()
        score_row = make_score_row()
        thesis = generate_thesis("GOOD", fundamentals, score_row)
        assert thesis.expected_return_bear < thesis.expected_return_1y < thesis.expected_return_bull

    def test_entry_and_target_prices(self):
        from src.thesis.generator import generate_thesis
        fundamentals = make_good_fundamentals()
        score_row = make_score_row()
        thesis = generate_thesis("GOOD", fundamentals, score_row)
        assert thesis.entry_price == 150.0
        assert thesis.target_price > thesis.entry_price, "Target should exceed entry in base case"

    def test_stop_loss_below_entry(self):
        from src.thesis.generator import generate_thesis
        fundamentals = make_good_fundamentals()
        score_row = make_score_row()
        thesis = generate_thesis("GOOD", fundamentals, score_row)
        assert thesis.stop_loss < thesis.entry_price, "Stop loss must be below entry"

    def test_catalysts_not_empty(self):
        from src.thesis.generator import generate_thesis
        fundamentals = make_good_fundamentals()
        score_row = make_score_row()
        thesis = generate_thesis("GOOD", fundamentals, score_row)
        assert len(thesis.primary_catalysts) > 0

    def test_risks_not_empty(self):
        from src.thesis.generator import generate_thesis
        fundamentals = make_good_fundamentals()
        score_row = make_score_row()
        thesis = generate_thesis("GOOD", fundamentals, score_row)
        assert len(thesis.key_risks) > 0

    def test_narrative_contains_ticker(self):
        from src.thesis.generator import generate_thesis
        fundamentals = make_good_fundamentals()
        score_row = make_score_row()
        thesis = generate_thesis("GOOD", fundamentals, score_row)
        assert "GOOD" in thesis.narrative

    def test_position_size_reasonable(self):
        from src.thesis.generator import generate_thesis
        fundamentals = make_good_fundamentals()
        score_row = make_score_row()
        thesis = generate_thesis("GOOD", fundamentals, score_row)
        assert 0.01 <= thesis.position_size_pct <= 0.15


class TestThesisChallenger:
    def test_challenge_returns_result(self):
        from src.thesis.generator import generate_thesis
        from src.thesis.challenger import challenge_thesis
        fundamentals = make_good_fundamentals()
        score_row = make_score_row()
        thesis = generate_thesis("GOOD", fundamentals, score_row)
        challenge = challenge_thesis(thesis, fundamentals, score_row)
        assert challenge is not None

    def test_risky_stock_higher_risk_score(self):
        from src.thesis.generator import generate_thesis
        from src.thesis.challenger import challenge_thesis
        good_fund = make_good_fundamentals()
        risky_fund = make_risky_fundamentals()
        good_score = make_score_row()
        risky_score = make_score_row()

        good_thesis = generate_thesis("GOOD", good_fund, good_score)
        risky_thesis = generate_thesis("RISKY", risky_fund, risky_score)

        good_challenge = challenge_thesis(good_thesis, good_fund, good_score)
        risky_challenge = challenge_thesis(risky_thesis, risky_fund, risky_score)

        assert risky_challenge.overall_risk_score >= good_challenge.overall_risk_score, (
            f"Risky stock should have higher risk score: "
            f"{risky_challenge.overall_risk_score:.2f} vs {good_challenge.overall_risk_score:.2f}"
        )

    def test_risk_score_in_range(self):
        from src.thesis.generator import generate_thesis
        from src.thesis.challenger import challenge_thesis
        fundamentals = make_risky_fundamentals()
        score_row = make_score_row()
        thesis = generate_thesis("RISKY", fundamentals, score_row)
        challenge = challenge_thesis(thesis, fundamentals, score_row)
        assert 0 <= challenge.overall_risk_score <= 1

    def test_adjustment_factor_reduces_return(self):
        from src.thesis.generator import generate_thesis
        from src.thesis.challenger import challenge_thesis
        fundamentals = make_risky_fundamentals()
        score_row = make_score_row()
        thesis = generate_thesis("RISKY", fundamentals, score_row)
        challenge = challenge_thesis(thesis, fundamentals, score_row)
        assert challenge.adjustment_factor <= 1.0, "Challenging should never increase expected return"

    def test_red_flags_detected_for_risky(self):
        from src.thesis.generator import generate_thesis
        from src.thesis.challenger import challenge_thesis
        fundamentals = make_risky_fundamentals()
        score_row = make_score_row()
        thesis = generate_thesis("RISKY", fundamentals, score_row)
        challenge = challenge_thesis(thesis, fundamentals, score_row)
        assert len(challenge.red_flags) > 0, "Risky stock should trigger red flags"

    def test_mitigants_present(self):
        from src.thesis.generator import generate_thesis
        from src.thesis.challenger import challenge_thesis
        fundamentals = make_good_fundamentals()
        score_row = make_score_row()
        thesis = generate_thesis("GOOD", fundamentals, score_row)
        challenge = challenge_thesis(thesis, fundamentals, score_row)
        assert len(challenge.mitigants) > 0

    def test_valid_risk_ratings(self):
        from src.thesis.generator import generate_thesis
        from src.thesis.challenger import challenge_thesis
        for fund in [make_good_fundamentals(), make_risky_fundamentals()]:
            score_row = make_score_row()
            thesis = generate_thesis(fund["ticker"], fund, score_row)
            challenge = challenge_thesis(thesis, fund, score_row)
            assert challenge.risk_rating in ("LOW", "MEDIUM", "HIGH", "VERY HIGH")
