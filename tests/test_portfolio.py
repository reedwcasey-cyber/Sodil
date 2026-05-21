"""Tests for portfolio manager and tracker."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock


class TestPortfolioDB:
    def test_create_and_retrieve_portfolio(self):
        from src.portfolio.db import create_portfolio, get_portfolio_id, init_db
        pid = create_portfolio("test_portfolio_unit", 50_000.0)
        assert isinstance(pid, int)
        assert pid > 0

        retrieved = get_portfolio_id("test_portfolio_unit")
        assert retrieved == pid

    def test_record_and_retrieve_trade(self):
        from src.portfolio.db import create_portfolio, record_trade, get_trade_history
        pid = create_portfolio("test_trade_unit", 100_000.0)
        record_trade(pid, "AAPL", "BUY", 10.0, 150.0, commission=0.0, reason="test")
        trades = get_trade_history(pid)
        assert any(t["ticker"] == "AAPL" and t["action"] == "BUY" for t in trades)

    def test_upsert_and_get_position(self):
        from src.portfolio.db import create_portfolio, upsert_position, get_positions
        pid = create_portfolio("test_position_unit", 100_000.0)
        upsert_position(pid, "MSFT", 5.0, 300.0, conviction="HIGH")
        positions = get_positions(pid)
        assert any(p["ticker"] == "MSFT" and abs(p["shares"] - 5.0) < 0.01 for p in positions)

    def test_delete_position_on_zero_shares(self):
        from src.portfolio.db import create_portfolio, upsert_position, get_positions
        pid = create_portfolio("test_delete_unit", 100_000.0)
        upsert_position(pid, "GOOG", 3.0, 140.0)
        upsert_position(pid, "GOOG", 0.0, 140.0)  # should delete
        positions = get_positions(pid)
        assert not any(p["ticker"] == "GOOG" for p in positions)

    def test_save_opportunity(self):
        from src.portfolio.db import save_opportunity, get_opportunities
        save_opportunity("NVDA", 0.95, "HIGH", 0.35, {"test": True}, {}, {})
        opps = get_opportunities(10)
        assert any(o["ticker"] == "NVDA" for o in opps)


class TestPortfolioManager:
    def _make_manager(self, name: str = "pm_test") -> "PortfolioManager":
        from src.portfolio.manager import PortfolioManager
        import time
        unique_name = f"{name}_{int(time.time() * 1000) % 100000}"
        return PortfolioManager(name=unique_name, initial_cash=100_000.0)

    def test_initial_state(self):
        pm = self._make_manager()
        assert pm.cash > 0
        assert isinstance(pm.positions, list)

    @patch("src.portfolio.manager.PortfolioManager._current_price", return_value=150.0)
    def test_buy_reduces_cash(self, mock_price):
        pm = self._make_manager()
        initial_cash = pm.cash
        result = pm.buy("AAPL", 0.05)
        assert result["success"], f"Buy failed: {result.get('reason')}"
        assert pm.cash < initial_cash

    @patch("src.portfolio.manager.PortfolioManager._current_price", return_value=150.0)
    def test_buy_creates_position(self, mock_price):
        pm = self._make_manager()
        pm.buy("TSLA", 0.05)
        assert any(p["ticker"] == "TSLA" for p in pm.positions)

    @patch("src.portfolio.manager.PortfolioManager._current_price", return_value=150.0)
    def test_sell_removes_position(self, mock_price):
        pm = self._make_manager()
        pm.buy("META", 0.05)
        result = pm.sell("META")
        assert result["success"]
        assert not any(p["ticker"] == "META" for p in pm.positions)

    @patch("src.portfolio.manager.PortfolioManager._current_price", return_value=150.0)
    def test_sell_increases_cash(self, mock_price):
        pm = self._make_manager()
        pm.buy("GOOG", 0.05)
        cash_before = pm.cash
        pm.sell("GOOG")
        assert pm.cash > cash_before

    def test_sell_nonexistent_position_fails(self):
        pm = self._make_manager()
        result = pm.sell("DOESNOTEXIST")
        assert not result["success"]

    @patch("src.portfolio.manager.PortfolioManager._current_price", return_value=80.0)
    def test_stop_loss_triggered(self, mock_price):
        pm = self._make_manager()
        # Manually add a position with stop at 90
        from src.portfolio.db import upsert_position
        upsert_position(pm.portfolio_id, "STOP_TEST", 10.0, 100.0, stop_loss=90.0)
        pm.positions.append({
            "ticker": "STOP_TEST", "shares": 10.0, "avg_cost": 100.0,
            "stop_loss": 90.0, "target_price": None,
        })
        triggered = pm.enforce_stop_losses()
        assert any(t["ticker"] == "STOP_TEST" for t in triggered)

    @patch("src.portfolio.manager.PortfolioManager._current_price", return_value=110.0)
    def test_stop_loss_not_triggered_above(self, mock_price):
        pm = self._make_manager()
        from src.portfolio.db import upsert_position
        upsert_position(pm.portfolio_id, "SAFE_STOCK", 10.0, 100.0, stop_loss=90.0)
        pm.positions.append({
            "ticker": "SAFE_STOCK", "shares": 10.0, "avg_cost": 100.0,
            "stop_loss": 90.0, "target_price": None,
        })
        triggered = pm.enforce_stop_losses()
        assert not any(t["ticker"] == "SAFE_STOCK" for t in triggered)

    @patch("src.portfolio.manager.PortfolioManager._current_price", return_value=150.0)
    def test_get_portfolio_value(self, mock_price):
        pm = self._make_manager()
        pm.buy("NVDA", 0.10)
        total, pos_val, cash = pm.get_portfolio_value()
        assert total == pytest.approx(pos_val + cash, rel=0.01)
        assert total > 0


class TestRealizedPnL:
    def test_profitable_trade(self):
        from src.portfolio.tracker import compute_realized_pnl
        trades = [
            {"ticker": "AAPL", "action": "BUY", "shares": 10, "price": 100.0},
            {"ticker": "AAPL", "action": "SELL", "shares": 10, "price": 120.0},
        ]
        pnl, win_rate, avg_win, avg_loss = compute_realized_pnl(trades)
        assert abs(pnl - 200.0) < 0.01, f"Expected $200 profit, got ${pnl:.2f}"
        assert win_rate == 1.0

    def test_losing_trade(self):
        from src.portfolio.tracker import compute_realized_pnl
        trades = [
            {"ticker": "AAPL", "action": "BUY", "shares": 10, "price": 100.0},
            {"ticker": "AAPL", "action": "SELL", "shares": 10, "price": 80.0},
        ]
        pnl, win_rate, avg_win, avg_loss = compute_realized_pnl(trades)
        assert pnl < 0
        assert win_rate == 0.0

    def test_no_trades(self):
        from src.portfolio.tracker import compute_realized_pnl
        pnl, win_rate, avg_win, avg_loss = compute_realized_pnl([])
        assert pnl == 0.0

    def test_mixed_trades(self):
        from src.portfolio.tracker import compute_realized_pnl
        trades = [
            {"ticker": "AAPL", "action": "BUY", "shares": 10, "price": 100.0},
            {"ticker": "AAPL", "action": "SELL", "shares": 10, "price": 120.0},
            {"ticker": "MSFT", "action": "BUY", "shares": 5, "price": 200.0},
            {"ticker": "MSFT", "action": "SELL", "shares": 5, "price": 180.0},
        ]
        pnl, win_rate, avg_win, avg_loss = compute_realized_pnl(trades)
        # AAPL: +200, MSFT: -100 → net +100
        assert abs(pnl - 100.0) < 0.01
        assert win_rate == 0.5
