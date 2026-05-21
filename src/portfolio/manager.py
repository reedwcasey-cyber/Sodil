"""
Mock portfolio manager: paper trading with position sizing,
rebalancing, stop-loss enforcement, and P&L tracking.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from .db import (
    create_portfolio, get_portfolio_id, get_positions,
    record_trade, upsert_position, save_snapshot, get_snapshots, get_trade_history,
)
from ..thesis.generator import InvestmentThesis
from ..thesis.challenger import ChallengeResult
from ..data.fetcher import fetch_price_history

logger = logging.getLogger(__name__)

COMMISSION_PER_TRADE = 0.0  # Modern zero-commission brokers
SLIPPAGE_BPS = 5             # 5 basis points slippage per trade


@dataclass
class PortfolioState:
    portfolio_id: int
    name: str
    cash: float
    initial_cash: float
    positions: list[dict] = field(default_factory=list)
    total_value: float = 0.0
    cumulative_return: float = 0.0


class PortfolioManager:
    """
    Paper trading portfolio with full position lifecycle management.
    """

    def __init__(self, name: str = "sodil_paper", initial_cash: float = 100_000.0):
        self.name = name
        self.initial_cash = initial_cash
        pid = get_portfolio_id(name)
        if pid is None:
            pid = create_portfolio(name, initial_cash)
        self.portfolio_id = pid
        self._load_state()

    def _load_state(self):
        positions = get_positions(self.portfolio_id)
        snapshots = get_snapshots(self.portfolio_id)

        if snapshots:
            latest = snapshots[-1]
            self.cash = latest["cash"]
        else:
            from .db import get_conn
            with get_conn() as conn:
                row = conn.execute(
                    "SELECT initial_cash FROM portfolios WHERE id=?",
                    (self.portfolio_id,),
                ).fetchone()
                self.cash = row["initial_cash"] if row else self.initial_cash

        self.positions = positions

    def _current_price(self, ticker: str) -> Optional[float]:
        try:
            df = fetch_price_history(ticker, period="5d")
            if df.empty:
                return None
            close = df["Close"].squeeze() if "Close" in df.columns else df.iloc[:, 0].squeeze()
            return float(close.iloc[-1])
        except Exception:
            return None

    def get_portfolio_value(self) -> tuple[float, float, float]:
        """Returns (total_value, positions_value, cash)."""
        positions_value = 0.0
        for pos in self.positions:
            price = self._current_price(pos["ticker"])
            if price:
                positions_value += pos["shares"] * price
        total = positions_value + self.cash
        return total, positions_value, self.cash

    def buy(
        self,
        ticker: str,
        position_pct: float,
        thesis: Optional[InvestmentThesis] = None,
        challenge: Optional[ChallengeResult] = None,
        force_price: Optional[float] = None,
    ) -> dict:
        """
        Execute a paper buy order.
        position_pct: fraction of total portfolio to allocate (e.g., 0.05 = 5%)
        """
        total_value, _, _ = self.get_portfolio_value()
        target_value = total_value * position_pct

        price = force_price or self._current_price(ticker)
        if not price:
            return {"success": False, "reason": f"Could not fetch price for {ticker}"}

        # Apply slippage
        price_with_slippage = price * (1 + SLIPPAGE_BPS / 10000)
        max_affordable = self.cash / price_with_slippage
        target_shares = min(target_value / price_with_slippage, max_affordable)

        if target_shares < 0.001:
            return {"success": False, "reason": f"Insufficient cash (${self.cash:.2f}) for {ticker}"}

        total_cost = target_shares * price_with_slippage

        # Check if already in position
        existing = next((p for p in self.positions if p["ticker"] == ticker), None)
        if existing:
            # Average into existing position
            old_shares = existing["shares"]
            old_cost = existing["avg_cost"]
            new_shares = old_shares + target_shares
            new_avg = (old_shares * old_cost + target_shares * price_with_slippage) / new_shares
            upsert_position(
                self.portfolio_id, ticker, new_shares, new_avg,
                conviction=thesis.conviction if thesis else "",
                stop_loss=thesis.stop_loss if thesis else None,
                target_price=thesis.target_price if thesis else None,
            )
            for pos in self.positions:
                if pos["ticker"] == ticker:
                    pos["shares"] = new_shares
                    pos["avg_cost"] = new_avg
        else:
            upsert_position(
                self.portfolio_id, ticker, target_shares, price_with_slippage,
                conviction=thesis.conviction if thesis else "",
                stop_loss=thesis.stop_loss if thesis else None,
                target_price=thesis.target_price if thesis else None,
            )
            self.positions.append({
                "ticker": ticker,
                "shares": target_shares,
                "avg_cost": price_with_slippage,
                "entry_date": datetime.now().isoformat(),
                "thesis_conviction": thesis.conviction if thesis else "",
                "stop_loss": thesis.stop_loss if thesis else None,
                "target_price": thesis.target_price if thesis else None,
            })

        self.cash -= total_cost
        record_trade(
            self.portfolio_id, ticker, "BUY", target_shares, price_with_slippage,
            commission=COMMISSION_PER_TRADE,
            reason=f"Thesis: {thesis.conviction if thesis else 'manual'} conviction",
        )

        return {
            "success": True,
            "ticker": ticker,
            "shares": target_shares,
            "price": price_with_slippage,
            "cost": total_cost,
            "cash_remaining": self.cash,
        }

    def sell(
        self,
        ticker: str,
        shares: Optional[float] = None,
        reason: str = "",
        force_price: Optional[float] = None,
    ) -> dict:
        """Execute a paper sell order. If shares=None, sells entire position."""
        existing = next((p for p in self.positions if p["ticker"] == ticker), None)
        if not existing:
            return {"success": False, "reason": f"No position in {ticker}"}

        price = force_price or self._current_price(ticker)
        if not price:
            return {"success": False, "reason": f"Could not fetch price for {ticker}"}

        sell_shares = shares or existing["shares"]
        price_with_slippage = price * (1 - SLIPPAGE_BPS / 10000)
        proceeds = sell_shares * price_with_slippage

        pnl = sell_shares * (price_with_slippage - existing["avg_cost"])
        pnl_pct = (price_with_slippage / existing["avg_cost"] - 1)

        remaining_shares = existing["shares"] - sell_shares
        upsert_position(
            self.portfolio_id, ticker, max(remaining_shares, 0), existing["avg_cost"],
        )

        if remaining_shares <= 0.001:
            self.positions = [p for p in self.positions if p["ticker"] != ticker]
        else:
            for pos in self.positions:
                if pos["ticker"] == ticker:
                    pos["shares"] = remaining_shares

        self.cash += proceeds
        record_trade(
            self.portfolio_id, ticker, "SELL", sell_shares, price_with_slippage,
            commission=COMMISSION_PER_TRADE,
            reason=reason or "Manual sell",
        )

        return {
            "success": True,
            "ticker": ticker,
            "shares": sell_shares,
            "price": price_with_slippage,
            "proceeds": proceeds,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "cash_remaining": self.cash,
        }

    def enforce_stop_losses(self) -> list[dict]:
        """Check all positions against stop-loss levels and sell if triggered."""
        triggered = []
        for pos in list(self.positions):
            stop = pos.get("stop_loss")
            if not stop:
                continue
            price = self._current_price(pos["ticker"])
            if price and price <= stop:
                result = self.sell(
                    pos["ticker"],
                    reason=f"Stop-loss triggered at ${price:.2f} (stop: ${stop:.2f})",
                )
                if result["success"]:
                    triggered.append({
                        "ticker": pos["ticker"],
                        "stop_price": stop,
                        "exit_price": price,
                        "pnl": result.get("pnl", 0),
                    })
        return triggered

    def rebalance(self, target_weights: dict[str, float]) -> list[dict]:
        """
        Rebalance portfolio to target weights.
        target_weights: {ticker: fraction} (must sum to <= 1.0)
        """
        actions = []
        total_value, _, _ = self.get_portfolio_value()

        for ticker, target_pct in target_weights.items():
            target_value = total_value * target_pct
            existing = next((p for p in self.positions if p["ticker"] == ticker), None)
            price = self._current_price(ticker)
            if not price:
                continue

            current_value = (existing["shares"] * price) if existing else 0
            diff = target_value - current_value

            if abs(diff) < total_value * 0.01:  # < 1% deviation, skip
                continue

            if diff > 0:
                # Need to buy more
                shares_to_buy = diff / price
                if self.cash >= shares_to_buy * price:
                    result = self.buy(ticker, diff / total_value, force_price=price)
                    actions.append({"action": "BUY", "ticker": ticker, **result})
            else:
                # Need to sell some
                shares_to_sell = abs(diff) / price
                result = self.sell(ticker, shares_to_sell, reason="Rebalance")
                actions.append({"action": "SELL", "ticker": ticker, **result})

        return actions

    def take_snapshot(self, spy_return: float = 0.0) -> dict:
        """Record portfolio snapshot for tracking."""
        total, pos_val, cash = self.get_portfolio_value()
        cum_ret = (total / self.initial_cash) - 1

        positions_data = []
        for pos in self.positions:
            price = self._current_price(pos["ticker"])
            if price:
                current_val = pos["shares"] * price
                unrealized_pnl = current_val - pos["shares"] * pos["avg_cost"]
                positions_data.append({
                    "ticker": pos["ticker"],
                    "shares": pos["shares"],
                    "avg_cost": pos["avg_cost"],
                    "current_price": price,
                    "current_value": current_val,
                    "unrealized_pnl": unrealized_pnl,
                    "unrealized_pnl_pct": (price / pos["avg_cost"] - 1) * 100,
                })

        snapshots = get_snapshots(self.portfolio_id)
        prev_total = snapshots[-1]["total_value"] if snapshots else self.initial_cash
        daily_pnl = total - prev_total

        save_snapshot(
            self.portfolio_id, total, cash, pos_val, daily_pnl, cum_ret, spy_return, positions_data
        )

        return {
            "total_value": total,
            "cash": cash,
            "positions_value": pos_val,
            "cumulative_return": cum_ret,
            "daily_pnl": daily_pnl,
            "positions": positions_data,
        }

    def get_performance_summary(self) -> dict:
        """Compute performance metrics from snapshot history."""
        snapshots = get_snapshots(self.portfolio_id)
        if not snapshots:
            total, _, cash = self.get_portfolio_value()
            return {"total_value": total, "cash": cash, "cumulative_return": 0, "num_snapshots": 0}

        values = [s["total_value"] for s in snapshots]
        dates = [s["snapshot_date"] for s in snapshots]
        returns = pd.Series(values).pct_change().dropna()

        from ..risk.metrics import compute_risk_metrics
        if len(returns) > 1:
            metrics = compute_risk_metrics(returns)
        else:
            metrics = None

        total, pos_val, cash = self.get_portfolio_value()
        cum_ret = (total / self.initial_cash) - 1
        trades = get_trade_history(self.portfolio_id)

        return {
            "total_value": total,
            "initial_value": self.initial_cash,
            "cash": cash,
            "positions_value": pos_val,
            "cumulative_return": cum_ret,
            "cumulative_pnl": total - self.initial_cash,
            "num_trades": len(trades),
            "num_positions": len(self.positions),
            "num_snapshots": len(snapshots),
            "sharpe_ratio": metrics.sharpe_ratio if metrics else None,
            "max_drawdown": metrics.max_drawdown if metrics else None,
            "volatility": metrics.volatility_ann if metrics else None,
            "best_month": metrics.best_month if metrics else None,
            "worst_month": metrics.worst_month if metrics else None,
            "positions": self.positions,
            "recent_snapshots": snapshots[-30:],
        }
