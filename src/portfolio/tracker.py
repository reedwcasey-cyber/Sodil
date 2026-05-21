"""
Portfolio performance tracker: P&L attribution, benchmark comparison,
performance attribution, and reporting.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from .db import get_snapshots, get_positions, get_trade_history
from ..data.fetcher import fetch_price_history
from ..risk.metrics import compute_risk_metrics, RiskMetrics


@dataclass
class PositionPerformance:
    ticker: str
    shares: float
    avg_cost: float
    current_price: float
    current_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    weight_in_portfolio: float
    contribution_to_return: float


@dataclass
class PortfolioReport:
    portfolio_name: str
    total_value: float
    initial_value: float
    cumulative_return: float
    benchmark_return: float
    alpha: float
    risk_metrics: Optional[RiskMetrics]
    position_performance: list[PositionPerformance]
    top_winner: Optional[PositionPerformance]
    top_loser: Optional[PositionPerformance]
    realized_pnl: float
    unrealized_pnl: float
    trade_count: int
    win_rate: float
    avg_win: float
    avg_loss: float
    best_day: float
    worst_day: float
    equity_curve: list[tuple[str, float]] = field(default_factory=list)


def compute_realized_pnl(trades: list[dict]) -> tuple[float, float, float, float]:
    """
    Compute realized P&L from trade history.
    Returns (total_pnl, win_rate, avg_win, avg_loss).
    """
    buys = {}
    realized = []

    for trade in trades:
        ticker = trade["ticker"]
        if trade["action"] == "BUY":
            if ticker not in buys:
                buys[ticker] = []
            buys[ticker].append({
                "shares": trade["shares"],
                "price": trade["price"],
            })
        elif trade["action"] == "SELL" and ticker in buys:
            sell_shares = trade["shares"]
            sell_price = trade["price"]
            cost_basis = 0.0
            remaining = sell_shares

            for buy in list(buys[ticker]):
                if remaining <= 0:
                    break
                used = min(remaining, buy["shares"])
                cost_basis += used * buy["price"]
                buy["shares"] -= used
                remaining -= used

            buys[ticker] = [b for b in buys[ticker] if b["shares"] > 0.001]
            trade_pnl = sell_shares * sell_price - cost_basis
            realized.append(trade_pnl)

    if not realized:
        return 0.0, 0.0, 0.0, 0.0

    wins = [p for p in realized if p > 0]
    losses = [p for p in realized if p <= 0]
    win_rate = len(wins) / len(realized)
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0

    return sum(realized), win_rate, avg_win, avg_loss


def generate_report(
    portfolio_id: int,
    portfolio_name: str,
    initial_cash: float,
    spy_ticker: str = "SPY",
) -> PortfolioReport:
    """Generate comprehensive portfolio performance report."""
    snapshots = get_snapshots(portfolio_id)
    positions = get_positions(portfolio_id)
    trades = get_trade_history(portfolio_id)

    # Current portfolio value from positions
    total_position_value = 0.0
    pos_perf = []

    for pos in positions:
        try:
            df = fetch_price_history(pos["ticker"], period="5d")
            if df.empty:
                continue
            close = df["Close"].squeeze() if "Close" in df.columns else df.iloc[:, 0].squeeze()
            price = float(close.iloc[-1])
        except Exception:
            price = pos["avg_cost"]

        value = pos["shares"] * price
        total_position_value += value
        unreal_pnl = value - pos["shares"] * pos["avg_cost"]
        unreal_pct = (price / pos["avg_cost"] - 1) * 100 if pos["avg_cost"] > 0 else 0

        pos_perf.append(PositionPerformance(
            ticker=pos["ticker"],
            shares=pos["shares"],
            avg_cost=pos["avg_cost"],
            current_price=price,
            current_value=value,
            unrealized_pnl=unreal_pnl,
            unrealized_pnl_pct=unreal_pct,
            weight_in_portfolio=0,  # filled below
            contribution_to_return=0,  # filled below
        ))

    # Determine cash from latest snapshot or initial
    if snapshots:
        latest_snap = snapshots[-1]
        cash = latest_snap["cash"]
    else:
        cash = initial_cash

    total_value = total_position_value + cash
    cum_return = (total_value / initial_cash) - 1

    # Fill weights and contributions
    for pp in pos_perf:
        pp.weight_in_portfolio = pp.current_value / total_value if total_value > 0 else 0
        pp.contribution_to_return = pp.unrealized_pnl_pct / 100 * pp.weight_in_portfolio

    # Risk metrics from snapshot history
    risk_metrics = None
    equity_curve = []
    best_day = 0.0
    worst_day = 0.0
    if len(snapshots) >= 2:
        values = pd.Series([s["total_value"] for s in snapshots])
        dates = [s["snapshot_date"][:10] for s in snapshots]
        daily_returns = values.pct_change().dropna()
        risk_metrics = compute_risk_metrics(daily_returns)
        equity_curve = list(zip(dates, values.tolist()))
        best_day = float(daily_returns.max())
        worst_day = float(daily_returns.min())

    # Benchmark (SPY) return
    try:
        spy_df = fetch_price_history(spy_ticker, period="1y")
        spy_close = spy_df["Close"].squeeze() if "Close" in spy_df.columns else spy_df.iloc[:, 0].squeeze()
        bench_return = float(spy_close.iloc[-1] / spy_close.iloc[0] - 1)
    except Exception:
        bench_return = 0.0

    realized_pnl, win_rate, avg_win, avg_loss = compute_realized_pnl(trades)
    unrealized_pnl = sum(pp.unrealized_pnl for pp in pos_perf)

    top_winner = max(pos_perf, key=lambda p: p.unrealized_pnl_pct) if pos_perf else None
    top_loser = min(pos_perf, key=lambda p: p.unrealized_pnl_pct) if pos_perf else None

    return PortfolioReport(
        portfolio_name=portfolio_name,
        total_value=total_value,
        initial_value=initial_cash,
        cumulative_return=cum_return,
        benchmark_return=bench_return,
        alpha=cum_return - bench_return,
        risk_metrics=risk_metrics,
        position_performance=sorted(pos_perf, key=lambda p: p.unrealized_pnl, reverse=True),
        top_winner=top_winner,
        top_loser=top_loser,
        realized_pnl=realized_pnl,
        unrealized_pnl=unrealized_pnl,
        trade_count=len(trades),
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        best_day=best_day,
        worst_day=worst_day,
        equity_curve=equity_curve,
    )
