"""
Backtesting engine: walk-forward strategy validation.
Tests if the quantitative thesis actually worked historically.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import warnings

warnings.filterwarnings("ignore")


@dataclass
class BacktestResult:
    ticker: str
    strategy: str
    total_return: float
    annualized_return: float
    benchmark_return: float
    alpha: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_holding_days: float
    best_trade: float
    worst_trade: float
    monthly_returns: list[float] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    passed: bool = True
    failure_reason: str = ""


def _compute_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().dropna()


def _sharpe_ratio(returns: pd.Series, rf: float = 0.05) -> float:
    if returns.empty or returns.std() == 0:
        return 0.0
    excess = returns - rf / 252
    return float(np.sqrt(252) * excess.mean() / excess.std())


def _max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.expanding().max()
    drawdown = (equity - roll_max) / roll_max
    return float(drawdown.min())


def _momentum_strategy_signals(prices: pd.DataFrame, lookback: int = 63) -> pd.Series:
    """
    Generate buy/sell signals based on momentum crossover.
    Long when price > rolling average, short/flat when below.
    """
    close = prices["Close"].squeeze() if "Close" in prices.columns else prices.iloc[:, 0].squeeze()
    sma = close.rolling(lookback).mean()
    signal = pd.Series(0.0, index=close.index)
    signal[close > sma] = 1.0
    signal[close < sma * 0.97] = -1.0  # slight buffer before going short
    return signal.shift(1)  # lag 1 day to avoid look-ahead


def _value_reversion_signals(prices: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Mean-reversion signal using Bollinger Bands.
    Buy when price drops below lower band, sell at upper band.
    """
    close = prices["Close"].squeeze() if "Close" in prices.columns else prices.iloc[:, 0].squeeze()
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    lower = sma - 2 * std
    upper = sma + 2 * std

    signal = pd.Series(0.0, index=close.index)
    signal[close < lower] = 1.0   # oversold → buy
    signal[close > upper] = -1.0  # overbought → sell
    signal[abs(close - sma) < std] = 0.0  # near mean → neutral
    return signal.shift(1)


def _apply_transaction_costs(returns: pd.Series, signal: pd.Series, cost_bps: float = 5) -> pd.Series:
    """Apply transaction costs on signal changes."""
    cost = abs(signal.diff().fillna(0)) * (cost_bps / 10000)
    return returns - cost


def _compute_equity_curve(strategy_returns: pd.Series, initial: float = 100.0) -> pd.Series:
    return (1 + strategy_returns).cumprod() * initial


def backtest_momentum(
    ticker: str,
    prices: pd.DataFrame,
    benchmark_prices: Optional[pd.DataFrame] = None,
    lookback: int = 63,
    cost_bps: float = 5,
) -> BacktestResult:
    """Walk-forward momentum strategy backtest."""
    if prices.empty or len(prices) < lookback + 50:
        return BacktestResult(
            ticker=ticker, strategy="momentum",
            total_return=0, annualized_return=0, benchmark_return=0,
            alpha=0, sharpe_ratio=0, max_drawdown=0,
            win_rate=0, profit_factor=0, num_trades=0,
            avg_holding_days=0, best_trade=0, worst_trade=0,
            passed=False, failure_reason="Insufficient price history",
        )

    close = prices["Close"].squeeze() if "Close" in prices.columns else prices.iloc[:, 0].squeeze()
    returns = _compute_returns(close)
    signal = _momentum_strategy_signals(prices, lookback).reindex(returns.index).fillna(0)

    strategy_returns = returns * signal
    strategy_returns = _apply_transaction_costs(strategy_returns, signal, cost_bps)

    equity = _compute_equity_curve(strategy_returns)
    n_years = len(returns) / 252

    total_ret = float(equity.iloc[-1] / 100 - 1)
    ann_ret = float((1 + total_ret) ** (1 / n_years) - 1) if n_years > 0 else 0

    # Benchmark
    if benchmark_prices is not None and not benchmark_prices.empty:
        bench_close = benchmark_prices["Close"].squeeze() if "Close" in benchmark_prices.columns else benchmark_prices.iloc[:, 0].squeeze()
        bench_ret = float(bench_close.iloc[-1] / bench_close.iloc[0] - 1)
    else:
        bench_ret = 0.10 * n_years  # assume 10%/year

    # Trade analysis
    positions = signal.diff().fillna(0)
    trade_entries = positions[positions != 0].index

    trades = []
    in_trade = False
    entry_price = None
    entry_date = None
    for date in returns.index:
        sig = signal.get(date, 0)
        price = close.get(date, np.nan)
        if sig != 0 and not in_trade and not np.isnan(price):
            in_trade = True
            entry_price = price
            entry_date = date
        elif sig == 0 and in_trade and entry_price and not np.isnan(price):
            trade_ret = price / entry_price - 1
            holding = (date - entry_date).days if entry_date else 0
            trades.append({"return": trade_ret, "holding_days": holding})
            in_trade = False

    win_rate = np.mean([t["return"] > 0 for t in trades]) if trades else 0
    wins = [t["return"] for t in trades if t["return"] > 0]
    losses = [abs(t["return"]) for t in trades if t["return"] <= 0]
    profit_factor = (np.sum(wins) / np.sum(losses)) if losses and wins else float("inf")
    avg_holding = np.mean([t["holding_days"] for t in trades]) if trades else 0

    monthly = strategy_returns.resample("ME").apply(lambda x: (1+x).prod() - 1).tolist()

    return BacktestResult(
        ticker=ticker,
        strategy="momentum",
        total_return=total_ret,
        annualized_return=ann_ret,
        benchmark_return=bench_ret,
        alpha=ann_ret - bench_ret / n_years if n_years > 0 else 0,
        sharpe_ratio=_sharpe_ratio(strategy_returns),
        max_drawdown=_max_drawdown(equity),
        win_rate=float(win_rate),
        profit_factor=float(profit_factor),
        num_trades=len(trades),
        avg_holding_days=float(avg_holding),
        best_trade=float(max((t["return"] for t in trades), default=0)),
        worst_trade=float(min((t["return"] for t in trades), default=0)),
        monthly_returns=monthly,
        equity_curve=equity.tolist(),
        passed=ann_ret > 0 and _sharpe_ratio(strategy_returns) > 0.3,
    )


def backtest_buy_and_hold(
    ticker: str,
    prices: pd.DataFrame,
    benchmark_prices: Optional[pd.DataFrame] = None,
) -> BacktestResult:
    """Simple buy-and-hold backtest for comparison."""
    if prices.empty:
        return BacktestResult(
            ticker=ticker, strategy="buy_and_hold",
            total_return=0, annualized_return=0, benchmark_return=0,
            alpha=0, sharpe_ratio=0, max_drawdown=0,
            win_rate=0, profit_factor=0, num_trades=1,
            avg_holding_days=365, best_trade=0, worst_trade=0,
            passed=False, failure_reason="No price data",
        )

    close = prices["Close"].squeeze() if "Close" in prices.columns else prices.iloc[:, 0].squeeze()
    returns = _compute_returns(close)
    n_years = len(returns) / 252

    total_ret = float(close.iloc[-1] / close.iloc[0] - 1)
    ann_ret = float((1 + total_ret) ** (1 / n_years) - 1) if n_years > 0 else 0
    equity = _compute_equity_curve(returns)

    if benchmark_prices is not None and not benchmark_prices.empty:
        bench_close = benchmark_prices["Close"].squeeze() if "Close" in benchmark_prices.columns else benchmark_prices.iloc[:, 0].squeeze()
        bench_total = float(bench_close.iloc[-1] / bench_close.iloc[0] - 1)
        bench_ann = float((1 + bench_total) ** (1 / n_years) - 1) if n_years > 0 else 0
    else:
        bench_ann = 0.10

    monthly = returns.resample("ME").apply(lambda x: (1+x).prod() - 1).tolist()

    return BacktestResult(
        ticker=ticker,
        strategy="buy_and_hold",
        total_return=total_ret,
        annualized_return=ann_ret,
        benchmark_return=bench_ann,
        alpha=ann_ret - bench_ann,
        sharpe_ratio=_sharpe_ratio(returns),
        max_drawdown=_max_drawdown(equity),
        win_rate=float(total_ret > 0),
        profit_factor=float(max(total_ret, 0)) / float(max(abs(min(total_ret, 0)), 0.001)),
        num_trades=1,
        avg_holding_days=len(returns),
        best_trade=total_ret if total_ret > 0 else 0,
        worst_trade=total_ret if total_ret <= 0 else 0,
        monthly_returns=monthly,
        equity_curve=equity.tolist(),
        passed=ann_ret > bench_ann,
    )


def run_full_backtest(
    ticker: str,
    prices: pd.DataFrame,
    spy_prices: Optional[pd.DataFrame] = None,
) -> dict[str, BacktestResult]:
    """Run all backtest strategies and return results dict."""
    return {
        "momentum": backtest_momentum(ticker, prices, spy_prices),
        "buy_and_hold": backtest_buy_and_hold(ticker, prices, spy_prices),
    }
