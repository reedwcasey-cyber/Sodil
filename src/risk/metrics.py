"""Risk metrics: VaR, CVaR, Sharpe, Sortino, Calmar, drawdown analysis."""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from scipy import stats


@dataclass
class RiskMetrics:
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    avg_drawdown: float
    var_95: float           # 1-day 95% VaR
    cvar_95: float          # 1-day 95% CVaR (Expected Shortfall)
    var_99: float           # 1-day 99% VaR
    volatility_ann: float   # annualized volatility
    downside_vol: float     # downside volatility
    beta: float
    alpha_ann: float
    information_ratio: float
    skewness: float
    kurtosis: float
    best_month: float
    worst_month: float
    positive_months_pct: float


def compute_risk_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    rf: float = 0.05,
) -> RiskMetrics:
    """
    Compute comprehensive risk metrics for a return series.
    All returns should be daily.
    """
    r = portfolio_returns.dropna()
    if r.empty:
        return _empty_metrics()

    rf_daily = rf / 252
    excess = r - rf_daily

    # Volatility
    vol_ann = float(r.std() * np.sqrt(252))
    downside = r[r < rf_daily]
    downside_vol = float(downside.std() * np.sqrt(252)) if len(downside) > 1 else vol_ann

    # Sharpe
    sharpe = float(excess.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0

    # Sortino: penalize only downside volatility
    sortino = float(excess.mean() / (downside.std() if len(downside) > 1 else r.std()) * np.sqrt(252))

    # Equity curve and drawdown
    equity = (1 + r).cumprod()
    roll_max = equity.expanding().max()
    drawdowns = (equity - roll_max) / roll_max
    max_dd = float(drawdowns.min())
    avg_dd = float(drawdowns[drawdowns < 0].mean()) if (drawdowns < 0).any() else 0

    # Calmar: annualized return / max drawdown
    ann_ret = float((1 + r.sum()) ** (252 / len(r)) - 1)
    calmar = float(ann_ret / abs(max_dd)) if max_dd != 0 else 0

    # VaR / CVaR (parametric using historical simulation)
    var_95 = float(np.percentile(r, 5))     # 5th percentile loss
    var_99 = float(np.percentile(r, 1))     # 1st percentile loss
    cvar_95 = float(r[r <= var_95].mean()) if (r <= var_95).any() else var_95

    # Beta and alpha vs benchmark
    beta = 1.0
    alpha_ann = 0.0
    ir = 0.0
    if benchmark_returns is not None:
        bench = benchmark_returns.dropna().reindex(r.index).dropna()
        if len(bench) > 20:
            aligned_r = r.reindex(bench.index).dropna()
            if len(aligned_r) > 20:
                cov = np.cov(aligned_r, bench)[0, 1]
                bench_var = bench.var()
                beta = float(cov / bench_var) if bench_var > 0 else 1.0
                alpha_daily = aligned_r.mean() - beta * bench.mean()
                alpha_ann = float(alpha_daily * 252)
                active_returns = aligned_r - bench
                ir = float(active_returns.mean() / active_returns.std() * np.sqrt(252)) if active_returns.std() > 0 else 0

    # Monthly stats
    monthly = r.resample("ME").apply(lambda x: (1+x).prod() - 1)
    best_month = float(monthly.max()) if not monthly.empty else 0
    worst_month = float(monthly.min()) if not monthly.empty else 0
    pos_months = float((monthly > 0).mean()) if not monthly.empty else 0

    # Distribution
    skewness = float(stats.skew(r))
    kurt = float(stats.kurtosis(r))

    return RiskMetrics(
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_dd,
        avg_drawdown=avg_dd,
        var_95=var_95,
        cvar_95=cvar_95,
        var_99=var_99,
        volatility_ann=vol_ann,
        downside_vol=downside_vol,
        beta=beta,
        alpha_ann=alpha_ann,
        information_ratio=ir,
        skewness=skewness,
        kurtosis=kurt,
        best_month=best_month,
        worst_month=worst_month,
        positive_months_pct=pos_months,
    )


def _empty_metrics() -> RiskMetrics:
    return RiskMetrics(
        sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
        max_drawdown=0, avg_drawdown=0, var_95=0, cvar_95=0, var_99=0,
        volatility_ann=0, downside_vol=0, beta=1, alpha_ann=0,
        information_ratio=0, skewness=0, kurtosis=0,
        best_month=0, worst_month=0, positive_months_pct=0,
    )


def position_var(
    position_value: float,
    returns: pd.Series,
    confidence: float = 0.95,
) -> dict:
    """Dollar VaR and CVaR for a specific position."""
    r = returns.dropna()
    if r.empty:
        return {"var_1d": 0, "cvar_1d": 0, "var_10d": 0}

    pct_var = float(np.percentile(r, (1 - confidence) * 100))
    pct_cvar = float(r[r <= pct_var].mean()) if (r <= pct_var).any() else pct_var

    return {
        "var_1d": abs(pct_var * position_value),
        "cvar_1d": abs(pct_cvar * position_value),
        "var_10d": abs(pct_var * position_value * np.sqrt(10)),
    }
