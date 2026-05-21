"""Momentum signals: price and earnings momentum."""

import numpy as np
import pandas as pd
from scipy import stats


def momentum_12_1(prices: pd.DataFrame) -> pd.Series:
    """
    Classic 12-1 month momentum: 12-month return excluding the last month.
    Avoids short-term reversal contamination.
    """
    if len(prices) < 252:
        return pd.Series(dtype=float)
    close = prices["Close"] if "Close" in prices.columns else prices.iloc[:, 0]
    ret_12m = close.pct_change(252)
    ret_1m = close.pct_change(21)
    return ret_12m - ret_1m


def momentum_6_1(prices: pd.DataFrame) -> float:
    """6-1 month momentum."""
    if len(prices) < 126:
        return np.nan
    close = prices["Close"] if "Close" in prices.columns else prices.iloc[:, 0]
    ret_6m = close.iloc[-1] / close.iloc[-126] - 1
    ret_1m = close.iloc[-1] / close.iloc[-21] - 1
    return ret_6m - ret_1m


def momentum_score(prices: pd.DataFrame) -> float:
    """
    Composite momentum score combining multiple lookback windows.
    Returns z-scored composite.
    """
    if prices.empty or len(prices) < 63:
        return np.nan

    close = prices["Close"].squeeze() if "Close" in prices.columns else prices.iloc[:, 0].squeeze()
    close = close.dropna()

    if len(close) < 63:
        return np.nan

    scores = {}

    # Price momentum at different horizons
    for days, label in [(63, "3m"), (126, "6m"), (252, "12m")]:
        if len(close) > days:
            scores[label] = close.iloc[-1] / close.iloc[-days] - 1

    if not scores:
        return np.nan

    # Trend strength via linear regression slope
    if len(close) >= 63:
        x = np.arange(min(63, len(close)))
        y = close.iloc[-len(x):].values
        if len(y) >= 10:
            slope, _, r, _, _ = stats.linregress(x, y)
            scores["trend_r2"] = r ** 2 * np.sign(slope)

    return float(np.nanmean(list(scores.values())))


def rsi(prices: pd.DataFrame, period: int = 14) -> float:
    """Relative Strength Index."""
    if prices.empty:
        return np.nan
    close = prices["Close"].squeeze() if "Close" in prices.columns else prices.iloc[:, 0].squeeze()
    close = close.dropna()
    if len(close) < period + 1:
        return np.nan

    delta = close.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean().iloc[-1]
    avg_loss = loss.rolling(period).mean().iloc[-1]

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - 100 / (1 + rs))


def macd_signal(prices: pd.DataFrame) -> float:
    """
    MACD signal: positive means bullish momentum.
    Returns MACD line - Signal line (normalized by price).
    """
    close = prices["Close"].squeeze() if "Close" in prices.columns else prices.iloc[:, 0].squeeze()
    close = close.dropna()
    if len(close) < 26:
        return np.nan

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = (macd_line - signal_line).iloc[-1]

    return float(histogram / close.iloc[-1])  # normalize by price


def bollinger_position(prices: pd.DataFrame, window: int = 20) -> float:
    """
    Position within Bollinger Bands: -1 (lower band) to +1 (upper band).
    Positive = price above midband (bullish).
    """
    close = prices["Close"].squeeze() if "Close" in prices.columns else prices.iloc[:, 0].squeeze()
    close = close.dropna()
    if len(close) < window:
        return np.nan

    sma = close.rolling(window).mean().iloc[-1]
    std = close.rolling(window).std().iloc[-1]
    price = close.iloc[-1]

    if std == 0:
        return 0.0
    return float((price - sma) / (2 * std))


def volume_trend(prices: pd.DataFrame) -> float:
    """Rising volume on up days vs down days (OBV-style signal)."""
    if prices.empty or "Close" not in prices.columns or "Volume" not in prices.columns:
        return np.nan
    close = prices["Close"].squeeze().dropna()
    volume = prices["Volume"].squeeze().dropna()

    if len(close) < 21:
        return np.nan

    returns = close.pct_change().dropna()
    # Align volume to returns index to avoid index mismatch
    volume_aligned = volume.reindex(returns.index).dropna()
    returns_aligned = returns.reindex(volume_aligned.index).dropna()
    volume_aligned = volume_aligned.reindex(returns_aligned.index)

    up_vol = volume_aligned[returns_aligned > 0].tail(21).mean()
    down_vol = volume_aligned[returns_aligned < 0].tail(21).mean()

    if np.isnan(up_vol) or np.isnan(down_vol) or (up_vol + down_vol) == 0:
        return 0.0
    return float((up_vol - down_vol) / (up_vol + down_vol))
