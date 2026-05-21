"""Market data helpers using yfinance."""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


def get_ticker_fundamentals(symbol: str) -> dict:
    """Return key fundamental fields for a symbol."""
    try:
        info = yf.Ticker(symbol).info
        return {
            "symbol": symbol,
            "name": info.get("longName", symbol),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "market_cap": info.get("marketCap", 0) or 0,
            "pe_ratio": info.get("trailingPE", None),
            "fwd_pe": info.get("forwardPE", None),
            "peg": info.get("pegRatio", None),
            "price_to_book": info.get("priceToBook", None),
            "revenue_growth": info.get("revenueGrowth", None),
            "earnings_growth": info.get("earningsGrowth", None),
            "profit_margin": info.get("profitMargins", None),
            "debt_to_equity": info.get("debtToEquity", None),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice", 0),
            "52w_high": info.get("fiftyTwoWeekHigh", 0),
            "52w_low": info.get("fiftyTwoWeekLow", 0),
            "avg_volume": info.get("averageVolume", 0) or 0,
            "beta": info.get("beta", 1.0) or 1.0,
            "dividend_yield": info.get("dividendYield", 0) or 0,
            "short_ratio": info.get("shortRatio", 0) or 0,
        }
    except Exception:
        return {"symbol": symbol, "name": symbol, "sector": "Unknown"}


def get_price_history(symbol: str, period: str = "1y") -> pd.DataFrame:
    """Return OHLCV price history."""
    try:
        df = yf.download(symbol, period=period, progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


def compute_technicals(hist: pd.DataFrame) -> dict:
    """Compute RSI, MACD, Bollinger Bands, and trend info from price history."""
    if hist.empty or len(hist) < 20:
        return {}

    close = hist["close"].squeeze() if "close" in hist.columns else hist.iloc[:, 0]

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = float((100 - 100 / (1 + rs)).iloc[-1])

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line

    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper_band = sma20 + 2 * std20
    lower_band = sma20 - 2 * std20
    bb_pct = float(((close - lower_band) / (upper_band - lower_band)).iloc[-1])

    # SMAs
    sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else float(close.mean())
    sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else float(close.mean())
    price = float(close.iloc[-1])

    # Trend
    if price > sma50 > sma200:
        trend = "Strong Uptrend"
    elif price > sma50:
        trend = "Uptrend"
    elif price < sma50 < sma200:
        trend = "Strong Downtrend"
    elif price < sma50:
        trend = "Downtrend"
    else:
        trend = "Sideways"

    # Momentum (20-day)
    momentum_20d = float((close.pct_change(20) * 100).iloc[-1])
    momentum_5d = float((close.pct_change(5) * 100).iloc[-1])

    # Volatility (20-day annualized)
    vol_20d = float(close.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100)

    return {
        "rsi": round(rsi, 1),
        "macd": round(float(macd_line.iloc[-1]), 4),
        "macd_signal": round(float(signal_line.iloc[-1]), 4),
        "macd_hist": round(float(macd_hist.iloc[-1]), 4),
        "macd_bullish": float(macd_hist.iloc[-1]) > 0,
        "bb_pct": round(bb_pct, 3),
        "sma20": round(float(sma20.iloc[-1]), 2),
        "sma50": round(sma50, 2),
        "sma200": round(sma200, 2),
        "price": round(price, 2),
        "trend": trend,
        "above_sma50": price > sma50,
        "above_sma200": price > sma200,
        "momentum_20d": round(momentum_20d, 2),
        "momentum_5d": round(momentum_5d, 2),
        "volatility_20d": round(vol_20d, 2),
    }


def get_options_chain(symbol: str) -> pd.DataFrame:
    """Return the nearest-expiry options chain for a symbol."""
    try:
        tk = yf.Ticker(symbol)
        exps = tk.options
        if not exps:
            return pd.DataFrame()

        calls = tk.option_chain(exps[0]).calls
        puts = tk.option_chain(exps[0]).puts
        calls["option_type"] = "call"
        puts["option_type"] = "put"
        chain = pd.concat([calls, puts], ignore_index=True)
        chain["expiry"] = exps[0]
        chain["underlying"] = symbol
        return chain
    except Exception:
        return pd.DataFrame()
