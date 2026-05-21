"""
Analyzes your trade history to extract your investment process:
- Win rate by sector, hold duration, entry signal type
- Best/worst performing patterns
- Behavioral tendencies (FOMO buys, patience, position sizing)
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


def build_trade_pairs(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Match buy→sell orders per symbol into round trips.
    Returns a DataFrame of completed trades with P&L.
    """
    if orders.empty:
        return pd.DataFrame()

    pairs = []
    for symbol, grp in orders.groupby("symbol"):
        grp = grp.sort_values("created_at").reset_index(drop=True)
        buy_queue: list[dict] = []

        for _, row in grp.iterrows():
            if row["side"] == "buy":
                buy_queue.append(row.to_dict())
            elif row["side"] == "sell" and buy_queue:
                buy = buy_queue.pop(0)
                hold_days = max(1, (row["updated_at"] - buy["created_at"]).days)
                pnl_pct = (row["avg_price"] - buy["avg_price"]) / buy["avg_price"] * 100 if buy["avg_price"] else 0
                pnl_dollar = (row["avg_price"] - buy["avg_price"]) * min(row["quantity"], buy["quantity"])
                pairs.append({
                    "symbol": symbol,
                    "buy_date": buy["created_at"],
                    "sell_date": row["updated_at"],
                    "hold_days": hold_days,
                    "buy_price": buy["avg_price"],
                    "sell_price": row["avg_price"],
                    "quantity": min(row["quantity"], buy["quantity"]),
                    "pnl_dollar": pnl_dollar,
                    "pnl_pct": pnl_pct,
                    "win": pnl_pct > 0,
                })

    if not pairs:
        return pd.DataFrame()

    df = pd.DataFrame(pairs)
    df = _enrich_with_market_context(df)
    return df


def _enrich_with_market_context(trades: pd.DataFrame) -> pd.DataFrame:
    """Add sector, market-cap bucket, volatility, and entry-RSI to each trade."""
    symbols = trades["symbol"].unique().tolist()
    sector_map: dict[str, str] = {}
    mcap_map: dict[str, str] = {}

    for sym in symbols:
        try:
            info = yf.Ticker(sym).info
            sector_map[sym] = info.get("sector", "Unknown")
            mc = info.get("marketCap", 0) or 0
            if mc >= 200_000_000_000:
                mcap_map[sym] = "Mega"
            elif mc >= 10_000_000_000:
                mcap_map[sym] = "Large"
            elif mc >= 2_000_000_000:
                mcap_map[sym] = "Mid"
            elif mc >= 300_000_000:
                mcap_map[sym] = "Small"
            else:
                mcap_map[sym] = "Micro"
        except Exception:
            sector_map[sym] = "Unknown"
            mcap_map[sym] = "Unknown"

    trades["sector"] = trades["symbol"].map(sector_map)
    trades["market_cap_bucket"] = trades["symbol"].map(mcap_map)

    # Entry RSI (14-day) computed from price history around buy date
    entry_rsies = []
    entry_regimes = []
    for _, row in trades.iterrows():
        rsi, regime = _get_entry_rsi_and_regime(row["symbol"], row["buy_date"])
        entry_rsies.append(rsi)
        entry_regimes.append(regime)

    trades["entry_rsi"] = entry_rsies
    trades["entry_regime"] = entry_regimes

    # Hold duration bucket
    trades["hold_bucket"] = pd.cut(
        trades["hold_days"],
        bins=[-1, 1, 7, 30, 90, 365, 9999],
        labels=["Intraday", "Swing (2-7d)", "Monthly (1mo)", "Quarterly (3mo)", "Annual (1yr)", "Long-term"],
    )

    return trades


def _get_entry_rsi_and_regime(symbol: str, buy_date: pd.Timestamp) -> tuple[float, str]:
    try:
        start = buy_date - pd.Timedelta(days=60)
        end = buy_date + pd.Timedelta(days=1)
        hist = yf.download(symbol, start=start.date(), end=end.date(), progress=False, auto_adjust=True)
        if len(hist) < 16:
            return 50.0, "Unknown"

        close = hist["Close"].squeeze()
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi_series = 100 - (100 / (1 + rs))
        rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0

        sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else float(close.mean())
        sma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else float(close.mean())
        price = float(close.iloc[-1])

        if price > sma50 and sma20 > sma50:
            regime = "Uptrend"
        elif price < sma50 and sma20 < sma50:
            regime = "Downtrend"
        else:
            regime = "Sideways"

        return round(rsi, 1), regime
    except Exception:
        return 50.0, "Unknown"


def analyze_process(trades: pd.DataFrame) -> dict:
    """
    Extract quantitative patterns from your trade history.
    Returns a rich stats dict used for display and recommendations.
    """
    if trades.empty:
        return {}

    stats: dict = {}

    # ── Overall metrics ──────────────────────────────────────────────────────
    stats["total_trades"] = len(trades)
    stats["win_rate"] = trades["win"].mean()
    stats["avg_pnl_pct"] = trades["pnl_pct"].mean()
    stats["median_pnl_pct"] = trades["pnl_pct"].median()
    stats["total_pnl"] = trades["pnl_dollar"].sum()
    stats["best_trade"] = trades.loc[trades["pnl_pct"].idxmax()].to_dict()
    stats["worst_trade"] = trades.loc[trades["pnl_pct"].idxmin()].to_dict()
    stats["avg_hold_days"] = trades["hold_days"].mean()

    # ── Profit factor ────────────────────────────────────────────────────────
    gains = trades.loc[trades["pnl_dollar"] > 0, "pnl_dollar"].sum()
    losses = abs(trades.loc[trades["pnl_dollar"] < 0, "pnl_dollar"].sum())
    stats["profit_factor"] = gains / losses if losses > 0 else float("inf")

    # ── By sector ────────────────────────────────────────────────────────────
    stats["by_sector"] = _group_stats(trades, "sector")

    # ── By hold duration ─────────────────────────────────────────────────────
    stats["by_hold_bucket"] = _group_stats(trades, "hold_bucket")

    # ── By market-cap bucket ──────────────────────────────────────────────────
    stats["by_market_cap"] = _group_stats(trades, "market_cap_bucket")

    # ── By entry regime ───────────────────────────────────────────────────────
    stats["by_entry_regime"] = _group_stats(trades, "entry_regime")

    # ── Entry RSI bands ───────────────────────────────────────────────────────
    trades["rsi_band"] = pd.cut(
        trades["entry_rsi"],
        bins=[0, 30, 45, 55, 70, 100],
        labels=["Oversold(<30)", "Low(30-45)", "Neutral(45-55)", "High(55-70)", "Overbought(>70)"],
    )
    stats["by_rsi_band"] = _group_stats(trades, "rsi_band")

    # ── Behavioral flags ──────────────────────────────────────────────────────
    # Chasing: buy after >10% run-up in same week
    stats["avg_loss_cut_days"] = (
        trades.loc[~trades["win"], "hold_days"].mean() if (~trades["win"]).any() else 0
    )
    stats["avg_winner_hold_days"] = (
        trades.loc[trades["win"], "hold_days"].mean() if trades["win"].any() else 0
    )
    # Let winners run vs cutting losers fast
    stats["patience_ratio"] = (
        stats["avg_winner_hold_days"] / stats["avg_loss_cut_days"]
        if stats["avg_loss_cut_days"] > 0 else 1.0
    )

    # ── Top winning symbols ───────────────────────────────────────────────────
    sym_stats = (
        trades.groupby("symbol")
        .agg(
            trades_count=("pnl_pct", "count"),
            win_rate=("win", "mean"),
            avg_pnl_pct=("pnl_pct", "mean"),
            total_pnl=("pnl_dollar", "sum"),
        )
        .sort_values("total_pnl", ascending=False)
        .reset_index()
    )
    stats["top_symbols"] = sym_stats.head(10)
    stats["worst_symbols"] = sym_stats.tail(5)

    # ── Edge profile ──────────────────────────────────────────────────────────
    stats["edge_profile"] = _build_edge_profile(stats)

    return stats


def _group_stats(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame()
    g = (
        df.groupby(col, observed=True)
        .agg(
            count=("pnl_pct", "count"),
            win_rate=("win", "mean"),
            avg_pnl_pct=("pnl_pct", "mean"),
            total_pnl=("pnl_dollar", "sum"),
        )
        .sort_values("total_pnl", ascending=False)
        .reset_index()
    )
    g["win_rate_pct"] = (g["win_rate"] * 100).round(1)
    g["avg_pnl_pct"] = g["avg_pnl_pct"].round(2)
    return g


def _build_edge_profile(stats: dict) -> dict:
    """Summarize WHERE the user has a measurable statistical edge."""
    edge = {}

    # Best sector by avg pnl
    sector_df = stats.get("by_sector", pd.DataFrame())
    if not sector_df.empty and "avg_pnl_pct" in sector_df.columns:
        top = sector_df.iloc[0]
        edge["best_sector"] = top.get("sector", "")
        edge["best_sector_win_rate"] = top.get("win_rate_pct", 0)
        edge["best_sector_avg_pnl"] = top.get("avg_pnl_pct", 0)

    # Best hold duration
    hold_df = stats.get("by_hold_bucket", pd.DataFrame())
    if not hold_df.empty:
        top = hold_df.iloc[0]
        edge["best_hold_duration"] = str(top.iloc[0])
        edge["best_hold_win_rate"] = top.get("win_rate_pct", 0)

    # Best RSI band
    rsi_df = stats.get("by_rsi_band", pd.DataFrame())
    if not rsi_df.empty:
        top = rsi_df.iloc[0]
        edge["best_rsi_band"] = str(top.iloc[0])

    # Best market cap
    mc_df = stats.get("by_market_cap", pd.DataFrame())
    if not mc_df.empty:
        top = mc_df.iloc[0]
        edge["best_market_cap"] = str(top.iloc[0])

    return edge
