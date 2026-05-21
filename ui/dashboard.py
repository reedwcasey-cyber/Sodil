"""
Rich terminal dashboard — renders portfolio, process analysis, and recommendations.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()


# ── Portfolio Summary ─────────────────────────────────────────────────────────

def render_portfolio_summary(portfolio: dict, positions: pd.DataFrame) -> None:
    equity = portfolio.get("equity", 0)
    cash = portfolio.get("cash", 0)
    buying_power = portfolio.get("buying_power", 0)
    day_pnl = sum(positions["day_return_pct"] * positions["market_value"] / 100) if not positions.empty else 0

    table = Table(box=box.ROUNDED, show_header=False, padding=(0, 2))
    table.add_column("Label", style="bold cyan", min_width=22)
    table.add_column("Value", style="bold white", justify="right", min_width=14)

    table.add_row("Portfolio Value", f"${equity:,.2f}")
    table.add_row("Cash", f"${cash:,.2f}")
    table.add_row("Buying Power", f"${buying_power:,.2f}")
    day_color = "green" if day_pnl >= 0 else "red"
    sign = "+" if day_pnl >= 0 else ""
    table.add_row("Today's P&L (est.)", f"[{day_color}]{sign}${day_pnl:,.2f}[/]")

    if not positions.empty:
        total_invested = (positions["avg_cost"] * positions["quantity"]).sum()
        total_value = positions["market_value"].sum()
        total_ret = total_value - total_invested
        ret_pct = total_ret / total_invested * 100 if total_invested else 0
        ret_color = "green" if total_ret >= 0 else "red"
        sign2 = "+" if total_ret >= 0 else ""
        table.add_row("Position Return", f"[{ret_color}]{sign2}${total_ret:,.2f} ({sign2}{ret_pct:.1f}%)[/]")

    console.print(Panel(table, title="[bold blue]Portfolio Overview[/]", border_style="blue"))


def render_positions(positions: pd.DataFrame) -> None:
    if positions.empty:
        console.print("[yellow]No open positions.[/]")
        return

    t = Table(box=box.SIMPLE_HEAD, show_lines=False)
    t.add_column("Symbol", style="bold cyan", min_width=8)
    t.add_column("Qty", justify="right", min_width=8)
    t.add_column("Avg Cost", justify="right", min_width=10)
    t.add_column("Price", justify="right", min_width=10)
    t.add_column("Value", justify="right", min_width=12)
    t.add_column("Return $", justify="right", min_width=12)
    t.add_column("Return %", justify="right", min_width=10)
    t.add_column("Day %", justify="right", min_width=9)

    for _, row in positions.iterrows():
        ret_color = "green" if row["return_pct"] >= 0 else "red"
        day_color = "green" if row["day_return_pct"] >= 0 else "red"
        sign = "+" if row["return_pct"] >= 0 else ""
        day_sign = "+" if row["day_return_pct"] >= 0 else ""
        t.add_row(
            row["symbol"],
            f"{row['quantity']:.2f}",
            f"${row['avg_cost']:.2f}",
            f"${row['current_price']:.2f}",
            f"${row['market_value']:,.2f}",
            f"[{ret_color}]{sign}${row['total_return']:,.2f}[/]",
            f"[{ret_color}]{sign}{row['return_pct']:.1f}%[/]",
            f"[{day_color}]{day_sign}{row['day_return_pct']:.1f}%[/]",
        )

    console.print(Panel(t, title="[bold blue]Current Positions[/]", border_style="blue"))


# ── Process Analysis ──────────────────────────────────────────────────────────

def render_process_stats(stats: dict) -> None:
    if not stats:
        console.print("[yellow]No trade history to analyze.[/]")
        return

    # Top-level stats row
    wrate = stats.get("win_rate", 0) * 100
    pf = stats.get("profit_factor", 0)
    avg_pnl = stats.get("avg_pnl_pct", 0)
    total_pnl = stats.get("total_pnl", 0)
    n = stats.get("total_trades", 0)
    patience = stats.get("patience_ratio", 1.0)

    header_table = Table(box=box.ROUNDED, show_header=False, padding=(0, 2))
    header_table.add_column("", min_width=16, style="bold cyan")
    header_table.add_column("", justify="right", min_width=12, style="bold white")
    header_table.add_column("", min_width=16, style="bold cyan")
    header_table.add_column("", justify="right", min_width=12, style="bold white")

    wrate_color = "green" if wrate >= 50 else "yellow" if wrate >= 40 else "red"
    pf_color = "green" if pf >= 1.5 else "yellow" if pf >= 1.0 else "red"

    header_table.add_row(
        "Trades Analyzed", str(n),
        "Win Rate", f"[{wrate_color}]{wrate:.1f}%[/]",
    )
    header_table.add_row(
        "Profit Factor", f"[{pf_color}]{pf:.2f}[/]",
        "Avg Trade P&L", f"{'+'if avg_pnl>=0 else ''}{avg_pnl:.1f}%",
    )
    pnl_color = "green" if total_pnl >= 0 else "red"
    header_table.add_row(
        "Total P&L", f"[{pnl_color}]{'+'if total_pnl>=0 else ''}${total_pnl:,.2f}[/]",
        "Patience Ratio", f"{patience:.2f}x",
    )

    console.print(Panel(header_table, title="[bold magenta]Your Trade Process — Stats[/]", border_style="magenta"))

    # By sector
    _render_breakdown(stats.get("by_sector"), "Sector Breakdown", "sector")
    # By hold duration
    _render_breakdown(stats.get("by_hold_bucket"), "Hold Duration Breakdown", "hold_bucket")
    # By RSI band
    _render_breakdown(stats.get("by_rsi_band"), "Entry RSI Band Breakdown", "rsi_band")
    # By regime
    _render_breakdown(stats.get("by_entry_regime"), "Entry Market Regime", "entry_regime")

    # Edge profile summary
    edge = stats.get("edge_profile", {})
    if edge:
        _render_edge_profile(edge, stats)

    # Best/worst trades
    _render_best_worst(stats)


def _render_breakdown(df: Optional[pd.DataFrame], title: str, col: str) -> None:
    if df is None or df.empty:
        return

    t = Table(box=box.SIMPLE_HEAD, show_lines=False, title=f"[bold]{title}[/]")
    label_col = df.columns[0]
    t.add_column(label_col.replace("_", " ").title(), min_width=20)
    t.add_column("Trades", justify="right", min_width=8)
    t.add_column("Win %", justify="right", min_width=8)
    t.add_column("Avg P&L %", justify="right", min_width=10)
    t.add_column("Total P&L", justify="right", min_width=12)

    for _, row in df.iterrows():
        wr = row.get("win_rate_pct", 0)
        pnl = row.get("avg_pnl_pct", 0)
        total = row.get("total_pnl", 0)
        wr_color = "green" if wr >= 50 else "yellow" if wr >= 40 else "red"
        pnl_color = "green" if pnl >= 0 else "red"
        total_color = "green" if total >= 0 else "red"
        t.add_row(
            str(row.iloc[0]),
            str(int(row.get("count", 0))),
            f"[{wr_color}]{wr:.1f}%[/]",
            f"[{pnl_color}]{'+' if pnl>=0 else ''}{pnl:.1f}%[/]",
            f"[{total_color}]{'+' if total>=0 else ''}${total:,.0f}[/]",
        )

    console.print(t)
    console.print()


def _render_edge_profile(edge: dict, stats: dict) -> None:
    t = Table(box=box.ROUNDED, show_header=False, padding=(0, 2),
              title="[bold green]Your Statistical Edge[/]")
    t.add_column("Edge Factor", style="bold cyan", min_width=24)
    t.add_column("Value", style="bold green", min_width=24)
    t.add_column("Insight", style="italic", min_width=36)

    best_s = edge.get("best_sector", "N/A")
    best_s_wr = edge.get("best_sector_win_rate", 0)
    best_s_pnl = edge.get("best_sector_avg_pnl", 0)
    if best_s and best_s != "N/A":
        t.add_row("Best Sector", best_s, f"{best_s_wr:.0f}% win rate, avg {best_s_pnl:+.1f}% per trade")

    best_h = edge.get("best_hold_duration", "N/A")
    best_h_wr = edge.get("best_hold_win_rate", 0)
    if best_h and best_h != "N/A":
        t.add_row("Best Hold Duration", best_h, f"{best_h_wr:.0f}% win rate in this timeframe")

    best_r = edge.get("best_rsi_band", "N/A")
    if best_r and best_r != "N/A":
        t.add_row("Best Entry RSI", best_r, "Historically most profitable entry zone")

    best_mc = edge.get("best_market_cap", "N/A")
    if best_mc and best_mc != "N/A":
        t.add_row("Best Market Cap", best_mc + " cap", "Where you have the most alpha")

    avg_win_hold = stats.get("avg_winner_hold_days", 0)
    avg_loss_hold = stats.get("avg_loss_cut_days", 0)
    patience = stats.get("patience_ratio", 1.0)
    p_color = "green" if patience >= 1.5 else "yellow" if patience >= 0.9 else "red"
    p_note = "Letting winners run ✓" if patience >= 1.5 else "Cutting losers fast ✓" if patience >= 1.0 else "Cutting winners early — improve this"
    t.add_row(
        "Patience Ratio",
        f"[{p_color}]{patience:.2f}x[/]",
        f"Winners held {avg_win_hold:.0f}d vs losers {avg_loss_hold:.0f}d — {p_note}",
    )

    console.print(Panel(t, border_style="green"))


def _render_best_worst(stats: dict) -> None:
    best = stats.get("best_trade", {})
    worst = stats.get("worst_trade", {})

    if not best and not worst:
        return

    t = Table(box=box.SIMPLE_HEAD, show_header=True)
    t.add_column("", style="bold", min_width=8)
    t.add_column("Symbol", min_width=8)
    t.add_column("P&L %", justify="right", min_width=10)
    t.add_column("P&L $", justify="right", min_width=12)
    t.add_column("Hold Days", justify="right", min_width=10)
    t.add_column("Sector", min_width=16)

    if best:
        t.add_row(
            "[green]Best[/]",
            str(best.get("symbol", "")),
            f"[green]+{best.get('pnl_pct', 0):.1f}%[/]",
            f"[green]+${best.get('pnl_dollar', 0):,.2f}[/]",
            str(int(best.get("hold_days", 0))),
            str(best.get("sector", "")),
        )
    if worst:
        t.add_row(
            "[red]Worst[/]",
            str(worst.get("symbol", "")),
            f"[red]{worst.get('pnl_pct', 0):.1f}%[/]",
            f"[red]${worst.get('pnl_dollar', 0):,.2f}[/]",
            str(int(worst.get("hold_days", 0))),
            str(worst.get("sector", "")),
        )

    console.print(Panel(t, title="[bold]Best & Worst Completed Trades[/]", border_style="yellow"))


# ── Recommendations ───────────────────────────────────────────────────────────

def render_recommendations(recs: pd.DataFrame, title: str = "Recommendations") -> None:
    if recs is None or recs.empty:
        console.print("[yellow]No recommendations generated.[/]")
        return

    t = Table(box=box.ROUNDED, show_lines=True, title=f"[bold yellow]{title}[/]")
    t.add_column("Score", justify="center", min_width=7, style="bold")
    t.add_column("Symbol", style="bold cyan", min_width=7)
    t.add_column("Name", min_width=20)
    t.add_column("Sector", min_width=18)
    t.add_column("Price", justify="right", min_width=9)
    t.add_column("RSI", justify="right", min_width=6)
    t.add_column("Trend", min_width=16)
    t.add_column("20d Mom", justify="right", min_width=9)
    t.add_column("MACD", justify="center", min_width=7)
    t.add_column("Rationale", min_width=40)

    for _, row in recs.iterrows():
        score = row.get("score", 0)
        score_color = "green" if score >= 70 else "yellow" if score >= 50 else "white"
        rsi = row.get("rsi", 50)
        rsi_color = "green" if 30 <= rsi <= 50 else "yellow" if rsi < 65 else "red"
        trend = str(row.get("trend", ""))
        trend_color = "green" if "Up" in trend else "red" if "Down" in trend else "white"
        mom = row.get("momentum_20d", 0)
        mom_color = "green" if mom > 5 else "red" if mom < -5 else "white"
        macd = "✓" if row.get("macd_bullish", False) else "✗"
        macd_color = "green" if row.get("macd_bullish", False) else "dim red"
        price = row.get("current_price", 0)
        name = str(row.get("name", ""))[:20]

        t.add_row(
            f"[{score_color}]{score:.0f}[/]",
            row.get("symbol", ""),
            name,
            str(row.get("sector", "")),
            f"${price:.2f}" if price else "N/A",
            f"[{rsi_color}]{rsi:.0f}[/]",
            f"[{trend_color}]{trend}[/]",
            f"[{mom_color}]{mom:+.1f}%[/]" if mom else "N/A",
            f"[{macd_color}]{macd}[/]",
            str(row.get("rationale", ""))[:60],
        )

    console.print(t)


def render_options_recommendations(recs: pd.DataFrame) -> None:
    if recs is None or recs.empty:
        console.print("[yellow]No options recommendations generated.[/]")
        return

    t = Table(box=box.ROUNDED, show_lines=True, title="[bold yellow]Options Recommendations[/]")
    t.add_column("Score", justify="center", min_width=7, style="bold")
    t.add_column("Symbol", style="bold cyan", min_width=7)
    t.add_column("Strategy", style="bold yellow", min_width=20)
    t.add_column("Price", justify="right", min_width=9)
    t.add_column("RSI", justify="right", min_width=6)
    t.add_column("IV (Vol)", justify="right", min_width=9)
    t.add_column("Trend", min_width=14)
    t.add_column("Options Rationale", min_width=40)

    for _, row in recs.iterrows():
        score = row.get("score", 0)
        score_color = "green" if score >= 70 else "yellow" if score >= 50 else "white"
        rsi = row.get("rsi", 50)
        price = row.get("current_price", 0)
        vol = row.get("volatility_20d", 0)
        trend = str(row.get("trend", ""))
        trend_color = "green" if "Up" in trend else "red" if "Down" in trend else "white"

        t.add_row(
            f"[{score_color}]{score:.0f}[/]",
            row.get("symbol", ""),
            str(row.get("options_strategy", "")),
            f"${price:.2f}" if price else "N/A",
            f"{rsi:.0f}",
            f"{vol:.1f}%" if vol else "N/A",
            f"[{trend_color}]{trend}[/]",
            str(row.get("options_rationale", ""))[:50],
        )

    console.print(t)


# ── Spinner / Progress ────────────────────────────────────────────────────────

def make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True,
    )
