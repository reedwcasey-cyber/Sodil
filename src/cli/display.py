"""Rich terminal display components for SODIL."""

from __future__ import annotations
import numpy as np
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich.rule import Rule
from rich import box

console = Console()


def header(title: str, subtitle: str = ""):
    console.print()
    console.print(Rule(f"[bold cyan]{title}[/bold cyan]", style="cyan"))
    if subtitle:
        console.print(f"  [dim]{subtitle}[/dim]")
    console.print()


def pct(val: Optional[float], decimals: int = 1, color: bool = True) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "[dim]N/A[/dim]"
    p = val * 100
    formatted = f"{p:+.{decimals}f}%"
    if color:
        return f"[green]{formatted}[/green]" if p >= 0 else f"[red]{formatted}[/red]"
    return formatted


def fmt_score(val: Optional[float]) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "[dim]N/A[/dim]"
    color = "green" if val > 0 else "red" if val < -0.1 else "yellow"
    return f"[{color}]{val:+.3f}[/{color}]"


def fmt_currency(val: Optional[float]) -> str:
    if val is None:
        return "N/A"
    return f"${val:,.2f}"


def conviction_badge(conviction: str) -> str:
    colors = {"HIGH": "green", "MEDIUM": "yellow", "LOW": "red"}
    c = colors.get(conviction, "white")
    return f"[bold {c}]{conviction}[/bold {c}]"


def risk_badge(rating: str) -> str:
    colors = {"LOW": "green", "MEDIUM": "yellow", "HIGH": "red", "VERY HIGH": "bold red"}
    c = colors.get(rating, "white")
    return f"[{c}]{rating}[/{c}]"


def print_opportunity_table(ranked_df, fundamentals: dict):
    """Print ranked opportunities as a rich table."""
    table = Table(
        title="[bold]Top Quantitative Opportunities[/bold]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Rank", style="dim", width=5, justify="center")
    table.add_column("Ticker", style="bold white", width=7)
    table.add_column("Name", width=22)
    table.add_column("Sector", width=16)
    table.add_column("Score", justify="center", width=8)
    table.add_column("Momentum", justify="center", width=10)
    table.add_column("Value", justify="center", width=8)
    table.add_column("Quality", justify="center", width=9)
    table.add_column("RSI", justify="center", width=6)
    table.add_column("Analyst ↑", justify="center", width=10)

    for ticker, row in ranked_df.iterrows():
        info = fundamentals.get(ticker, {})
        name = (info.get("name", ticker) or ticker)[:20]
        sector = (info.get("sector", "?") or "?")[:14]
        score = row.get("composite_score", np.nan)
        mom_z = row.get("momentum_z", np.nan)
        val_z = row.get("value_z", np.nan)
        qual_z = row.get("quality_z", np.nan)
        rsi = row.get("rsi", np.nan)
        upside = row.get("analyst_upside", np.nan)

        rsi_str = f"{rsi:.0f}" if not np.isnan(rsi) else "N/A"
        rsi_color = "green" if 40 <= rsi <= 65 else "yellow" if rsi < 40 else "red"

        table.add_row(
            str(int(row.get("rank", 0))),
            ticker,
            name,
            sector,
            fmt_score(score),
            fmt_score(mom_z),
            fmt_score(val_z),
            fmt_score(qual_z),
            f"[{rsi_color}]{rsi_str}[/{rsi_color}]",
            pct(upside if not np.isnan(upside) else None),
        )

    console.print(table)


def print_thesis(thesis, challenge=None, backtest=None):
    """Print full investment thesis with challenge and backtest results."""
    header(
        f"Investment Thesis: {thesis.ticker}",
        thesis.name
    )

    # Summary panel
    summary_text = Text()
    summary_text.append(f"Conviction: ", style="bold")
    summary_text.append(conviction_badge(thesis.conviction) + "\n")
    summary_text.append(f"Sector: {thesis.sector}\n", style="dim")
    summary_text.append(f"Entry: {fmt_currency(thesis.entry_price)}  |  ")
    summary_text.append(f"Target: {fmt_currency(thesis.target_price)}  |  ")
    summary_text.append(f"Stop: {fmt_currency(thesis.stop_loss)}\n")
    summary_text.append(f"Position Size: {thesis.position_size_pct*100:.0f}% of portfolio\n")

    console.print(Panel(
        thesis.narrative,
        title="[bold cyan]Thesis Narrative[/bold cyan]",
        border_style="cyan",
    ))

    # Expected returns
    ret_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    ret_table.add_column("Scenario", style="bold")
    ret_table.add_column("1-Year Return", justify="center")
    ret_table.add_column("Implied Price", justify="center")

    price = thesis.entry_price or 0
    ret_table.add_row(
        "Bull Case", pct(thesis.expected_return_bull),
        fmt_currency(price * (1 + thesis.expected_return_bull)) if price else "N/A"
    )
    ret_table.add_row(
        "Base Case", pct(thesis.expected_return_1y),
        fmt_currency(price * (1 + thesis.expected_return_1y)) if price else "N/A"
    )
    ret_table.add_row(
        "Bear Case", pct(thesis.expected_return_bear),
        fmt_currency(price * (1 + thesis.expected_return_bear)) if price else "N/A"
    )
    console.print(Panel(ret_table, title="[bold green]Expected Return Scenarios[/bold green]", border_style="green"))

    # Catalysts and risks side by side
    cat_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    cat_table.add_column("Catalysts", style="green")
    for i, c in enumerate(thesis.primary_catalysts, 1):
        cat_table.add_row(f"[green]{i}.[/green] {c}")

    risk_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    risk_table.add_column("Risks", style="red")
    for i, r in enumerate(thesis.key_risks, 1):
        risk_table.add_row(f"[red]{i}.[/red] {r}")

    console.print(Columns([
        Panel(cat_table, title="[bold green]Bull Catalysts[/bold green]", border_style="green"),
        Panel(risk_table, title="[bold red]Key Risks[/bold red]", border_style="red"),
    ]))

    # Valuation and quality metrics
    val_table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
    val_table.add_column("Metric")
    val_table.add_column("Value", justify="right")

    for k, v in thesis.valuation.items():
        val = str(v) if v is not None else "[dim]N/A[/dim]"
        val_table.add_row(k, val)

    qual_table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
    qual_table.add_column("Metric")
    qual_table.add_column("Value", justify="right")

    for k, v in thesis.quality_metrics.items():
        val = str(v) if v is not None else "[dim]N/A[/dim]"
        qual_table.add_row(k, val)

    console.print(Columns([
        Panel(val_table, title="[bold]Valuation[/bold]"),
        Panel(qual_table, title="[bold]Business Quality[/bold]"),
    ]))

    # Challenge results
    if challenge:
        print_challenge(challenge)

    # Backtest results
    if backtest:
        print_backtest(backtest)


def print_challenge(challenge):
    """Print thesis challenge/devil's advocate analysis."""
    header("Devil's Advocate Analysis", f"Risk Rating: {challenge.risk_rating}")

    console.print(Panel(
        f"Overall Risk Score: [bold]{challenge.overall_risk_score:.2f}[/bold] | "
        f"Risk Rating: {risk_badge(challenge.risk_rating)} | "
        f"Revised Conviction: {conviction_badge(challenge.revised_conviction)} | "
        f"Return Adjustment: [bold]{challenge.adjustment_factor:.2f}x[/bold]",
        border_style="yellow",
    ))

    if challenge.red_flags:
        rf_text = "\n".join(f"  [red]⚠[/red] {rf}" for rf in challenge.red_flags)
        console.print(Panel(rf_text, title="[bold red]Red Flags[/bold red]", border_style="red"))

    if challenge.challenges:
        ch_text = "\n".join(f"  [yellow]•[/yellow] {c}" for c in challenge.challenges)
        console.print(Panel(ch_text, title="[bold yellow]Challenges[/bold yellow]", border_style="yellow"))

    if challenge.mitigants:
        mit_text = "\n".join(f"  [green]✓[/green] {m}" for m in challenge.mitigants)
        console.print(Panel(mit_text, title="[bold green]Mitigants[/bold green]", border_style="green"))


def print_backtest(backtest_results: dict):
    """Print backtest results for all strategies."""
    header("Backtesting Results", "Historical strategy validation")

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("Strategy", style="bold")
    table.add_column("Total Return", justify="center")
    table.add_column("Ann. Return", justify="center")
    table.add_column("vs Benchmark", justify="center")
    table.add_column("Sharpe", justify="center")
    table.add_column("Max DD", justify="center")
    table.add_column("Win Rate", justify="center")
    table.add_column("# Trades", justify="center")
    table.add_column("Result", justify="center")

    for strategy, result in backtest_results.items():
        table.add_row(
            strategy.replace("_", " ").title(),
            pct(result.total_return),
            pct(result.annualized_return),
            pct(result.alpha),
            f"{result.sharpe_ratio:.2f}",
            pct(result.max_drawdown),
            f"{result.win_rate*100:.0f}%",
            str(result.num_trades),
            "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]",
        )

    console.print(table)


def print_stress_test(stress_result):
    """Print stress test and Monte Carlo results."""
    header("Stress Test & Monte Carlo", stress_result.ticker)

    # Historical scenarios
    scen_table = Table(box=box.ROUNDED, show_header=True, header_style="bold yellow")
    scen_table.add_column("Crisis Scenario", style="bold")
    scen_table.add_column("Market Drop", justify="center")
    scen_table.add_column("Position Drop (β-adj)", justify="center")
    scen_table.add_column("Dollar Loss", justify="center")
    scen_table.add_column("Recovery Est.", justify="center")

    for scenario in stress_result.scenarios:
        scen_table.add_row(
            scenario.name,
            pct(scenario.market_drop),
            pct(scenario.position_drop),
            f"${scenario.max_loss_dollars:,.0f}",
            f"~{scenario.recovery_time_days} days",
        )

    console.print(Panel(scen_table, title="[bold yellow]Historical Crisis Scenarios[/bold yellow]", border_style="yellow"))

    # Monte Carlo
    mc = stress_result.monte_carlo
    if mc:
        mc_table = Table(box=box.SIMPLE, show_header=False)
        mc_table.add_column("Metric", style="bold")
        mc_table.add_column("Value", justify="right")

        mc_table.add_row("Simulated Paths", f"{mc.simulated_paths:,}")
        mc_table.add_row("Expected Return (mean)", pct(mc.mean_return_1y))
        mc_table.add_row("Median Return", pct(mc.median_return_1y))
        mc_table.add_row("95th Percentile (Bull)", pct(mc.percentile_95))
        mc_table.add_row("75th Percentile", pct(mc.percentile_75))
        mc_table.add_row("25th Percentile", pct(mc.percentile_25))
        mc_table.add_row("5th Percentile (Bear)", pct(mc.percentile_5))
        mc_table.add_row("Probability of Loss", pct(mc.prob_loss))
        mc_table.add_row("P(Return > 20%)", pct(mc.prob_gain_20pct))
        mc_table.add_row("P(Return > 50%)", pct(mc.prob_gain_50pct))

        console.print(Panel(mc_table, title="[bold blue]Monte Carlo (10,000 paths, fat tails)[/bold blue]", border_style="blue"))


def print_portfolio(report):
    """Print portfolio performance report."""
    header("Portfolio Dashboard", report.portfolio_name)

    # Key stats
    ret_color = "green" if report.cumulative_return >= 0 else "red"
    alpha_color = "green" if report.alpha >= 0 else "red"

    summary = Table(box=box.SIMPLE, show_header=False)
    summary.add_column("", style="bold", width=22)
    summary.add_column("", justify="right", width=15)
    summary.add_column("", style="bold", width=22)
    summary.add_column("", justify="right", width=15)

    summary.add_row(
        "Total Value", fmt_currency(report.total_value),
        "Initial Value", fmt_currency(report.initial_value),
    )
    summary.add_row(
        "Total Return", f"[{ret_color}]{report.cumulative_return*100:+.2f}%[/{ret_color}]",
        "vs SPY", f"[{alpha_color}]{report.alpha*100:+.2f}%[/{alpha_color}]",
    )
    summary.add_row(
        "Unrealized P&L", fmt_currency(report.unrealized_pnl),
        "Realized P&L", fmt_currency(report.realized_pnl),
    )
    summary.add_row(
        "# Positions", str(report.trade_count),
        "Win Rate", f"{report.win_rate*100:.0f}%" if report.win_rate else "N/A",
    )

    console.print(Panel(summary, title="[bold cyan]Performance Summary[/bold cyan]", border_style="cyan"))

    # Risk metrics
    if report.risk_metrics:
        rm = report.risk_metrics
        risk_table = Table(box=box.SIMPLE, show_header=False)
        risk_table.add_column("", style="bold", width=20)
        risk_table.add_column("", justify="right", width=12)
        risk_table.add_column("", style="bold", width=20)
        risk_table.add_column("", justify="right", width=12)

        risk_table.add_row(
            "Sharpe Ratio", f"{rm.sharpe_ratio:.2f}",
            "Sortino Ratio", f"{rm.sortino_ratio:.2f}",
        )
        risk_table.add_row(
            "Max Drawdown", pct(rm.max_drawdown),
            "Volatility (ann.)", pct(rm.volatility_ann),
        )
        risk_table.add_row(
            "1-Day VaR (95%)", pct(rm.var_95),
            "1-Day CVaR (95%)", pct(rm.cvar_95),
        )
        risk_table.add_row(
            "Beta", f"{rm.beta:.2f}",
            "Alpha (ann.)", pct(rm.alpha_ann),
        )
        risk_table.add_row(
            "Best Month", pct(rm.best_month),
            "Worst Month", pct(rm.worst_month),
        )
        console.print(Panel(risk_table, title="[bold yellow]Risk Metrics[/bold yellow]", border_style="yellow"))

    # Position table
    if report.position_performance:
        pos_table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
        pos_table.add_column("Ticker", style="bold", width=8)
        pos_table.add_column("Shares", justify="right", width=10)
        pos_table.add_column("Avg Cost", justify="right", width=10)
        pos_table.add_column("Current", justify="right", width=10)
        pos_table.add_column("Value", justify="right", width=12)
        pos_table.add_column("Unreal. P&L", justify="right", width=12)
        pos_table.add_column("Return", justify="center", width=10)
        pos_table.add_column("Weight", justify="center", width=8)

        for pp in report.position_performance:
            pos_table.add_row(
                pp.ticker,
                f"{pp.shares:.2f}",
                fmt_currency(pp.avg_cost),
                fmt_currency(pp.current_price),
                fmt_currency(pp.current_value),
                f"[green]+{pp.unrealized_pnl:,.2f}[/green]" if pp.unrealized_pnl >= 0
                    else f"[red]{pp.unrealized_pnl:,.2f}[/red]",
                pct(pp.unrealized_pnl_pct / 100),
                f"{pp.weight_in_portfolio*100:.1f}%",
            )

        console.print(pos_table)

    if report.top_winner:
        console.print(f"\n  [green]Best position:[/green] {report.top_winner.ticker} "
                      f"({pct(report.top_winner.unrealized_pnl_pct/100)})")
    if report.top_loser:
        console.print(f"  [red]Worst position:[/red] {report.top_loser.ticker} "
                      f"({pct(report.top_loser.unrealized_pnl_pct/100)})")
