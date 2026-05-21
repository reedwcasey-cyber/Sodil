"""
SODIL — Systematic Opportunity Discovery & Investment Lab
Main CLI entry point.
"""

from __future__ import annotations
import sys
import logging
from typing import Optional

import typer
import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .display import (
    console, header, print_opportunity_table, print_thesis,
    print_challenge, print_backtest, print_stress_test, print_portfolio,
    conviction_badge, fmt_currency, pct,
)

app = typer.Typer(
    name="sodil",
    help="SODIL — Systematic Opportunity Discovery & Investment Lab",
    add_completion=False,
    rich_markup_mode="rich",
)

logging.basicConfig(level=logging.WARNING)


@app.command("scan")
def scan(
    universe: str = typer.Option("sp500", help="Universe: sp500, custom"),
    tickers: Optional[str] = typer.Option(None, help="Comma-separated tickers for custom universe"),
    top_n: int = typer.Option(10, help="Number of top opportunities to surface"),
    min_market_cap: float = typer.Option(2e9, help="Minimum market cap ($)"),
):
    """
    Scan the market for top quantitative opportunities.
    Scores stocks on momentum, value, quality, and technical signals.
    """
    header("SODIL — Market Scanner", "Identifying high-return opportunities")

    # Build universe
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  console=console, transient=True) as prog:
        task = prog.add_task("Loading stock universe...", total=None)

        from ..data.universe import build_universe, enrich_with_prices
        from ..data.fetcher import get_sp500_tickers

        if universe == "custom" and tickers:
            ticker_list = [t.strip().upper() for t in tickers.split(",")]
        else:
            ticker_list = get_sp500_tickers()

        prog.update(task, description=f"Fetching fundamentals for {min(len(ticker_list), 60)} stocks...")
        univ_df = build_universe(ticker_list, min_market_cap=min_market_cap, verbose=False)

        if univ_df.empty:
            console.print("[red]No stocks passed universe filters.[/red]")
            raise typer.Exit(1)

        prog.update(task, description="Fetching price history...")
        price_data = enrich_with_prices(univ_df, period="2y")

        # Optionally fetch SPY for benchmark
        from ..data.fetcher import fetch_price_history
        spy_prices = fetch_price_history("SPY", period="2y")

    # Compute composite scores
    console.print(f"[dim]Scoring {len(univ_df)} stocks across 4 factor dimensions...[/dim]")
    from ..signals.composite import compute_composite_score, rank_opportunities
    from ..signals.momentum import rsi as compute_rsi, macd_signal, volume_trend

    all_scores = []
    for ticker in univ_df.index:
        fundamentals = univ_df.loc[ticker].to_dict()
        fundamentals["ticker"] = ticker
        prices = price_data.get(ticker, pd.DataFrame())

        score = compute_composite_score(ticker, fundamentals, prices)
        all_scores.append(score)

    ranked = rank_opportunities(all_scores, top_n=top_n)
    fundamentals_map = {t: univ_df.loc[t].to_dict() for t in univ_df.index if t in ranked.index}

    console.print()
    print_opportunity_table(ranked, fundamentals_map)

    console.print(f"\n[dim]Universe: {len(univ_df)} stocks | Scored on: Momentum (35%), Value (30%), Quality (25%), Technical (10%)[/dim]")
    console.print(f"[dim]Run [bold]sodil analyze <TICKER>[/bold] for deep-dive thesis + backtest[/dim]")

    return ranked


@app.command("analyze")
def analyze(
    ticker: str = typer.Argument(..., help="Stock ticker to analyze"),
    invest: bool = typer.Option(False, "--invest", help="Add to paper portfolio if conviction >= MEDIUM"),
    portfolio: str = typer.Option("sodil_paper", help="Portfolio name"),
    initial_cash: float = typer.Option(100_000.0, help="Portfolio initial cash (for new portfolios)"),
):
    """
    Deep-dive analysis: thesis generation, challenge, backtest, stress test.
    Optionally paper-trade the position.
    """
    ticker = ticker.upper()
    header(f"Deep Analysis: {ticker}", "Thesis · Challenge · Backtest · Stress Test")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  console=console, transient=True) as prog:
        task = prog.add_task(f"Fetching data for {ticker}...", total=None)

        from ..data.fetcher import fetch_price_history, fetch_fundamentals
        from ..data.fetcher import fetch_price_history as fph

        fundamentals = fetch_fundamentals(ticker)
        prices = fetch_price_history(ticker, period="2y")
        spy_prices = fph("SPY", period="2y")

        prog.update(task, description="Computing factor scores...")

        from ..signals.composite import compute_composite_score, rank_opportunities
        score = compute_composite_score(ticker, fundamentals, prices)
        # Build a mini ranking (just this stock) to get z-scores
        ranked = rank_opportunities([score], top_n=1)

        if ranked.empty:
            console.print(f"[red]Could not compute scores for {ticker}[/red]")
            raise typer.Exit(1)

        score_row = ranked.iloc[0]

        prog.update(task, description="Generating investment thesis...")
        from ..thesis.generator import generate_thesis
        thesis = generate_thesis(ticker, fundamentals, score_row)

        prog.update(task, description="Challenging the thesis...")
        from ..thesis.challenger import challenge_thesis
        challenge = challenge_thesis(thesis, fundamentals, score_row)

        prog.update(task, description="Running backtests...")
        from ..thesis.backtest import run_full_backtest
        backtest = run_full_backtest(ticker, prices, spy_prices)

        prog.update(task, description="Running stress tests...")
        from ..risk.stress_test import run_stress_test
        vol = float(prices["Close"].squeeze().pct_change().std() * np.sqrt(252)) if not prices.empty else 0.25
        stress = run_stress_test(
            ticker, fundamentals,
            expected_return=thesis.expected_return_1y,
            annual_vol=vol,
            position_value=10_000,
        )

    # Display everything
    print_thesis(thesis, challenge, backtest)
    print_stress_test(stress)

    # Save to database
    from ..portfolio.db import save_opportunity
    save_opportunity(
        ticker=ticker,
        composite_score=float(score_row.get("composite_score", 0)),
        conviction=thesis.conviction,
        expected_return=thesis.expected_return_1y,
        thesis={
            "narrative": thesis.narrative,
            "catalysts": thesis.primary_catalysts,
            "entry": thesis.entry_price,
            "target": thesis.target_price,
            "stop": thesis.stop_loss,
        },
        challenge={
            "risk_rating": challenge.risk_rating,
            "risk_score": challenge.overall_risk_score,
            "red_flags": challenge.red_flags,
            "revised_conviction": challenge.revised_conviction,
        },
        backtest={
            s: {
                "ann_return": r.annualized_return,
                "sharpe": r.sharpe_ratio,
                "max_dd": r.max_drawdown,
                "passed": r.passed,
            }
            for s, r in backtest.items()
        },
    )

    # Auto-invest if requested and conviction warrants it
    if invest:
        if thesis.conviction in ("HIGH", "MEDIUM") and challenge.revised_conviction != "LOW":
            console.print(f"\n[bold green]Auto-investing in {ticker}...[/bold green]")
            from ..portfolio.manager import PortfolioManager
            pm = PortfolioManager(name=portfolio, initial_cash=initial_cash)
            result = pm.buy(
                ticker,
                position_pct=thesis.position_size_pct * challenge.adjustment_factor,
                thesis=thesis,
                challenge=challenge,
            )
            if result["success"]:
                console.print(
                    f"[green]Bought {result['shares']:.2f} shares of {ticker} @ "
                    f"{fmt_currency(result['price'])} | Cost: {fmt_currency(result['cost'])} | "
                    f"Cash remaining: {fmt_currency(result['cash_remaining'])}[/green]"
                )
            else:
                console.print(f"[red]Trade failed: {result['reason']}[/red]")
        else:
            console.print(f"\n[yellow]Skipping investment: conviction={thesis.conviction}, "
                          f"revised={challenge.revised_conviction}[/yellow]")


@app.command("screen-and-invest")
def screen_and_invest(
    top_n: int = typer.Option(5, help="Number of top picks to invest in"),
    portfolio: str = typer.Option("sodil_paper", help="Portfolio name"),
    initial_cash: float = typer.Option(100_000.0, help="Initial portfolio cash"),
    min_conviction: str = typer.Option("MEDIUM", help="Minimum conviction to invest: HIGH, MEDIUM, LOW"),
):
    """
    Full pipeline: scan → analyze top picks → run tests → paper trade the best ones.
    This is the full automated investment workflow.
    """
    header("Full Investment Pipeline", "Scan → Analyze → Test → Trade")

    from ..data.universe import build_universe, enrich_with_prices
    from ..data.fetcher import get_sp500_tickers, fetch_price_history
    from ..signals.composite import compute_composite_score, rank_opportunities
    from ..thesis.generator import generate_thesis
    from ..thesis.challenger import challenge_thesis
    from ..thesis.backtest import run_full_backtest
    from ..portfolio.manager import PortfolioManager

    conviction_rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
    min_rank = conviction_rank.get(min_conviction.upper(), 2)

    pm = PortfolioManager(name=portfolio, initial_cash=initial_cash)

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(),
        console=console, transient=False,
    ) as prog:
        ticker_list = get_sp500_tickers()[:60]  # Top 60 for speed
        task1 = prog.add_task("Building universe...", total=None)
        univ_df = build_universe(ticker_list, verbose=False)
        prog.update(task1, description=f"[green]Universe: {len(univ_df)} stocks[/green]", completed=1, total=1)

        task2 = prog.add_task("Fetching price histories...", total=None)
        price_data = enrich_with_prices(univ_df, period="2y")
        spy_prices = fetch_price_history("SPY", period="2y")
        prog.update(task2, description="[green]Price data loaded[/green]", completed=1, total=1)

        task3 = prog.add_task("Scoring stocks...", total=len(univ_df))
        all_scores = []
        for ticker in univ_df.index:
            fundamentals = univ_df.loc[ticker].to_dict()
            fundamentals["ticker"] = ticker
            prices = price_data.get(ticker, pd.DataFrame())
            score = compute_composite_score(ticker, fundamentals, prices)
            all_scores.append(score)
            prog.advance(task3)

        ranked = rank_opportunities(all_scores, top_n=top_n * 3)

    console.print(f"\n[bold]Top {top_n * 3} candidates identified. Running deep analysis...[/bold]\n")
    print_opportunity_table(ranked, {t: univ_df.loc[t].to_dict() for t in ranked.index if t in univ_df.index})

    invested = []
    console.print(f"\n[bold cyan]Running analysis on top {top_n} candidates...[/bold cyan]\n")

    for i, (ticker, row) in enumerate(ranked.head(top_n).iterrows()):
        if ticker not in univ_df.index:
            continue

        fundamentals = univ_df.loc[ticker].to_dict()
        fundamentals["ticker"] = ticker
        prices = price_data.get(ticker, pd.DataFrame())

        console.print(f"\n[bold]({i+1}/{top_n}) Analyzing {ticker}...[/bold]")

        thesis = generate_thesis(ticker, fundamentals, row)
        challenge = challenge_thesis(thesis, fundamentals, row)
        backtest = run_full_backtest(ticker, prices, spy_prices)

        # Investment decision
        thesis_rank = conviction_rank.get(thesis.conviction, 1)
        challenge_rank = conviction_rank.get(challenge.revised_conviction, 1)
        effective_rank = min(thesis_rank, challenge_rank)

        passes_backtest = any(r.passed for r in backtest.values())

        console.print(
            f"  Conviction: {conviction_badge(thesis.conviction)} → "
            f"Challenged: {conviction_badge(challenge.revised_conviction)} | "
            f"Backtest: {'[green]PASS[/green]' if passes_backtest else '[red]FAIL[/red]'} | "
            f"Expected Return: {pct(thesis.expected_return_1y)}"
        )

        if effective_rank >= min_rank and passes_backtest:
            position_pct = thesis.position_size_pct * challenge.adjustment_factor
            result = pm.buy(ticker, position_pct, thesis, challenge)
            if result["success"]:
                console.print(
                    f"  [green]✓ Invested: {result['shares']:.2f} shares @ "
                    f"{fmt_currency(result['price'])} | Cost: {fmt_currency(result['cost'])}[/green]"
                )
                invested.append(ticker)
            else:
                console.print(f"  [yellow]⚠ Trade skipped: {result['reason']}[/yellow]")
        else:
            reason = "Below conviction threshold" if effective_rank < min_rank else "Failed backtest"
            console.print(f"  [dim]Skipped: {reason}[/dim]")

    # Take a portfolio snapshot
    snap = pm.take_snapshot()

    console.print(f"\n[bold cyan]Pipeline complete.[/bold cyan]")
    console.print(f"  Invested in: {', '.join(invested) if invested else 'None'}")
    console.print(f"  Portfolio value: {fmt_currency(snap['total_value'])}")
    console.print(f"  Cash remaining: {fmt_currency(snap['cash'])}")
    console.print(f"\n[dim]Run [bold]sodil portfolio[/bold] to view full dashboard[/dim]")


@app.command("portfolio")
def portfolio_cmd(
    name: str = typer.Option("sodil_paper", help="Portfolio name"),
    initial_cash: float = typer.Option(100_000.0, help="Initial cash (for new portfolios)"),
    snapshot: bool = typer.Option(False, "--snapshot", help="Take and save a snapshot now"),
):
    """View portfolio performance dashboard with P&L, risk metrics, and positions."""
    from ..portfolio.manager import PortfolioManager
    from ..portfolio.tracker import generate_report
    from ..portfolio.db import get_portfolio_id

    pm = PortfolioManager(name=name, initial_cash=initial_cash)

    if snapshot:
        snap = pm.take_snapshot()
        console.print(f"[green]Snapshot saved.[/green] Total value: {fmt_currency(snap['total_value'])}")

    report = generate_report(pm.portfolio_id, name, pm.initial_cash)
    print_portfolio(report)


@app.command("buy")
def buy_cmd(
    ticker: str = typer.Argument(..., help="Ticker to buy"),
    pct_: float = typer.Option(0.05, "--pct", help="Fraction of portfolio to allocate (0.05 = 5%)"),
    portfolio: str = typer.Option("sodil_paper", help="Portfolio name"),
):
    """Manually paper-buy a position."""
    from ..portfolio.manager import PortfolioManager
    pm = PortfolioManager(name=portfolio)
    result = pm.buy(ticker.upper(), pct_)
    if result["success"]:
        console.print(
            f"[green]Bought {result['shares']:.4f} shares of {ticker.upper()} @ "
            f"{fmt_currency(result['price'])} | Total: {fmt_currency(result['cost'])}[/green]"
        )
    else:
        console.print(f"[red]Trade failed: {result['reason']}[/red]")


@app.command("sell")
def sell_cmd(
    ticker: str = typer.Argument(..., help="Ticker to sell"),
    shares: Optional[float] = typer.Option(None, help="Shares to sell (default: all)"),
    portfolio: str = typer.Option("sodil_paper", help="Portfolio name"),
):
    """Manually paper-sell a position."""
    from ..portfolio.manager import PortfolioManager
    pm = PortfolioManager(name=portfolio)
    result = pm.sell(ticker.upper(), shares=shares, reason="Manual sell via CLI")
    if result["success"]:
        pnl = result.get("pnl", 0)
        color = "green" if pnl >= 0 else "red"
        console.print(
            f"[{color}]Sold {result['shares']:.4f} shares of {ticker.upper()} @ "
            f"{fmt_currency(result['price'])} | P&L: {fmt_currency(pnl)} "
            f"({pct(result.get('pnl_pct', 0))})[/{color}]"
        )
    else:
        console.print(f"[red]Sell failed: {result['reason']}[/red]")


@app.command("check-stops")
def check_stops(
    portfolio: str = typer.Option("sodil_paper", help="Portfolio name"),
):
    """Check all positions against stop-loss levels and execute triggered stops."""
    from ..portfolio.manager import PortfolioManager
    pm = PortfolioManager(name=portfolio)
    triggered = pm.enforce_stop_losses()
    if triggered:
        for t in triggered:
            console.print(
                f"[red]STOP TRIGGERED: {t['ticker']} @ ${t['exit_price']:.2f} "
                f"(stop: ${t['stop_price']:.2f}) | P&L: {fmt_currency(t['pnl'])}[/red]"
            )
    else:
        console.print("[green]No stop-losses triggered.[/green]")


@app.command("opportunities")
def list_opportunities(
    limit: int = typer.Option(20, help="Number of past opportunities to show"),
):
    """List all previously discovered investment opportunities."""
    from ..portfolio.db import get_opportunities
    from rich.table import Table
    from rich import box

    opps = get_opportunities(limit)
    if not opps:
        console.print("[yellow]No opportunities discovered yet. Run 'sodil scan' first.[/yellow]")
        return

    table = Table(title="Past Opportunities", box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Date", width=12)
    table.add_column("Ticker", style="bold", width=8)
    table.add_column("Score", justify="center", width=8)
    table.add_column("Conviction", width=10)
    table.add_column("Exp. Return", justify="center", width=12)

    for opp in opps:
        date = opp["discovered_at"][:10]
        score = opp.get("composite_score")
        table.add_row(
            date,
            opp["ticker"],
            f"{score:.3f}" if score else "N/A",
            conviction_badge(opp.get("conviction", "?")),
            pct(opp.get("expected_return")),
        )

    console.print(table)


@app.command("stress")
def stress_cmd(
    ticker: str = typer.Argument(..., help="Ticker to stress test"),
    position_value: float = typer.Option(10_000.0, help="Position size in dollars"),
):
    """Run standalone stress test and Monte Carlo for a ticker."""
    ticker = ticker.upper()

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  console=console, transient=True) as prog:
        task = prog.add_task(f"Fetching data for {ticker}...", total=None)
        from ..data.fetcher import fetch_price_history, fetch_fundamentals
        fundamentals = fetch_fundamentals(ticker)
        prices = fetch_price_history(ticker, period="2y")
        prog.update(task, description="Running stress tests...")

        from ..risk.stress_test import run_stress_test
        close = prices["Close"].squeeze() if not prices.empty and "Close" in prices.columns else pd.Series()
        vol = float(close.pct_change().std() * np.sqrt(252)) if not close.empty else 0.25
        exp_ret = 0.10  # conservative base

        from ..data.fetcher import fetch_fundamentals as ff
        fwd_pe = fundamentals.get("forward_pe")
        pe = fundamentals.get("pe_ratio")
        if fwd_pe and pe and pe > 0:
            exp_ret = max((pe / fwd_pe - 1) * 0.5, 0.05)

        stress = run_stress_test(ticker, fundamentals, exp_ret, vol, position_value)

    print_stress_test(stress)


def main():
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
