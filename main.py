#!/usr/bin/env python3
"""
Sodil — Robinhood Portfolio Intelligence Tool
============================================
Tracks your portfolio, identifies your edge, and surfaces new opportunities.

Usage:
  python main.py                    # Full interactive session
  python main.py --portfolio        # Portfolio snapshot only
  python main.py --process          # Trade process analysis only
  python main.py --recommend        # Recommendations only (stock)
  python main.py --options          # Options recommendations
  python main.py --demo             # Run on synthetic demo data (no login)
"""
from __future__ import annotations

import argparse
import sys
import warnings
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich.prompt import Confirm

warnings.filterwarnings("ignore")

console = Console()


BANNER = """
[bold blue]
 ███████╗ ██████╗ ██████╗ ██╗██╗
 ██╔════╝██╔═══██╗██╔══██╗██║██║
 ███████╗██║   ██║██║  ██║██║██║
 ╚════██║██║   ██║██║  ██║██║██║
 ███████║╚██████╔╝██████╔╝██║███████╗
 ╚══════╝ ╚═════╝ ╚═════╝ ╚═╝╚══════╝
[/bold blue]
[cyan]Robinhood Portfolio Intelligence[/cyan]  [dim]— Know your edge. Find your next move.[/dim]
"""


def run_demo() -> None:
    """Run with synthetic demo data so the tool can be tested without credentials."""
    import pandas as pd
    import numpy as np
    from ui.dashboard import (
        render_portfolio_summary, render_positions,
        render_process_stats, render_recommendations,
        render_options_recommendations,
    )
    from portfolio.analyzer import analyze_process
    from recommendations.engine import score_candidates, score_options_candidates

    console.print("[bold yellow]Running in DEMO mode — synthetic data[/]\n")

    # Synthetic portfolio
    portfolio = {
        "equity": 47_823.44,
        "cash": 3_210.00,
        "buying_power": 6_420.00,
        "total_return_pct": 12.4,
    }

    positions = pd.DataFrame([
        {"symbol": "NVDA", "quantity": 15,  "avg_cost": 420.00, "current_price": 875.32, "market_value": 13129.80, "total_return": 6829.80, "return_pct": 108.3, "day_return_pct": 2.1},
        {"symbol": "AAPL", "quantity": 25,  "avg_cost": 168.50, "current_price": 189.25, "market_value": 4731.25, "total_return": 518.75, "return_pct": 12.3, "day_return_pct": -0.4},
        {"symbol": "TSLA", "quantity": 10,  "avg_cost": 235.00, "current_price": 178.40, "market_value": 1784.00, "total_return": -566.00, "return_pct": -24.1, "day_return_pct": -1.8},
        {"symbol": "META", "quantity": 8,   "avg_cost": 290.00, "current_price": 491.00, "market_value": 3928.00, "total_return": 1608.00, "return_pct": 69.3, "day_return_pct": 0.9},
        {"symbol": "AMD",  "quantity": 30,  "avg_cost": 95.00,  "current_price": 162.50, "market_value": 4875.00, "total_return": 2025.00, "return_pct": 71.1, "day_return_pct": 1.5},
        {"symbol": "MSFT", "quantity": 12,  "avg_cost": 330.00, "current_price": 415.80, "market_value": 4989.60, "total_return": 1029.60, "return_pct": 26.0, "day_return_pct": 0.3},
    ])

    # Synthetic trade history
    np.random.seed(42)
    symbols = ["NVDA","AAPL","TSLA","META","AMD","MSFT","GOOGL","AMZN","NFLX","CRWD","SNOW","PLTR","DDOG","ZS","NET"]
    sectors = {
        "NVDA":"Technology","AAPL":"Technology","TSLA":"Consumer Cyclical","META":"Communication Services",
        "AMD":"Technology","MSFT":"Technology","GOOGL":"Communication Services","AMZN":"Consumer Cyclical",
        "NFLX":"Communication Services","CRWD":"Technology","SNOW":"Technology","PLTR":"Technology",
        "DDOG":"Technology","ZS":"Technology","NET":"Technology",
    }
    trades_rows = []
    base_date = pd.Timestamp("2023-01-01")
    for i in range(60):
        sym = np.random.choice(symbols)
        sector = sectors[sym]
        hold = int(np.random.choice([1, 3, 7, 14, 30, 60, 90, 180], p=[0.05,0.10,0.15,0.20,0.25,0.15,0.07,0.03]))
        buy_price = round(np.random.uniform(50, 500), 2)
        # Tech winners, TSLA/AMZN mixed
        if sector == "Technology":
            pnl_pct = round(np.random.normal(14, 18), 2)
        elif sym in ("TSLA", "AMZN"):
            pnl_pct = round(np.random.normal(-2, 22), 2)
        else:
            pnl_pct = round(np.random.normal(4, 12), 2)
        sell_price = round(buy_price * (1 + pnl_pct / 100), 2)
        qty = round(np.random.uniform(5, 50), 2)
        buy_date = base_date + pd.Timedelta(days=i * 6)
        sell_date = buy_date + pd.Timedelta(days=hold)
        mc = np.random.choice(["Mega","Large","Mid"], p=[0.5,0.3,0.2])
        entry_rsi = round(np.random.uniform(28, 72), 1)
        regime = np.random.choice(["Uptrend","Sideways","Downtrend"], p=[0.55,0.30,0.15])
        trades_rows.append({
            "symbol": sym, "sector": sector, "buy_date": buy_date, "sell_date": sell_date,
            "hold_days": hold, "buy_price": buy_price, "sell_price": sell_price,
            "quantity": qty, "pnl_dollar": (sell_price - buy_price) * qty,
            "pnl_pct": pnl_pct, "win": pnl_pct > 0,
            "market_cap_bucket": mc, "entry_rsi": entry_rsi, "entry_regime": regime,
        })

    trades = pd.DataFrame(trades_rows)
    trades["hold_bucket"] = pd.cut(
        trades["hold_days"],
        bins=[-1,1,7,30,90,365,9999],
        labels=["Intraday","Swing (2-7d)","Monthly (1mo)","Quarterly (3mo)","Annual (1yr)","Long-term"],
    )
    trades["rsi_band"] = pd.cut(
        trades["entry_rsi"],
        bins=[0,30,45,55,70,100],
        labels=["Oversold(<30)","Low(30-45)","Neutral(45-55)","High(55-70)","Overbought(>70)"],
    )

    # Synthetic screened candidates
    candidates = pd.DataFrame([
        {"symbol":"CRWD","name":"CrowdStrike Holdings","sector":"Technology","current_price":325.0,"market_cap":80e9,"rsi":48.2,"trend":"Uptrend","macd_bullish":True,"momentum_20d":8.4,"momentum_5d":2.1,"volatility_20d":32.1,"pe_ratio":None,"fwd_pe":55,"revenue_growth":0.33,"beta":1.4,"52w_high":395,"52w_low":130,"avg_volume":4e6},
        {"symbol":"DDOG","name":"Datadog Inc","sector":"Technology","current_price":142.0,"market_cap":45e9,"rsi":42.5,"trend":"Uptrend","macd_bullish":True,"momentum_20d":5.2,"momentum_5d":1.8,"volatility_20d":38.4,"pe_ratio":None,"fwd_pe":70,"revenue_growth":0.27,"beta":1.6,"52w_high":175,"52w_low":90,"avg_volume":3.2e6},
        {"symbol":"NET","name":"Cloudflare Inc","sector":"Technology","current_price":98.0,"market_cap":32e9,"rsi":44.0,"trend":"Uptrend","macd_bullish":True,"momentum_20d":6.1,"momentum_5d":0.9,"volatility_20d":41.2,"pe_ratio":None,"fwd_pe":None,"revenue_growth":0.30,"beta":1.7,"52w_high":120,"52w_low":55,"avg_volume":5.1e6},
        {"symbol":"PANW","name":"Palo Alto Networks","sector":"Technology","current_price":358.0,"market_cap":115e9,"rsi":52.1,"trend":"Strong Uptrend","macd_bullish":True,"momentum_20d":12.3,"momentum_5d":3.2,"volatility_20d":28.6,"pe_ratio":None,"fwd_pe":48,"revenue_growth":0.22,"beta":1.2,"52w_high":380,"52w_low":200,"avg_volume":3.8e6},
        {"symbol":"ZS","name":"Zscaler Inc","sector":"Technology","current_price":210.0,"market_cap":32e9,"rsi":38.0,"trend":"Uptrend","macd_bullish":False,"momentum_20d":-2.1,"momentum_5d":1.2,"volatility_20d":44.1,"pe_ratio":None,"fwd_pe":62,"revenue_growth":0.35,"beta":1.9,"52w_high":250,"52w_low":120,"avg_volume":2.8e6},
        {"symbol":"SNOW","name":"Snowflake Inc","sector":"Technology","current_price":178.0,"market_cap":59e9,"rsi":41.3,"trend":"Sideways","macd_bullish":False,"momentum_20d":-4.2,"momentum_5d":-1.0,"volatility_20d":46.8,"pe_ratio":None,"fwd_pe":None,"revenue_growth":0.29,"beta":1.8,"52w_high":230,"52w_low":120,"avg_volume":4.2e6},
        {"symbol":"ISRG","name":"Intuitive Surgical","sector":"Healthcare","current_price":415.0,"market_cap":147e9,"rsi":56.0,"trend":"Strong Uptrend","macd_bullish":True,"momentum_20d":9.1,"momentum_5d":1.8,"volatility_20d":22.3,"pe_ratio":55,"fwd_pe":45,"revenue_growth":0.14,"beta":0.9,"52w_high":430,"52w_low":290,"avg_volume":1.1e6},
        {"symbol":"INTU","name":"Intuit Inc","sector":"Technology","current_price":635.0,"market_cap":178e9,"rsi":50.4,"trend":"Uptrend","macd_bullish":True,"momentum_20d":4.8,"momentum_5d":0.6,"volatility_20d":24.2,"pe_ratio":52,"fwd_pe":35,"revenue_growth":0.15,"beta":1.1,"52w_high":690,"52w_low":430,"avg_volume":1.5e6},
        {"symbol":"AXON","name":"Axon Enterprise","sector":"Industrials","current_price":295.0,"market_cap":19e9,"rsi":46.2,"trend":"Uptrend","macd_bullish":True,"momentum_20d":7.3,"momentum_5d":2.4,"volatility_20d":35.5,"pe_ratio":80,"fwd_pe":55,"revenue_growth":0.30,"beta":1.3,"52w_high":330,"52w_low":165,"avg_volume":1.2e6},
        {"symbol":"COIN","name":"Coinbase Global","sector":"Financial Services","current_price":220.0,"market_cap":55e9,"rsi":45.0,"trend":"Uptrend","macd_bullish":True,"momentum_20d":11.2,"momentum_5d":3.5,"volatility_20d":68.4,"pe_ratio":None,"fwd_pe":None,"revenue_growth":None,"beta":2.8,"52w_high":283,"52w_low":80,"avg_volume":8.1e6},
    ])

    # Render everything
    console.print(Panel(BANNER.strip(), border_style="blue", padding=(0, 2)))
    console.rule("[bold blue]PORTFOLIO[/]")
    render_portfolio_summary(portfolio, positions)
    render_positions(positions)

    console.rule("[bold magenta]PROCESS ANALYSIS[/]")
    stats = analyze_process(trades)
    render_process_stats(stats)

    console.rule("[bold yellow]STOCK RECOMMENDATIONS[/]")
    recs = score_candidates(candidates, stats, positions)
    render_recommendations(recs, title="New Opportunities — Matching Your Edge")

    console.rule("[bold yellow]OPTIONS RECOMMENDATIONS[/]")
    opts = score_options_candidates(candidates, stats)
    render_options_recommendations(opts)

    console.print()
    console.print(Panel(
        "[bold green]Demo complete![/]\n\n"
        "To connect your real Robinhood account:\n"
        "  1. Copy [cyan].env.example[/] → [cyan].env[/]\n"
        "  2. Set [cyan]RH_USERNAME[/] and [cyan]RH_PASSWORD[/]\n"
        "  3. Run [cyan]python main.py[/]\n",
        border_style="green",
        title="Next Steps",
    ))


def run_live(mode: str = "all") -> None:
    """Full live run connected to Robinhood."""
    from auth.robinhood import login
    from portfolio.tracker import get_portfolio, get_positions, get_order_history
    from portfolio.analyzer import build_trade_pairs, analyze_process
    from market.screener import build_candidate_universe, screen_candidates
    from recommendations.engine import score_candidates, score_options_candidates
    from ui.dashboard import (
        render_portfolio_summary, render_positions,
        render_process_stats, render_recommendations,
        render_options_recommendations, make_progress,
    )

    console.print(Panel(BANNER.strip(), border_style="blue", padding=(0, 2)))

    if not login():
        console.print("[bold red]Could not authenticate. Exiting.[/]")
        sys.exit(1)

    # ── Portfolio ──────────────────────────────────────────────────────────────
    if mode in ("all", "portfolio"):
        console.rule("[bold blue]PORTFOLIO[/]")
        with console.status("[blue]Loading portfolio…"):
            portfolio = get_portfolio()
            positions = get_positions()
        render_portfolio_summary(portfolio, positions)
        render_positions(positions)
        if mode == "portfolio":
            return

    # ── Process analysis ───────────────────────────────────────────────────────
    if mode in ("all", "process"):
        console.rule("[bold magenta]PROCESS ANALYSIS[/]")
        with console.status("[magenta]Fetching order history…"):
            orders = get_order_history(limit=500)

        if orders.empty:
            console.print("[yellow]No completed orders found — process analysis unavailable.[/]")
            stats = {}
        else:
            console.print(f"[dim]Analyzing {len(orders)} orders…[/]")
            with console.status("[magenta]Enriching trades with market context (this may take a minute)…"):
                trades = build_trade_pairs(orders)
            if not trades.empty:
                stats = analyze_process(trades)
                render_process_stats(stats)
            else:
                stats = {}
                console.print("[yellow]No round-trip trades found yet.[/]")

        if mode == "process":
            return
    else:
        stats = {}
        with console.status("[blue]Loading positions for filtering…"):
            positions = get_positions()

    # ── Recommendations ────────────────────────────────────────────────────────
    if mode in ("all", "recommend", "options"):
        if not stats:
            console.print("[yellow]Limited process data — using default scoring weights.[/]")

        edge = stats.get("edge_profile", {})
        best_sectors = ([edge["best_sector"]] if edge.get("best_sector") else None)
        symbols = build_candidate_universe(target_sectors=best_sectors)

        console.rule("[bold yellow]SCREENING CANDIDATES[/]")
        screened_rows = []

        with make_progress() as progress:
            task = progress.add_task("Screening universe…", total=len(symbols))
            def cb(i, total, sym):
                progress.update(task, completed=i, description=f"[blue]Analyzing {sym}…")
            from market.screener import screen_candidates
            candidates = screen_candidates(symbols, edge, progress_callback=cb)

        console.print(f"[dim]Screened {len(candidates)} candidates[/]")

        if mode in ("all", "recommend"):
            console.rule("[bold yellow]STOCK RECOMMENDATIONS[/]")
            recs = score_candidates(candidates, stats, positions if 'positions' in dir() else None)
            render_recommendations(recs, title="New Opportunities — Matching Your Edge")

        if mode in ("all", "options"):
            console.rule("[bold yellow]OPTIONS RECOMMENDATIONS[/]")
            opts = score_options_candidates(candidates, stats)
            render_options_recommendations(opts)


def main():
    parser = argparse.ArgumentParser(
        description="Sodil — Robinhood Portfolio Intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--portfolio", action="store_true", help="Portfolio snapshot only")
    parser.add_argument("--process",   action="store_true", help="Trade process analysis only")
    parser.add_argument("--recommend", action="store_true", help="Stock recommendations only")
    parser.add_argument("--options",   action="store_true", help="Options recommendations only")
    parser.add_argument("--demo",      action="store_true", help="Run on demo data (no login)")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.portfolio:
        run_live("portfolio")
    elif args.process:
        run_live("process")
    elif args.recommend:
        run_live("recommend")
    elif args.options:
        run_live("options")
    else:
        run_live("all")


if __name__ == "__main__":
    main()
