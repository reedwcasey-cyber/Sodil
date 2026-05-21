"""
Sodil conversational agent — wraps the portfolio intelligence pipeline
behind a Claude-powered chat interface.

Bug fixed: "400 messages: text content blocks must be non-empty"
Root cause: appending extracted text (which can be "") as the assistant
message content when Claude replies with tool_use blocks only.
Fix: always append response.content (the full list) so no empty text
blocks are created.
"""
from __future__ import annotations

import json
import os
import warnings
from typing import Any

import anthropic
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

MODEL = "claude-opus-4-7"

SYSTEM = """You are Sodil, a quantitative investment assistant with access to
real-time portfolio and market data tools. Help the user understand their
portfolio, analyze their trading process, and discover new opportunities.

Use tools whenever a question involves current data. When answering:
- Be concise and specific — quote numbers from the data you fetched.
- When the user asks about their edge, use analyze_trades.
- When the user asks for stock ideas, use screen_market then get_recommendations.
- When the user asks about their positions or P&L, use get_portfolio.
"""

# ── Synthetic demo data (same as main.py demo mode) ──────────────────────────

def _demo_portfolio() -> dict:
    return {
        "equity": 47_823.44,
        "cash": 3_210.00,
        "buying_power": 6_420.00,
        "total_return_pct": 12.4,
    }


def _demo_positions() -> list[dict]:
    return [
        {"symbol": "NVDA", "quantity": 15,  "avg_cost": 420.00, "current_price": 875.32, "market_value": 13129.80, "total_return": 6829.80, "return_pct": 108.3},
        {"symbol": "AAPL", "quantity": 25,  "avg_cost": 168.50, "current_price": 189.25, "market_value": 4731.25,  "total_return": 518.75,  "return_pct": 12.3},
        {"symbol": "TSLA", "quantity": 10,  "avg_cost": 235.00, "current_price": 178.40, "market_value": 1784.00,  "total_return": -566.00, "return_pct": -24.1},
        {"symbol": "META", "quantity": 8,   "avg_cost": 290.00, "current_price": 491.00, "market_value": 3928.00,  "total_return": 1608.00, "return_pct": 69.3},
        {"symbol": "AMD",  "quantity": 30,  "avg_cost": 95.00,  "current_price": 162.50, "market_value": 4875.00,  "total_return": 2025.00, "return_pct": 71.1},
        {"symbol": "MSFT", "quantity": 12,  "avg_cost": 330.00, "current_price": 415.80, "market_value": 4989.60,  "total_return": 1029.60, "return_pct": 26.0},
    ]


def _demo_trades() -> pd.DataFrame:
    np.random.seed(42)
    symbols = ["NVDA","AAPL","TSLA","META","AMD","MSFT","GOOGL","AMZN","NFLX","CRWD","SNOW","PLTR","DDOG","ZS","NET"]
    sectors = {
        "NVDA":"Technology","AAPL":"Technology","TSLA":"Consumer Cyclical","META":"Communication Services",
        "AMD":"Technology","MSFT":"Technology","GOOGL":"Communication Services","AMZN":"Consumer Cyclical",
        "NFLX":"Communication Services","CRWD":"Technology","SNOW":"Technology","PLTR":"Technology",
        "DDOG":"Technology","ZS":"Technology","NET":"Technology",
    }
    rows = []
    base = pd.Timestamp("2023-01-01")
    for i in range(60):
        sym = np.random.choice(symbols)
        sector = sectors[sym]
        hold = int(np.random.choice([1,3,7,14,30,60,90,180], p=[0.05,0.10,0.15,0.20,0.25,0.15,0.07,0.03]))
        buy = round(np.random.uniform(50, 500), 2)
        pnl = round(np.random.normal(14 if sector=="Technology" else -2 if sym in ("TSLA","AMZN") else 4,
                                     18 if sector=="Technology" else 22 if sym in ("TSLA","AMZN") else 12), 2)
        sell = round(buy * (1 + pnl/100), 2)
        qty = round(np.random.uniform(5, 50), 2)
        buy_date = base + pd.Timedelta(days=i*6)
        rows.append({
            "symbol": sym, "sector": sector,
            "buy_date": buy_date, "sell_date": buy_date + pd.Timedelta(days=hold),
            "hold_days": hold, "buy_price": buy, "sell_price": sell, "quantity": qty,
            "pnl_dollar": (sell - buy) * qty, "pnl_pct": pnl, "win": pnl > 0,
            "market_cap_bucket": np.random.choice(["Mega","Large","Mid"], p=[0.5,0.3,0.2]),
            "entry_rsi": round(np.random.uniform(28, 72), 1),
            "entry_regime": np.random.choice(["Uptrend","Sideways","Downtrend"], p=[0.55,0.30,0.15]),
        })
    return pd.DataFrame(rows)


def _demo_candidates() -> list[dict]:
    return [
        {"symbol":"CRWD","name":"CrowdStrike Holdings","sector":"Technology","current_price":325.0,"rsi":48.2,"trend":"Uptrend","macd_bullish":True,"momentum_20d":8.4,"volatility_20d":32.1,"revenue_growth":0.33,"beta":1.4},
        {"symbol":"DDOG","name":"Datadog Inc","sector":"Technology","current_price":142.0,"rsi":42.5,"trend":"Uptrend","macd_bullish":True,"momentum_20d":5.2,"volatility_20d":38.4,"revenue_growth":0.27,"beta":1.6},
        {"symbol":"NET","name":"Cloudflare Inc","sector":"Technology","current_price":98.0,"rsi":44.0,"trend":"Uptrend","macd_bullish":True,"momentum_20d":6.1,"volatility_20d":41.2,"revenue_growth":0.30,"beta":1.7},
        {"symbol":"PANW","name":"Palo Alto Networks","sector":"Technology","current_price":358.0,"rsi":52.1,"trend":"Strong Uptrend","macd_bullish":True,"momentum_20d":12.3,"volatility_20d":28.6,"revenue_growth":0.22,"beta":1.2},
        {"symbol":"COIN","name":"Coinbase Global","sector":"Financial Services","current_price":220.0,"rsi":45.0,"trend":"Uptrend","macd_bullish":True,"momentum_20d":11.2,"volatility_20d":68.4,"revenue_growth":None,"beta":2.8},
    ]


# ── Tool implementations ──────────────────────────────────────────────────────

def _tool_get_portfolio(use_demo: bool) -> dict:
    if use_demo:
        return {
            "portfolio": _demo_portfolio(),
            "positions": _demo_positions(),
        }
    try:
        from auth.robinhood import login
        from portfolio.tracker import get_portfolio, get_positions
        if not login():
            return {"error": "Robinhood login failed"}
        portfolio = get_portfolio()
        positions = get_positions()
        return {
            "portfolio": portfolio,
            "positions": positions.to_dict(orient="records") if hasattr(positions, "to_dict") else positions,
        }
    except Exception as exc:
        return {"error": str(exc)}


def _tool_analyze_trades(use_demo: bool) -> dict:
    if use_demo:
        trades = _demo_trades()
    else:
        try:
            from auth.robinhood import login
            from portfolio.tracker import get_order_history
            from portfolio.analyzer import build_trade_pairs
            if not login():
                return {"error": "Robinhood login failed"}
            orders = get_order_history(limit=500)
            if orders.empty:
                return {"error": "No completed orders found"}
            trades = build_trade_pairs(orders)
        except Exception as exc:
            return {"error": str(exc)}

    from portfolio.analyzer import analyze_process
    stats = analyze_process(trades)

    # Serialize DataFrames to plain dicts for JSON
    result: dict[str, Any] = {}
    for key, val in stats.items():
        if isinstance(val, pd.DataFrame):
            result[key] = val.to_dict(orient="records")
        elif isinstance(val, (np.integer, np.floating)):
            result[key] = float(val)
        else:
            result[key] = val
    return result


def _tool_screen_market(use_demo: bool) -> dict:
    if use_demo:
        return {"candidates": _demo_candidates()}
    try:
        from market.screener import build_candidate_universe, screen_candidates
        symbols = build_candidate_universe()
        candidates = screen_candidates(symbols, {})
        return {"candidates": candidates.to_dict(orient="records") if hasattr(candidates, "to_dict") else candidates}
    except Exception as exc:
        return {"error": str(exc)}


def _tool_get_recommendations(candidates_json: str, stats_json: str) -> dict:
    try:
        candidates = pd.DataFrame(json.loads(candidates_json))
        stats = json.loads(stats_json)

        # Restore list-of-dicts back to DataFrames (they were serialized from DFs)
        df_keys = ("by_sector", "by_hold_bucket", "by_rsi_band", "by_entry_regime",
                   "by_market_cap", "best_trades", "worst_trades")
        for key in df_keys:
            if key in stats and isinstance(stats[key], list):
                stats[key] = pd.DataFrame(stats[key])

        from recommendations.engine import score_candidates, score_options_candidates
        recs = score_candidates(candidates, stats)
        opts = score_options_candidates(candidates, stats)
        return {
            "stock_recommendations": recs.to_dict(orient="records") if not recs.empty else [],
            "options_recommendations": opts[["symbol","score","options_strategy","options_rationale"]].to_dict(orient="records") if not opts.empty else [],
        }
    except Exception as exc:
        return {"error": str(exc)}


# ── Tool schema definitions ───────────────────────────────────────────────────

TOOLS = [
    {
        "name": "get_portfolio",
        "description": "Fetch the user's current portfolio value, cash, buying power, and all open positions with their P&L.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "analyze_trades",
        "description": "Analyze the user's historical trade history to compute win rate, profit factor, best sectors, optimal hold durations, and their statistical edge.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "screen_market",
        "description": "Screen the market for candidate stocks matching broad technical and fundamental criteria. Returns a list of candidates with RSI, trend, momentum, and fundamentals.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_recommendations",
        "description": "Score market candidates against the user's historical edge profile and return ranked stock and options recommendations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "candidates_json": {
                    "type": "string",
                    "description": "JSON string of candidates list from screen_market",
                },
                "stats_json": {
                    "type": "string",
                    "description": "JSON string of trade stats from analyze_trades",
                },
            },
            "required": ["candidates_json", "stats_json"],
        },
    },
]


def _execute_tool(name: str, tool_input: dict, use_demo: bool) -> str:
    if name == "get_portfolio":
        result = _tool_get_portfolio(use_demo)
    elif name == "analyze_trades":
        result = _tool_analyze_trades(use_demo)
    elif name == "screen_market":
        result = _tool_screen_market(use_demo)
    elif name == "get_recommendations":
        result = _tool_get_recommendations(
            tool_input.get("candidates_json", "[]"),
            tool_input.get("stats_json", "{}"),
        )
    else:
        result = {"error": f"Unknown tool: {name}"}
    return json.dumps(result, default=str)


# ── Main agent loop ───────────────────────────────────────────────────────────

def run_chat(use_demo: bool = True) -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set. Add it to your .env file.")
        return

    client = anthropic.Anthropic(api_key=api_key)
    messages: list[dict] = []

    mode_label = "DEMO" if use_demo else "LIVE"
    print(f"\nSodil Agent [{mode_label}] — type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye.")
            break

        messages.append({"role": "user", "content": user_input})

        # Agentic loop: keep going until Claude stops calling tools
        while True:
            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=SYSTEM,
                tools=TOOLS,
                messages=messages,
            )

            # ── THE BUG FIX ──────────────────────────────────────────────────
            # Always append response.content (the full list of content blocks)
            # as the assistant turn — NEVER extract just the text string.
            #
            # If Claude responds with tool_use blocks only (no text), extracting
            # text yields "". Appending {"role":"assistant","content":""} causes:
            #   400 messages: text content blocks must be non-empty
            #
            # Appending the full content list preserves tool_use blocks and
            # avoids creating empty text blocks entirely.
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                # Extract and print the final text response
                for block in response.content:
                    if block.type == "text" and block.text:
                        print(f"\nSodil: {block.text}\n")
                break

            if response.stop_reason == "tool_use":
                # Execute each requested tool and collect results
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        print(f"  [tool: {block.name}]")
                        result_str = _execute_tool(block.name, block.input, use_demo)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_str,
                        })

                messages.append({"role": "user", "content": tool_results})
                # Loop back — let Claude process the tool results
                continue

            # Any other stop reason — print what we have and break
            for block in response.content:
                if block.type == "text" and block.text:
                    print(f"\nSodil: {block.text}\n")
            break
