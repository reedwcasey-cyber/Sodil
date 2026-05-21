"""
Sodil Web Dashboard
Run with:  python launch.py
Or:        streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import anthropic

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sodil",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Sodil — Quantitative Investment Intelligence"},
)

st.markdown("""
<style>
#MainMenu, footer { visibility: hidden; }
div[data-testid="metric-container"] {
    background: rgba(0,212,170,0.08);
    border: 1px solid rgba(0,212,170,0.2);
    border-radius: 10px;
    padding: 14px 18px;
}
div[data-testid="stTabs"] button[data-baseweb="tab"] {
    font-size: 0.95rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL = "claude-opus-4-7"
SYSTEM = """You are Sodil, a quantitative investment assistant with access to
real-time portfolio and market data tools. Help the user understand their
portfolio, analyze their trading process, and discover new opportunities.

Use tools whenever a question involves current data. Be concise and specific —
quote numbers from the data you fetched.
- Edge analysis: use analyze_trades
- Stock ideas: use screen_market then get_recommendations
- Portfolio / positions / P&L: use get_portfolio"""

TOOLS = [
    {
        "name": "get_portfolio",
        "description": "Fetch current portfolio value, cash, buying power, and all open positions with P&L.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "analyze_trades",
        "description": "Analyze historical trade history to compute win rate, profit factor, best sectors, hold durations, and statistical edge.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "screen_market",
        "description": "Screen the market for candidate stocks with technical and fundamental criteria.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_recommendations",
        "description": "Score market candidates against edge profile and return ranked stock and options recommendations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "candidates_json": {"type": "string", "description": "JSON string of candidates from screen_market"},
                "stats_json": {"type": "string", "description": "JSON string of stats from analyze_trades"},
            },
            "required": ["candidates_json", "stats_json"],
        },
    },
]

# ── Session state ─────────────────────────────────────────────────────────────
_DEFAULTS: dict[str, Any] = {
    "use_demo": True,
    "portfolio": None,
    "positions": None,
    "trades_df": None,
    "stats": None,
    "candidates": None,
    "recs": None,
    "opts_recs": None,
    "chat_messages": [],
    "api_messages": [],
    "data_loaded": False,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Demo data ─────────────────────────────────────────────────────────────────
def _demo_portfolio() -> dict:
    return {"equity": 47_823.44, "cash": 3_210.00, "buying_power": 6_420.00, "total_return_pct": 12.4}


def _demo_positions() -> list[dict]:
    return [
        {"symbol": "NVDA", "quantity": 15, "avg_cost": 420.00, "current_price": 875.32, "market_value": 13129.80, "total_return": 6829.80, "return_pct": 108.3, "day_return_pct": 2.1},
        {"symbol": "AAPL", "quantity": 25, "avg_cost": 168.50, "current_price": 189.25, "market_value": 4731.25, "total_return": 518.75, "return_pct": 12.3, "day_return_pct": -0.3},
        {"symbol": "TSLA", "quantity": 10, "avg_cost": 235.00, "current_price": 178.40, "market_value": 1784.00, "total_return": -566.00, "return_pct": -24.1, "day_return_pct": -1.8},
        {"symbol": "META", "quantity": 8, "avg_cost": 290.00, "current_price": 491.00, "market_value": 3928.00, "total_return": 1608.00, "return_pct": 69.3, "day_return_pct": 0.7},
        {"symbol": "AMD", "quantity": 30, "avg_cost": 95.00, "current_price": 162.50, "market_value": 4875.00, "total_return": 2025.00, "return_pct": 71.1, "day_return_pct": 1.2},
        {"symbol": "MSFT", "quantity": 12, "avg_cost": 330.00, "current_price": 415.80, "market_value": 4989.60, "total_return": 1029.60, "return_pct": 26.0, "day_return_pct": 0.4},
    ]


def _demo_trades() -> pd.DataFrame:
    np.random.seed(42)
    symbols = ["NVDA","AAPL","TSLA","META","AMD","MSFT","GOOGL","AMZN","NFLX","CRWD","SNOW","PLTR","DDOG","ZS","NET"]
    sectors = {
        "NVDA": "Technology", "AAPL": "Technology", "TSLA": "Consumer Cyclical",
        "META": "Communication Services", "AMD": "Technology", "MSFT": "Technology",
        "GOOGL": "Communication Services", "AMZN": "Consumer Cyclical",
        "NFLX": "Communication Services", "CRWD": "Technology", "SNOW": "Technology",
        "PLTR": "Technology", "DDOG": "Technology", "ZS": "Technology", "NET": "Technology",
    }
    rows = []
    base = pd.Timestamp("2023-01-01")
    for i in range(60):
        sym = np.random.choice(symbols)
        sector = sectors[sym]
        hold = int(np.random.choice([1,3,7,14,30,60,90,180], p=[0.05,0.10,0.15,0.20,0.25,0.15,0.07,0.03]))
        buy = round(np.random.uniform(50, 500), 2)
        pnl = round(np.random.normal(
            14 if sector == "Technology" else -2 if sym in ("TSLA", "AMZN") else 4,
            18 if sector == "Technology" else 22 if sym in ("TSLA", "AMZN") else 12,
        ), 2)
        sell = round(buy * (1 + pnl / 100), 2)
        qty = round(np.random.uniform(5, 50), 2)
        buy_date = base + pd.Timedelta(days=i * 6)
        rows.append({
            "symbol": sym, "sector": sector,
            "buy_date": buy_date, "sell_date": buy_date + pd.Timedelta(days=hold),
            "hold_days": hold, "buy_price": buy, "sell_price": sell, "quantity": qty,
            "pnl_dollar": (sell - buy) * qty, "pnl_pct": pnl, "win": pnl > 0,
            "market_cap_bucket": np.random.choice(["Mega","Large","Mid"], p=[0.5, 0.3, 0.2]),
            "entry_rsi": round(np.random.uniform(28, 72), 1),
            "entry_regime": np.random.choice(["Uptrend","Sideways","Downtrend"], p=[0.55, 0.30, 0.15]),
            "hold_bucket": np.random.choice(["Swing (2-7d)","Monthly (1mo)","Quarterly (3mo)"], p=[0.3, 0.5, 0.2]),
            "rsi_band": np.random.choice(["Oversold(<30)","Low(30-45)","Neutral(45-55)","High(55-70)"], p=[0.1, 0.3, 0.4, 0.2]),
        })
    return pd.DataFrame(rows)


def _demo_candidates() -> pd.DataFrame:
    return pd.DataFrame([
        {"symbol":"CRWD","name":"CrowdStrike","sector":"Technology","current_price":325.0,"rsi":48.2,"trend":"Uptrend","macd_bullish":True,"momentum_20d":8.4,"volatility_20d":32.1,"revenue_growth":0.33,"beta":1.4},
        {"symbol":"DDOG","name":"Datadog","sector":"Technology","current_price":142.0,"rsi":42.5,"trend":"Uptrend","macd_bullish":True,"momentum_20d":5.2,"volatility_20d":38.4,"revenue_growth":0.27,"beta":1.6},
        {"symbol":"NET","name":"Cloudflare","sector":"Technology","current_price":98.0,"rsi":44.0,"trend":"Uptrend","macd_bullish":True,"momentum_20d":6.1,"volatility_20d":41.2,"revenue_growth":0.30,"beta":1.7},
        {"symbol":"PANW","name":"Palo Alto Networks","sector":"Technology","current_price":358.0,"rsi":52.1,"trend":"Strong Uptrend","macd_bullish":True,"momentum_20d":12.3,"volatility_20d":28.6,"revenue_growth":0.22,"beta":1.2},
        {"symbol":"COIN","name":"Coinbase","sector":"Financial Services","current_price":220.0,"rsi":45.0,"trend":"Uptrend","macd_bullish":True,"momentum_20d":11.2,"volatility_20d":68.4,"revenue_growth":None,"beta":2.8},
        {"symbol":"SNOW","name":"Snowflake","sector":"Technology","current_price":185.0,"rsi":38.0,"trend":"Uptrend","macd_bullish":False,"momentum_20d":3.1,"volatility_20d":44.1,"revenue_growth":0.35,"beta":1.9},
        {"symbol":"PLTR","name":"Palantir","sector":"Technology","current_price":42.0,"rsi":55.0,"trend":"Strong Uptrend","macd_bullish":True,"momentum_20d":18.2,"volatility_20d":52.3,"revenue_growth":0.21,"beta":2.1},
    ])


# ── Risk / performance metrics ────────────────────────────────────────────────
def compute_risk_metrics(trades_df: pd.DataFrame) -> dict:
    if trades_df is None or trades_df.empty:
        return {}
    wins = trades_df[trades_df["win"]]
    losses = trades_df[~trades_df["win"]]
    win_rate = float(trades_df["win"].mean())
    avg_win = float(wins["pnl_pct"].mean()) if not wins.empty else 0.0
    avg_loss = float(abs(losses["pnl_pct"].mean())) if not losses.empty else 1.0
    b = avg_win / avg_loss if avg_loss > 0 else 0
    kelly_raw = (win_rate * b - (1 - win_rate)) / b if b > 0 else 0
    half_kelly = max(0.0, kelly_raw / 2)
    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss
    returns = trades_df["pnl_pct"].values
    sharpe = float(returns.mean() / returns.std() * np.sqrt(12)) if returns.std() > 0 else 0.0
    cumulative = trades_df["pnl_dollar"].cumsum().values
    peak = np.maximum.accumulate(cumulative)
    drawdown = (peak - cumulative) / np.where(peak == 0, 1, peak)
    gross_profit = float(wins["pnl_dollar"].sum()) if not wins.empty else 0.0
    gross_loss = float(abs(losses["pnl_dollar"].sum())) if not losses.empty else 1.0
    return {
        "half_kelly_pct": half_kelly * 100,
        "expectancy_pct": expectancy,
        "sharpe": sharpe,
        "max_drawdown_pct": float(drawdown.max() * 100),
        "profit_factor": gross_profit / gross_loss,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "win_rate": win_rate,
    }


# ── Tool execution ────────────────────────────────────────────────────────────
def _execute_tool(name: str, tool_input: dict, use_demo: bool) -> str:
    try:
        if name == "get_portfolio":
            if use_demo:
                result: Any = {"portfolio": _demo_portfolio(), "positions": _demo_positions()}
            else:
                from auth.robinhood import login
                from portfolio.tracker import get_portfolio, get_positions
                if not login():
                    result = {"error": "Robinhood login failed"}
                else:
                    pos = get_positions()
                    result = {
                        "portfolio": get_portfolio(),
                        "positions": pos.to_dict(orient="records") if hasattr(pos, "to_dict") else pos,
                    }

        elif name == "analyze_trades":
            if use_demo:
                trades = _demo_trades()
            else:
                from auth.robinhood import login
                from portfolio.tracker import get_order_history
                from portfolio.analyzer import build_trade_pairs
                if not login():
                    return json.dumps({"error": "Robinhood login failed"})
                orders = get_order_history(limit=500)
                trades = build_trade_pairs(orders)
            from portfolio.analyzer import analyze_process
            raw = analyze_process(trades)
            result = {}
            for k, v in raw.items():
                if isinstance(v, pd.DataFrame):
                    result[k] = v.to_dict(orient="records")
                elif isinstance(v, (np.integer, np.floating)):
                    result[k] = float(v)
                else:
                    result[k] = v

        elif name == "screen_market":
            if use_demo:
                result = {"candidates": _demo_candidates().to_dict(orient="records")}
            else:
                from market.screener import build_candidate_universe, screen_candidates
                syms = build_candidate_universe()
                cands = screen_candidates(syms, {})
                result = {"candidates": cands.to_dict(orient="records") if hasattr(cands, "to_dict") else cands}

        elif name == "get_recommendations":
            candidates = pd.DataFrame(json.loads(tool_input.get("candidates_json", "[]")))
            stats = json.loads(tool_input.get("stats_json", "{}"))
            for key in ("by_sector","by_hold_bucket","by_rsi_band","by_entry_regime","by_market_cap","best_trades","worst_trades"):
                if key in stats and isinstance(stats[key], list):
                    stats[key] = pd.DataFrame(stats[key])
            from recommendations.engine import score_candidates, score_options_candidates
            recs = score_candidates(candidates, stats)
            opts = score_options_candidates(candidates, stats)
            result = {
                "stock_recommendations": recs.to_dict(orient="records") if not recs.empty else [],
                "options_recommendations": opts[["symbol","score","options_strategy","options_rationale"]].to_dict(orient="records") if not opts.empty else [],
            }
        else:
            result = {"error": f"Unknown tool: {name}"}
    except Exception as exc:
        result = {"error": str(exc)}

    return json.dumps(result, default=str)


# ── Agent turn ────────────────────────────────────────────────────────────────
def run_agent_turn(client: anthropic.Anthropic, user_message: str) -> tuple[str, list[str]]:
    use_demo = st.session_state.use_demo
    msgs = list(st.session_state.api_messages)
    msgs.append({"role": "user", "content": user_message})
    tools_used: list[str] = []

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM,
            tools=TOOLS,
            messages=msgs,
        )
        msgs.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            st.session_state.api_messages = msgs
            text = next((b.text for b in response.content if b.type == "text" and b.text), "")
            return text, tools_used

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tools_used.append(block.name)
                    result_str = _execute_tool(block.name, block.input, use_demo)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str,
                    })
            msgs.append({"role": "user", "content": tool_results})
            continue

        st.session_state.api_messages = msgs
        text = next((b.text for b in response.content if b.type == "text" and b.text), "")
        return text, tools_used


# ── Data loading ──────────────────────────────────────────────────────────────
def load_all_data() -> None:
    use_demo = st.session_state.use_demo
    with st.spinner("Loading data..."):
        if use_demo:
            st.session_state.portfolio = _demo_portfolio()
            st.session_state.positions = pd.DataFrame(_demo_positions())
            trades = _demo_trades()
        else:
            try:
                from auth.robinhood import login
                from portfolio.tracker import get_portfolio, get_positions, get_order_history
                from portfolio.analyzer import build_trade_pairs
                if not login():
                    st.error("Robinhood login failed. Check credentials in the sidebar.")
                    return
                st.session_state.portfolio = get_portfolio()
                raw_pos = get_positions()
                st.session_state.positions = raw_pos if hasattr(raw_pos, "columns") else pd.DataFrame(raw_pos)
                orders = get_order_history(limit=500)
                if orders.empty:
                    st.warning("No completed orders found — using demo trade data for analysis.")
                    trades = _demo_trades()
                else:
                    trades = build_trade_pairs(orders)
            except Exception as exc:
                st.error(f"Error connecting: {exc}")
                return

        st.session_state.trades_df = trades

        from portfolio.analyzer import analyze_process
        st.session_state.stats = analyze_process(trades)

        # Candidates (demo instant; live requires manual trigger)
        if use_demo:
            st.session_state.candidates = _demo_candidates()
            _compute_recommendations()

        st.session_state.data_loaded = True


def _compute_recommendations() -> None:
    cands = st.session_state.candidates
    stats = st.session_state.stats
    if cands is None or cands.empty or not stats:
        return
    try:
        from recommendations.engine import score_candidates, score_options_candidates
        st.session_state.recs = score_candidates(cands, stats)
        st.session_state.opts_recs = score_options_candidates(cands, stats)
    except Exception as exc:
        st.warning(f"Recommendations unavailable: {exc}")


# ── Formatting helpers ────────────────────────────────────────────────────────
def fmt_pct(v: float) -> str:
    return f"+{v:.1f}%" if v > 0 else f"{v:.1f}%"


def fmt_dollar(v: float) -> str:
    return f"+${v:,.0f}" if v > 0 else f"-${abs(v):,.0f}"


def _plotly_base() -> dict:
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8eaf0"),
        margin=dict(t=40, b=30, l=10, r=10),
    )


GREEN = "#00d4aa"
RED = "#ff5566"
AMBER = "#ffaa00"
BLUE = "#4488ff"
TEAL_SCALE = [RED, AMBER, GREEN]


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📈 Sodil")
    st.caption("Quantitative Investment Intelligence")
    st.divider()

    mode = st.radio("Data Source", ["Demo", "Live"], horizontal=True,
                    index=0 if st.session_state.use_demo else 1)
    use_demo = mode == "Demo"
    if use_demo != st.session_state.use_demo:
        st.session_state.use_demo = use_demo
        st.session_state.data_loaded = False

    if not use_demo:
        with st.expander("Robinhood Credentials", expanded=True):
            rh_user = st.text_input("Email", placeholder="you@example.com")
            rh_pass = st.text_input("Password", type="password")
            rh_mfa = st.text_input("MFA Secret (optional)", placeholder="base32 TOTP")
            if rh_user:
                os.environ["RH_USERNAME"] = rh_user
            if rh_pass:
                os.environ["RH_PASSWORD"] = rh_pass
            if rh_mfa:
                os.environ["RH_MFA_SECRET"] = rh_mfa

    st.divider()

    api_key_input = st.text_input(
        "Anthropic API Key",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        type="password",
        help="Required for the AI Chat tab",
    )
    if api_key_input:
        os.environ["ANTHROPIC_API_KEY"] = api_key_input

    st.divider()

    if st.button("Load / Refresh Data", type="primary", use_container_width=True):
        load_all_data()

    if not st.session_state.data_loaded:
        st.info("Click **Load / Refresh Data** to begin.")
    else:
        st.success("Data loaded")
        if st.session_state.use_demo:
            st.caption("Demo mode — synthetic data")
        else:
            st.caption("Live mode — Robinhood")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_port, tab_trades, tab_screen, tab_recs, tab_chat = st.tabs([
    "📊  Portfolio",
    "🔬  Trade Analysis",
    "🔍  Screener",
    "💡  Recommendations",
    "🤖  AI Chat",
])

# ── TAB 1: Portfolio ──────────────────────────────────────────────────────────
with tab_port:
    if not st.session_state.data_loaded:
        st.markdown("### Welcome to Sodil")
        st.markdown(
            "Use the sidebar to select **Demo** (no credentials needed) or **Live** mode, "
            "then click **Load / Refresh Data** to populate the dashboard."
        )
    else:
        p = st.session_state.portfolio or {}
        pos_df = st.session_state.positions

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Equity", f"${p.get('equity', 0):,.2f}")
        c2.metric("Cash", f"${p.get('cash', 0):,.2f}")
        c3.metric("Buying Power", f"${p.get('buying_power', 0):,.2f}")
        ret = p.get("total_return_pct", 0)
        c4.metric("Total Return", fmt_pct(ret), delta=fmt_pct(ret))

        if st.session_state.trades_df is not None:
            risk = compute_risk_metrics(st.session_state.trades_df)
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Sharpe Ratio", f"{risk.get('sharpe', 0):.2f}", help="Annualized trade-level Sharpe")
            r2.metric("Profit Factor", f"{risk.get('profit_factor', 0):.2f}", help="Gross profit / gross loss")
            r3.metric("Expectancy", f"{risk.get('expectancy_pct', 0):+.1f}%", help="Average expected return per trade")
            r4.metric("Max Drawdown", f"-{risk.get('max_drawdown_pct', 0):.1f}%")

        st.divider()

        if pos_df is not None and len(pos_df) > 0:
            df = pos_df.copy() if hasattr(pos_df, "copy") else pd.DataFrame(pos_df)

            left, right = st.columns([1, 2])
            with left:
                fig_pie = px.pie(
                    df, values="market_value", names="symbol",
                    title="Allocation",
                    color_discrete_sequence=px.colors.qualitative.Safe,
                    hole=0.45,
                )
                fig_pie.update_traces(textposition="inside", textinfo="percent+label")
                fig_pie.update_layout(**_plotly_base(), showlegend=False)
                st.plotly_chart(fig_pie, use_container_width=True)

            with right:
                colors = [GREEN if v >= 0 else RED for v in df["total_return"]]
                fig_pnl = go.Figure(go.Bar(
                    x=df["symbol"],
                    y=df["total_return"],
                    marker_color=colors,
                    text=df["return_pct"].apply(lambda v: fmt_pct(v)),
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>P&L: $%{y:,.0f}<extra></extra>",
                ))
                fig_pnl.update_layout(
                    title="Total Return by Position",
                    yaxis_title="P&L ($)",
                    **_plotly_base(),
                )
                st.plotly_chart(fig_pnl, use_container_width=True)

            st.subheader("Positions")
            display = df[["symbol","quantity","avg_cost","current_price","market_value","total_return","return_pct"]].copy()
            display.columns = ["Symbol","Qty","Avg Cost","Price","Value","Total P&L","Return %"]

            def _style_num(val):
                try:
                    return f"color: {GREEN}" if float(val) > 0 else f"color: {RED}"
                except Exception:
                    return ""

            styled = (
                display.style
                .applymap(_style_num, subset=["Total P&L", "Return %"])
                .format({
                    "Avg Cost": "${:,.2f}",
                    "Price": "${:,.2f}",
                    "Value": "${:,.2f}",
                    "Total P&L": "${:,.2f}",
                    "Return %": "{:+.1f}%",
                    "Qty": "{:.0f}",
                })
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)


# ── TAB 2: Trade Analysis ─────────────────────────────────────────────────────
with tab_trades:
    if not st.session_state.data_loaded or st.session_state.stats is None:
        st.info("Load data to see your trade analysis.")
    else:
        stats = st.session_state.stats
        trades_df = st.session_state.trades_df
        risk = compute_risk_metrics(trades_df)

        t1, t2, t3, t4, t5 = st.columns(5)
        wr = stats.get("win_rate", 0)
        t1.metric("Win Rate", f"{wr * 100:.1f}%")
        t2.metric("Total Trades", stats.get("total_trades", 0))
        t3.metric("Avg P&L / Trade", f"{stats.get('avg_pnl_pct', 0):+.1f}%")
        t4.metric("Total P&L", fmt_dollar(stats.get("total_pnl", 0)))
        kelly = risk.get("half_kelly_pct", 0)
        t5.metric("Half-Kelly Size", f"{kelly:.1f}%", help="Suggested max position size as % of portfolio")

        st.divider()

        def _bar_chart(df_or_list, group_col: str, title: str):
            if df_or_list is None:
                return
            df = pd.DataFrame(df_or_list) if isinstance(df_or_list, list) else df_or_list.copy()
            if df.empty or "win_rate_pct" not in df.columns or group_col not in df.columns:
                return
            df[group_col] = df[group_col].astype(str)
            fig = px.bar(
                df.sort_values("win_rate_pct"),
                x="win_rate_pct", y=group_col, orientation="h",
                title=title,
                color="win_rate_pct",
                color_continuous_scale=TEAL_SCALE,
                range_color=[0, 100],
                text="win_rate_pct",
            )
            fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
            fig.update_layout(**_plotly_base(), coloraxis_showscale=False, xaxis_range=[0, 110])
            st.plotly_chart(fig, use_container_width=True)

        row1_l, row1_r = st.columns(2)
        with row1_l:
            _bar_chart(stats.get("by_sector"), "sector", "Win Rate by Sector (%)")
        with row1_r:
            _bar_chart(stats.get("by_hold_bucket"), "hold_bucket", "Win Rate by Hold Duration (%)")

        row2_l, row2_r = st.columns(2)
        with row2_l:
            _bar_chart(stats.get("by_rsi_band"), "rsi_band", "Win Rate by RSI Entry Band (%)")
        with row2_r:
            _bar_chart(stats.get("by_entry_regime"), "entry_regime", "Win Rate by Market Regime (%)")

        if trades_df is not None and not trades_df.empty:
            st.divider()
            st.subheader("Cumulative P&L")
            cum = trades_df.sort_values("sell_date")[["sell_date","pnl_dollar"]].copy()
            cum["Cumulative P&L"] = cum["pnl_dollar"].cumsum()
            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(
                x=cum["sell_date"], y=cum["Cumulative P&L"],
                fill="tozeroy",
                line=dict(color=GREEN, width=2),
                fillcolor="rgba(0,212,170,0.15)",
                name="Cumulative P&L",
            ))
            fig_cum.update_layout(
                xaxis_title="Date", yaxis_title="P&L ($)",
                **_plotly_base(),
            )
            st.plotly_chart(fig_cum, use_container_width=True)

        edge = stats.get("edge_profile", {})
        if edge:
            st.divider()
            st.subheader("Your Statistical Edge")
            e1, e2, e3, e4 = st.columns(4)
            e1.info(f"**Best Sector**\n\n{edge.get('best_sector', '—')}")
            e2.info(f"**Best Hold**\n\n{edge.get('best_hold_duration', '—')}")
            e3.info(f"**Best RSI Entry**\n\n{edge.get('best_rsi_band', '—')}")
            e4.info(f"**Best Market Cap**\n\n{edge.get('best_market_cap', '—')}")

            pr = stats.get("patience_ratio", 1.0)
            if pr >= 1.2:
                st.success(f"You let winners run ({pr:.1f}x longer than losers). This is good discipline.")
            elif pr < 0.8:
                st.warning(f"You cut winners short ({pr:.1f}x hold ratio). Consider wider profit targets.")


# ── TAB 3: Screener ───────────────────────────────────────────────────────────
with tab_screen:
    st.subheader("Market Screener")
    if not st.session_state.data_loaded:
        st.info("Load data first.")
    else:
        if st.session_state.use_demo:
            cands = _demo_candidates()
            st.session_state.candidates = cands
            st.caption("Showing 7 pre-screened demo candidates. Switch to Live mode to run a real screen.")
        else:
            if st.button("Run Live Screen (2-3 min)", type="primary"):
                with st.spinner("Screening 150+ stocks across 10 sectors..."):
                    try:
                        from market.screener import build_candidate_universe, screen_candidates
                        edge_profile = st.session_state.stats.get("edge_profile", {}) if st.session_state.stats else {}
                        syms = build_candidate_universe()
                        raw = screen_candidates(syms, edge_profile)
                        cands = raw if hasattr(raw, "columns") else pd.DataFrame(raw)
                        st.session_state.candidates = cands
                        _compute_recommendations()
                        st.success(f"Found {len(cands)} candidates.")
                    except Exception as exc:
                        st.error(f"Screen failed: {exc}")
                        cands = st.session_state.candidates
            else:
                cands = st.session_state.candidates

        if cands is not None and len(cands) > 0:
            df = cands.copy()

            if "rsi" in df.columns and "momentum_20d" in df.columns:
                trend_colors = {
                    "Strong Uptrend": GREEN, "Uptrend": "#66ddbb",
                    "Sideways": AMBER, "Downtrend": "#ff8888", "Strong Downtrend": RED,
                }
                size_col = "volatility_20d" if "volatility_20d" in df.columns else None
                fig_sc = px.scatter(
                    df, x="rsi", y="momentum_20d",
                    text="symbol",
                    color="trend" if "trend" in df.columns else None,
                    size=size_col,
                    size_max=30,
                    title="RSI vs 20-Day Momentum  (bubble size = volatility)",
                    labels={"rsi": "RSI", "momentum_20d": "20d Momentum (%)"},
                    color_discrete_map=trend_colors,
                )
                fig_sc.update_traces(textposition="top center", marker=dict(opacity=0.85))
                fig_sc.add_vline(x=50, line_dash="dot", line_color="gray", opacity=0.4)
                fig_sc.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4)
                fig_sc.update_layout(**_plotly_base())
                st.plotly_chart(fig_sc, use_container_width=True)

            show_df = df.copy()
            for col in ["current_price"]:
                if col in show_df.columns:
                    show_df[col] = show_df[col].apply(lambda v: f"${v:,.2f}" if pd.notna(v) else "—")
            for col in ["rsi","momentum_20d","volatility_20d","beta"]:
                if col in show_df.columns:
                    show_df[col] = show_df[col].apply(lambda v: f"{v:.1f}" if pd.notna(v) else "—")
            if "revenue_growth" in show_df.columns:
                show_df["revenue_growth"] = show_df["revenue_growth"].apply(
                    lambda v: f"{v * 100:.0f}%" if pd.notna(v) else "—"
                )
            if "macd_bullish" in show_df.columns:
                show_df["macd_bullish"] = show_df["macd_bullish"].map({True: "Yes", False: "No"})
            st.dataframe(show_df, use_container_width=True, hide_index=True)


# ── TAB 4: Recommendations ────────────────────────────────────────────────────
with tab_recs:
    st.subheader("Recommendations")
    if not st.session_state.data_loaded:
        st.info("Load data first.")
    else:
        risk_data = compute_risk_metrics(st.session_state.trades_df) if st.session_state.trades_df is not None else {}
        kelly = risk_data.get("half_kelly_pct", 0)
        if kelly > 0:
            st.info(
                f"**Position Sizing (Half-Kelly):** Based on your win rate "
                f"({risk_data.get('win_rate', 0) * 100:.0f}%) and edge, "
                f"risk **{kelly:.1f}% of portfolio** per trade."
            )

        recs = st.session_state.recs
        opts = st.session_state.opts_recs

        st.subheader("Stock Picks")
        if recs is None or (hasattr(recs, "empty") and recs.empty):
            st.caption("No recommendations yet — run the screener or load demo data.")
        else:
            recs_df = recs if hasattr(recs, "columns") else pd.DataFrame(recs)

            if "score" in recs_df.columns and "symbol" in recs_df.columns:
                fig_bar = go.Figure(go.Bar(
                    x=recs_df["score"].values,
                    y=recs_df["symbol"].values,
                    orientation="h",
                    marker=dict(
                        color=recs_df["score"].values,
                        colorscale=[[0, RED], [0.5, AMBER], [1, GREEN]],
                        showscale=False,
                    ),
                    text=[f"{s:.0f}" for s in recs_df["score"].values],
                    textposition="auto",
                    hovertemplate="<b>%{y}</b>  Score: %{x:.0f}<extra></extra>",
                ))
                fig_bar.update_layout(
                    title="Recommendation Score (0–100)",
                    xaxis_range=[0, 110],
                    yaxis={"categoryorder": "total ascending"},
                    **_plotly_base(),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            show_cols = [c for c in ["symbol","sector","score","current_price","rsi","trend","momentum_20d","rationale"] if c in recs_df.columns]
            st.dataframe(recs_df[show_cols], use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Options Plays")
        if opts is None or (hasattr(opts, "empty") and opts.empty):
            st.caption("No options recommendations available.")
        else:
            opts_df = opts if hasattr(opts, "columns") else pd.DataFrame(opts)
            show_cols = [c for c in ["symbol","options_strategy","score","options_rationale"] if c in opts_df.columns]
            if show_cols:
                st.dataframe(opts_df[show_cols], use_container_width=True, hide_index=True)


# ── TAB 5: AI Chat ────────────────────────────────────────────────────────────
with tab_chat:
    st.subheader("Ask Sodil Anything")
    st.caption("Powered by Claude Opus — uses your portfolio & market tools automatically.")

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.warning("Enter your **Anthropic API Key** in the sidebar to use AI Chat.")
        st.stop()

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("tools_used"):
                st.caption(f"Tools: {', '.join(msg['tools_used'])}")

    if prompt := st.chat_input("What's my win rate in tech? Show me top opportunities..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.status("Working...", expanded=False) as status_box:
                try:
                    client = anthropic.Anthropic(api_key=api_key)
                    response_text, tools_used = run_agent_turn(client, prompt)
                    label = f"Used: {', '.join(tools_used)}" if tools_used else "Done"
                    status_box.update(label=label, state="complete")
                except Exception as exc:
                    response_text = f"Something went wrong: {exc}"
                    tools_used = []
                    status_box.update(label="Error", state="error")
            st.markdown(response_text)
            if tools_used:
                st.caption(f"Tools: {', '.join(tools_used)}")

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": response_text,
            "tools_used": tools_used,
        })

    if st.session_state.chat_messages:
        if st.button("Clear Chat"):
            st.session_state.chat_messages = []
            st.session_state.api_messages = []
            st.rerun()
