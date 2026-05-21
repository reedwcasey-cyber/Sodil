"""
Sodil — Investment Intelligence Dashboard
Launch:  python launch.py   (or: streamlit run dashboard/app.py)
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
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import anthropic

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sodil",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={"About": "Sodil — Quantitative Investment Intelligence"},
)

st.markdown("""
<style>
/* ── Global ──────────────────────────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px; }
*, *::before, *::after { box-sizing: border-box; }

/* ── Typography scale ────────────────────────────────────────────────────── */
/* Enforce max font sizes so nothing overflows columns */
h1, h2, h3, h4, h5, h6 { line-height: 1.2; }
p, span, div { max-width: 100%; }

/* ── Truncation utilities ─────────────────────────────────────────────────── */
.truncate-1 {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 100%;
}
.clamp-2 {
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    line-height: 1.45;
}
.clamp-3 {
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
    line-height: 1.45;
}

/* ── Position / Rec cards ────────────────────────────────────────────────── */
.pos-card {
    background: #131929;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 8px;
    transition: border-color 0.2s;
    min-width: 0;        /* allow flex children to shrink */
    overflow: hidden;
}
.pos-card:hover { border-color: rgba(0,212,170,0.4); }

.pos-sym {
    font-size: 1rem;
    font-weight: 700;
    color: #fff;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 80px;
}
.pos-name {
    font-size: 0.72rem;
    color: #8892a4;
    margin-bottom: 5px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 140px;
}
.pos-price { font-size: 1.15rem; font-weight: 700; color: #fff; white-space: nowrap; }
.pos-ret-pos { font-size: 0.8rem; color: #00d4aa; font-weight: 600; white-space: nowrap; }
.pos-ret-neg { font-size: 0.8rem; color: #ff5566; font-weight: 600; white-space: nowrap; }
.pos-meta  { font-size: 0.72rem; color: #8892a4; margin-top: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* Card inner flex rows must not overflow */
.card-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    min-width: 0;
    gap: 8px;
}
.card-row > div { min-width: 0; }

/* Rec card score number */
.rec-score {
    font-size: 1.25rem;
    font-weight: 800;
    white-space: nowrap;
    flex-shrink: 0;
}

/* Card rationale — 2-line clamp */
.card-rationale {
    font-size: 0.75rem;
    color: #8892a4;
    line-height: 1.45;
    margin-top: 8px;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

/* ── KPI metric containers ───────────────────────────────────────────────── */
div[data-testid="metric-container"] {
    background: #131929;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 12px 14px;
    overflow: hidden;
    min-width: 0;
}
/* Prevent metric labels from overflowing */
div[data-testid="metric-container"] label {
    font-size: 0.72rem !important;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 100%;
    display: block;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.1rem !important;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* ── Tab font ────────────────────────────────────────────────────────────── */
div[data-testid="stTabs"] button {
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.01em;
    white-space: nowrap;
}

/* ── Signal badges ───────────────────────────────────────────────────────── */
.badge-bull {
    display: inline-block;
    background: rgba(0,212,170,0.15);
    color: #00d4aa;
    border: 1px solid rgba(0,212,170,0.3);
    border-radius: 5px;
    padding: 2px 7px;
    font-size: 0.72rem;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 160px;
}
.badge-bear {
    display: inline-block;
    background: rgba(255,85,102,0.15);
    color: #ff5566;
    border: 1px solid rgba(255,85,102,0.3);
    border-radius: 5px;
    padding: 2px 7px;
    font-size: 0.72rem;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 160px;
}
.badge-neutral {
    display: inline-block;
    background: rgba(255,170,0,0.15);
    color: #ffaa00;
    border: 1px solid rgba(255,170,0,0.3);
    border-radius: 5px;
    padding: 2px 7px;
    font-size: 0.72rem;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 160px;
}
/* Badge row: always wraps — never overflows card */
.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin-top: 8px;
    overflow: hidden;
}

/* ── Score bar ───────────────────────────────────────────────────────────── */
.score-bar-wrap {
    background: #1e2130;
    border-radius: 5px;
    height: 6px;
    width: 100%;
    overflow: hidden;
}
.score-bar { height: 6px; border-radius: 5px; }

/* ── AI thesis box ───────────────────────────────────────────────────────── */
.thesis-box {
    max-height: 460px;
    overflow-y: auto;
    overflow-x: hidden;
    word-break: break-word;
    line-height: 1.65;
    font-size: 0.9rem;
    color: #e8eaf0;
    border-radius: 12px;
    padding: 18px 20px;
}
/* Scrollbar styling */
.thesis-box::-webkit-scrollbar { width: 4px; }
.thesis-box::-webkit-scrollbar-track { background: transparent; }
.thesis-box::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 4px; }

/* ── Price / equity headers ──────────────────────────────────────────────── */
.equity-num {
    font-size: clamp(1.6rem, 3vw, 2.4rem);
    font-weight: 800;
    color: #fff;
    letter-spacing: -1px;
    line-height: 1;
}
.price-num {
    font-size: clamp(1.3rem, 2.5vw, 2rem);
    font-weight: 800;
    color: #fff;
    white-space: nowrap;
}
.price-chg {
    font-size: clamp(0.85rem, 1.5vw, 1.05rem);
    font-weight: 600;
    white-space: nowrap;
}
.stock-name-header {
    font-size: 0.9rem;
    color: #8892a4;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 100%;
    margin-bottom: 2px;
}

/* ── Info / edge boxes ───────────────────────────────────────────────────── */
div[data-testid="stAlert"] {
    overflow: hidden;
}
div[data-testid="stAlert"] p {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* ── Quant Lab stock header ──────────────────────────────────────────────── */
.lab-header {
    display: flex;
    align-items: baseline;
    flex-wrap: wrap;
    gap: 8px;
    margin: 10px 0 18px;
    min-width: 0;
}
.lab-sym  { font-size: 1.3rem; font-weight: 800; color: #fff; flex-shrink: 0; }
.lab-name {
    font-size: 0.88rem;
    color: #8892a4;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 280px;
    flex-shrink: 1;
}
.lab-price { font-size: 1.3rem; font-weight: 700; color: #fff; flex-shrink: 0; white-space: nowrap; }

/* ── Probability panel ───────────────────────────────────────────────────── */
.prob-panel {
    background: #131929;
    border-radius: 10px;
    padding: 14px 16px;
    border: 1px solid rgba(255,255,255,0.08);
}
.prob-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
    gap: 8px;
}
.prob-label { font-size: 0.78rem; color: #c8d0e0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.prob-val   { font-size: 0.78rem; font-weight: 700; white-space: nowrap; flex-shrink: 0; }

/* ── Streamlit column overflow guard ─────────────────────────────────────── */
[data-testid="column"] { min-width: 0; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Color constants ───────────────────────────────────────────────────────────
GREEN, RED, AMBER, BLUE = "#00d4aa", "#ff5566", "#ffaa00", "#4488ff"
TEAL_SCALE = [[0, RED], [0.5, AMBER], [1, GREEN]]

MODEL = "claude-opus-4-7"
SYSTEM = """You are Sodil, a world-class quantitative investment advisor. You have access to the user's live portfolio, trade history, and market data.

Be concise and numbers-driven. Always quote specific figures. When the user asks about their edge, use analyze_trades. For opportunities, use screen_market then get_recommendations. For portfolio/P&L, use get_portfolio.

Give clear buy/hold/avoid verdicts when analyzing specific stocks."""

TOOLS = [
    {"name": "get_portfolio", "description": "Fetch portfolio value, cash, buying power, and all open positions with P&L.", "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "analyze_trades", "description": "Compute win rate, profit factor, best sectors/hold-durations/RSI-bands, and full statistical edge from trade history.", "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "screen_market", "description": "Screen 150+ stocks for technically and fundamentally sound candidates.", "input_schema": {"type": "object", "properties": {}, "required": []}},
    {
        "name": "get_recommendations",
        "description": "Score candidates against the user's personal edge profile and return ranked picks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "candidates_json": {"type": "string"},
                "stats_json": {"type": "string"},
            },
            "required": ["candidates_json", "stats_json"],
        },
    },
]

PERIOD_MAP = {"1W": "5d", "1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "5Y": "5y"}
SUGGESTED_QUESTIONS = [
    "What is my win rate and biggest edge?",
    "Show me my top 3 opportunities right now",
    "Which positions should I be worried about?",
    "What's my best sector and why?",
    "How should I size my next trade?",
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
    "research_ticker": "NVDA",
    "research_period": "1Y",
    "watchlist": ["NVDA", "AAPL", "META", "MSFT"],
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ══════════════════════════════════════════════════════════════════════════════
# DEMO DATA
# ══════════════════════════════════════════════════════════════════════════════
def _demo_portfolio() -> dict:
    return {"equity": 47_823.44, "cash": 3_210.00, "buying_power": 6_420.00, "total_return_pct": 12.4}


def _demo_positions() -> list[dict]:
    return [
        {"symbol": "NVDA", "name": "NVIDIA Corp",       "quantity": 15, "avg_cost": 420.00, "current_price": 875.32, "market_value": 13129.80, "total_return": 6829.80, "return_pct": 108.3, "day_return_pct": 2.1},
        {"symbol": "AAPL", "name": "Apple Inc",          "quantity": 25, "avg_cost": 168.50, "current_price": 189.25, "market_value": 4731.25,  "total_return": 518.75,  "return_pct": 12.3,  "day_return_pct": -0.3},
        {"symbol": "TSLA", "name": "Tesla Inc",          "quantity": 10, "avg_cost": 235.00, "current_price": 178.40, "market_value": 1784.00,  "total_return": -566.00, "return_pct": -24.1, "day_return_pct": -1.8},
        {"symbol": "META", "name": "Meta Platforms",     "quantity": 8,  "avg_cost": 290.00, "current_price": 491.00, "market_value": 3928.00,  "total_return": 1608.00, "return_pct": 69.3,  "day_return_pct": 0.7},
        {"symbol": "AMD",  "name": "Advanced Micro Devices","quantity": 30, "avg_cost": 95.00, "current_price": 162.50, "market_value": 4875.00, "total_return": 2025.00, "return_pct": 71.1,  "day_return_pct": 1.2},
        {"symbol": "MSFT", "name": "Microsoft Corp",     "quantity": 12, "avg_cost": 330.00, "current_price": 415.80, "market_value": 4989.60,  "total_return": 1029.60, "return_pct": 26.0,  "day_return_pct": 0.4},
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
    rows, base = [], pd.Timestamp("2023-01-01")
    for i in range(60):
        sym = np.random.choice(symbols)
        sector = sectors[sym]
        hold = int(np.random.choice([1,3,7,14,30,60,90,180], p=[0.05,0.10,0.15,0.20,0.25,0.15,0.07,0.03]))
        buy = round(np.random.uniform(50, 500), 2)
        pnl = round(np.random.normal(
            14 if sector == "Technology" else -2 if sym in ("TSLA","AMZN") else 4,
            18 if sector == "Technology" else 22 if sym in ("TSLA","AMZN") else 12,
        ), 2)
        sell = round(buy * (1 + pnl / 100), 2)
        qty = round(np.random.uniform(5, 50), 2)
        buy_date = base + pd.Timedelta(days=i * 6)
        rows.append({
            "symbol": sym, "sector": sector,
            "buy_date": buy_date,
            "sell_date": buy_date + pd.Timedelta(days=hold),
            "hold_days": hold, "buy_price": buy, "sell_price": sell, "quantity": qty,
            "pnl_dollar": (sell - buy) * qty, "pnl_pct": pnl, "win": pnl > 0,
            "market_cap_bucket": np.random.choice(["Mega","Large","Mid"], p=[0.5,0.3,0.2]),
            "entry_rsi": round(np.random.uniform(28, 72), 1),
            "entry_regime": np.random.choice(["Uptrend","Sideways","Downtrend"], p=[0.55,0.30,0.15]),
        })
    df = pd.DataFrame(rows)
    df["hold_bucket"] = pd.cut(
        df["hold_days"], bins=[-1,1,7,30,90,365,9999],
        labels=["Intraday","Swing (2-7d)","Monthly (1mo)","Quarterly (3mo)","Annual (1yr)","Long-term"],
    )
    df["rsi_band"] = pd.cut(
        df["entry_rsi"], bins=[0,30,45,55,70,100],
        labels=["Oversold(<30)","Low(30-45)","Neutral(45-55)","High(55-70)","Overbought(>70)"],
    )
    return df


def _demo_candidates() -> pd.DataFrame:
    return pd.DataFrame([
        {"symbol":"CRWD","name":"CrowdStrike","sector":"Technology","current_price":325.0,"market_cap":80e9,"rsi":48.2,"trend":"Uptrend","macd_bullish":True,"momentum_20d":8.4,"momentum_5d":2.1,"volatility_20d":32.1,"revenue_growth":0.33,"beta":1.4,"52w_high":395,"52w_low":130},
        {"symbol":"DDOG","name":"Datadog","sector":"Technology","current_price":142.0,"market_cap":45e9,"rsi":42.5,"trend":"Uptrend","macd_bullish":True,"momentum_20d":5.2,"momentum_5d":1.8,"volatility_20d":38.4,"revenue_growth":0.27,"beta":1.6,"52w_high":175,"52w_low":90},
        {"symbol":"NET","name":"Cloudflare","sector":"Technology","current_price":98.0,"market_cap":32e9,"rsi":44.0,"trend":"Uptrend","macd_bullish":True,"momentum_20d":6.1,"momentum_5d":0.9,"volatility_20d":41.2,"revenue_growth":0.30,"beta":1.7,"52w_high":120,"52w_low":55},
        {"symbol":"PANW","name":"Palo Alto Networks","sector":"Technology","current_price":358.0,"market_cap":115e9,"rsi":52.1,"trend":"Strong Uptrend","macd_bullish":True,"momentum_20d":12.3,"momentum_5d":3.2,"volatility_20d":28.6,"revenue_growth":0.22,"beta":1.2,"52w_high":380,"52w_low":200},
        {"symbol":"ISRG","name":"Intuitive Surgical","sector":"Healthcare","current_price":415.0,"market_cap":147e9,"rsi":56.0,"trend":"Strong Uptrend","macd_bullish":True,"momentum_20d":9.1,"momentum_5d":1.8,"volatility_20d":22.3,"revenue_growth":0.14,"beta":0.9,"52w_high":430,"52w_low":290},
        {"symbol":"COIN","name":"Coinbase","sector":"Financial Services","current_price":220.0,"market_cap":55e9,"rsi":45.0,"trend":"Uptrend","macd_bullish":True,"momentum_20d":11.2,"momentum_5d":3.5,"volatility_20d":68.4,"revenue_growth":None,"beta":2.8,"52w_high":283,"52w_low":80},
        {"symbol":"AXON","name":"Axon Enterprise","sector":"Industrials","current_price":295.0,"market_cap":19e9,"rsi":46.2,"trend":"Uptrend","macd_bullish":True,"momentum_20d":7.3,"momentum_5d":2.4,"volatility_20d":35.5,"revenue_growth":0.30,"beta":1.3,"52w_high":330,"52w_low":165},
    ])


# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
def compute_risk_metrics(trades_df: pd.DataFrame) -> dict:
    if trades_df is None or trades_df.empty:
        return {}
    wins = trades_df[trades_df["win"]]
    losses = trades_df[~trades_df["win"]]
    wr = float(trades_df["win"].mean())
    avg_win = float(wins["pnl_pct"].mean()) if not wins.empty else 0.0
    avg_loss = float(abs(losses["pnl_pct"].mean())) if not losses.empty else 1.0
    b = avg_win / avg_loss if avg_loss > 0 else 0
    kelly = max(0.0, ((wr * b - (1 - wr)) / b) / 2) if b > 0 else 0
    expectancy = wr * avg_win - (1 - wr) * avg_loss
    ret = trades_df["pnl_pct"].values
    sharpe = float(ret.mean() / ret.std() * np.sqrt(12)) if ret.std() > 0 else 0.0
    cum = trades_df["pnl_dollar"].cumsum().values
    peak = np.maximum.accumulate(cum)
    dd = (peak - cum) / np.where(peak == 0, 1, peak)
    gp = float(wins["pnl_dollar"].sum()) if not wins.empty else 0.0
    gl = float(abs(losses["pnl_dollar"].sum())) if not losses.empty else 1.0
    return {
        "half_kelly_pct": kelly * 100,
        "expectancy_pct": expectancy,
        "sharpe": sharpe,
        "max_drawdown_pct": float(dd.max() * 100),
        "profit_factor": gp / gl,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "win_rate": wr,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _base_layout(**kwargs) -> dict:
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c8d0e0", size=12),
        margin=dict(t=32, b=28, l=8, r=8),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0, font=dict(size=11)),
    )
    base.update(kwargs)
    return base


def _axis_style(show_grid: bool = True) -> dict:
    return dict(
        showgrid=show_grid,
        gridcolor="rgba(255,255,255,0.06)",
        zeroline=False,
        showline=False,
        tickfont=dict(size=11, color="#8892a4"),
    )


@st.cache_data(ttl=300, show_spinner=False)
def fetch_chart(symbol: str, period: str) -> tuple[pd.DataFrame, dict]:
    """Fetch OHLCV + fundamentals for any ticker. Cached 5 min."""
    try:
        import yfinance as yf
        tk = yf.Ticker(symbol.upper())
        hist = tk.history(period=period, auto_adjust=True)
        if not hist.empty:
            hist.index = hist.index.tz_localize(None)
        info = tk.info or {}
        return hist, info
    except Exception:
        return pd.DataFrame(), {}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_technicals(symbol: str) -> dict:
    """Fetch and compute full technical indicators for a ticker."""
    try:
        from market.data import get_price_history, compute_technicals
        hist = get_price_history(symbol, period="1y")
        if hist.empty:
            return {}
        return compute_technicals(hist)
    except Exception:
        return {}


def build_price_chart(
    hist: pd.DataFrame,
    symbol: str,
    trades_df: pd.DataFrame | None = None,
    show_volume: bool = True,
) -> go.Figure:
    """
    Robinhood-style area chart with buy/sell trade overlays.
    Gradient fill: green if price up over period, red if down.
    """
    if hist.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False, font=dict(color="#8892a4"))
        fig.update_layout(**_base_layout())
        return fig

    close = hist["Close"] if "Close" in hist.columns else hist.iloc[:, 3]
    is_up = float(close.iloc[-1]) >= float(close.iloc[0])
    line_color = GREEN if is_up else RED
    fill_color = "rgba(0,212,170,0.12)" if is_up else "rgba(255,85,102,0.12)"

    from plotly.subplots import make_subplots
    rows = 2 if (show_volume and "Volume" in hist.columns) else 1
    row_heights = [0.75, 0.25] if rows == 2 else [1]
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
    )

    # Price area
    fig.add_trace(go.Scatter(
        x=hist.index, y=close,
        fill="tozeroy",
        fillcolor=fill_color,
        line=dict(color=line_color, width=2.5),
        mode="lines",
        name="Price",
        hovertemplate="<b>%{x|%b %d '%y}</b>  $%{y:,.2f}<extra></extra>",
    ), row=1, col=1)

    # SMA overlays (from longer period history only)
    if len(hist) >= 50:
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        fig.add_trace(go.Scatter(
            x=hist.index, y=sma20,
            line=dict(color="rgba(255,170,0,0.6)", width=1.2, dash="dot"),
            mode="lines", name="SMA20",
            hovertemplate="SMA20: $%{y:,.2f}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=hist.index, y=sma50,
            line=dict(color="rgba(68,136,255,0.6)", width=1.2, dash="dot"),
            mode="lines", name="SMA50",
            hovertemplate="SMA50: $%{y:,.2f}<extra></extra>",
        ), row=1, col=1)

    # Trade markers
    if trades_df is not None and not trades_df.empty:
        sym_trades = trades_df[trades_df["symbol"] == symbol.upper()].copy()
        if len(sym_trades) > 0:
            # Filter to visible date range
            start_dt = hist.index.min()
            end_dt = hist.index.max()

            buys = sym_trades[
                (pd.to_datetime(sym_trades["buy_date"]) >= start_dt) &
                (pd.to_datetime(sym_trades["buy_date"]) <= end_dt)
            ]
            sells = sym_trades[
                (pd.to_datetime(sym_trades["sell_date"]) >= start_dt) &
                (pd.to_datetime(sym_trades["sell_date"]) <= end_dt)
            ]

            if len(buys) > 0:
                hover_buy = [
                    f"<b>BUY {r.symbol}</b><br>${r.buy_price:,.2f} × {r.quantity:.0f} shares"
                    for _, r in buys.iterrows()
                ]
                fig.add_trace(go.Scatter(
                    x=pd.to_datetime(buys["buy_date"]),
                    y=buys["buy_price"],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up", size=16, color=GREEN,
                        line=dict(color="white", width=1.5),
                    ),
                    name="Your Buy",
                    hovertemplate="%{text}<extra></extra>",
                    text=hover_buy,
                ), row=1, col=1)

            if len(sells) > 0:
                hover_sell = [
                    f"<b>SELL {r.symbol}</b><br>${r.sell_price:,.2f}  P&L: {r.pnl_pct:+.1f}%"
                    for _, r in sells.iterrows()
                ]
                fig.add_trace(go.Scatter(
                    x=pd.to_datetime(sells["sell_date"]),
                    y=sells["sell_price"],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-down", size=16, color=RED,
                        line=dict(color="white", width=1.5),
                    ),
                    name="Your Sell",
                    hovertemplate="%{text}<extra></extra>",
                    text=hover_sell,
                ), row=1, col=1)

    # Volume bars
    if rows == 2 and "Volume" in hist.columns:
        vol_colors = [GREEN if c >= o else RED
                      for c, o in zip(hist["Close"], hist["Open"])]
        fig.add_trace(go.Bar(
            x=hist.index, y=hist["Volume"],
            marker_color=vol_colors,
            marker_opacity=0.5,
            name="Volume",
            hovertemplate="Vol: %{y:,.0f}<extra></extra>",
        ), row=2, col=1)

    # Layout
    fig.update_xaxes(**_axis_style(show_grid=False))
    fig.update_yaxes(**_axis_style(show_grid=True))
    if rows == 2:
        fig.update_yaxes(title_text="Price ($)", row=1, col=1, title_font=dict(size=11, color="#8892a4"))
        fig.update_yaxes(title_text="Volume", row=2, col=1, showticklabels=False, title_font=dict(size=10, color="#8892a4"))
    fig.update_layout(
        **_base_layout(margin=dict(t=16, b=8, l=8, r=8)),
        showlegend=True,
    )
    return fig


def build_pnl_timeline(trades_df: pd.DataFrame) -> go.Figure:
    """Cumulative P&L area chart with individual trade hover markers."""
    if trades_df is None or trades_df.empty:
        return go.Figure()
    df = trades_df.sort_values("sell_date").copy()
    df["cumulative"] = df["pnl_dollar"].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["sell_date"], y=df["cumulative"],
        fill="tozeroy",
        fillcolor="rgba(0,212,170,0.1)",
        line=dict(color=GREEN, width=2.5),
        mode="lines",
        name="Cumulative P&L",
        hovertemplate="<b>%{x|%b %d '%y}</b>  Cumulative: $%{y:,.0f}<extra></extra>",
    ))

    wins = df[df["win"]]
    losses = df[~df["win"]]

    if len(wins) > 0:
        fig.add_trace(go.Scatter(
            x=wins["sell_date"], y=wins["cumulative"],
            mode="markers",
            marker=dict(color=GREEN, size=8, opacity=0.8, line=dict(color="white", width=1)),
            name="Win",
            hovertemplate="<b>%{customdata[0]}</b><br>+$%{customdata[1]:,.0f} (+%{customdata[2]:.1f}%)<extra></extra>",
            customdata=np.stack([wins["symbol"], wins["pnl_dollar"], wins["pnl_pct"]], axis=-1),
        ))

    if len(losses) > 0:
        fig.add_trace(go.Scatter(
            x=losses["sell_date"], y=losses["cumulative"],
            mode="markers",
            marker=dict(color=RED, size=8, opacity=0.8, line=dict(color="white", width=1)),
            name="Loss",
            hovertemplate="<b>%{customdata[0]}</b><br>$%{customdata[1]:,.0f} (%{customdata[2]:.1f}%)<extra></extra>",
            customdata=np.stack([losses["symbol"], losses["pnl_dollar"], losses["pnl_pct"]], axis=-1),
        ))

    fig.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_dash="dash")
    fig.update_xaxes(**_axis_style(show_grid=False))
    fig.update_yaxes(**_axis_style(show_grid=True), tickprefix="$")
    fig.update_layout(**_base_layout(title="Cumulative P&L  ·  Hover trades for detail"))
    return fig


def build_breakdown_chart(data: Any, group_col: str, title: str) -> go.Figure | None:
    if data is None:
        return None
    df = pd.DataFrame(data) if isinstance(data, list) else data.copy()
    if df.empty or group_col not in df.columns or "win_rate_pct" not in df.columns:
        return None
    df[group_col] = df[group_col].astype(str)
    df = df[df[group_col] != "nan"].sort_values("win_rate_pct", ascending=True)

    bar_colors = [
        GREEN if v >= 55 else AMBER if v >= 45 else RED
        for v in df["win_rate_pct"]
    ]
    fig = go.Figure(go.Bar(
        x=df["win_rate_pct"], y=df[group_col], orientation="h",
        marker_color=bar_colors,
        text=df["win_rate_pct"].apply(lambda v: f"{v:.0f}%"),
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>Win Rate: %{x:.1f}%<br>"
            + (f"Avg P&L: %{{customdata:.1f}}%" if "avg_pnl_pct" in df.columns else "")
            + "<extra></extra>"
        ),
        customdata=df["avg_pnl_pct"].values if "avg_pnl_pct" in df.columns else None,
    ))
    fig.add_vline(x=50, line_color="rgba(255,255,255,0.2)", line_dash="dash", annotation_text="50%", annotation_font_size=10)
    fig.update_xaxes(**_axis_style(show_grid=False), range=[0, 115])
    fig.update_yaxes(**_axis_style(show_grid=False))
    fig.update_layout(**_base_layout(title=title, margin=dict(t=36, b=8, l=8, r=32)))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# TOOL EXECUTION
# ══════════════════════════════════════════════════════════════════════════════
def _execute_tool(name: str, tool_input: dict, use_demo: bool) -> str:
    try:
        if name == "get_portfolio":
            result: Any = (
                {"portfolio": _demo_portfolio(), "positions": _demo_positions()}
                if use_demo else _live_portfolio()
            )

        elif name == "analyze_trades":
            trades = _demo_trades() if use_demo else _live_trades()
            from portfolio.analyzer import analyze_process
            raw = analyze_process(trades)
            result = {
                k: (v.to_dict(orient="records") if isinstance(v, pd.DataFrame)
                    else float(v) if isinstance(v, (np.integer, np.floating)) else v)
                for k, v in raw.items()
            }

        elif name == "screen_market":
            result = (
                {"candidates": _demo_candidates().to_dict(orient="records")}
                if use_demo else _live_screen()
            )

        elif name == "get_recommendations":
            cands = pd.DataFrame(json.loads(tool_input.get("candidates_json", "[]")))
            stats = json.loads(tool_input.get("stats_json", "{}"))
            for key in ("by_sector","by_hold_bucket","by_rsi_band","by_entry_regime","by_market_cap"):
                if key in stats and isinstance(stats[key], list):
                    stats[key] = pd.DataFrame(stats[key])
            from recommendations.engine import score_candidates, score_options_candidates
            recs = score_candidates(cands, stats)
            opts = score_options_candidates(cands, stats)
            result = {
                "stock_recommendations": recs.to_dict(orient="records") if not recs.empty else [],
                "options_recommendations": opts[["symbol","score","options_strategy","options_rationale"]].to_dict(orient="records") if not opts.empty else [],
            }
        else:
            result = {"error": f"Unknown tool: {name}"}
    except Exception as exc:
        result = {"error": str(exc)}
    return json.dumps(result, default=str)


def _live_portfolio() -> dict:
    from auth.robinhood import login
    from portfolio.tracker import get_portfolio, get_positions
    if not login():
        return {"error": "Login failed"}
    pos = get_positions()
    return {
        "portfolio": get_portfolio(),
        "positions": pos.to_dict(orient="records") if hasattr(pos, "to_dict") else pos,
    }


def _live_trades() -> pd.DataFrame:
    from auth.robinhood import login
    from portfolio.tracker import get_order_history
    from portfolio.analyzer import build_trade_pairs
    if not login():
        return pd.DataFrame()
    orders = get_order_history(limit=500)
    return build_trade_pairs(orders) if not orders.empty else pd.DataFrame()


def _live_screen() -> dict:
    from market.screener import build_candidate_universe, screen_candidates
    syms = build_candidate_universe()
    cands = screen_candidates(syms, {})
    return {"candidates": cands.to_dict(orient="records") if hasattr(cands, "to_dict") else cands}


# ══════════════════════════════════════════════════════════════════════════════
# AGENT
# ══════════════════════════════════════════════════════════════════════════════
def run_agent_turn(client: anthropic.Anthropic, user_message: str) -> tuple[str, list[str]]:
    use_demo = st.session_state.use_demo
    msgs = list(st.session_state.api_messages)
    msgs.append({"role": "user", "content": user_message})
    tools_used: list[str] = []

    while True:
        response = client.messages.create(
            model=MODEL, max_tokens=4096, system=SYSTEM, tools=TOOLS, messages=msgs,
        )
        msgs.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            st.session_state.api_messages = msgs
            return next((b.text for b in response.content if b.type == "text" and b.text), ""), tools_used

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tools_used.append(block.name)
                    result_str = _execute_tool(block.name, block.input, use_demo)
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result_str})
            msgs.append({"role": "user", "content": tool_results})
            continue

        st.session_state.api_messages = msgs
        return next((b.text for b in response.content if b.type == "text" and b.text), ""), tools_used


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_all_data() -> bool:
    use_demo = st.session_state.use_demo
    try:
        if use_demo:
            st.session_state.portfolio = _demo_portfolio()
            st.session_state.positions = pd.DataFrame(_demo_positions())
            trades = _demo_trades()
        else:
            from auth.robinhood import login
            from portfolio.tracker import get_portfolio, get_positions, get_order_history
            from portfolio.analyzer import build_trade_pairs
            if not login():
                st.error("Robinhood login failed. Check credentials.")
                return False
            st.session_state.portfolio = get_portfolio()
            raw_pos = get_positions()
            st.session_state.positions = raw_pos if hasattr(raw_pos, "columns") else pd.DataFrame(raw_pos)
            orders = get_order_history(limit=500)
            trades = build_trade_pairs(orders) if not orders.empty else _demo_trades()

        st.session_state.trades_df = trades
        from portfolio.analyzer import analyze_process
        st.session_state.stats = analyze_process(trades)

        if use_demo:
            st.session_state.candidates = _demo_candidates()
            _compute_recs()

        st.session_state.data_loaded = True
        return True
    except Exception as exc:
        st.error(f"Error loading data: {exc}")
        return False


def _compute_recs() -> None:
    cands = st.session_state.candidates
    stats = st.session_state.stats
    if cands is None or (hasattr(cands, "empty") and cands.empty) or not stats:
        return
    try:
        from recommendations.engine import score_candidates, score_options_candidates
        st.session_state.recs = score_candidates(cands, stats)
        st.session_state.opts_recs = score_options_candidates(cands, stats)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def fmt_pct(v: float) -> str:
    return f"+{v:.1f}%" if v > 0 else f"{v:.1f}%"


def fmt_dollar(v: float) -> str:
    return f"+${v:,.0f}" if v > 0 else f"-${abs(v):,.0f}"


def trend_badge(trend: str) -> str:
    if "Up" in trend:
        return f'<span class="badge-bull">{trend}</span>'
    elif "Down" in trend:
        return f'<span class="badge-bear">{trend}</span>'
    return f'<span class="badge-neutral">{trend}</span>'


def score_html(score: float) -> str:
    color = GREEN if score >= 65 else AMBER if score >= 45 else RED
    return (
        f'<div class="score-bar-wrap">'
        f'<div class="score-bar" style="width:{score:.0f}%;background:{color};"></div>'
        f'</div><div style="font-size:0.8rem;color:{color};margin-top:2px;">{score:.0f}/100</div>'
    )


def rsi_color(rsi: float) -> str:
    if rsi < 30:
        return GREEN   # oversold = buy opportunity
    elif rsi > 70:
        return RED     # overbought = caution
    return "#e8eaf0"


def _position_card(pos: dict) -> None:
    sym = pos.get("symbol", "")
    name = pos.get("name", sym)
    qty = pos.get("quantity", 0)
    price = pos.get("current_price", 0)
    total_ret = pos.get("total_return", 0)
    ret_pct = pos.get("return_pct", 0)
    day_ret = pos.get("day_return_pct", 0)
    mktval = pos.get("market_value", 0)
    ret_class = "pos-ret-pos" if ret_pct >= 0 else "pos-ret-neg"
    day_class = "pos-ret-pos" if day_ret >= 0 else "pos-ret-neg"
    # Truncate long company names at display level
    display_name = name if len(name) <= 22 else name[:20] + "…"
    st.markdown(f"""
    <div class="pos-card">
      <div class="card-row">
        <div style="min-width:0;flex:1;">
          <div class="pos-sym">{sym}</div>
          <div class="pos-name" title="{name}">{display_name}</div>
        </div>
        <div style="text-align:right;flex-shrink:0;">
          <div class="pos-price">${price:,.2f}</div>
          <div class="{day_class}">Today: {fmt_pct(day_ret)}</div>
        </div>
      </div>
      <div class="card-row" style="margin-top:8px;">
        <div class="pos-meta">{qty:.0f} sh · ${mktval:,.0f}</div>
        <div class="{ret_class}" style="text-align:right;">{fmt_pct(ret_pct)}&nbsp;({fmt_dollar(total_ret)})</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def _rec_card(rec: pd.Series, col_idx: int) -> None:
    sym = rec.get("symbol", "")
    name = rec.get("name", sym)
    score = float(rec.get("score", 0))
    price = rec.get("current_price", 0)
    rsi = rec.get("rsi", 50)
    trend = str(rec.get("trend", ""))
    rationale = str(rec.get("rationale", ""))
    mom = rec.get("momentum_20d", 0)
    color = GREEN if score >= 65 else AMBER if score >= 45 else RED
    display_name = name if len(name) <= 20 else name[:18] + "…"
    mom_cls = "badge-bull" if mom >= 0 else "badge-bear"
    mom_arrow = "▲" if mom >= 0 else "▼"
    # Truncate rationale beyond 120 chars — tooltip shows full text
    short_rat = rationale if len(rationale) <= 120 else rationale[:117] + "…"
    st.markdown(f"""
    <div class="pos-card">
      <div class="card-row" style="margin-bottom:6px;">
        <div style="min-width:0;flex:1;">
          <div class="pos-sym">{sym}</div>
          <div class="pos-name" title="{name}">{display_name}</div>
        </div>
        <div class="rec-score" style="color:{color};">{score:.0f}</div>
      </div>
      {score_html(score)}
      <div class="badge-row">
        {trend_badge(trend)}
        <span class="badge-neutral">RSI {rsi:.0f}</span>
        <span class="{mom_cls}">{mom_arrow} {abs(mom):.1f}%</span>
      </div>
      <div class="card-rationale" title="{rationale}">{short_rat}</div>
      <div style="margin-top:6px;font-size:0.82rem;color:#c8d0e0;font-weight:600;">${price:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)
    # Expand icon if rationale is long
    if len(rationale) > 120:
        with st.expander("ℹ️ Full rationale"):
            st.caption(rationale)


# ══════════════════════════════════════════════════════════════════════════════
# TOP NAV / HEADER
# ══════════════════════════════════════════════════════════════════════════════
hdr_l, hdr_r = st.columns([3, 1])
with hdr_l:
    st.markdown("<h2 style='margin:0;padding:0;color:#fff;letter-spacing:-0.5px;'>📈 Sodil</h2>", unsafe_allow_html=True)
    st.caption("Quantitative Investment Intelligence")
with hdr_r:
    mode = st.radio("", ["Demo", "Live"], horizontal=True,
                    index=0 if st.session_state.use_demo else 1, key="top_mode")
    if (mode == "Demo") != st.session_state.use_demo:
        st.session_state.use_demo = (mode == "Demo")
        st.session_state.data_loaded = False
        st.rerun()

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR (credentials + load button)
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### Settings")

    if not st.session_state.use_demo:
        with st.expander("Robinhood Credentials", expanded=True):
            rh_user = st.text_input("Email", placeholder="you@example.com")
            rh_pass = st.text_input("Password", type="password")
            rh_mfa  = st.text_input("MFA Secret (optional)")
            if rh_user: os.environ["RH_USERNAME"] = rh_user
            if rh_pass: os.environ["RH_PASSWORD"] = rh_pass
            if rh_mfa:  os.environ["RH_MFA_SECRET"] = rh_mfa

    api_key_input = st.text_input(
        "Anthropic API Key",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        type="password",
        help="For AI Advisor tab",
    )
    if api_key_input:
        os.environ["ANTHROPIC_API_KEY"] = api_key_input

    st.divider()
    btn_label = "Load Demo Data" if st.session_state.use_demo else "Connect & Load"
    if st.button(btn_label, type="primary", use_container_width=True):
        with st.spinner("Loading..."):
            load_all_data()

    if st.session_state.data_loaded:
        st.success("✓ Data ready")
    else:
        st.info("Click above to load data")

    st.divider()
    st.caption("**Watchlist**")
    new_ticker = st.text_input("Add ticker", placeholder="e.g. TSLA", key="wl_add").upper()
    if new_ticker and new_ticker not in st.session_state.watchlist:
        if st.button("Add"):
            st.session_state.watchlist.append(new_ticker)
    for wl_sym in st.session_state.watchlist:
        if st.button(f"📊 {wl_sym}", key=f"wl_{wl_sym}", use_container_width=True):
            st.session_state.research_ticker = wl_sym


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_home, tab_research, tab_trades, tab_opps, tab_lab, tab_ai = st.tabs([
    "🏠  Portfolio",
    "🔎  Research",
    "📊  My Trades",
    "🎯  Opportunities",
    "⚡  Quant Lab",
    "🤖  AI Advisor",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PORTFOLIO (Robinhood-style)
# ══════════════════════════════════════════════════════════════════════════════
with tab_home:
    if not st.session_state.data_loaded:
        st.markdown("""
        <div style="max-width:520px;padding:32px 0 8px;">
          <div style="font-size:1.15rem;font-weight:700;color:#fff;margin-bottom:6px;">Welcome to Sodil</div>
          <div style="font-size:0.88rem;color:#8892a4;line-height:1.6;margin-bottom:18px;">
            The smarter way to understand your trades and find your next move.
          </div>
          <div style="font-size:0.82rem;color:#c8d0e0;line-height:2;">
            <b style="color:#00d4aa;">1.</b>&nbsp; Choose <b>Demo</b> (top right) — no credentials needed<br>
            <b style="color:#00d4aa;">2.</b>&nbsp; Click <b>Load Demo Data</b> in the sidebar<br>
            <b style="color:#00d4aa;">3.</b>&nbsp; Explore, research stocks, and ask the AI anything
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        p = st.session_state.portfolio or {}
        pos_df = st.session_state.positions
        trades_df = st.session_state.trades_df

        # ── Big equity number ──────────────────────────────────────────────────
        equity = p.get("equity", 0)
        ret_pct = p.get("total_return_pct", 0)
        ret_color = GREEN if ret_pct >= 0 else RED
        st.markdown(f"""
        <div style="margin-bottom:18px;">
          <div style="font-size:0.75rem;color:#8892a4;font-weight:500;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:2px;">Portfolio Value</div>
          <div class="equity-num">${equity:,.2f}</div>
          <div style="font-size:0.9rem;color:{ret_color};font-weight:600;margin-top:3px;">
            {fmt_pct(ret_pct)} total return
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── KPI row ────────────────────────────────────────────────────────────
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Cash", f"${p.get('cash',0):,.0f}")
        k2.metric("Buying Power", f"${p.get('buying_power',0):,.0f}")
        risk = compute_risk_metrics(trades_df)
        k3.metric("Sharpe Ratio", f"{risk.get('sharpe',0):.2f}", help="Annualized trade Sharpe")
        k4.metric("Profit Factor", f"{risk.get('profit_factor',0):.2f}", help="Gross wins / gross losses")
        k5.metric("Max Drawdown", f"-{risk.get('max_drawdown_pct',0):.1f}%")

        st.divider()

        # ── Portfolio chart (reconstructed from cumulative P&L) ────────────────
        if trades_df is not None and not trades_df.empty:
            fig_port = build_pnl_timeline(trades_df)
            st.plotly_chart(fig_port, use_container_width=True, config={"displayModeBar": False})
        else:
            # Fallback: allocation pie only
            pass

        st.divider()

        # ── Position cards ─────────────────────────────────────────────────────
        st.markdown("#### Positions")
        if pos_df is not None and len(pos_df) > 0:
            df = pos_df if hasattr(pos_df, "iterrows") else pd.DataFrame(pos_df)
            pos_list = df.to_dict(orient="records")
            n = len(pos_list)
            cols_per_row = 3
            for row_start in range(0, n, cols_per_row):
                row_items = pos_list[row_start: row_start + cols_per_row]
                cols = st.columns(cols_per_row)
                for col, pos in zip(cols, row_items):
                    with col:
                        _position_card(pos)
                        if st.button(f"Research {pos['symbol']}", key=f"res_{pos['symbol']}", use_container_width=True):
                            st.session_state.research_ticker = pos["symbol"]
                            # Jump to research tab happens on next rerun
        else:
            st.caption("No positions loaded.")

        # ── Allocation pie ────────────────────────────────────────────────────
        if pos_df is not None and len(pos_df) > 0:
            st.divider()
            pie_col, _ = st.columns([1, 1])
            with pie_col:
                df2 = pos_df if hasattr(pos_df, "columns") else pd.DataFrame(pos_df)
                fig_pie = go.Figure(go.Pie(
                    labels=df2["symbol"], values=df2["market_value"],
                    hole=0.5,
                    marker=dict(
                        colors=px.colors.qualitative.Safe,
                        line=dict(color="#0a0e1a", width=2),
                    ),
                    textinfo="label+percent",
                    textposition="inside",
                ))
                fig_pie.update_layout(
                    **_base_layout(title="Portfolio Allocation", margin=dict(t=36, b=8, l=8, r=8)),
                    showlegend=False,
                )
                st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RESEARCH (Stock Picker + Trade Visualization)
# ══════════════════════════════════════════════════════════════════════════════
with tab_research:
    st.markdown("#### Research Any Stock")
    st.caption("Your historical trades appear as ▲ buy / ▼ sell markers on the chart.")

    # ── Search row ────────────────────────────────────────────────────────────
    s_col, p_col = st.columns([2, 3])
    with s_col:
        ticker_input = st.text_input(
            "Ticker symbol",
            value=st.session_state.research_ticker,
            placeholder="NVDA, AAPL, TSLA...",
            label_visibility="collapsed",
        ).upper().strip()
        if ticker_input:
            st.session_state.research_ticker = ticker_input

    with p_col:
        period_label = st.radio(
            "Period", list(PERIOD_MAP.keys()), index=4,
            horizontal=True, label_visibility="collapsed",
            key="research_period_radio",
        )

    symbol = st.session_state.research_ticker
    yf_period = PERIOD_MAP[period_label]

    if symbol:
        with st.spinner(f"Loading {symbol}..."):
            hist, info = fetch_chart(symbol, yf_period)
            techs = fetch_technicals(symbol)

        if hist.empty:
            st.error(f"No data found for **{symbol}**. Check the ticker and try again.")
        else:
            # ── Price header ──────────────────────────────────────────────────
            cur_price = float(hist["Close"].iloc[-1]) if "Close" in hist.columns else 0
            first_price = float(hist["Close"].iloc[0]) if "Close" in hist.columns else 0
            chg = cur_price - first_price
            chg_pct = (chg / first_price * 100) if first_price != 0 else 0
            is_pos = chg >= 0
            chg_color = GREEN if is_pos else RED
            name = info.get("longName") or info.get("shortName") or symbol

            display_name = name if len(name) <= 40 else name[:38] + "…"
            st.markdown(f"""
            <div style="margin-bottom:14px;min-width:0;">
              <div class="stock-name-header" title="{name}">{display_name}</div>
              <div style="display:flex;align-items:baseline;flex-wrap:wrap;gap:10px;">
                <span class="price-num">${cur_price:,.2f}</span>
                <span class="price-chg" style="color:{chg_color};">
                  {'+' if chg >= 0 else ''}{chg:,.2f} ({chg_pct:+.2f}%) {period_label}
                </span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Main chart ────────────────────────────────────────────────────
            trades_for_overlay = st.session_state.trades_df if st.session_state.data_loaded else None
            fig_stock = build_price_chart(hist, symbol, trades_for_overlay, show_volume=True)
            st.plotly_chart(fig_stock, use_container_width=True, config={"displayModeBar": False})

            # ── Technicals row ────────────────────────────────────────────────
            if techs:
                st.divider()
                st.markdown("##### Technical Snapshot")
                t1, t2, t3, t4, t5, t6 = st.columns(6)

                rsi_val = techs.get("rsi", 50)
                t1.metric("RSI 14", f"{rsi_val:.1f}", help="<30 oversold, >70 overbought")
                t2.metric("Trend", techs.get("trend", "—"))
                t3.metric("20d Mom.", f"{techs.get('momentum_20d', 0):+.1f}%", help="20-day price momentum")
                t4.metric("MACD", "Bull ✓" if techs.get("macd_bullish") else "Bear ✗", help="MACD histogram signal")
                t5.metric("Vol. 20d", f"{techs.get('volatility_20d', 0):.1f}%", help="20-day annualised volatility")
                t6.metric("vs SMA50", "Above ▲" if techs.get("above_sma50") else "Below ▼", help="Price vs 50-day moving average")

                # Signal badges
                signals = []
                rsi_v = techs.get("rsi", 50)
                if rsi_v < 35:   signals.append(("Oversold — potential entry", "bull"))
                elif rsi_v > 70: signals.append(("Overbought — caution", "bear"))
                if techs.get("macd_bullish"): signals.append(("MACD Bullish Cross", "bull"))
                if techs.get("above_sma50") and techs.get("above_sma200", True):
                    signals.append(("Above SMA50 & SMA200", "bull"))
                if techs.get("momentum_20d", 0) > 10: signals.append(("Strong momentum", "bull"))
                if techs.get("momentum_20d", 0) < -10: signals.append(("Weak momentum", "bear"))

                if signals:
                    badge_html = " ".join(f'<span class="badge-{cls}">{msg}</span>' for msg, cls in signals)
                    st.markdown(badge_html, unsafe_allow_html=True)

            # ── Your history with this stock ──────────────────────────────────
            if st.session_state.data_loaded and st.session_state.trades_df is not None:
                sym_trades = st.session_state.trades_df[
                    st.session_state.trades_df["symbol"] == symbol
                ].copy()
                if len(sym_trades) > 0:
                    st.divider()
                    st.markdown(f"##### Your {symbol} Trade History")
                    total_pnl = sym_trades["pnl_dollar"].sum()
                    wr = sym_trades["win"].mean()
                    h1, h2, h3 = st.columns(3)
                    h1.metric("Trades", len(sym_trades))
                    h2.metric("Win Rate", f"{wr*100:.0f}%")
                    h3.metric("Total P&L", fmt_dollar(total_pnl))

                    display = sym_trades[["buy_date","sell_date","hold_days","buy_price","sell_price","quantity","pnl_dollar","pnl_pct"]].copy()
                    display.columns = ["Buy Date","Sell Date","Hold (d)","Buy $","Sell $","Qty","P&L $","P&L %"]
                    display["Buy Date"] = pd.to_datetime(display["Buy Date"]).dt.strftime("%b %d '%y")
                    display["Sell Date"] = pd.to_datetime(display["Sell Date"]).dt.strftime("%b %d '%y")

                    def _pnl_style(val):
                        try:
                            return f"color: {GREEN}" if float(val) > 0 else f"color: {RED}"
                        except Exception:
                            return ""

                    styled = (
                        display.style
                        .applymap(_pnl_style, subset=["P&L $", "P&L %"])
                        .format({"Buy $": "${:,.2f}", "Sell $": "${:,.2f}", "P&L $": "${:,.0f}", "P&L %": "{:+.1f}%", "Qty": "{:.0f}"})
                    )
                    st.dataframe(styled, use_container_width=True, hide_index=True)

            # ── AI Quick Analysis ─────────────────────────────────────────────
            st.divider()
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
            if api_key:
                if st.button(f"🤖 AI Analysis of {symbol}", type="primary"):
                    stats = st.session_state.stats or {}
                    edge = stats.get("edge_profile", {})
                    risk = compute_risk_metrics(st.session_state.trades_df) if st.session_state.trades_df is not None else {}
                    sym_trades_for_ai = (
                        st.session_state.trades_df[st.session_state.trades_df["symbol"] == symbol]
                        if st.session_state.data_loaded and st.session_state.trades_df is not None
                        else pd.DataFrame()
                    )
                    history_note = ""
                    if len(sym_trades_for_ai) > 0:
                        history_note = (
                            f"User has traded {symbol} {len(sym_trades_for_ai)}x. "
                            f"Win rate: {sym_trades_for_ai['win'].mean()*100:.0f}%, "
                            f"Avg P&L: {sym_trades_for_ai['pnl_pct'].mean():+.1f}%, "
                            f"Total P&L: ${sym_trades_for_ai['pnl_dollar'].sum():,.0f}."
                        )
                    prompt = f"""Analyze {symbol} ({name}) as a trade opportunity for this specific user.

User edge profile:
- Best sector: {edge.get('best_sector', 'Not computed')}
- Best RSI entry band: {edge.get('best_rsi_band', 'Not computed')}
- Best hold duration: {edge.get('best_hold_duration', 'Not computed')}
- Overall win rate: {risk.get('win_rate', 0)*100:.0f}%
- Half-Kelly position size: {risk.get('half_kelly_pct', 0):.1f}% of portfolio
{history_note}

{symbol} current technicals:
- Price: ${cur_price:,.2f}
- RSI: {techs.get('rsi', 'N/A')}
- Trend: {techs.get('trend', 'N/A')}
- 20d Momentum: {techs.get('momentum_20d', 'N/A')}%
- MACD Bullish: {techs.get('macd_bullish', 'N/A')}
- Volatility: {techs.get('volatility_20d', 'N/A')}%

Give: (1) BUY / HOLD / AVOID verdict, (2) specific entry/exit levels or conditions, (3) how this fits or clashes with their edge. Max 4 sentences. Be direct."""
                    with st.spinner("Analyzing..."):
                        try:
                            client = anthropic.Anthropic(api_key=api_key)
                            resp = client.messages.create(
                                model=MODEL, max_tokens=512,
                                messages=[{"role": "user", "content": prompt}],
                            )
                            st.info(resp.content[0].text)
                        except Exception as exc:
                            st.error(f"Analysis failed: {exc}")
            else:
                st.caption("Add your Anthropic API Key in the sidebar for AI analysis.")

            # ── Fundamentals ──────────────────────────────────────────────────
            if info:
                with st.expander("Fundamentals"):
                    f1, f2, f3, f4 = st.columns(4)
                    f1.metric("Market Cap", f"${info.get('marketCap',0)/1e9:.1f}B" if info.get('marketCap') else "—")
                    f2.metric("Fwd P/E", f"{info.get('forwardPE',0):.1f}" if info.get('forwardPE') else "—")
                    f3.metric("Revenue Growth", f"{info.get('revenueGrowth',0)*100:.1f}%" if info.get('revenueGrowth') else "—")
                    f4.metric("Beta", f"{info.get('beta',0):.2f}" if info.get('beta') else "—")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MY TRADES
# ══════════════════════════════════════════════════════════════════════════════
with tab_trades:
    if not st.session_state.data_loaded or st.session_state.stats is None:
        st.info("Load data from the sidebar to see your trade analysis.")
    else:
        stats = st.session_state.stats
        trades_df = st.session_state.trades_df
        risk = compute_risk_metrics(trades_df)

        # ── KPI row ────────────────────────────────────────────────────────────
        wr = stats.get("win_rate", 0)
        t1, t2, t3, t4, t5, t6 = st.columns(6)
        t1.metric("Win Rate", f"{wr*100:.1f}%")
        t2.metric("Total Trades", stats.get("total_trades", 0))
        t3.metric("Profit Factor", f"{risk.get('profit_factor',0):.2f}")
        t4.metric("Avg P&L / Trade", f"{stats.get('avg_pnl_pct',0):+.1f}%")
        t5.metric("Expectancy", f"{risk.get('expectancy_pct',0):+.1f}%")
        t6.metric("Total P&L", fmt_dollar(stats.get("total_pnl", 0)))

        st.divider()

        # ── Half-Kelly insight ─────────────────────────────────────────────────
        kelly = risk.get("half_kelly_pct", 0)
        if kelly > 0:
            st.success(
                f"**Optimal Position Size (Half-Kelly):** Risk **{kelly:.1f}%** of your portfolio per trade. "
                f"Based on your {wr*100:.0f}% win rate and {risk.get('avg_win_pct',0):.1f}% avg win / "
                f"{risk.get('avg_loss_pct',0):.1f}% avg loss."
            )

        # ── Cumulative P&L chart ───────────────────────────────────────────────
        fig_pnl = build_pnl_timeline(trades_df)
        st.plotly_chart(fig_pnl, use_container_width=True, config={"displayModeBar": False})

        # ── Edge profile ───────────────────────────────────────────────────────
        edge = stats.get("edge_profile", {})
        if edge:
            st.divider()
            st.markdown("##### Your Statistical Edge")
            e1, e2, e3, e4 = st.columns(4)
            def _edge_box(col, label: str, value: str) -> None:
                short_val = value if len(value) <= 20 else value[:18] + "…"
                col.markdown(f"""
                <div style="background:#131929;border:1px solid rgba(255,255,255,0.08);
                            border-radius:8px;padding:12px 14px;overflow:hidden;">
                  <div style="font-size:0.68rem;color:#8892a4;font-weight:700;letter-spacing:0.06em;
                              text-transform:uppercase;margin-bottom:4px;">{label}</div>
                  <div style="font-size:0.9rem;color:#c8d0e0;font-weight:600;overflow:hidden;
                              text-overflow:ellipsis;white-space:nowrap;" title="{value}">{short_val}</div>
                </div>
                """, unsafe_allow_html=True)

            _edge_box(e1, "Best Sector",       str(edge.get("best_sector", "—")))
            _edge_box(e2, "Best Hold Duration", str(edge.get("best_hold_duration", "—")))
            _edge_box(e3, "Best RSI Entry",     str(edge.get("best_rsi_band", "—")))
            _edge_box(e4, "Best Market Cap",    str(edge.get("best_market_cap", "—")))

            pr = stats.get("patience_ratio", 1.0)
            if pr >= 1.2:
                st.success(f"You let winners run {pr:.1f}× longer than losers — excellent trade discipline.")
            elif pr < 0.8:
                st.warning(f"You're cutting winners short ({pr:.1f}× ratio). Consider holding winning trades longer.")

        # ── Breakdown charts (2×2 grid) ────────────────────────────────────────
        st.divider()
        st.markdown("##### Performance by Condition")
        row1_l, row1_r = st.columns(2)
        with row1_l:
            fig = build_breakdown_chart(stats.get("by_sector"), "sector", "Win Rate by Sector (%)")
            if fig: st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with row1_r:
            fig = build_breakdown_chart(stats.get("by_hold_bucket"), "hold_bucket", "Win Rate by Hold Duration (%)")
            if fig: st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        row2_l, row2_r = st.columns(2)
        with row2_l:
            fig = build_breakdown_chart(stats.get("by_rsi_band"), "rsi_band", "Win Rate by RSI Entry Band (%)")
            if fig: st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with row2_r:
            fig = build_breakdown_chart(stats.get("by_entry_regime"), "entry_regime", "Win Rate by Market Regime (%)")
            if fig: st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # ── Trade log ──────────────────────────────────────────────────────────
        st.divider()
        with st.expander("Full Trade Log"):
            if trades_df is not None and not trades_df.empty:
                log = trades_df[["symbol","sector","buy_date","sell_date","hold_days","buy_price","sell_price","pnl_dollar","pnl_pct","win"]].copy()
                log["buy_date"]  = pd.to_datetime(log["buy_date"]).dt.strftime("%b %d '%y")
                log["sell_date"] = pd.to_datetime(log["sell_date"]).dt.strftime("%b %d '%y")
                log["win"] = log["win"].map({True: "✓", False: "✗"})
                log.columns = ["Symbol","Sector","Buy","Sell","Days","Buy $","Sell $","P&L $","P&L %","W"]

                def _pnl_style2(val):
                    try:
                        return f"color: {GREEN}" if float(val) > 0 else f"color: {RED}"
                    except Exception:
                        return ""

                styled_log = (
                    log.style
                    .applymap(_pnl_style2, subset=["P&L $","P&L %"])
                    .format({"Buy $":"${:,.2f}","Sell $":"${:,.2f}","P&L $":"${:,.0f}","P&L %":"{:+.1f}%"})
                )
                st.dataframe(styled_log, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — OPPORTUNITIES
# ══════════════════════════════════════════════════════════════════════════════
with tab_opps:
    if not st.session_state.data_loaded:
        st.info("Load data first to see opportunities.")
    else:
        # Live screener button
        if not st.session_state.use_demo:
            if st.button("Run Live Market Screen (2-3 min)", type="primary"):
                with st.spinner("Screening 150+ stocks..."):
                    try:
                        from market.screener import build_candidate_universe, screen_candidates
                        edge_profile = st.session_state.stats.get("edge_profile", {}) if st.session_state.stats else {}
                        syms = build_candidate_universe()
                        raw = screen_candidates(syms, edge_profile)
                        st.session_state.candidates = raw if hasattr(raw, "columns") else pd.DataFrame(raw)
                        _compute_recs()
                        st.success(f"Found {len(st.session_state.candidates)} candidates.")
                    except Exception as exc:
                        st.error(f"Screen failed: {exc}")

        recs = st.session_state.recs
        risk = compute_risk_metrics(st.session_state.trades_df) if st.session_state.trades_df is not None else {}

        # ── Position sizing ────────────────────────────────────────────────────
        kelly = risk.get("half_kelly_pct", 0)
        port_equity = (st.session_state.portfolio or {}).get("equity", 50000)
        if kelly > 0:
            dollar_size = port_equity * kelly / 100
            st.info(
                f"**Sizing Guide (Half-Kelly):** Risk ${dollar_size:,.0f} per trade ({kelly:.1f}% of ${port_equity:,.0f} portfolio). "
                f"Based on {risk.get('win_rate',0)*100:.0f}% win rate and {risk.get('profit_factor',0):.2f}× profit factor."
            )

        # ── Recommendation cards ───────────────────────────────────────────────
        st.divider()
        st.markdown("#### Top Stock Picks")
        st.caption("Scored against your personal edge — the higher the score, the better the match to your winning trade patterns.")

        if recs is None or (hasattr(recs, "empty") and recs.empty):
            st.caption("No recommendations yet.")
        else:
            recs_df = recs if hasattr(recs, "iterrows") else pd.DataFrame(recs)
            n_recs = len(recs_df)
            cols_per_row = 3
            for row_start in range(0, n_recs, cols_per_row):
                row_recs = list(recs_df.iloc[row_start: row_start + cols_per_row].iterrows())
                cols = st.columns(cols_per_row)
                for col, (_, rec_row) in zip(cols, row_recs):
                    with col:
                        _rec_card(rec_row, row_start)
                        if st.button(f"Research {rec_row.get('symbol','')}", key=f"opp_res_{rec_row.get('symbol','')}_{row_start}", use_container_width=True):
                            st.session_state.research_ticker = rec_row.get("symbol", "")

        # ── Options plays ──────────────────────────────────────────────────────
        opts = st.session_state.opts_recs
        if opts is not None and not (hasattr(opts, "empty") and opts.empty) and len(opts) > 0:
            st.divider()
            st.markdown("#### Options Plays")
            opts_df = opts if hasattr(opts, "columns") else pd.DataFrame(opts)
            show_cols = [c for c in ["symbol","options_strategy","score","options_rationale"] if c in opts_df.columns]
            if show_cols:
                st.dataframe(opts_df[show_cols], use_container_width=True, hide_index=True)

        # ── Screener scatter ──────────────────────────────────────────────────
        cands = st.session_state.candidates
        if cands is not None and len(cands) > 0:
            st.divider()
            st.markdown("#### Market Map")
            df = cands.copy()
            if "rsi" in df.columns and "momentum_20d" in df.columns:
                trend_colors = {
                    "Strong Uptrend": GREEN, "Uptrend": "#66ddbb",
                    "Sideways": AMBER, "Downtrend": "#ff8888", "Strong Downtrend": RED,
                }
                fig_sc = go.Figure()
                for trend_val, group in df.groupby("trend") if "trend" in df.columns else [("All", df)]:
                    color = trend_colors.get(str(trend_val), BLUE)
                    fig_sc.add_trace(go.Scatter(
                        x=group["rsi"], y=group["momentum_20d"],
                        mode="markers+text",
                        text=group["symbol"],
                        textposition="top center",
                        textfont=dict(size=10),
                        marker=dict(
                            size=group["volatility_20d"] / 2 if "volatility_20d" in group.columns else 12,
                            color=color, opacity=0.85,
                            line=dict(color="rgba(255,255,255,0.2)", width=1),
                        ),
                        name=str(trend_val),
                        hovertemplate="<b>%{text}</b><br>RSI: %{x:.0f}<br>20d Mom: %{y:.1f}%<extra></extra>",
                    ))
                fig_sc.add_vline(x=50, line_dash="dot", line_color="rgba(255,255,255,0.2)")
                fig_sc.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)")
                fig_sc.add_annotation(x=35, y=fig_sc.layout.yaxis.range[1] if fig_sc.layout.yaxis.range else 15,
                                      text="Potential entries", showarrow=False, font=dict(color="#8892a4", size=10))
                fig_sc.update_xaxes(title_text="RSI", **_axis_style(show_grid=False))
                fig_sc.update_yaxes(title_text="20-Day Momentum (%)", **_axis_style(show_grid=True))
                fig_sc.update_layout(
                    **_base_layout(
                        title="RSI vs Momentum  (bubble size = volatility, color = trend)",
                        margin=dict(t=40, b=28, l=8, r=8),
                    ),
                )
                st.plotly_chart(fig_sc, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — QUANT LAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_lab:
    st.markdown("#### ⚡ Quant Lab — State-of-the-Art Stock Analysis")
    st.caption(
        "Monte Carlo simulation (3,000 paths) · Hurst exponent · GBM + Jump Diffusion · "
        "Support/Resistance clustering · Kelly sizing · VaR/CVaR · AI synthesis"
    )

    # ── Inputs ────────────────────────────────────────────────────────────────
    col_sym, col_inv, col_hor, col_btn = st.columns([2, 2, 2, 1])
    with col_sym:
        lab_ticker = st.text_input(
            "Stock", value=st.session_state.get("lab_ticker", "NVDA"),
            placeholder="NVDA, AAPL, TSLA...", key="lab_ticker_input",
        ).upper().strip()
    with col_inv:
        lab_invest = st.number_input(
            "Investment ($)", min_value=100, max_value=10_000_000,
            value=st.session_state.get("lab_invest", 10000), step=500, key="lab_invest_input",
        )
    with col_hor:
        horizon_label = st.selectbox(
            "Horizon", ["1 Month", "3 Months", "6 Months", "1 Year", "2 Years"],
            index=3, key="lab_horizon_sel",
        )
        horizon_map = {"1 Month": 21, "3 Months": 63, "6 Months": 126, "1 Year": 252, "2 Years": 504}
        lab_horizon = horizon_map[horizon_label]
    with col_btn:
        st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
        run_lab = st.button("🚀 Analyze", type="primary", use_container_width=True)

    # Store inputs in session state
    if lab_ticker:
        st.session_state.lab_ticker = lab_ticker
    st.session_state.lab_invest = lab_invest

    if run_lab and lab_ticker:
        st.session_state.lab_results = None
        st.session_state.lab_hist = None
        st.session_state.lab_info = None

        with st.spinner(f"Running full quantitative analysis on {lab_ticker}..."):
            # Fetch 2 years for better drift estimation (CAPM shrinkage uses data length)
            hist_lab, info_lab = fetch_chart(lab_ticker, "2y")
            if hist_lab.empty:
                hist_lab, info_lab = fetch_chart(lab_ticker, "1y")  # fallback
            if hist_lab.empty:
                st.error(f"No data for {lab_ticker}. Check the ticker symbol.")
            else:
                from analytics.quant import run_full_analysis
                stock_beta = float(info_lab.get("beta") or 1.0)
                results = run_full_analysis(
                    hist=hist_lab,
                    investment=float(lab_invest),
                    horizon_days=lab_horizon,
                    n_paths=3000,
                    beta=stock_beta,
                )
                if results:
                    st.session_state.lab_results = results
                    st.session_state.lab_hist = hist_lab
                    st.session_state.lab_info = info_lab
                    st.session_state.lab_ai_analysis = None  # reset AI
                else:
                    st.error("Analysis failed — insufficient price history.")

    # ── Results ───────────────────────────────────────────────────────────────
    res = st.session_state.get("lab_results")
    hist_lab = st.session_state.get("lab_hist")
    info_lab = st.session_state.get("lab_info", {}) or {}

    if res is None:
        st.info("Enter a ticker and click **Analyze** to run the full quantitative model.")
        with st.expander("What does this analyze?"):
            st.markdown("""
| Model | What it tells you |
|---|---|
| **Geometric Brownian Motion + Jump Diffusion** | Simulates 3,000 possible futures for the stock price |
| **Hurst Exponent (R/S Analysis)** | Is this stock trending, mean-reverting, or random? |
| **Adaptive Volatility** | Blends short-term and long-term volatility for accuracy |
| **Monte Carlo Percentile Cone** | Shows the 80% confidence range of where price could go |
| **Support & Resistance** | Key price levels where the stock historically bounces |
| **Kelly Criterion** | Optimal position size based on your edge |
| **VaR / CVaR** | How much can you lose in a bad day (95% confidence) |
| **Entry Score** | RSI + MACD + Bollinger + trend fusion (0–100) |
| **AI Synthesis** | Claude reads all the data and gives you a clear verdict |
            """)
    else:
        sym = st.session_state.get("lab_ticker", "")
        name = info_lab.get("longName") or info_lab.get("shortName") or sym
        cur = res["current_price"]

        # ── Header ────────────────────────────────────────────────────────────
        display_lab_name = name if len(name) <= 36 else name[:34] + "…"
        st.markdown(f"""
        <div class="lab-header">
          <span class="lab-sym">{sym}</span>
          <span class="lab-name" title="{name}">{display_lab_name}</span>
          <span class="lab-price">${cur:,.2f}</span>
        </div>
        """, unsafe_allow_html=True)

        # ── KPI rows — split 4+3 to prevent cramping ──────────────────────────
        hurst_val = res["hurst"]
        if hurst_val > 0.58:
            hurst_label = f"Trending ({hurst_val:.2f})"
        elif hurst_val < 0.42:
            hurst_label = f"Mean-Rev ({hurst_val:.2f})"
        else:
            hurst_label = f"Random ({hurst_val:.2f})"

        entry_val = res["entry_score"]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Hurst", hurst_label, help=f"Fractal regime: {res['regime_note']}")
        k2.metric("Sharpe", f"{res['sharpe']:.2f}", help="Risk-adjusted return vs 5% risk-free rate")
        k3.metric("Sortino", f"{res['sortino']:.2f}", help="Downside-only risk ratio")
        k4.metric("Ann. Volatility", f"{res['sigma_annual_pct']:.0f}%", help="Annualised price volatility")

        k5, k6, k7, k8 = st.columns(4)
        k5.metric("VaR 95%", f"-{res['var_95_pct']:.1f}%", help="Max expected daily loss, 19/20 days")
        k6.metric("Entry Score", f"{entry_val:.0f} / 100", help="RSI + MACD + Bollinger + Trend fusion")
        k7.metric("P(Profit)", f"{res['prob_profit']:.0f}%", help=f"Probability above current price in {horizon_label}")
        k8.metric("Median Return", f"{res['median_return_pct']:+.1f}%", help="P50 of 3,000 simulations — half of outcomes above this, half below")

        # ── Model assumptions disclosure ──────────────────────────────────────
        n_days_used = len(hist_lab.dropna())
        n_yrs = n_days_used / 252
        hist_w = min(0.60, n_yrs * 0.20)
        capm_w = 1 - hist_w
        st.markdown(f"""
        <div style="background:#0d1117;border:1px solid rgba(255,255,255,0.07);border-radius:8px;
                    padding:10px 14px;margin-bottom:12px;display:flex;gap:20px;flex-wrap:wrap;
                    font-size:0.72rem;color:#8892a4;">
          <span>📐 <b style="color:#c8d0e0;">Drift model:</b>
            {capm_w*100:.0f}% CAPM ({res['mu_capm_pct']:+.1f}%) +
            {hist_w*100:.0f}% historical ({res['mu_historical_pct']:+.1f}%) =
            <b style="color:#00d4aa;">{res['mu_adjusted_pct']:+.1f}% adj. drift</b></span>
          <span>📊 <b style="color:#c8d0e0;">Data:</b> {n_days_used} trading days ({n_yrs:.1f} yrs)</span>
          <span>🎲 <b style="color:#c8d0e0;">Simulations:</b> 3,000 GBM + Jump Diffusion paths</span>
          <span>📉 <b style="color:#c8d0e0;">Returns shown:</b> Median (P50) — half of simulations finish above, half below</span>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── Prediction chart ───────────────────────────────────────────────────
        st.markdown(f"##### Price Forecast — {lab_ticker} next {horizon_label}")
        st.caption("Historical price + Monte Carlo prediction cone (80% confidence interval). Green ▲ / Red ▼ = your historical trades.")

        # Build prediction chart
        import pandas.tseries.offsets as offsets
        close_col = "Close" if "Close" in hist_lab.columns else hist_lab.columns[3]
        close_hist = hist_lab[close_col].squeeze()

        last_date = hist_lab.index[-1]
        future_dates = pd.bdate_range(start=last_date, periods=lab_horizon + 1)

        pp = res["percentiles"]
        fig_pred = go.Figure()

        # Historical line
        fig_pred.add_trace(go.Scatter(
            x=hist_lab.index, y=close_hist,
            line=dict(color="#c8d0e0", width=2),
            mode="lines", name="Historical Price",
            hovertemplate="<b>%{x|%b %d '%y}</b>  $%{y:,.2f}<extra></extra>",
        ))

        # 10-90 band
        fig_pred.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates[::-1]),
            y=list(pp[90]) + list(pp[10][::-1]),
            fill="toself", fillcolor="rgba(0,212,170,0.07)",
            line=dict(color="rgba(0,0,0,0)"), name="80% Range", hoverinfo="skip",
        ))

        # 25-75 band
        fig_pred.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates[::-1]),
            y=list(pp[75]) + list(pp[25][::-1]),
            fill="toself", fillcolor="rgba(0,212,170,0.15)",
            line=dict(color="rgba(0,0,0,0)"), name="50% Range", hoverinfo="skip",
        ))

        # Median forecast
        fig_pred.add_trace(go.Scatter(
            x=future_dates, y=pp[50],
            line=dict(color=GREEN, width=2.5, dash="dash"),
            name="Median Forecast",
            hovertemplate="<b>Forecast %{x|%b %d '%y}</b>  $%{y:,.2f}<extra></extra>",
        ))

        # Bull & Bear lines
        fig_pred.add_trace(go.Scatter(
            x=future_dates, y=pp[90],
            line=dict(color="#66ddbb", width=1, dash="dot"),
            name="Bull (90th %ile)",
            hovertemplate="Bull: $%{y:,.2f}<extra></extra>",
        ))
        fig_pred.add_trace(go.Scatter(
            x=future_dates, y=pp[10],
            line=dict(color=RED, width=1, dash="dot"),
            name="Bear (10th %ile)",
            hovertemplate="Bear: $%{y:,.2f}<extra></extra>",
        ))

        # Your trades on this stock as overlays
        if st.session_state.data_loaded and st.session_state.trades_df is not None:
            sym_trades = st.session_state.trades_df[st.session_state.trades_df["symbol"] == sym].copy()
            if len(sym_trades) > 0:
                buys = sym_trades[pd.to_datetime(sym_trades["buy_date"]) >= hist_lab.index.min()]
                sells = sym_trades[pd.to_datetime(sym_trades["sell_date"]) <= hist_lab.index.max()]
                if len(buys) > 0:
                    fig_pred.add_trace(go.Scatter(
                        x=pd.to_datetime(buys["buy_date"]), y=buys["buy_price"],
                        mode="markers",
                        marker=dict(symbol="triangle-up", size=14, color=GREEN, line=dict(color="white", width=1.5)),
                        name="Your Buy",
                        hovertemplate="<b>BUY</b> $%{y:,.2f}<extra></extra>",
                    ))
                if len(sells) > 0:
                    fig_pred.add_trace(go.Scatter(
                        x=pd.to_datetime(sells["sell_date"]), y=sells["sell_price"],
                        mode="markers",
                        marker=dict(symbol="triangle-down", size=14, color=RED, line=dict(color="white", width=1.5)),
                        name="Your Sell",
                        hovertemplate="<b>SELL</b> $%{y:,.2f}  P&L: %{customdata:+.1f}%<extra></extra>",
                        customdata=sells["pnl_pct"].values,
                    ))

        # Support / Resistance levels
        for lvl in res.get("support", []):
            fig_pred.add_hline(
                y=lvl, line_color=GREEN, line_dash="dot", line_width=1, opacity=0.6,
                annotation_text=f"  S ${lvl:,.0f}",
                annotation_position="top left",
                annotation_font=dict(size=10, color=GREEN),
            )
        for lvl in res.get("resistance", []):
            fig_pred.add_hline(
                y=lvl, line_color=RED, line_dash="dot", line_width=1, opacity=0.6,
                annotation_text=f"  R ${lvl:,.0f}",
                annotation_position="top left",
                annotation_font=dict(size=10, color=RED),
            )

        # Today marker
        fig_pred.add_vline(
            x=last_date.timestamp() * 1000,
            line_color="rgba(255,255,255,0.25)", line_dash="dash",
            annotation_text=" Now", annotation_position="top",
            annotation_font=dict(size=11, color="#8892a4"),
        )

        # End-of-forecast annotations
        sc = res["scenarios"]
        fig_pred.add_annotation(
            x=future_dates[-1], y=pp[90][-1],
            text=f"Bull ${sc['bull_price']:,.0f}", showarrow=False,
            font=dict(size=10, color=GREEN), xanchor="left",
        )
        fig_pred.add_annotation(
            x=future_dates[-1], y=pp[50][-1],
            text=f"Base ${sc['base_price']:,.0f}", showarrow=False,
            font=dict(size=10, color="#c8d0e0"), xanchor="left",
        )
        fig_pred.add_annotation(
            x=future_dates[-1], y=pp[10][-1],
            text=f"Bear ${sc['bear_price']:,.0f}", showarrow=False,
            font=dict(size=10, color=RED), xanchor="left",
        )

        fig_pred.update_xaxes(**_axis_style(show_grid=False))
        fig_pred.update_yaxes(**_axis_style(show_grid=True), tickprefix="$")
        fig_pred.update_layout(**_base_layout(margin=dict(t=16, b=8, l=8, r=80)))
        st.plotly_chart(fig_pred, use_container_width=True, config={"displayModeBar": False})

        st.divider()

        # ── Investment projection + Distribution ───────────────────────────────
        left_col, right_col = st.columns([3, 2])

        with left_col:
            st.markdown(f"##### ${lab_invest:,.0f} Investment — {horizon_label} Projection")
            sc = res["scenarios"]
            shares = res["shares"]

            # Investment value paths from percentiles
            bull_vals = pp[90] * shares
            base_vals = pp[50] * shares
            bear_vals = pp[10] * shares

            fig_inv = go.Figure()

            # Confidence band
            fig_inv.add_trace(go.Scatter(
                x=list(future_dates) + list(future_dates[::-1]),
                y=list(bull_vals) + list(bear_vals[::-1]),
                fill="toself", fillcolor="rgba(0,212,170,0.1)",
                line=dict(color="rgba(0,0,0,0)"), name="80% Range", hoverinfo="skip",
            ))

            fig_inv.add_trace(go.Scatter(
                x=future_dates, y=base_vals,
                line=dict(color=GREEN, width=2.5),
                name="Expected",
                hovertemplate="Expected: $%{y:,.0f}<extra></extra>",
            ))
            fig_inv.add_trace(go.Scatter(
                x=future_dates, y=bull_vals,
                line=dict(color="#66ddbb", width=1.2, dash="dot"),
                name="Bull",
                hovertemplate="Bull: $%{y:,.0f}<extra></extra>",
            ))
            fig_inv.add_trace(go.Scatter(
                x=future_dates, y=bear_vals,
                line=dict(color=RED, width=1.2, dash="dot"),
                name="Bear",
                hovertemplate="Bear: $%{y:,.0f}<extra></extra>",
            ))

            # Starting value line
            fig_inv.add_hline(
                y=float(lab_invest),
                line_color="rgba(255,255,255,0.2)", line_dash="dash",
                annotation_text=f"  Invested ${lab_invest:,.0f}",
                annotation_font=dict(size=10, color="#8892a4"),
            )

            fig_inv.update_xaxes(**_axis_style(show_grid=False))
            fig_inv.update_yaxes(**_axis_style(show_grid=True), tickprefix="$")
            fig_inv.update_layout(**_base_layout(margin=dict(t=16, b=8, l=8, r=8)))
            st.plotly_chart(fig_inv, use_container_width=True, config={"displayModeBar": False})

            # Scenario table
            bear_col, base_col_ui, bull_col = st.columns(3)
            bear_delta = sc["bear_value"] - lab_invest
            base_delta = sc["base_value"] - lab_invest
            bull_delta = sc["bull_value"] - lab_invest
            bear_col.metric("Bear (10th %ile)", f"${sc['bear_value']:,.0f}", f"{bear_delta:+,.0f} ({sc['bear_return']:+.1f}%)")
            base_col_ui.metric("Expected (50th)", f"${sc['base_value']:,.0f}", f"{base_delta:+,.0f} ({sc['base_return']:+.1f}%)")
            bull_col.metric("Bull (90th %ile)", f"${sc['bull_value']:,.0f}", f"{bull_delta:+,.0f} ({sc['bull_return']:+.1f}%)")

        with right_col:
            st.markdown("##### Return Distribution (3,000 Simulations)")
            final_ret_pct = res["final_returns"] * 100

            fig_hist = go.Figure()
            # Two traces for proper green/red coloring (Plotly histogram doesn't support per-point colors)
            pos_mask = final_ret_pct > 0
            fig_hist.add_trace(go.Histogram(
                x=final_ret_pct[pos_mask],
                nbinsx=40,
                marker=dict(color=GREEN, line=dict(width=0)),
                opacity=0.85,
                name="Gain",
                hovertemplate="Return: %{x:.1f}%<br>Paths: %{y}<extra></extra>",
            ))
            fig_hist.add_trace(go.Histogram(
                x=final_ret_pct[~pos_mask],
                nbinsx=20,
                marker=dict(color=RED, line=dict(width=0)),
                opacity=0.85,
                name="Loss",
                hovertemplate="Return: %{x:.1f}%<br>Paths: %{y}<extra></extra>",
            ))
            fig_hist.update_layout(barmode="overlay")

            # Percentile markers
            for pct, label, color in [(10, "P10", RED), (50, "P50", "#fff"), (90, "P90", GREEN)]:
                val = float(np.percentile(final_ret_pct, pct))
                fig_hist.add_vline(
                    x=val, line_color=color, line_dash="dash", line_width=1.5,
                    annotation_text=f" {label}: {val:+.1f}%",
                    annotation_position="top",
                    annotation_font=dict(size=9, color=color),
                )

            fig_hist.add_vline(x=0, line_color="rgba(255,255,255,0.3)", line_width=1)
            fig_hist.update_xaxes(title_text="Return (%)", **_axis_style(show_grid=False))
            fig_hist.update_yaxes(title_text="Paths", **_axis_style(show_grid=True))
            fig_hist.update_layout(**_base_layout(margin=dict(t=16, b=8, l=8, r=8)))
            st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

            # Probability breakdown — using tightly-controlled CSS classes
            st.markdown(f"""
            <div class="prob-panel">
              <div style="font-size:0.68rem;color:#8892a4;margin-bottom:10px;font-weight:700;letter-spacing:0.07em;text-transform:uppercase;">Probability Breakdown</div>
              <div class="prob-row">
                <span class="prob-label">Any profit</span>
                <span class="prob-val" style="color:{GREEN};">{res['prob_profit']:.0f}%</span>
              </div>
              <div class="prob-row">
                <span class="prob-label">+10% or more</span>
                <span class="prob-val" style="color:{GREEN};">{res['prob_10pct']:.0f}%</span>
              </div>
              <div class="prob-row">
                <span class="prob-label">+20% or more</span>
                <span class="prob-val" style="color:{GREEN};">{res['prob_20pct']:.0f}%</span>
              </div>
              <div class="prob-row" style="margin-bottom:0;">
                <span class="prob-label">Loss &gt;20%</span>
                <span class="prob-val" style="color:{RED};">{res['prob_loss_20']:.0f}%</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # ── AI Synthesis ───────────────────────────────────────────────────────
        st.markdown("##### 🤖 AI Investment Thesis")
        api_key = os.getenv("ANTHROPIC_API_KEY", "")

        if not api_key:
            st.warning("Add your Anthropic API Key in the sidebar to unlock the AI Synthesis.")
        else:
            if st.session_state.get("lab_ai_analysis") is None:
                run_ai = st.button("Generate AI Thesis", type="primary", key="lab_ai_btn")
            else:
                run_ai = False

            if run_ai or st.session_state.get("lab_ai_analysis") is None and st.session_state.get("lab_auto_ai"):
                sc = res["scenarios"]
                prompt = f"""You are a world-class quantitative analyst. Produce a concise, direct investment thesis for {sym} ({name}).

QUANTITATIVE DATA:
• Current Price: ${cur:,.2f}
• Horizon: {horizon_label}
• Investment: ${lab_invest:,.0f}

PHYSICS & STATS:
• Hurst Exponent: {res['hurst']:.3f} → {res['regime']} market ({res['regime_note']})
• Annual Volatility: {res['sigma_annual_pct']:.1f}%
• Historical Annual Return: {res['mu_annual_pct']:+.1f}%
• Sharpe Ratio: {res['sharpe']:.2f}
• Sortino Ratio: {res['sortino']:.2f}
• VaR 95% (daily): -{res['var_95_pct']:.2f}%

TECHNICAL SIGNALS:
• RSI: {res['rsi']:.1f} {'(oversold)' if res['rsi'] < 35 else '(overbought)' if res['rsi'] > 70 else '(neutral)'}
• MACD: {'Bullish ✓' if res['macd_bullish'] else 'Bearish ✗'}
• Trend: {res['trend']}
• 20-day Momentum: {res['momentum_20d']:+.1f}%
• Entry Score: {res['entry_score']:.0f}/100

MONTE CARLO (3,000 paths, {horizon_label}):
• P(any profit): {res['prob_profit']:.0f}%
• P(+10% gain): {res['prob_10pct']:.0f}%
• P(+20% gain): {res['prob_20pct']:.0f}%
• P(loss >20%): {res['prob_loss_20']:.0f}%
• Expected return: {res['expected_return_pct']:+.1f}%
• Bear (10th %ile): ${sc['bear_value']:,.0f} ({sc['bear_return']:+.1f}%)
• Expected (50th): ${sc['base_value']:,.0f} ({sc['base_return']:+.1f}%)
• Bull (90th %ile): ${sc['bull_value']:,.0f} ({sc['bull_return']:+.1f}%)

KEY LEVELS:
• Support: {', '.join(f'${s:,.0f}' for s in res.get('support',[])[:3]) or 'N/A'}
• Resistance: {', '.join(f'${r:,.0f}' for r in res.get('resistance',[])[:3]) or 'N/A'}

Provide exactly 5 sections:
1. **VERDICT** — BUY / HOLD / AVOID + conviction (High/Medium/Low) + one sentence why
2. **THE EDGE** — What the quantitative data reveals that most investors miss
3. **ENTRY STRATEGY** — Specific price level or condition to enter; how many shares for ${lab_invest:,.0f}
4. **RISK MANAGEMENT** — Stop loss level, max acceptable loss, when to exit
5. **{horizon_label.upper()} TARGET** — Specific dollar outcome for ${lab_invest:,.0f} across bear/base/bull cases

Be specific, use numbers, be direct. No disclaimers. Max 250 words total."""

                with st.spinner("Generating AI thesis..."):
                    try:
                        client = anthropic.Anthropic(api_key=api_key)
                        ai_resp = client.messages.create(
                            model=MODEL, max_tokens=600,
                            messages=[{"role": "user", "content": prompt}],
                        )
                        st.session_state.lab_ai_analysis = ai_resp.content[0].text
                    except Exception as exc:
                        st.session_state.lab_ai_analysis = f"AI analysis error: {exc}"

            if st.session_state.get("lab_ai_analysis"):
                verdict_text = st.session_state.lab_ai_analysis
                if "BUY" in verdict_text[:200]:
                    box_color = "rgba(0,212,170,0.1)"
                    border_color = "rgba(0,212,170,0.3)"
                elif "AVOID" in verdict_text[:200]:
                    box_color = "rgba(255,85,102,0.1)"
                    border_color = "rgba(255,85,102,0.3)"
                else:
                    box_color = "rgba(255,170,0,0.08)"
                    border_color = "rgba(255,170,0,0.25)"

                # thesis-box class provides max-height + scrollbar + word-break
                import html as _html
                safe_text = _html.escape(verdict_text).replace("\n", "<br>")
                st.markdown(f"""
                <div class="thesis-box" style="background:{box_color};border:1px solid {border_color};">
{safe_text}
                </div>
                """, unsafe_allow_html=True)

                if st.button("↺ Regenerate", key="lab_regen"):
                    st.session_state.lab_ai_analysis = None
                    st.session_state.lab_auto_ai = True
                    st.rerun()
            else:
                st.markdown("""
                <div style="background:#131929;border:1px dashed rgba(255,255,255,0.15);
                            border-radius:12px;padding:18px;text-align:center;color:#8892a4;font-size:0.88rem;">
                  Click <strong style="color:#00d4aa;">Generate AI Thesis</strong> for a structured investment analysis
                </div>
                """, unsafe_allow_html=True)


# TAB 6 — AI ADVISOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_ai:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")

    if not api_key:
        st.markdown("#### AI Advisor — Powered by Claude Opus")
        st.warning("Add your **Anthropic API Key** in the sidebar to unlock the AI Advisor.")
        with st.expander("What can the AI Advisor do?"):
            st.markdown("""
            - Analyze your complete trade history and identify your statistical edge
            - Find stock opportunities tailored to YOUR winning patterns
            - Explain why specific positions are up or down
            - Give position sizing recommendations based on Kelly criterion
            - Answer any question about your portfolio in plain English
            """)
    else:
        st.markdown("#### AI Advisor")
        st.caption("Ask anything. The AI knows your portfolio and trade history, and will pull live data automatically.")

        # ── Suggested questions ────────────────────────────────────────────────
        if not st.session_state.chat_messages:
            st.markdown("**Try asking:**")
            cols = st.columns(3)
            for i, q in enumerate(SUGGESTED_QUESTIONS):
                with cols[i % 3]:
                    if st.button(q, key=f"suggest_{i}", use_container_width=True):
                        st.session_state._pending_question = q
                        st.rerun()

        # Handle suggested question click
        if hasattr(st.session_state, "_pending_question") and st.session_state._pending_question:
            pending = st.session_state._pending_question
            st.session_state._pending_question = None
            st.session_state.chat_messages.append({"role": "user", "content": pending})
            with st.spinner("Thinking..."):
                try:
                    client = anthropic.Anthropic(api_key=api_key)
                    resp_text, tools_used = run_agent_turn(client, pending)
                except Exception as exc:
                    resp_text, tools_used = f"Error: {exc}", []
            st.session_state.chat_messages.append({"role": "assistant", "content": resp_text, "tools_used": tools_used})

        # ── Message history ────────────────────────────────────────────────────
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("tools_used"):
                    st.caption(f"Data sources: {', '.join(msg['tools_used'])}")

        # ── Chat input ─────────────────────────────────────────────────────────
        if prompt := st.chat_input("Ask about your portfolio, trades, or market opportunities..."):
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.chat_messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                with st.status("Working...", expanded=False) as status_box:
                    try:
                        client = anthropic.Anthropic(api_key=api_key)
                        resp_text, tools_used = run_agent_turn(client, prompt)
                        label = f"Used: {', '.join(tools_used)}" if tools_used else "Done"
                        status_box.update(label=label, state="complete")
                    except Exception as exc:
                        resp_text, tools_used = f"Something went wrong: {exc}", []
                        status_box.update(label="Error", state="error")
                st.markdown(resp_text)
                if tools_used:
                    st.caption(f"Data sources: {', '.join(tools_used)}")

            st.session_state.chat_messages.append({
                "role": "assistant", "content": resp_text, "tools_used": tools_used,
            })

        if st.session_state.chat_messages:
            if st.button("Clear Chat", key="clear_chat_btn"):
                st.session_state.chat_messages = []
                st.session_state.api_messages = []
                st.rerun()
