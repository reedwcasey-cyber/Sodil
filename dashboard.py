"""
SODIL Dashboard — Systematic Opportunity Discovery & Investment Lab
Run with: streamlit run dashboard.py
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SODIL — Investment Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; }
[data-testid="stMetricLabel"] { font-size: 0.75rem; color: #888; }
.stTabs [data-baseweb="tab"] { font-size: 0.9rem; font-weight: 600; padding: 8px 20px; }
div[data-testid="metric-container"] {
    background: #1a1a2e; border-radius: 8px; padding: 12px 16px; border: 1px solid #2d2d4e;
}
.block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def pct(val, decimals=1):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val*100:+.{decimals}f}%"


def fmt_usd(val):
    if val is None:
        return "N/A"
    if abs(val) >= 1e9:
        return f"${val/1e9:.2f}B"
    if abs(val) >= 1e6:
        return f"${val/1e6:.1f}M"
    return f"${val:,.2f}"


def conviction_color(c):
    return {"HIGH": "#00e676", "MEDIUM": "#ffb300", "LOW": "#ef5350"}.get(c, "#aaa")


def risk_color(r):
    return {"LOW": "#00e676", "MEDIUM": "#ffb300", "HIGH": "#ef5350", "VERY HIGH": "#b71c1c"}.get(r, "#aaa")


def score_color(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "#888"
    return "#00e676" if v > 0.3 else "#ef5350" if v < -0.3 else "#ffb300"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 SODIL")
    st.markdown("*Systematic Opportunity Discovery & Investment Lab*")
    st.divider()
    page = st.radio(
        "Navigate",
        ["🔭 Market Scanner", "🔬 Deep Analyzer", "💼 Portfolio", "🧪 Test Suite"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown(
        "<small style='color:#666'>Data via Yahoo Finance · Paper trading only · Not financial advice</small>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: MARKET SCANNER
# ─────────────────────────────────────────────────────────────────────────────
if page == "🔭 Market Scanner":
    st.title("🔭 Market Scanner")
    st.markdown("Multi-factor quantitative screen across the S&P 500. Ranks stocks on momentum, value, quality, and technical signals.")

    col1, col2, col3 = st.columns(3)
    with col1:
        top_n = st.number_input("Top N opportunities", 5, 30, 10)
    with col2:
        min_cap = st.selectbox("Min market cap", ["$2B", "$5B", "$10B", "$50B"], index=0)
        cap_map = {"$2B": 2e9, "$5B": 5e9, "$10B": 10e9, "$50B": 50e9}
        min_cap_val = cap_map[min_cap]
    with col3:
        use_custom = st.checkbox("Custom tickers")

    custom_tickers = None
    if use_custom:
        raw = st.text_input("Tickers (comma-separated)", "AAPL,MSFT,NVDA,GOOGL,META,AMZN,TSLA,JPM,V,UNH")
        custom_tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]

    run_scan = st.button("🚀 Run Scan", type="primary", use_container_width=True)

    if run_scan or "scan_results" in st.session_state:
        if run_scan:
            with st.spinner("Fetching universe and scoring stocks…"):
                from src.data.universe import build_universe, enrich_with_prices
                from src.data.fetcher import get_sp500_tickers, fetch_price_history
                from src.signals.composite import compute_composite_score, rank_opportunities

                ticker_list = custom_tickers or get_sp500_tickers()[:60]
                univ_df = build_universe(ticker_list, min_market_cap=min_cap_val, verbose=False)

                if univ_df.empty:
                    st.error("No stocks passed filters.")
                    st.stop()

                price_data = enrich_with_prices(univ_df, period="2y")

                all_scores = []
                prog = st.progress(0, "Scoring stocks…")
                tickers = univ_df.index.tolist()
                for i, ticker in enumerate(tickers):
                    fund = univ_df.loc[ticker].to_dict()
                    fund["ticker"] = ticker
                    prices = price_data.get(ticker, pd.DataFrame())
                    score = compute_composite_score(ticker, fund, prices)
                    all_scores.append(score)
                    prog.progress((i + 1) / len(tickers), f"Scored {ticker} ({i+1}/{len(tickers)})")
                prog.empty()

                ranked = rank_opportunities(all_scores, top_n=int(top_n))
                fund_map = {t: univ_df.loc[t].to_dict() for t in ranked.index if t in univ_df.index}

                st.session_state["scan_results"] = ranked
                st.session_state["scan_fundamentals"] = fund_map
                st.session_state["scan_prices"] = price_data
                st.session_state["scan_universe"] = univ_df

        ranked = st.session_state["scan_results"]
        fund_map = st.session_state["scan_fundamentals"]
        price_data = st.session_state.get("scan_prices", {})

        # ── Summary metrics ──────────────────────────────────────────────
        st.subheader("Top Opportunities")
        m1, m2, m3, m4 = st.columns(4)
        high = sum(1 for _, r in ranked.iterrows() if r.get("composite_score", 0) > 0.5)
        med = sum(1 for _, r in ranked.iterrows() if 0 < r.get("composite_score", 0) <= 0.5)
        avg_score = ranked["composite_score"].mean() if "composite_score" in ranked else 0
        m1.metric("Stocks Surfaced", len(ranked))
        m2.metric("Strong Signal (>0.5)", high)
        m3.metric("Moderate Signal", med)
        m4.metric("Avg Composite Score", f"{avg_score:.3f}")

        # ── Factor heatmap table ─────────────────────────────────────────
        display_rows = []
        for ticker, row in ranked.iterrows():
            info = fund_map.get(ticker, {})
            display_rows.append({
                "Rank": int(row.get("rank", 0)),
                "Ticker": ticker,
                "Name": (info.get("name", ticker) or ticker)[:22],
                "Sector": (info.get("sector", "?") or "?")[:14],
                "Composite": round(row.get("composite_score", 0), 3),
                "Momentum Z": round(row.get("momentum_z", 0) or 0, 2),
                "Value Z": round(row.get("value_z", 0) or 0, 2),
                "Quality Z": round(row.get("quality_z", 0) or 0, 2),
                "RSI": round(row.get("rsi", 50) or 50, 1),
                "Analyst ↑": pct(row.get("analyst_upside")),
                "Mkt Cap": fmt_usd(info.get("market_cap")),
            })

        df_display = pd.DataFrame(display_rows).set_index("Rank")

        # Color-code composite column
        def color_composite(val):
            color = "#00e676" if val > 0.3 else "#ef5350" if val < -0.1 else "#ffb300"
            return f"color: {color}; font-weight: bold"

        def color_z(val):
            try:
                v = float(val)
                color = "#00e676" if v > 0.5 else "#ef5350" if v < -0.5 else "#ffb300"
                return f"color: {color}"
            except Exception:
                return ""

        styled = (
            df_display.style
            .applymap(color_composite, subset=["Composite"])
            .applymap(color_z, subset=["Momentum Z", "Value Z", "Quality Z"])
        )
        st.dataframe(styled, use_container_width=True, height=420)

        # ── Factor scatter chart ─────────────────────────────────────────
        st.subheader("Factor Map — Momentum vs Value")
        chart_data = []
        for ticker, row in ranked.iterrows():
            info = fund_map.get(ticker, {})
            chart_data.append({
                "Ticker": ticker,
                "Momentum Z": row.get("momentum_z", 0) or 0,
                "Value Z": row.get("value_z", 0) or 0,
                "Quality Z": row.get("quality_z", 0) or 0,
                "Composite": row.get("composite_score", 0) or 0,
                "Sector": (info.get("sector", "Unknown") or "Unknown"),
            })
        chart_df = pd.DataFrame(chart_data)

        fig = px.scatter(
            chart_df, x="Value Z", y="Momentum Z",
            size=chart_df["Quality Z"].clip(lower=0.1) + 0.5,
            color="Composite", color_continuous_scale="RdYlGn",
            text="Ticker", hover_data=["Sector", "Composite"],
            template="plotly_dark",
        )
        fig.update_traces(textposition="top center", marker=dict(line=dict(width=1, color="#333")))
        fig.add_hline(y=0, line_dash="dash", line_color="#555")
        fig.add_vline(x=0, line_dash="dash", line_color="#555")
        fig.update_layout(height=450, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

        # ── Price sparklines for top 5 ───────────────────────────────────
        st.subheader("Price Momentum — Top 5")
        top5 = list(ranked.head(5).index)
        fig2 = go.Figure()
        for ticker in top5:
            df_p = price_data.get(ticker, pd.DataFrame())
            if df_p.empty:
                continue
            close = df_p["Close"].squeeze() if "Close" in df_p.columns else df_p.iloc[:, 0].squeeze()
            close = close.tail(252)
            normalized = close / close.iloc[0] * 100
            fig2.add_trace(go.Scatter(
                x=normalized.index, y=normalized.values,
                mode="lines", name=ticker, line=dict(width=2),
            ))
        fig2.update_layout(
            template="plotly_dark", height=300,
            yaxis_title="Indexed (100 = 1yr ago)",
            legend=dict(orientation="h", y=1.1),
            margin=dict(t=10, b=20),
        )
        st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: DEEP ANALYZER
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔬 Deep Analyzer":
    st.title("🔬 Deep Analyzer")
    st.markdown("Full thesis generation, devil's advocate challenge, backtesting, and stress testing for any ticker.")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        ticker_input = st.text_input("Enter ticker", value="NVDA", placeholder="e.g. AAPL, MSFT, NVDA")
    with col_b:
        st.markdown("<br>", unsafe_allow_html=True)
        run_analysis = st.button("🔬 Analyze", type="primary", use_container_width=True)

    if run_analysis and ticker_input:
        ticker = ticker_input.strip().upper()
        st.session_state.pop("analysis", None)

        with st.spinner(f"Running full analysis on {ticker}…"):
            from src.data.fetcher import fetch_price_history, fetch_fundamentals
            from src.signals.composite import compute_composite_score, rank_opportunities
            from src.thesis.generator import generate_thesis
            from src.thesis.challenger import challenge_thesis
            from src.thesis.backtest import run_full_backtest
            from src.risk.stress_test import run_stress_test

            fundamentals = fetch_fundamentals(ticker)
            prices = fetch_price_history(ticker, period="2y")
            spy_prices = fetch_price_history("SPY", period="2y")

            score = compute_composite_score(ticker, fundamentals, prices)
            ranked = rank_opportunities([score], top_n=1)

            if ranked.empty:
                st.error(f"Could not compute scores for {ticker}.")
                st.stop()

            score_row = ranked.iloc[0]
            thesis = generate_thesis(ticker, fundamentals, score_row)
            challenge = challenge_thesis(thesis, fundamentals, score_row)
            backtest = run_full_backtest(ticker, prices, spy_prices)

            close = prices["Close"].squeeze() if not prices.empty and "Close" in prices.columns else pd.Series()
            vol = float(close.pct_change().std() * np.sqrt(252)) if not close.empty else 0.25
            stress = run_stress_test(ticker, fundamentals, thesis.expected_return_1y, vol, 10_000)

            st.session_state["analysis"] = {
                "ticker": ticker, "fundamentals": fundamentals, "prices": prices,
                "thesis": thesis, "challenge": challenge, "backtest": backtest,
                "stress": stress, "score_row": score_row, "spy_prices": spy_prices,
            }

    if "analysis" in st.session_state:
        a = st.session_state["analysis"]
        thesis = a["thesis"]
        challenge = a["challenge"]
        backtest = a["backtest"]
        stress = a["stress"]
        prices = a["prices"]
        spy_prices = a["spy_prices"]
        fundamentals = a["fundamentals"]
        ticker = a["ticker"]

        # ── Header ───────────────────────────────────────────────────────
        st.divider()
        h1, h2, h3, h4, h5 = st.columns(5)
        h1.metric("Entry Price", fmt_usd(thesis.entry_price))
        h2.metric("Target Price", fmt_usd(thesis.target_price))
        h3.metric("Stop Loss", fmt_usd(thesis.stop_loss))
        delta_pct = f"{thesis.expected_return_1y*100:+.1f}%"
        h4.metric("Expected Return", delta_pct)
        h5.metric("Position Size", f"{thesis.position_size_pct*100:.0f}%")

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["📄 Thesis", "⚔️ Challenge", "📊 Backtest", "🌊 Stress Test", "📉 Price Chart"]
        )

        # ── Tab 1: Thesis ─────────────────────────────────────────────────
        with tab1:
            conv_c = conviction_color(thesis.conviction)
            st.markdown(
                f"**Conviction:** <span style='color:{conv_c};font-weight:700'>{thesis.conviction}</span> &nbsp;|&nbsp; "
                f"**Sector:** {thesis.sector} &nbsp;|&nbsp; "
                f"**Name:** {thesis.name}",
                unsafe_allow_html=True,
            )
            st.info(thesis.narrative)

            # Return scenarios
            st.subheader("Return Scenarios")
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Bear Case", pct(thesis.expected_return_bear),
                       delta=None, help="Downside scenario")
            sc2.metric("Base Case", pct(thesis.expected_return_1y),
                       delta=None, help="Expected 1-year return")
            sc3.metric("Bull Case", pct(thesis.expected_return_bull),
                       delta=None, help="Upside scenario")

            # Scenario bar chart
            scen_fig = go.Figure(go.Bar(
                x=["Bear", "Base", "Bull"],
                y=[thesis.expected_return_bear * 100, thesis.expected_return_1y * 100, thesis.expected_return_bull * 100],
                marker_color=["#ef5350", "#ffb300", "#00e676"],
                text=[pct(v) for v in [thesis.expected_return_bear, thesis.expected_return_1y, thesis.expected_return_bull]],
                textposition="outside",
            ))
            scen_fig.update_layout(template="plotly_dark", height=280, showlegend=False,
                                   yaxis_title="1-Year Return (%)", margin=dict(t=10, b=10))
            st.plotly_chart(scen_fig, use_container_width=True)

            cat_col, risk_col = st.columns(2)
            with cat_col:
                st.markdown("**Bull Catalysts**")
                for i, c in enumerate(thesis.primary_catalysts, 1):
                    st.markdown(f"<span style='color:#00e676'>▶</span> {c}", unsafe_allow_html=True)
            with risk_col:
                st.markdown("**Key Risks**")
                for i, r in enumerate(thesis.key_risks, 1):
                    st.markdown(f"<span style='color:#ef5350'>▶</span> {r}", unsafe_allow_html=True)

            # Valuation and quality tables
            v_col, q_col = st.columns(2)
            with v_col:
                st.markdown("**Valuation**")
                v_df = pd.DataFrame(thesis.valuation.items(), columns=["Metric", "Value"])
                st.dataframe(v_df, use_container_width=True, hide_index=True)
            with q_col:
                st.markdown("**Business Quality**")
                q_df = pd.DataFrame(thesis.quality_metrics.items(), columns=["Metric", "Value"])
                st.dataframe(q_df, use_container_width=True, hide_index=True)

        # ── Tab 2: Challenge ──────────────────────────────────────────────
        with tab2:
            risk_c = risk_color(challenge.risk_rating)
            revised_c = conviction_color(challenge.revised_conviction)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Risk Score", f"{challenge.overall_risk_score:.2f}")
            m2.metric("Risk Rating", challenge.risk_rating)
            m3.metric("Revised Conviction", challenge.revised_conviction)
            m4.metric("Return Haircut", f"{(1 - challenge.adjustment_factor)*100:.0f}%")

            # Risk score gauge
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=challenge.overall_risk_score,
                title={"text": "Overall Risk Score"},
                gauge={
                    "axis": {"range": [0, 1]},
                    "bar": {"color": risk_c},
                    "steps": [
                        {"range": [0, 0.2], "color": "#1a3a1a"},
                        {"range": [0.2, 0.4], "color": "#3a3a1a"},
                        {"range": [0.4, 0.65], "color": "#3a1a1a"},
                        {"range": [0.65, 1.0], "color": "#5a0a0a"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 2}, "value": challenge.overall_risk_score},
                },
            ))
            gauge.update_layout(template="plotly_dark", height=250, margin=dict(t=30, b=10))
            st.plotly_chart(gauge, use_container_width=True)

            if challenge.red_flags:
                st.error("**Red Flags**")
                for rf in challenge.red_flags:
                    st.markdown(f"⚠️ {rf}")
            if challenge.challenges:
                st.warning("**Challenges**")
                for ch in challenge.challenges:
                    st.markdown(f"• {ch}")
            if challenge.mitigants:
                st.success("**Mitigants**")
                for m in challenge.mitigants:
                    st.markdown(f"✓ {m}")

        # ── Tab 3: Backtest ───────────────────────────────────────────────
        with tab3:
            bt_cols = st.columns(len(backtest))
            for col, (strategy_name, result) in zip(bt_cols, backtest.items()):
                label = strategy_name.replace("_", " ").title()
                status = "✅ PASS" if result.passed else "❌ FAIL"
                col.metric(f"{label} {status}", pct(result.annualized_return),
                           delta=f"α={pct(result.alpha)}")

            # Equity curves
            eq_fig = go.Figure()
            colors = ["#00e676", "#2196f3", "#ffb300", "#ff4081"]
            for i, (name, result) in enumerate(backtest.items()):
                if result.equity_curve:
                    label = name.replace("_", " ").title()
                    eq_fig.add_trace(go.Scatter(
                        y=result.equity_curve,
                        mode="lines", name=label,
                        line=dict(color=colors[i % len(colors)], width=2),
                    ))
            eq_fig.add_hline(y=100, line_dash="dash", line_color="#555", annotation_text="Start")
            eq_fig.update_layout(
                template="plotly_dark", height=350,
                yaxis_title="Portfolio Value (start=100)",
                legend=dict(orientation="h", y=1.1),
                margin=dict(t=10, b=20),
            )
            st.plotly_chart(eq_fig, use_container_width=True)

            # Stats table
            stats = []
            for name, result in backtest.items():
                stats.append({
                    "Strategy": name.replace("_", " ").title(),
                    "Total Return": pct(result.total_return),
                    "Ann. Return": pct(result.annualized_return),
                    "vs Benchmark": pct(result.alpha),
                    "Sharpe": f"{result.sharpe_ratio:.2f}",
                    "Max DD": pct(result.max_drawdown),
                    "Win Rate": f"{result.win_rate*100:.0f}%",
                    "# Trades": result.num_trades,
                    "Result": "PASS" if result.passed else "FAIL",
                })
            st.dataframe(pd.DataFrame(stats), use_container_width=True, hide_index=True)

        # ── Tab 4: Stress Test ────────────────────────────────────────────
        with tab4:
            mc = stress.monte_carlo

            if mc:
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Expected Return", pct(mc.mean_return_1y))
                mc2.metric("Prob. of Loss", pct(mc.prob_loss))
                mc3.metric("P(Return > 20%)", pct(mc.prob_gain_20pct))
                mc4.metric("P(Return > 50%)", pct(mc.prob_gain_50pct))

                # Monte Carlo distribution
                dist = np.array(mc.distribution) * 100
                mc_fig = go.Figure()
                mc_fig.add_trace(go.Histogram(
                    x=dist, nbinsx=80, name="Return distribution",
                    marker_color="#2196f3", opacity=0.7,
                ))
                mc_fig.add_vline(x=mc.mean_return_1y * 100, line_color="#ffb300",
                                 line_dash="dash", annotation_text="Mean")
                mc_fig.add_vline(x=mc.percentile_5 * 100, line_color="#ef5350",
                                 line_dash="dot", annotation_text="P5")
                mc_fig.add_vline(x=mc.percentile_95 * 100, line_color="#00e676",
                                 line_dash="dot", annotation_text="P95")
                mc_fig.update_layout(
                    template="plotly_dark", height=320,
                    title=f"Monte Carlo Distribution (10,000 paths · fat-tailed)",
                    xaxis_title="1-Year Return (%)",
                    margin=dict(t=40, b=20),
                )
                st.plotly_chart(mc_fig, use_container_width=True)

                # Percentile band chart
                pct_fig = go.Figure()
                pct_fig.add_trace(go.Bar(
                    x=["P5 (Bear)", "P25", "Median", "P75", "P95 (Bull)"],
                    y=[mc.percentile_5*100, mc.percentile_25*100,
                       mc.median_return_1y*100, mc.percentile_75*100, mc.percentile_95*100],
                    marker_color=["#b71c1c", "#ef5350", "#ffb300", "#00c853", "#00e676"],
                    text=[pct(v) for v in [mc.percentile_5, mc.percentile_25,
                                           mc.median_return_1y, mc.percentile_75, mc.percentile_95]],
                    textposition="outside",
                ))
                pct_fig.add_hline(y=0, line_color="#555")
                pct_fig.update_layout(template="plotly_dark", height=280,
                                      yaxis_title="Return (%)", margin=dict(t=10, b=10))
                st.plotly_chart(pct_fig, use_container_width=True)

            # Historical crises
            st.subheader("Historical Crisis Scenarios")
            crisis_rows = []
            for s in stress.scenarios:
                crisis_rows.append({
                    "Crisis": s.name,
                    "Market Drop": pct(s.market_drop),
                    "Position Drop (β-adj)": pct(s.position_drop),
                    "Dollar Loss ($10k pos.)": f"${s.max_loss_dollars:,.0f}",
                    "Recovery Est.": f"~{s.recovery_time_days} days",
                })
            crisis_df = pd.DataFrame(crisis_rows)

            def color_drop(val):
                try:
                    v = float(val.strip("%+"))
                    return f"color: {'#ef5350' if v < -20 else '#ffb300' if v < -10 else '#aaa'}"
                except Exception:
                    return ""

            st.dataframe(
                crisis_df.style.applymap(color_drop, subset=["Market Drop", "Position Drop (β-adj)"]),
                use_container_width=True,
                hide_index=True,
            )

        # ── Tab 5: Price Chart ────────────────────────────────────────────
        with tab5:
            if not prices.empty:
                close = prices["Close"].squeeze() if "Close" in prices.columns else prices.iloc[:, 0].squeeze()

                # Main price + MA
                ma50 = close.rolling(50).mean()
                ma200 = close.rolling(200).mean()

                p_fig = go.Figure()
                p_fig.add_trace(go.Scatter(x=close.index, y=close.values,
                                           mode="lines", name=ticker,
                                           line=dict(color="#2196f3", width=2)))
                p_fig.add_trace(go.Scatter(x=ma50.index, y=ma50.values,
                                           mode="lines", name="50-day MA",
                                           line=dict(color="#ffb300", width=1, dash="dot")))
                p_fig.add_trace(go.Scatter(x=ma200.index, y=ma200.values,
                                           mode="lines", name="200-day MA",
                                           line=dict(color="#ef5350", width=1, dash="dot")))

                if thesis.entry_price:
                    p_fig.add_hline(y=thesis.entry_price, line_color="#00e676",
                                    line_dash="dash", annotation_text="Entry")
                if thesis.target_price:
                    p_fig.add_hline(y=thesis.target_price, line_color="#00e676",
                                    annotation_text="Target")
                if thesis.stop_loss:
                    p_fig.add_hline(y=thesis.stop_loss, line_color="#ef5350",
                                    line_dash="dash", annotation_text="Stop")

                p_fig.update_layout(
                    template="plotly_dark", height=400,
                    title=f"{ticker} — 2 Year Price History",
                    yaxis_title="Price ($)",
                    legend=dict(orientation="h", y=1.1),
                    margin=dict(t=40, b=20),
                )
                st.plotly_chart(p_fig, use_container_width=True)

                # Volume bar
                if "Volume" in prices.columns:
                    volume = prices["Volume"].squeeze()
                    vol_fig = go.Figure(go.Bar(
                        x=volume.index, y=volume.values,
                        marker_color="#37474f", name="Volume",
                    ))
                    vol_fig.update_layout(template="plotly_dark", height=150,
                                          margin=dict(t=5, b=5), showlegend=False,
                                          yaxis_title="Volume")
                    st.plotly_chart(vol_fig, use_container_width=True)

                # RSI chart
                from src.signals.momentum import rsi as compute_rsi_series
                if len(close) >= 14:
                    delta = close.diff()
                    gain = delta.clip(lower=0).rolling(14).mean()
                    loss = (-delta.clip(upper=0)).rolling(14).mean()
                    rs = gain / loss.replace(0, np.nan)
                    rsi_series = 100 - 100 / (1 + rs)

                    rsi_fig = go.Figure()
                    rsi_fig.add_trace(go.Scatter(x=rsi_series.index, y=rsi_series.values,
                                                  mode="lines", line=dict(color="#9c27b0", width=1.5),
                                                  name="RSI(14)"))
                    rsi_fig.add_hline(y=70, line_color="#ef5350", line_dash="dash",
                                      annotation_text="Overbought")
                    rsi_fig.add_hline(y=30, line_color="#00e676", line_dash="dash",
                                      annotation_text="Oversold")
                    rsi_fig.update_layout(template="plotly_dark", height=180,
                                          yaxis_title="RSI", yaxis_range=[0, 100],
                                          margin=dict(t=5, b=5), showlegend=False)
                    st.plotly_chart(rsi_fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: PORTFOLIO
# ─────────────────────────────────────────────────────────────────────────────
elif page == "💼 Portfolio":
    st.title("💼 Portfolio Dashboard")

    col1, col2, col3 = st.columns(3)
    with col1:
        portfolio_name = st.text_input("Portfolio name", "sodil_paper")
    with col2:
        initial_cash = st.number_input("Initial cash ($)", 10_000, 10_000_000, 100_000, step=10_000)
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        snap_btn = st.button("📸 Take Snapshot", use_container_width=True)

    if snap_btn:
        from src.portfolio.manager import PortfolioManager
        pm = PortfolioManager(name=portfolio_name, initial_cash=initial_cash)
        snap = pm.take_snapshot()
        st.success(f"Snapshot saved. Portfolio value: {fmt_usd(snap['total_value'])}")

    refresh = st.button("🔄 Refresh Dashboard", type="primary", use_container_width=True)

    # ── Quick trade panel ─────────────────────────────────────────────────
    with st.expander("⚡ Quick Trade"):
        tc1, tc2, tc3, tc4, tc5 = st.columns([2, 1, 1, 1, 1])
        trade_ticker = tc1.text_input("Ticker", key="trade_ticker_input")
        trade_action = tc2.selectbox("Action", ["BUY", "SELL"], key="trade_action")
        trade_pct = tc3.slider("Allocation %", 1, 20, 5, key="trade_pct_slider")
        tc4.markdown("<br>", unsafe_allow_html=True)
        if tc4.button("Execute", type="primary", use_container_width=True):
            from src.portfolio.manager import PortfolioManager
            pm = PortfolioManager(name=portfolio_name, initial_cash=initial_cash)
            if trade_action == "BUY":
                result = pm.buy(trade_ticker.upper(), trade_pct / 100)
            else:
                result = pm.sell(trade_ticker.upper())
            if result["success"]:
                if trade_action == "BUY":
                    st.success(f"Bought {result['shares']:.4f} {trade_ticker.upper()} @ {fmt_usd(result['price'])}")
                else:
                    st.success(f"Sold {result['shares']:.4f} {trade_ticker.upper()} | P&L: {fmt_usd(result.get('pnl',0))}")
            else:
                st.error(result["reason"])
        tc5.markdown("<br>", unsafe_allow_html=True)
        if tc5.button("🛑 Check Stops", use_container_width=True):
            from src.portfolio.manager import PortfolioManager
            pm = PortfolioManager(name=portfolio_name, initial_cash=initial_cash)
            triggered = pm.enforce_stop_losses()
            if triggered:
                for t in triggered:
                    st.warning(f"Stop triggered: {t['ticker']} @ ${t['exit_price']:.2f}")
            else:
                st.success("No stop-losses triggered.")

    # ── Load report ───────────────────────────────────────────────────────
    from src.portfolio.manager import PortfolioManager
    from src.portfolio.tracker import generate_report

    pm = PortfolioManager(name=portfolio_name, initial_cash=initial_cash)
    report = generate_report(pm.portfolio_id, portfolio_name, pm.initial_cash)

    # ── KPIs ──────────────────────────────────────────────────────────────
    st.divider()
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    ret_pct = report.cumulative_return * 100
    alpha_pct = report.alpha * 100
    k1.metric("Total Value", fmt_usd(report.total_value))
    k2.metric("Total Return", f"{ret_pct:+.2f}%", delta=f"vs SPY: {alpha_pct:+.2f}%")
    k3.metric("Cash", fmt_usd(report.cash))
    k4.metric("Unrealized P&L", fmt_usd(report.unrealized_pnl))
    k5.metric("Realized P&L", fmt_usd(report.realized_pnl))
    k6.metric("# Trades", report.trade_count)

    # ── Risk metrics row ──────────────────────────────────────────────────
    if report.risk_metrics:
        rm = report.risk_metrics
        r1, r2, r3, r4, r5, r6 = st.columns(6)
        r1.metric("Sharpe", f"{rm.sharpe_ratio:.2f}")
        r2.metric("Sortino", f"{rm.sortino_ratio:.2f}")
        r3.metric("Max Drawdown", pct(rm.max_drawdown))
        r4.metric("Ann. Volatility", pct(rm.volatility_ann))
        r5.metric("VaR 95%", pct(rm.var_95))
        r6.metric("Beta", f"{rm.beta:.2f}")

    tab_pf1, tab_pf2, tab_pf3 = st.tabs(["📈 Equity Curve", "📋 Positions", "📜 Trade History"])

    with tab_pf1:
        if report.equity_curve and len(report.equity_curve) >= 2:
            dates, vals = zip(*report.equity_curve)
            eq_df = pd.DataFrame({"Date": dates, "Portfolio": vals})
            eq_df["Date"] = pd.to_datetime(eq_df["Date"])

            eq_fig = go.Figure()
            eq_fig.add_trace(go.Scatter(
                x=eq_df["Date"], y=eq_df["Portfolio"],
                mode="lines", fill="tozeroy",
                name="Portfolio Value",
                line=dict(color="#00e676", width=2),
                fillcolor="rgba(0,230,118,0.08)",
            ))
            eq_fig.add_hline(y=report.initial_value, line_dash="dash",
                              line_color="#555", annotation_text="Initial")
            eq_fig.update_layout(
                template="plotly_dark", height=380,
                yaxis_title="Portfolio Value ($)",
                margin=dict(t=10, b=20),
            )
            st.plotly_chart(eq_fig, use_container_width=True)
        else:
            st.info("Not enough snapshots yet. Take at least 2 snapshots to see the equity curve.")
            st.markdown("Use the **Take Snapshot** button above to record portfolio state.")

    with tab_pf2:
        if report.position_performance:
            pos_rows = []
            for pp in report.position_performance:
                pos_rows.append({
                    "Ticker": pp.ticker,
                    "Shares": f"{pp.shares:.2f}",
                    "Avg Cost": fmt_usd(pp.avg_cost),
                    "Current Price": fmt_usd(pp.current_price),
                    "Value": fmt_usd(pp.current_value),
                    "Unreal. P&L": pp.unrealized_pnl,
                    "Return %": pp.unrealized_pnl_pct,
                    "Weight": f"{pp.weight_in_portfolio*100:.1f}%",
                })
            pos_df = pd.DataFrame(pos_rows)

            def style_pnl(val):
                return f"color: {'#00e676' if val >= 0 else '#ef5350'}"

            styled_pos = (
                pos_df.style
                .applymap(style_pnl, subset=["Unreal. P&L", "Return %"])
                .format({"Unreal. P&L": "${:,.2f}", "Return %": "{:+.1f}%"})
            )
            st.dataframe(styled_pos, use_container_width=True, hide_index=True)

            # Position weight donut
            if len(report.position_performance) > 0:
                pie_data = [{"name": pp.ticker, "value": pp.current_value}
                            for pp in report.position_performance]
                cash_val = report.cash
                if cash_val > 0:
                    pie_data.append({"name": "Cash", "value": cash_val})
                pie_df = pd.DataFrame(pie_data)
                pie_fig = px.pie(pie_df, names="name", values="value",
                                 hole=0.45, template="plotly_dark",
                                 title="Portfolio Allocation")
                pie_fig.update_layout(height=320, margin=dict(t=40, b=10))
                st.plotly_chart(pie_fig, use_container_width=True)
        else:
            st.info("No open positions. Use Quick Trade above or run Screen & Invest from the CLI.")

    with tab_pf3:
        from src.portfolio.db import get_trade_history
        trades = get_trade_history(pm.portfolio_id)
        if trades:
            trade_df = pd.DataFrame(trades)[["trade_date", "ticker", "action", "shares", "price", "total_value", "reason"]]
            trade_df.columns = ["Date", "Ticker", "Action", "Shares", "Price", "Total", "Reason"]
            trade_df["Date"] = trade_df["Date"].str[:19]

            def color_action(val):
                return "color: #00e676; font-weight: bold" if val == "BUY" else "color: #ef5350; font-weight: bold"

            st.dataframe(
                trade_df.style.applymap(color_action, subset=["Action"]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No trades recorded yet.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4: TEST SUITE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🧪 Test Suite":
    st.title("🧪 Test Suite")
    st.markdown("Run the full unit test suite and inspect results. All tests cover signal logic, backtesting, risk math, thesis generation, and portfolio mechanics.")

    col1, col2 = st.columns([2, 1])
    with col1:
        test_module = st.selectbox(
            "Test scope",
            ["All tests", "test_signals", "test_backtest", "test_risk", "test_thesis", "test_portfolio"],
        )
    with col2:
        verbose = st.checkbox("Verbose output", value=True)
        show_coverage = st.checkbox("Coverage report", value=False)

    run_tests = st.button("▶  Run Tests", type="primary", use_container_width=True)

    if run_tests:
        cmd = [sys.executable, "-m", "pytest"]
        if test_module == "All tests":
            cmd += ["tests/"]
        else:
            cmd += [f"tests/{test_module}.py"]
        if verbose:
            cmd += ["-v"]
        cmd += ["--tb=short", "--no-header", "-q" if not verbose else ""]
        cmd = [c for c in cmd if c]

        if show_coverage:
            cmd += ["--cov=src", "--cov-report=term-missing"]

        with st.spinner("Running tests…"):
            t0 = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True, text=True,
                cwd="/home/user/Sodil",
            )
            elapsed = time.time() - t0

        # ── Results summary ───────────────────────────────────────────────
        stdout = result.stdout + result.stderr
        lines = stdout.splitlines()

        # Parse pass/fail counts from last line
        summary_line = next(
            (l for l in reversed(lines) if "passed" in l or "failed" in l or "error" in l),
            ""
        )

        passed = 0
        failed = 0
        errors = 0
        import re
        for m in re.finditer(r"(\d+)\s+(passed|failed|error)", summary_line):
            n, label = int(m.group(1)), m.group(2)
            if label == "passed":
                passed = n
            elif label == "failed":
                failed = n
            else:
                errors = n

        total = passed + failed + errors
        if total > 0:
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Total", total)
            s2.metric("Passed", passed, delta=None)
            s3.metric("Failed", failed, delta=None)
            s4.metric("Duration", f"{elapsed:.2f}s")

            pass_rate = passed / total * 100
            progress_color = "normal" if pass_rate == 100 else "inverse"
            st.progress(passed / total)

            if failed == 0 and errors == 0:
                st.success(f"✅ All {passed} tests passed in {elapsed:.2f}s")
            else:
                st.error(f"❌ {failed} test(s) failed — see output below")

        # ── Raw output ────────────────────────────────────────────────────
        with st.expander("📋 Full test output", expanded=(failed > 0 or verbose)):
            colored_output = stdout.replace(
                " PASSED", " ✅"
            ).replace(
                " FAILED", " ❌"
            ).replace(
                " ERROR", " 🔴"
            )
            st.code(colored_output, language="text")

        # ── Per-test table (verbose mode) ─────────────────────────────────
        if verbose and total > 0:
            test_lines = [l for l in lines if "PASSED" in l or "FAILED" in l or "ERROR" in l]
            if test_lines:
                rows = []
                for tl in test_lines:
                    if "PASSED" in tl:
                        status, name = "PASSED", tl.replace(" PASSED", "").strip()
                    elif "FAILED" in tl:
                        status, name = "FAILED", tl.replace(" FAILED", "").strip()
                    else:
                        status, name = "ERROR", tl.replace(" ERROR", "").strip()

                    # Parse module::class::test
                    parts = name.split("::")
                    module = parts[0].replace("tests/", "").replace(".py", "") if parts else ""
                    test_name = "::".join(parts[1:]) if len(parts) > 1 else name
                    rows.append({"Module": module, "Test": test_name, "Status": status})

                test_df = pd.DataFrame(rows)

                def color_status(val):
                    if val == "PASSED":
                        return "color: #00e676; font-weight: bold"
                    elif val == "FAILED":
                        return "color: #ef5350; font-weight: bold"
                    return "color: #ffb300; font-weight: bold"

                st.subheader(f"Test Results ({len(rows)} tests)")
                st.dataframe(
                    test_df.style.applymap(color_status, subset=["Status"]),
                    use_container_width=True,
                    hide_index=True,
                    height=min(35 * len(rows) + 40, 600),
                )

                # Breakdown by module
                if len(test_df["Module"].unique()) > 1:
                    st.subheader("Results by Module")
                    module_summary = test_df.groupby("Module")["Status"].value_counts().unstack(fill_value=0)
                    if "PASSED" not in module_summary:
                        module_summary["PASSED"] = 0
                    if "FAILED" not in module_summary:
                        module_summary["FAILED"] = 0
                    module_summary["Pass Rate"] = (
                        module_summary["PASSED"] / (module_summary["PASSED"] + module_summary["FAILED"]) * 100
                    ).round(0).astype(int).astype(str) + "%"
                    st.dataframe(module_summary, use_container_width=True)

    else:
        # Static overview when no run yet
        st.info("Click **Run Tests** to execute the test suite.")
        st.subheader("Test Coverage Overview")

        coverage_data = {
            "Module": ["test_signals", "test_backtest", "test_risk", "test_thesis", "test_portfolio"],
            "Tests": [17, 11, 17, 16, 20],
            "Covers": [
                "Momentum, value, quality signals, composite scoring",
                "Momentum strategy, buy-and-hold, full backtest pipeline",
                "Sharpe/Sortino/VaR/CVaR, stress scenarios, Monte Carlo",
                "Thesis generation, conviction, challenger, risk flags",
                "DB operations, buy/sell, stop-losses, realized P&L",
            ],
        }
        st.dataframe(pd.DataFrame(coverage_data), use_container_width=True, hide_index=True)

        # Show what's being tested as a bar chart
        fig = px.bar(
            pd.DataFrame(coverage_data),
            x="Module", y="Tests",
            color="Tests", color_continuous_scale="Blues",
            template="plotly_dark",
            text="Tests",
        )
        fig.update_layout(height=280, showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
