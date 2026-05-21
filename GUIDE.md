# SODIL — User Guide
### Systematic Opportunity Discovery & Investment Lab

> **Plain English guide** — no finance or coding background required.

---

## What Is SODIL?

SODIL is a **quantitative investment research tool**. "Quantitative" just means it uses math and data instead of gut feelings to find stocks worth looking at.

Think of it like this: instead of watching financial news and guessing which stocks might go up, SODIL automatically reads the numbers behind thousands of stocks, scores them, and surfaces the ones that look most promising — then challenges its own findings before showing them to you.

**It does not trade real money.** Everything is simulated ("paper trading") so you can experiment with zero financial risk.

---

## The Four Things SODIL Does

```
┌─────────────────────────────────────────────────────────────────┐
│  1. SCAN        Scores hundreds of stocks and ranks them        │
│  2. ANALYZE     Deep-dives one stock — thesis + devil's advocate│
│  3. PORTFOLIO   Tracks your simulated trades over time          │
│  4. TEST        Confirms the math engine is working correctly   │
└─────────────────────────────────────────────────────────────────┘
```

Each of these is a tab in the left sidebar of the dashboard.

---

## How to Start the Dashboard

1. Open your Codespace at **github.com/reedwcasey-cyber/Sodil**
2. In the terminal at the bottom, type:
   ```
   streamlit run dashboard.py
   ```
3. Click **"Open in Browser"** when the popup appears

The screen looks like this when it loads:

```
┌────────────────────────────────────────────────────────────────────────┐
│  SIDEBAR (left)          │  MAIN CONTENT (right)                       │
│  ─────────────────────   │  ─────────────────────────────────────────  │
│  📈 SODIL                │                                             │
│  Systematic Opportunity  │   (content changes based on which           │
│  Discovery & Investment  │    page you select on the left)             │
│  Lab                     │                                             │
│                          │                                             │
│  ○ 🔭 Market Scanner     │                                             │
│  ○ 🔬 Deep Analyzer      │                                             │
│  ○ 💼 Portfolio          │                                             │
│  ○ 🧪 Test Suite         │                                             │
│                          │                                             │
│  ─────────────────────   │                                             │
│  Data via Yahoo Finance  │                                             │
│  Paper trading only      │                                             │
│  Not financial advice    │                                             │
└────────────────────────────────────────────────────────────────────────┘
```

Click any item in the sidebar to switch pages.

---

---

# Page 1 — 🔭 Market Scanner

## What It Does
Scans up to 60 stocks from the S&P 500, scores each one across four factors, and shows you the top opportunities ranked best to worst.

## The Screen Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  Top N opportunities: [10 ▼]   Min market cap: [$2B ▼]   □ Custom  │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              🚀 Run Scan  (big blue button)                   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ── After scan completes: ───────────────────────────────────────   │
│                                                                      │
│  Stocks Surfaced: 10   Strong Signal: 3   Moderate: 5   Avg: 0.42   │
│                                                                      │
│  ┌──────────────────────────────────────── Results Table ───────┐   │
│  │ Rank │ Ticker │ Name        │ Score │ Momentum │ Value │ RSI  │   │
│  │  1   │ NVDA   │ NVIDIA...   │ 0.82  │  +1.8    │ +0.4  │ 58   │   │
│  │  2   │ AAPL   │ Apple...    │ 0.61  │  +1.1    │ +0.9  │ 52   │   │
│  │  3   │ MSFT   │ Microsoft.. │ 0.54  │  +0.8    │ +1.2  │ 49   │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ── Factor Map (scatter chart) ──────────────────────────────────   │
│  ── Price Chart: Top 5 stocks over past year ────────────────────   │
└─────────────────────────────────────────────────────────────────────┘
```

## Controls Explained

| Control | What It Does |
|---|---|
| **Top N** | How many stocks to show (5–30). Start with 10. |
| **Min Market Cap** | Filters out tiny companies. $2B = medium-large companies only. |
| **Custom Tickers** | Check this to type in specific stocks you want to compare instead of scanning the S&P 500. |

## Reading the Results Table

Each column tells you something different about the stock:

| Column | Plain English Meaning |
|---|---|
| **Score** | The overall grade. Higher = better opportunity. Green is good, red is bad. |
| **Momentum Z** | Is the stock trending upward recently? Positive = yes, heading up. |
| **Value Z** | Is the stock cheap relative to peers? Positive = looks undervalued. |
| **Quality Z** | Is this a financially healthy business? Positive = strong fundamentals. |
| **RSI** | A "temperature gauge" for the stock. 30–70 is healthy. Below 30 = possibly oversold (bargain?). Above 70 = possibly overheated. |
| **Analyst ↑** | How much upside Wall Street analysts are predicting. +15% means analysts think it'll rise 15%. |

**Colors in the table:**
- 🟢 Green numbers = positive signal
- 🔴 Red numbers = negative signal  
- 🟡 Yellow = neutral

## The Factor Map Chart

This scatter chart plots every stock by **Value** (x-axis) vs **Momentum** (y-axis):

```
         HIGH MOMENTUM
               │
   (Great momentum,   │   (Great momentum,
    cheap valuation)  │   expensive valuation)
               │
───────────────┼──────────────── HIGH VALUE
               │
   (Poor momentum,    │   (Poor momentum,
    cheap valuation)  │   expensive valuation)
               │
         LOW MOMENTUM
```

**Stocks in the top-right quadrant** (high momentum + high value) are the sweet spot. Bubble size = business quality. Color = overall composite score.

## How to Use This Page

1. Click **Run Scan**
2. Wait ~30–60 seconds
3. Look for stocks in the **top rows** with **green scores**
4. Note any tickers that interest you
5. Go to 🔬 **Deep Analyzer** to investigate further

---

---

# Page 2 — 🔬 Deep Analyzer

## What It Does
Takes one specific stock and runs a complete investigation:
- Builds an **investment thesis** (the case FOR buying it)
- **Challenges that thesis** (the case AGAINST)
- **Backtests** whether the strategy would have worked historically
- **Stress tests** what happens in a market crash

## The Screen Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  Enter ticker: [ NVDA          ]     🔬 Analyze  (button)           │
│                                                                      │
│  ── After analysis: ─────────────────────────────────────────────   │
│                                                                      │
│  Entry: $875    Target: $1,050    Stop: $805    Exp Return: +18%     │
│                                                                      │
│  ┌──────────┬────────────┬────────────┬──────────────┬───────────┐  │
│  │📄 Thesis │⚔️ Challenge│📊 Backtest │🌊 Stress Test│📉 Chart   │  │
│  └──────────┴────────────┴────────────┴──────────────┴───────────┘  │
│                                                                      │
│  (content of selected tab shown below)                               │
└─────────────────────────────────────────────────────────────────────┘
```

## The Five Tabs Explained

---

### 📄 Tab 1: Thesis

This is the **bull case** — the reasons this stock might be worth buying.

```
┌──────────────────────────────────────────────────────────────────┐
│  Conviction: HIGH   Sector: Technology   Name: NVIDIA Corp       │
│                                                                   │
│  📝 Thesis Narrative (plain English summary paragraph)           │
│                                                                   │
│  Return Scenarios:                                                │
│  ┌──────────┬────────────┬────────────┐                          │
│  │   Bear   │    Base    │    Bull    │                          │
│  │  -12%    │   +18%     │   +42%     │                          │
│  └──────────┴────────────┴────────────┘                          │
│                                                                   │
│  Bull Catalysts        │  Key Risks                              │
│  ▶ Strong momentum...  │  ▶ High beta (volatile)...              │
│  ▶ Revenue growing...  │  ▶ Expensive valuation...               │
│  ▶ ROE of 28%...       │  ▶ Market risk...                       │
│                                                                   │
│  Valuation Metrics     │  Business Quality                       │
│  P/E: 35x              │  ROE: 28%                               │
│  Fwd P/E: 28x          │  Profit Margin: 22%                     │
│  P/B: 18x              │  Revenue Growth: +35%                   │
└──────────────────────────────────────────────────────────────────┘
```

**Key terms:**

| Term | Plain English |
|---|---|
| **Conviction: HIGH/MEDIUM/LOW** | How confident the model is in this opportunity. HIGH = all signals agree. |
| **Bear / Base / Bull** | Three possible outcomes. Bear = things go wrong. Base = things go normally. Bull = things go very right. |
| **Catalyst** | A specific reason the stock might go up. |
| **P/E Ratio** | Price-to-Earnings. How much you're paying per dollar of profit. Lower = cheaper. 15x is cheap, 50x+ is expensive. |
| **Fwd P/E** | Same but using *next year's* expected earnings. If lower than current P/E, earnings are expected to grow. |
| **ROE (Return on Equity)** | How efficiently the company turns shareholder money into profit. 15%+ is good, 25%+ is excellent. |
| **Revenue Growth** | How fast sales are growing year over year. +20% means sales grew 20% vs last year. |
| **Profit Margin** | What percentage of sales becomes profit. 20% means they keep 20¢ of every $1 in sales. |

---

### ⚔️ Tab 2: Challenge

This is the **devil's advocate** — the model arguing against its own thesis. Healthy skepticism before committing to a trade.

```
┌──────────────────────────────────────────────────────────────────┐
│  Risk Score: 0.38   Risk Rating: MEDIUM   Revised: MEDIUM        │
│  Return Haircut: 12%  (expected return reduced by 12% for risk)  │
│                                                                   │
│  🕐 (Risk Gauge — needle pointing to 0.38 out of 1.0)           │
│                                                                   │
│  ⚠️ Red Flags (most serious concerns)                            │
│  ⚠️ Forward P/E higher than trailing — earnings may fall         │
│                                                                   │
│  ● Challenges (worth watching)                                   │
│  ● RSI above 70 — stock may be overbought short term            │
│  ● High beta amplifies losses in market downturns               │
│                                                                   │
│  ✓ Mitigants (reasons the risks are manageable)                  │
│  ✓ Strong balance sheet limits bankruptcy risk                   │
│  ✓ Position sized at 5% so max portfolio loss is small           │
└──────────────────────────────────────────────────────────────────┘
```

**Key terms:**

| Term | Plain English |
|---|---|
| **Risk Score** | 0 = very safe, 1 = very risky. Above 0.65 = be very careful. |
| **Risk Rating** | LOW / MEDIUM / HIGH / VERY HIGH — plain English version of the risk score. |
| **Revised Conviction** | After hearing the challenges, does the model still believe the thesis? May be lower than original. |
| **Return Haircut** | The model automatically reduces its expected return estimate based on risk. 12% haircut on a +20% expected return → adjusted to +17.6%. |
| **Red Flag** | A serious concern that could mean the thesis is wrong. |
| **Mitigant** | A reason not to panic — why the risk is manageable. |
| **Beta** | How much the stock moves relative to the market. Beta 2.0 = if the market drops 10%, this stock typically drops 20%. |

---

### 📊 Tab 3: Backtest

This answers the question: **"Would this strategy have worked in the past?"**

```
┌──────────────────────────────────────────────────────────────────┐
│  Momentum Strategy ✅ PASS    Ann. Return: +24%   α: +11%        │
│  Buy & Hold        ✅ PASS    Ann. Return: +18%   α: +5%         │
│                                                                   │
│  📈 Equity Curves (line chart showing portfolio growth over time) │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ $180                                      ╱ Momentum       │  │
│  │ $160                                  ╱╱╱                  │  │
│  │ $140                           ╱╱╱╱╱╱    Buy & Hold        │  │
│  │ $120                    ╱╱╱╱╱╱                             │  │
│  │ $100 ─────────────────╱                                    │  │
│  │      Jan 2023          Jan 2024          Jan 2025          │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  Backtest Stats Table                                             │
│  Sharpe: 1.4  │  Max Drawdown: -18%  │  Win Rate: 62%            │
└──────────────────────────────────────────────────────────────────┘
```

**Key terms:**

| Term | Plain English |
|---|---|
| **PASS / FAIL** | Did this strategy produce better-than-average returns historically? |
| **Ann. Return** | Average yearly return if you ran this strategy over the past 2 years. |
| **α (Alpha)** | How much better (or worse) than just holding the S&P 500 index. Positive = beat the market. |
| **Sharpe Ratio** | Return per unit of risk taken. Above 1.0 is good. Above 2.0 is excellent. Think of it as "bang for your buck" on risk. |
| **Max Drawdown** | The worst peak-to-trough loss the strategy experienced. -18% means at its worst point, you were down 18% from the high. |
| **Win Rate** | Percentage of trades that were profitable. 62% = 62 out of every 100 trades made money. |

---

### 🌊 Tab 4: Stress Test

This answers: **"How badly would this stock hurt me in a market crash?"**

Two parts:

**Part A — Monte Carlo (10,000 simulations)**

```
┌────────────────────────────────────────────────────────────────┐
│  Expected Return: +18%    Prob. of Loss: 28%                  │
│  P(Return > 20%): 45%     P(Return > 50%): 12%                │
│                                                                 │
│  📊 Distribution chart (bell curve of possible outcomes)       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │        P5 ↓        Mean ↓              P95 ↓             │  │
│  │    ────│───────────────│───────────────────│────         │  │
│  │   -25% │         +18%  │                +58%│            │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

The chart shows all 10,000 simulated outcomes. The further right, the better. The red line (P5) is the worst-case 5% of scenarios. The green line (P95) is the best-case 5%.

**Part B — Historical Crises**

| Crisis | Market Dropped | Your Position Would Drop |
|---|---|---|
| 2008 Financial Crisis | -57% | ~-68% (high beta) |
| 2020 COVID Crash | -34% | ~-41% |
| 2022 Rate Hike Bear | -25% | ~-30% |

This table is purely educational — it answers "if 2008 happened again tomorrow, how much money would I lose?" It's designed to keep you grounded about downside risk.

---

### 📉 Tab 5: Price Chart

A visual history of the stock's price over 2 years with key levels marked.

```
┌──────────────────────────────────────────────────────────────────┐
│  $1,100 ────────────────────────────────── Target ($1,050) ───  │
│  $1,000                              ╱╲╱╲                        │
│    $900                          ╱╲╱     ╲╱                      │
│    $875 ─────────────────────────────────────── Entry ($875) ── │
│    $800                      ╱╱                                  │
│    $805 ─────────────────────────────────────── Stop ($805) ──  │
│          Jan          Jul           Jan          May             │
│          2023         2023          2024          2024           │
│                                                                  │
│  Also shows: Volume bars, RSI indicator (momentum gauge)         │
└──────────────────────────────────────────────────────────────────┘
```

- **Blue line** = stock price
- **Yellow dotted** = 50-day moving average (short-term trend)
- **Red dotted** = 200-day moving average (long-term trend)
- **Green dashed line** = Entry price (where you'd buy)
- **Top green line** = Target price (where you'd sell for profit)
- **Bottom red line** = Stop-loss (where you'd sell to cut losses)

---

---

# Page 3 — 💼 Portfolio

## What It Does
Tracks all your simulated ("paper") trades in one place, shows your overall profit/loss, and measures how well you're doing compared to the S&P 500.

## The Screen Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  Portfolio: [sodil_paper]   Initial Cash: [$100,000]                │
│                                                                      │
│  📸 Take Snapshot     🔄 Refresh Dashboard                          │
│                                                                      │
│  ▼ Quick Trade (expandable panel)                                    │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Ticker: [NVDA]  Action: [BUY▼]  Allocation: [5%]  Execute  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ── KPI Row ──────────────────────────────────────────────────────  │
│  Total Value   │ Total Return   │ Cash    │ Unreal. P&L │ Trades    │
│  $104,200      │ +4.2% (+1.1%)  │ $52,000 │ +$4,200     │ 8        │
│                                                                      │
│  ── Risk Row ─────────────────────────────────────────────────────  │
│  Sharpe: 1.2  │ Sortino: 1.8  │ Max DD: -3.1%  │ VaR 95%: -1.2%   │
│                                                                      │
│  ┌─────────────┬──────────────┬──────────────────────────────────┐  │
│  │📈 Equity    │📋 Positions  │📜 Trade History                  │  │
│  └─────────────┴──────────────┴──────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Understanding the KPI Row

| Metric | Plain English |
|---|---|
| **Total Value** | Current value of everything (cash + stocks) |
| **Total Return** | How much you've made or lost since starting. +4.2% on $100k = you're up $4,200. The number in brackets is vs the S&P 500. |
| **Cash** | How much uninvested cash remains |
| **Unrealized P&L** | Paper profit/loss on positions you still hold (not yet sold) |
| **Trades** | Total number of buy/sell orders executed |

## Understanding the Risk Row

| Metric | Plain English |
|---|---|
| **Sharpe Ratio** | Quality of returns. Above 1.0 = good. Above 2.0 = excellent. Measures return relative to risk taken. |
| **Sortino Ratio** | Like Sharpe but only penalizes *bad* volatility (drops). Higher = better. |
| **Max Drawdown** | The worst losing streak so far. -3.1% means you were down 3.1% from your peak at the worst moment. |
| **VaR 95%** | Value at Risk. On your worst 5% of days, you'd lose this much or more. -1.2% on $100k = could lose $1,200 on a bad day. |

## The Three Tabs

**📈 Equity Curve** — A line chart of your portfolio value over time. If it's going up and to the right, you're doing well.

**📋 Positions** — A table of every stock you currently hold:
```
┌──────┬────────┬──────────┬─────────┬─────────┬────────────┬────────┐
│Ticker│ Shares │ Avg Cost │ Current │  Value  │ P&L        │ Weight │
│ NVDA │  5.71  │ $875.00  │ $935.00 │$5,339   │ +$342 +6.9%│  5.1%  │
│ AAPL │  13.2  │ $189.00  │ $198.00 │$2,614   │ +$119 +4.8%│  2.5%  │
└──────┴────────┴──────────┴─────────┴─────────┴────────────┴────────┘
```
- Green P&L = position is profitable
- Red P&L = position is losing money
- Weight = what % of your total portfolio this position represents

**📜 Trade History** — Every buy and sell you've made, in order.

## The Quick Trade Panel

Click "**⚡ Quick Trade**" to expand it and manually buy or sell:

- Type a ticker (e.g. `AAPL`)
- Choose BUY or SELL
- Set allocation % (e.g. 5% = invest 5% of portfolio value)
- Click **Execute**

The **🛑 Check Stops** button scans all open positions and automatically sells any that have fallen below their stop-loss price.

## Taking Snapshots

Click **📸 Take Snapshot** to record the current portfolio state. This is how the equity curve chart gets built over time — every snapshot is a data point. Click it once a day (or whenever you want to track progress) to build up a history.

---

---

# Page 4 — 🧪 Test Suite

## What It Does
Runs automated checks on all the math and logic inside SODIL to confirm everything is working correctly. Think of it like a self-diagnostic.

## The Screen Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  Test scope: [All tests ▼]   □ Verbose   □ Coverage                 │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │              ▶ Run Tests  (button)                              │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ── After running: ──────────────────────────────────────────────   │
│                                                                      │
│  Total: 81  │  Passed: 81  │  Failed: 0  │  Duration: 2.1s          │
│  ████████████████████████████████████████████  100%                 │
│                                                                      │
│  ✅ All 81 tests passed in 2.15s                                     │
│                                                                      │
│  ── Per-Test Table ──────────────────────────────────────────────   │
│  Module           │ Test Name                       │ Status        │
│  test_signals     │ test_rsi_range                  │ ✅ PASSED     │
│  test_backtest    │ test_trending_market_profitable  │ ✅ PASSED     │
│  test_risk        │ test_var_is_negative             │ ✅ PASSED     │
└─────────────────────────────────────────────────────────────────────┘
```

## What Gets Tested

| Module | What It Checks |
|---|---|
| **test_signals** | Momentum, value, and quality math (e.g. "does a trending stock actually score higher than a declining one?") |
| **test_backtest** | Historical simulation logic (e.g. "does a buy-and-hold strategy match the actual price return?") |
| **test_risk** | Risk calculations (e.g. "is VaR always a negative number? Does a riskier position have higher VaR?") |
| **test_thesis** | Thesis and challenge generation (e.g. "does a risky stock get more red flags than a healthy one?") |
| **test_portfolio** | Trade mechanics (e.g. "does cash decrease after a buy? Does stop-loss trigger at the right price?") |

## What to Do If Tests Fail
All 81 should pass. If any show ❌ FAILED:
1. The detailed output will show exactly which test failed and why
2. Share the failure message with your developer
3. Don't trust the scanner results until it's fixed

---

---

# The Full Workflow — Start to Finish

Here's how to use all four pages together in one session:

```
STEP 1 — SCAN
└─► Open 🔭 Market Scanner
└─► Click "Run Scan"
└─► Note the top 2–3 tickers with high green scores

STEP 2 — ANALYZE
└─► Open 🔬 Deep Analyzer
└─► Type in your top ticker → Click "Analyze"
└─► Read the Thesis tab → Do you believe the story?
└─► Read the Challenge tab → Are the red flags deal-breakers?
└─► Check the Backtest tab → Did this work historically?
└─► If all three look good → proceed to Step 3

STEP 3 — SIMULATE A TRADE
└─► Open 💼 Portfolio
└─► Expand "Quick Trade"
└─► Type the ticker, choose BUY, set 5% allocation → Execute
└─► Click "Take Snapshot" to record your portfolio state

STEP 4 — CHECK BACK OVER TIME
└─► Open 💼 Portfolio each day/week
└─► Click "Take Snapshot" each visit
└─► Watch the Equity Curve to see if your thesis is playing out
└─► Click "Check Stops" to enforce exit rules

STEP 5 — CONFIRM EVERYTHING IS WORKING
└─► Open 🧪 Test Suite occasionally
└─► Click "Run Tests"
└─► All 81 should show ✅
```

---

---

# Glossary — Key Terms in Plain English

| Term | Plain English |
|---|---|
| **Alpha** | Returns above and beyond what the market gives you. +5% alpha = you beat the market by 5%. |
| **Backtest** | Running a strategy against historical data to see if it would have worked. |
| **Beta** | A stock's sensitivity to market swings. Beta 1.5 = moves 50% more than the market in both directions. |
| **Conviction** | How strongly the model believes in an opportunity. HIGH = strong evidence. LOW = uncertain. |
| **CVaR** | Expected loss on your very worst days (worse than VaR). Always worse than VaR. |
| **Drawdown** | A decline from a peak. -20% drawdown = fell 20% from its highest point. |
| **Equity Curve** | A line chart of portfolio value over time. Going up = making money. |
| **Factor** | A measurable characteristic used to score stocks. SODIL uses momentum, value, quality, technical. |
| **Momentum** | The tendency for stocks that have been going up to keep going up (and vice versa). |
| **Monte Carlo** | Running thousands of random simulations to understand the range of possible outcomes. |
| **P/E Ratio** | Price ÷ Earnings per share. Tells you how expensive the stock is relative to profits. |
| **Paper Trading** | Simulated trading with fake money. Identical to real trading in every way except no real money moves. |
| **Position Size** | What percentage of your portfolio to allocate to one stock. 5% is conservative. |
| **Quantitative** | Using math and data (not gut feelings) to make decisions. |
| **RSI** | Relative Strength Index. A 0–100 gauge. Above 70 = possibly overbought. Below 30 = possibly oversold. |
| **Sharpe Ratio** | Return per unit of risk. The higher the better. 1.0+ = good. 2.0+ = excellent. |
| **Stop-Loss** | A predetermined exit price. If the stock falls to this level, you sell automatically to limit losses. |
| **Stress Test** | Simulating how your position would perform in extreme scenarios (crashes, recessions). |
| **Thesis** | The argument for why a stock is worth buying. A good thesis has specific, testable reasons. |
| **Universe** | The pool of stocks being considered for scanning. Default is the S&P 500. |
| **VaR** | Value at Risk. The most you'd expect to lose on a bad day (at a 95% confidence level). |
| **Z-Score** | How far above or below average something is, measured in standard deviations. +2.0 = top ~2%. |

---

---

# Important Reminders

> ⚠️ **SODIL is for research and education only.**
>
> - It uses **simulated (paper) money** — no real trades are ever placed
> - Past performance does not guarantee future results
> - The backtest results show historical patterns, not predictions
> - Always consult a qualified financial advisor before making real investment decisions
> - Market conditions can change rapidly in ways no model can fully predict

---

*SODIL — Systematic Opportunity Discovery & Investment Lab*
*Built on: Python · yfinance · Streamlit · SQLite*
