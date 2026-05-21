"""Market data fetching with caching and error handling."""

import os
import json
import time
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

CACHE_DIR = Path(os.path.expanduser("~/.sodil/cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_SECONDS = 3600  # 1 hour


def _cache_key(prefix: str, *args) -> Path:
    key = hashlib.md5(f"{prefix}{'_'.join(str(a) for a in args)}".encode()).hexdigest()
    return CACHE_DIR / f"{prefix}_{key}.json"


def _load_cache(path: Path):
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if time.time() - data.get("ts", 0) > CACHE_TTL_SECONDS:
            return None
        return data.get("payload")
    except Exception:
        return None


def _save_cache(path: Path, payload):
    try:
        with open(path, "w") as f:
            json.dump({"ts": time.time(), "payload": payload}, f)
    except Exception:
        pass


def fetch_price_history(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Return OHLCV DataFrame for ticker."""
    cache_path = _cache_key("prices", ticker, period, interval)
    if use_cache:
        cached = _load_cache(cache_path)
        if cached:
            df = pd.DataFrame(cached)
            df.index = pd.to_datetime(df.index)
            return df

    retries = 3
    for attempt in range(retries):
        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True,
            )
            if df.empty:
                return pd.DataFrame()
            if use_cache:
                _save_cache(cache_path, df.to_dict())
            return df
        except Exception as e:
            if attempt == retries - 1:
                logger.warning(f"Failed to fetch {ticker}: {e}")
                return pd.DataFrame()
            time.sleep(2 ** attempt)

    return pd.DataFrame()


def fetch_fundamentals(ticker: str, use_cache: bool = True) -> dict:
    """Return key fundamental data for a ticker."""
    cache_path = _cache_key("fundamentals", ticker)
    if use_cache:
        cached = _load_cache(cache_path)
        if cached:
            return cached

    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        fundamentals = {
            "ticker": ticker,
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "pb_ratio": info.get("priceToBook"),
            "ps_ratio": info.get("priceToSalesTrailing12Months"),
            "ev_ebitda": info.get("enterpriseToEbitda"),
            "debt_to_equity": info.get("debtToEquity"),
            "roe": info.get("returnOnEquity"),
            "roa": info.get("returnOnAssets"),
            "profit_margin": info.get("profitMargins"),
            "operating_margin": info.get("operatingMargins"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "free_cash_flow": info.get("freeCashflow"),
            "current_ratio": info.get("currentRatio"),
            "quick_ratio": info.get("quickRatio"),
            "beta": info.get("beta"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "avg_volume": info.get("averageVolume"),
            "short_ratio": info.get("shortRatio"),
            "analyst_target": info.get("targetMeanPrice"),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "dividend_yield": info.get("dividendYield"),
            "peg_ratio": info.get("pegRatio"),
        }

        if use_cache:
            _save_cache(cache_path, fundamentals)
        return fundamentals

    except Exception as e:
        logger.warning(f"Failed to fetch fundamentals for {ticker}: {e}")
        return {"ticker": ticker}


def fetch_batch_prices(
    tickers: list[str],
    period: str = "1y",
    max_workers: int = 8,
) -> dict[str, pd.DataFrame]:
    """Fetch price histories for multiple tickers."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_price_history, t, period): t for t in tickers
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                results[ticker] = future.result()
            except Exception as e:
                logger.warning(f"Error fetching {ticker}: {e}")
                results[ticker] = pd.DataFrame()
    return results


def fetch_batch_fundamentals(
    tickers: list[str],
    max_workers: int = 8,
) -> dict[str, dict]:
    """Fetch fundamentals for multiple tickers."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_fundamentals, t): t for t in tickers
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                results[ticker] = future.result()
            except Exception as e:
                logger.warning(f"Error fetching fundamentals for {ticker}: {e}")
                results[ticker] = {"ticker": ticker}
    return results


def get_sp500_tickers() -> list[str]:
    """Fetch current S&P 500 constituents from Wikipedia."""
    cache_path = _cache_key("sp500_tickers")
    if (cached := _load_cache(cache_path)):
        return cached

    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        tickers = tables[0]["Symbol"].tolist()
        tickers = [t.replace(".", "-") for t in tickers]
        _save_cache(cache_path, tickers)
        return tickers
    except Exception as e:
        logger.warning(f"Could not fetch S&P 500 list: {e}")
        # Fallback: representative large-cap universe
        return [
            "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B",
            "JPM", "V", "UNH", "XOM", "LLY", "JNJ", "MA", "AVGO", "PG", "HD",
            "MRK", "CVX", "COST", "ABBV", "AMD", "ORCL", "WMT", "BAC", "KO",
            "CRM", "PEP", "ACN", "MCD", "ADBE", "LIN", "TMO", "NFLX", "CSCO",
            "WFC", "DIS", "QCOM", "VZ", "IBM", "INTC", "TXN", "INTU", "SPGI",
            "GS", "CAT", "AXP", "BA", "GE", "HON", "UPS", "RTX", "MS",
            "ISRG", "NOW", "AMAT", "PANW", "KLAC", "LRCX", "ADI", "MRVL",
        ]
