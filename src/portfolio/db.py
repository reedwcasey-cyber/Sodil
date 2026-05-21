"""SQLite persistence layer for portfolio state."""

from __future__ import annotations
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

DB_PATH = Path.home() / ".sodil" / "portfolio.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create all tables if they don't exist."""
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS portfolios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            initial_cash REAL NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            shares REAL NOT NULL,
            avg_cost REAL NOT NULL,
            entry_date TEXT NOT NULL,
            thesis_conviction TEXT,
            stop_loss REAL,
            target_price REAL,
            notes TEXT,
            FOREIGN KEY (portfolio_id) REFERENCES portfolios(id)
        );

        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            action TEXT NOT NULL,
            shares REAL NOT NULL,
            price REAL NOT NULL,
            total_value REAL NOT NULL,
            commission REAL NOT NULL DEFAULT 0,
            trade_date TEXT NOT NULL,
            reason TEXT,
            FOREIGN KEY (portfolio_id) REFERENCES portfolios(id)
        );

        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id INTEGER NOT NULL,
            snapshot_date TEXT NOT NULL,
            total_value REAL NOT NULL,
            cash REAL NOT NULL,
            positions_value REAL NOT NULL,
            daily_pnl REAL,
            cumulative_return REAL,
            benchmark_return REAL,
            positions_json TEXT,
            FOREIGN KEY (portfolio_id) REFERENCES portfolios(id)
        );

        CREATE TABLE IF NOT EXISTS opportunities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            discovered_at TEXT NOT NULL,
            ticker TEXT NOT NULL,
            composite_score REAL,
            conviction TEXT,
            expected_return REAL,
            thesis_json TEXT,
            challenge_json TEXT,
            backtest_json TEXT,
            acted_on INTEGER DEFAULT 0
        );
        """)


init_db()


def create_portfolio(name: str, initial_cash: float) -> int:
    with get_conn() as conn:
        try:
            conn.execute(
                "INSERT INTO portfolios (name, initial_cash, created_at) VALUES (?, ?, ?)",
                (name, initial_cash, datetime.now().isoformat()),
            )
            conn.commit()
            row = conn.execute("SELECT id FROM portfolios WHERE name=?", (name,)).fetchone()
            return row["id"]
        except sqlite3.IntegrityError:
            row = conn.execute("SELECT id FROM portfolios WHERE name=?", (name,)).fetchone()
            return row["id"]


def get_portfolio_id(name: str) -> Optional[int]:
    with get_conn() as conn:
        row = conn.execute("SELECT id FROM portfolios WHERE name=?", (name,)).fetchone()
        return row["id"] if row else None


def get_all_portfolios() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM portfolios").fetchall()
        return [dict(r) for r in rows]


def record_trade(
    portfolio_id: int,
    ticker: str,
    action: str,
    shares: float,
    price: float,
    commission: float = 0.0,
    reason: str = "",
):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO trades
               (portfolio_id, ticker, action, shares, price, total_value, commission, trade_date, reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (portfolio_id, ticker, action, shares, price,
             shares * price, commission, datetime.now().isoformat(), reason),
        )
        conn.commit()


def upsert_position(
    portfolio_id: int,
    ticker: str,
    shares: float,
    avg_cost: float,
    conviction: str = "",
    stop_loss: Optional[float] = None,
    target_price: Optional[float] = None,
    notes: str = "",
):
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT id FROM positions WHERE portfolio_id=? AND ticker=?",
            (portfolio_id, ticker),
        ).fetchone()

        if existing:
            if shares == 0:
                conn.execute(
                    "DELETE FROM positions WHERE portfolio_id=? AND ticker=?",
                    (portfolio_id, ticker),
                )
            else:
                conn.execute(
                    """UPDATE positions SET shares=?, avg_cost=?, thesis_conviction=?,
                       stop_loss=?, target_price=?, notes=?
                       WHERE portfolio_id=? AND ticker=?""",
                    (shares, avg_cost, conviction, stop_loss, target_price, notes,
                     portfolio_id, ticker),
                )
        else:
            if shares > 0:
                conn.execute(
                    """INSERT INTO positions
                       (portfolio_id, ticker, shares, avg_cost, entry_date,
                        thesis_conviction, stop_loss, target_price, notes)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (portfolio_id, ticker, shares, avg_cost, datetime.now().isoformat(),
                     conviction, stop_loss, target_price, notes),
                )
        conn.commit()


def get_positions(portfolio_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM positions WHERE portfolio_id=?", (portfolio_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_trade_history(portfolio_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM trades WHERE portfolio_id=? ORDER BY trade_date",
            (portfolio_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def save_snapshot(
    portfolio_id: int,
    total_value: float,
    cash: float,
    positions_value: float,
    daily_pnl: float,
    cumulative_return: float,
    benchmark_return: float,
    positions: list[dict],
):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO snapshots
               (portfolio_id, snapshot_date, total_value, cash, positions_value,
                daily_pnl, cumulative_return, benchmark_return, positions_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (portfolio_id, datetime.now().isoformat(), total_value, cash,
             positions_value, daily_pnl, cumulative_return, benchmark_return,
             json.dumps(positions)),
        )
        conn.commit()


def get_snapshots(portfolio_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM snapshots WHERE portfolio_id=? ORDER BY snapshot_date",
            (portfolio_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def save_opportunity(
    ticker: str,
    composite_score: float,
    conviction: str,
    expected_return: float,
    thesis: dict,
    challenge: dict,
    backtest: dict,
):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO opportunities
               (discovered_at, ticker, composite_score, conviction,
                expected_return, thesis_json, challenge_json, backtest_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now().isoformat(), ticker, composite_score, conviction,
             expected_return, json.dumps(thesis, default=str),
             json.dumps(challenge, default=str), json.dumps(backtest, default=str)),
        )
        conn.commit()


def get_opportunities(limit: int = 50) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM opportunities ORDER BY discovered_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
