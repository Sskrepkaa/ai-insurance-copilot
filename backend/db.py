"""SQLite access layer for the AI Insurance Copilot backend.

Provides a small set of helpers for opening a connection and creating the
schema. This module has NO side effects at import time: no connections are
opened and no files are created just by importing it. Callers must
explicitly call `init_db()` (done by seed_data.py) or `get_connection()`.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "claims.db"


def get_connection() -> sqlite3.Connection:
    """Open a new SQLite connection to the claims database.

    Uses `sqlite3.Row` as the row factory so callers can access columns
    by name (and easily convert rows to dicts).
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the `claims` and `customer_history` tables if they don't exist."""
    conn = get_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS claims (
                id TEXT PRIMARY KEY,
                client TEXT NOT NULL,
                amount REAL NOT NULL,
                status TEXT NOT NULL,
                type TEXT NOT NULL,
                description TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS customer_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                client TEXT NOT NULL,
                claim_id TEXT NOT NULL,
                date TEXT NOT NULL,
                outcome TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()
