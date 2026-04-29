"""SQLite persistence layer for sentiment analysis history."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable


DATABASE_PATH = Path("sentiment_history.db")


def init_database(db_path: Path = DATABASE_PATH) -> sqlite3.Connection:
    """Create the history table if needed and return a database connection."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sentiment_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    return conn


def save_to_history(
    conn: sqlite3.Connection,
    text: str,
    sentiment: str,
    confidence: float,
    user_id: int | None = None,
) -> None:
    """Persist one sentiment analysis result."""
    conn.execute(
        "INSERT INTO sentiment_history (text, sentiment, confidence, user_id) VALUES (?, ?, ?, ?)",
        (text, sentiment, confidence, user_id),
    )
    conn.commit()


def get_history(
    conn: sqlite3.Connection,
    limit: int = 100,
    user_id: int | None = None,
    include_all: bool = False,
) -> Iterable[tuple]:
    """Return recent history records ordered by newest first."""
    if include_all:
        cursor = conn.execute(
            """
            SELECT text, sentiment, confidence, timestamp
            FROM sentiment_history
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        )
        return cursor.fetchall()

    cursor = conn.execute(
        """
        SELECT text, sentiment, confidence, timestamp
        FROM sentiment_history
        WHERE user_id = ? OR user_id IS NULL
        ORDER BY timestamp DESC, id DESC
        LIMIT ?
        """,
        (user_id, limit),
    )
    return cursor.fetchall()


def clear_history(conn: sqlite3.Connection) -> None:
    """Delete all saved analysis history."""
    conn.execute("DELETE FROM sentiment_history")
    conn.commit()
