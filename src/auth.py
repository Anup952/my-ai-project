"""Authentication helpers backed by SQLite."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from hmac import compare_digest


def init_auth_tables(conn: sqlite3.Connection) -> None:
    """Create user table and seed a default admin account."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sentiment_history_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            text TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    cursor = conn.execute("PRAGMA table_info(sentiment_history)")
    columns = {row[1] for row in cursor.fetchall()}
    if "user_id" not in columns:
        conn.execute("ALTER TABLE sentiment_history ADD COLUMN user_id INTEGER")

    if get_user_by_username(conn, "admin") is None:
        create_user(conn, "admin", "admin123", role="admin")
    conn.commit()


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """Hash a password with PBKDF2."""
    salt = salt or os.urandom(16).hex()
    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        120000,
    ).hex()
    return password_hash, salt


def get_user_by_username(conn: sqlite3.Connection, username: str) -> dict | None:
    """Return a user record by username."""
    cursor = conn.execute(
        "SELECT id, username, password_hash, salt, role, created_at FROM users WHERE username = ?",
        (username.strip().lower(),),
    )
    row = cursor.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "username": row[1],
        "password_hash": row[2],
        "salt": row[3],
        "role": row[4],
        "created_at": row[5],
    }


def create_user(conn: sqlite3.Connection, username: str, password: str, role: str = "user") -> None:
    """Create a new user account."""
    username = username.strip().lower()
    if len(username) < 3:
        raise ValueError("Username must be at least 3 characters.")
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")

    password_hash, salt = hash_password(password)
    conn.execute(
        "INSERT INTO users (username, password_hash, salt, role) VALUES (?, ?, ?, ?)",
        (username, password_hash, salt, role),
    )
    conn.commit()


def authenticate_user(conn: sqlite3.Connection, username: str, password: str) -> dict | None:
    """Validate login credentials and return safe user session data."""
    user = get_user_by_username(conn, username)
    if not user:
        return None

    password_hash, _ = hash_password(password, user["salt"])
    if not compare_digest(password_hash, user["password_hash"]):
        return None

    return {"id": user["id"], "username": user["username"], "role": user["role"]}


def list_users(conn: sqlite3.Connection) -> list[tuple]:
    """Return users for admin view."""
    cursor = conn.execute(
        """
        SELECT users.id, users.username, users.role, users.created_at, COUNT(sentiment_history.id) AS analyses
        FROM users
        LEFT JOIN sentiment_history ON sentiment_history.user_id = users.id
        GROUP BY users.id, users.username, users.role, users.created_at
        ORDER BY users.created_at DESC
        """
    )
    return cursor.fetchall()

