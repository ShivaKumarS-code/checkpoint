"""SQLite connection handling and numbered migrations."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1

MIGRATIONS: list[tuple[int, str]] = []


def _schema_sql() -> str:
    return Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")


def connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def migrate(conn: sqlite3.Connection) -> int:
    """Apply pending migrations and return the resulting schema version."""
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)")
    row = conn.execute("SELECT MAX(version) AS version FROM schema_version").fetchone()
    current = int(row["version"] or 0)
    if current == 0:
        conn.executescript(_schema_sql())
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        conn.commit()
        current = SCHEMA_VERSION
    for version, script in MIGRATIONS:
        if version > current:
            conn.executescript(script)
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
            conn.commit()
            current = version
    return current


def init_db(db_path: str | Path) -> sqlite3.Connection:
    """Open (creating if needed) a migrated cassette database."""
    conn = connect(db_path)
    migrate(conn)
    return conn
