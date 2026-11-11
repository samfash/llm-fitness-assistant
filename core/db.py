"""Storage layer: SQLite by default, PostgreSQL when DATABASE_URL is set.

    DATABASE_URL=postgresql://user:pass@host/dbname?sslmode=require     -> Postgres (e.g. Neon, Supabase)
    FITNESS_DB_PATH=data/fitness.db                                      -> SQLite (default)

Free Hugging Face Spaces have an ephemeral disk, so SQLite data is lost on restart; point DATABASE_URL at a
managed Postgres to keep it. Schema changes are versioned migrations (see _MIGRATIONS) applied automatically
the first time a process connects; they are safe to run concurrently on Postgres (advisory lock).

SQL is written once with `?` placeholders and translated for Postgres. Every statement here is portable
between SQLite 3.35+ and PostgreSQL (RETURNING, ON CONFLICT ... DO UPDATE).
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import date

_MIGRATIONS: list[tuple[int, list[str]]] = [
    (1, [
        "CREATE TABLE IF NOT EXISTS profiles (user_id TEXT PRIMARY KEY, data TEXT NOT NULL, "
        "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
        "CREATE TABLE IF NOT EXISTS workouts (id {pk}, user_id TEXT NOT NULL, day TEXT NOT NULL, "
        "exercise TEXT NOT NULL, sets INTEGER, reps INTEGER, weight_kg DOUBLE PRECISION, "
        "duration_min DOUBLE PRECISION, notes TEXT)",
        "CREATE TABLE IF NOT EXISTS meals (id {pk}, user_id TEXT NOT NULL, day TEXT NOT NULL, "
        "description TEXT NOT NULL, calories DOUBLE PRECISION, protein_g DOUBLE PRECISION, "
        "carbs_g DOUBLE PRECISION, fat_g DOUBLE PRECISION)",
        "CREATE INDEX IF NOT EXISTS ix_workouts ON workouts(user_id, day)",
        "CREATE INDEX IF NOT EXISTS ix_meals ON meals(user_id, day)",
    ]),
    (2, ["ALTER TABLE meals ADD COLUMN items TEXT"]),  # JSON list of foods behind a meal (USDA-backed logging)
    (3, [
        "CREATE TABLE IF NOT EXISTS api_keys (id {pk}, key_hash TEXT NOT NULL UNIQUE, user_id TEXT NOT NULL, label TEXT, "
        "prefix TEXT NOT NULL, revoked INTEGER NOT NULL DEFAULT 0, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
        "CREATE INDEX IF NOT EXISTS ix_api_keys_user ON api_keys(user_id)",
        "CREATE TABLE IF NOT EXISTS usage (user_id TEXT NOT NULL, day TEXT NOT NULL, "
        "requests INTEGER NOT NULL DEFAULT 0, prompt_tokens INTEGER NOT NULL DEFAULT 0, "
        "completion_tokens INTEGER NOT NULL DEFAULT 0, cost_usd DOUBLE PRECISION, PRIMARY KEY (user_id, day))",
    ]),
]
_DATA_TABLES = ["profiles", "workouts", "meals", "api_keys", "usage"]  # everything except schema_migrations

_initialized: set[str] = set()
_init_lock = threading.Lock()
_pools: dict[str, object] = {}


def database_url() -> str:
    return os.getenv("DATABASE_URL", "").strip()


def is_postgres() -> bool:
    return database_url().startswith(("postgres://", "postgresql://"))


def _target() -> str:
    return database_url() if is_postgres() else os.getenv("FITNESS_DB_PATH", "data/fitness.db")


class _Conn:
    """Thin wrapper so callers write `?` placeholders for both databases."""

    def __init__(self, raw, pg: bool):
        self.raw, self.pg = raw, pg

    def execute(self, sql: str, params=()):
        return self.raw.execute(sql.replace("?", "%s") if self.pg else sql, params)


def _render(stmt: str, pg: bool) -> str:
    return stmt.replace("{pk}", "BIGSERIAL PRIMARY KEY" if pg else "INTEGER PRIMARY KEY AUTOINCREMENT")


def _migrate(conn: _Conn) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations "
                 "(version INTEGER PRIMARY KEY, applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
    if conn.pg:
        conn.execute("SELECT pg_advisory_xact_lock(727274)")  # serialise concurrent first-start migrations
    done = {r["version"] for r in conn.execute("SELECT version FROM schema_migrations").fetchall()}
    for version, statements in _MIGRATIONS:
        if version in done:
            continue
        for st in statements:
            conn.execute(_render(st, conn.pg))
        conn.execute("INSERT INTO schema_migrations(version) VALUES (?)", (version,))


def _pg_pool(url: str):
    """A psycopg connection pool if psycopg_pool is installed (and DB_POOL != off), else None."""
    if os.getenv("DB_POOL", "on").lower() == "off":
        return None
    if url not in _pools:
        try:
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool

            _pools[url] = ConnectionPool(url, min_size=1, max_size=int(os.getenv("DB_POOL_MAX", "5")),
                                         kwargs={"row_factory": dict_row}, open=True)
        except ImportError:
            _pools[url] = None
    return _pools[url]


@contextmanager
def _raw_connection():
    if is_postgres():
        url = database_url()
        pool = _pg_pool(url)
        if pool is not None:
            with pool.connection() as conn:  # commits on success, rolls back on error
                yield conn, True
        else:
            import psycopg
            from psycopg.rows import dict_row

            with psycopg.connect(url, row_factory=dict_row) as conn:
                yield conn, True
    else:
        path = _target()
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn, False
            conn.commit()
        finally:
            conn.close()


@contextmanager
def connect():
    target = _target()
    if target not in _initialized:
        with _init_lock:
            if target not in _initialized:
                with _raw_connection() as (raw, pg):
                    _migrate(_Conn(raw, pg))
                _initialized.add(target)
    with _raw_connection() as (raw, pg):
        yield _Conn(raw, pg)


def close_pools() -> None:
    for pool in _pools.values():
        if pool is not None:
            pool.close()
    _pools.clear()
    _initialized.clear()


def reset_for_tests() -> None:
    """Delete all rows (keeps schema). Test helper."""
    with connect() as c:
        for t in _DATA_TABLES:
            c.execute(f"DELETE FROM {t}")


# ------------------------------------------------------------------------------------------ profiles
def get_profile(user_id: str) -> dict:
    with connect() as c:
        row = c.execute("SELECT data FROM profiles WHERE user_id=?", (user_id,)).fetchone()
    return json.loads(row["data"]) if row else {}


def save_profile(user_id: str, updates: dict) -> dict:
    merged = {**get_profile(user_id), **{k: v for k, v in updates.items() if v is not None}}
    with connect() as c:
        c.execute(
            "INSERT INTO profiles(user_id, data) VALUES(?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET data=excluded.data, updated_at=CURRENT_TIMESTAMP",
            (user_id, json.dumps(merged)),
        )
    return merged


# ------------------------------------------------------------------------------------ workouts / meals
def add_workout(user_id: str, e: dict) -> int:
    with connect() as c:
        cur = c.execute(
            "INSERT INTO workouts(user_id, day, exercise, sets, reps, weight_kg, duration_min, notes) "
            "VALUES(?,?,?,?,?,?,?,?) RETURNING id",
            (user_id, e.get("date") or date.today().isoformat(), e["exercise"].strip().lower(),
             e["sets"], e["reps"], e["weight_kg"], e.get("duration_min"), e.get("notes")),
        )
        return cur.fetchone()["id"]


def add_meal(user_id: str, e: dict) -> int:
    items = e.get("items")
    with connect() as c:
        cur = c.execute(
            "INSERT INTO meals(user_id, day, description, calories, protein_g, carbs_g, fat_g, items) "
            "VALUES(?,?,?,?,?,?,?,?) RETURNING id",
            (user_id, e.get("date") or date.today().isoformat(), e["description"].strip(),
             e["calories"], e["protein_g"], e["carbs_g"], e["fat_g"], json.dumps(items) if items else None),
        )
        return cur.fetchone()["id"]


def workouts_since(user_id: str, since: str) -> list[dict]:
    with connect() as c:
        rows = c.execute("SELECT * FROM workouts WHERE user_id=? AND day>=? ORDER BY day, id", (user_id, since))
        return [dict(r) for r in rows.fetchall()]


def meals_since(user_id: str, since: str) -> list[dict]:
    with connect() as c:
        rows = c.execute("SELECT * FROM meals WHERE user_id=? AND day>=? ORDER BY day, id", (user_id, since))
        return [dict(r) for r in rows.fetchall()]
