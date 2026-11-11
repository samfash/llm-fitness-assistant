"""SQLite persistence (stdlib only). Set FITNESS_DB_PATH to relocate the file.

NOTE: a free Hugging Face Space has an ephemeral disk, so data resets on restart.
Use a Space with persistent storage, or point this at Postgres, for real use.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import date

SCHEMA = """
CREATE TABLE IF NOT EXISTS profiles (
    user_id TEXT PRIMARY KEY, data TEXT NOT NULL, updated_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS workouts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, day TEXT NOT NULL,
    exercise TEXT NOT NULL, sets INTEGER, reps INTEGER, weight_kg REAL, duration_min REAL, notes TEXT);
CREATE TABLE IF NOT EXISTS meals (
    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, day TEXT NOT NULL,
    description TEXT NOT NULL, calories REAL, protein_g REAL, carbs_g REAL, fat_g REAL);
CREATE INDEX IF NOT EXISTS ix_workouts ON workouts(user_id, day);
CREATE INDEX IF NOT EXISTS ix_meals ON meals(user_id, day);
"""


@contextmanager
def connect():
    path = os.getenv("FITNESS_DB_PATH", "data/fitness.db")
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


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


def add_workout(user_id: str, e: dict) -> int:
    with connect() as c:
        cur = c.execute(
            "INSERT INTO workouts(user_id, day, exercise, sets, reps, weight_kg, duration_min, notes) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (user_id, e.get("date") or date.today().isoformat(), e["exercise"].strip().lower(),
             e["sets"], e["reps"], e["weight_kg"], e.get("duration_min"), e.get("notes")),
        )
        return cur.lastrowid


def add_meal(user_id: str, e: dict) -> int:
    with connect() as c:
        cur = c.execute(
            "INSERT INTO meals(user_id, day, description, calories, protein_g, carbs_g, fat_g) "
            "VALUES(?,?,?,?,?,?,?)",
            (user_id, e.get("date") or date.today().isoformat(), e["description"].strip(),
             e["calories"], e["protein_g"], e["carbs_g"], e["fat_g"]),
        )
        return cur.lastrowid


def workouts_since(user_id: str, since: str) -> list[dict]:
    with connect() as c:
        rows = c.execute("SELECT * FROM workouts WHERE user_id=? AND day>=? ORDER BY day", (user_id, since))
        return [dict(r) for r in rows]


def meals_since(user_id: str, since: str) -> list[dict]:
    with connect() as c:
        rows = c.execute("SELECT * FROM meals WHERE user_id=? AND day>=? ORDER BY day", (user_id, since))
        return [dict(r) for r in rows]
