import sqlite3

import pytest

from core import db


def test_backend_is_what_the_fixture_says(backend):
    assert db.is_postgres() == (backend == "postgres")


def test_roundtrip_all_tables_and_ordering():
    db.save_profile("u", {"age": 30})
    db.save_profile("u", {"sex": "male", "age": None})  # None never overwrites
    assert db.get_profile("u") == {"age": 30, "sex": "male"}
    a = db.add_workout("u", {"exercise": " Bench ", "sets": 3, "reps": 8, "weight_kg": 80.5, "date": "2026-01-02"})
    b = db.add_workout("u", {"exercise": "squat", "sets": 1, "reps": 5, "weight_kg": 100, "date": "2026-01-01"})
    assert b == a + 1
    rows = db.workouts_since("u", "2026-01-01")
    assert [r["exercise"] for r in rows] == ["squat", "bench"] and rows[1]["weight_kg"] == 80.5  # float precision kept
    assert db.workouts_since("other", "2000-01-01") == []


def test_meal_items_roundtrip_as_json():
    items = [{"fdc_id": 1, "name": "Egg", "grams": 100}]
    db.add_meal("u", {"description": "eggs", "calories": 143, "protein_g": 12.6, "carbs_g": 0.7, "fat_g": 9.5, "items": items})
    db.add_meal("u", {"description": "plain", "calories": 10, "protein_g": 0, "carbs_g": 0, "fat_g": 0})
    import json
    rows = db.meals_since("u", "2000-01-01")
    assert json.loads(rows[0]["items"]) == items and rows[1]["items"] is None


def test_migrations_recorded_and_idempotent():
    with db.connect() as c:
        versions = [r["version"] for r in c.execute("SELECT version FROM schema_migrations ORDER BY version").fetchall()]
    assert versions == [v for v, _ in db._MIGRATIONS]
    db._initialized.clear()  # simulate a fresh process connecting again
    with db.connect() as c:
        again = c.execute("SELECT COUNT(*) AS n FROM schema_migrations").fetchone()["n"]
    assert again == len(db._MIGRATIONS)


def test_transaction_rolls_back_on_error():
    with pytest.raises(RuntimeError), db.connect() as c:
        c.execute("INSERT INTO profiles(user_id, data) VALUES (?, ?)", ("boom", "{}"))
        raise RuntimeError("fail mid-transaction")
    assert db.get_profile("boom") == {}


def test_legacy_sqlite_database_is_upgraded_in_place(tmp_path, monkeypatch):
    """A database created by the pre-migration app (no schema_migrations, no meals.items) keeps its data."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    path = tmp_path / "legacy.db"
    raw = sqlite3.connect(path)
    raw.executescript("""
        CREATE TABLE profiles (user_id TEXT PRIMARY KEY, data TEXT NOT NULL, updated_at TEXT DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE workouts (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, day TEXT NOT NULL,
            exercise TEXT NOT NULL, sets INTEGER, reps INTEGER, weight_kg REAL, duration_min REAL, notes TEXT);
        CREATE TABLE meals (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, day TEXT NOT NULL,
            description TEXT NOT NULL, calories REAL, protein_g REAL, carbs_g REAL, fat_g REAL);
        INSERT INTO profiles(user_id, data) VALUES ('old', '{"age": 41}');
        INSERT INTO meals(user_id, day, description, calories, protein_g, carbs_g, fat_g)
            VALUES ('old', '2026-01-01', 'toast', 100, 3, 20, 1);
    """)
    raw.commit()
    raw.close()
    monkeypatch.setenv("FITNESS_DB_PATH", str(path))
    assert db.get_profile("old") == {"age": 41}
    meal = db.meals_since("old", "2000-01-01")[0]
    assert meal["description"] == "toast" and meal["items"] is None
    db.add_meal("old", {"description": "new", "calories": 1, "protein_g": 0, "carbs_g": 0, "fat_g": 0, "items": [{"x": 1}]})


def test_eval_runner_never_touches_a_configured_database_url(monkeypatch):
    from core.agent import AssistantTurn
    from evals.runner import run_case

    class Quiet:
        def chat(self, messages, tools):
            return AssistantTurn(content="ok")

    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody@127.0.0.1:1/never")  # would fail loudly if used
    case = {"id": "x", "category": "t", "prompt": "What's a good warm up?", "expect": {}}
    assert run_case(case, Quiet()).passed
    assert __import__("os").environ["DATABASE_URL"].endswith("/never")  # and restored afterwards
