import json
from datetime import date, timedelta

from core import db
from core.tools import execute_tool, progress_summary, tool_specs


def run(uid, name, args):
    return json.loads(execute_tool(uid, name, args))


def test_specs_are_flat_and_valid():
    specs = tool_specs()
    assert {s["function"]["name"] for s in specs} >= {"log_workout", "log_meal", "update_profile"}
    blob = json.dumps(specs)
    assert "anyOf" not in blob and "$defs" not in blob


def test_profile_merges_partial_updates():
    run("u1", "update_profile", {"age": 30, "sex": "male"})
    out = run("u1", "update_profile", {"weight_kg": 80})
    assert out["profile"] == {"age": 30, "sex": "male", "weight_kg": 80}


def test_log_workout_persists_and_normalises():
    out = run("u1", "log_workout", {"exercise": "Bench Press", "sets": 3, "reps": 8, "weight_kg": 80})
    assert out["saved"]
    row = db.workouts_since("u1", "2000-01-01")[0]
    assert row["exercise"] == "bench press" and row["sets"] == 3


def test_validation_error_returned_not_raised():
    out = run("u1", "log_workout", {"exercise": "squat", "sets": 0, "reps": 5})
    assert out["error"] == "Invalid arguments" and any("sets" in d for d in out["details"])
    assert db.workouts_since("u1", "2000-01-01") == []


def test_bad_date_and_bad_json_and_unknown_tool():
    assert "error" in run("u1", "log_meal", {"description": "rice", "calories": 300, "date": "yesterday"})
    assert "error" in run("u1", "log_meal", "{not json")
    assert "Unknown tool" in run("u1", "drop_tables", {})["error"]


def test_users_are_isolated():
    run("alice", "log_workout", {"exercise": "squat", "sets": 1, "reps": 5, "weight_kg": 100})
    assert db.workouts_since("bob", "2000-01-01") == []


def test_progress_summary_numbers():
    old = (date.today() - timedelta(days=30)).isoformat()
    run("u1", "log_workout", {"exercise": "bench", "sets": 3, "reps": 8, "weight_kg": 80})
    run("u1", "log_workout", {"exercise": "bench", "sets": 5, "reps": 5, "weight_kg": 90, "date": old})  # outside window
    run("u1", "log_meal", {"description": "lunch", "calories": 600, "protein_g": 40})
    run("u1", "log_meal", {"description": "dinner", "calories": 400, "protein_g": 20})
    s = progress_summary("u1", 7)
    assert s["workout_sessions"] == 1 and s["total_volume_kg"] == 3 * 8 * 80
    assert s["avg_daily_calories"] == 1000 and s["avg_daily_protein_g"] == 60
    assert s["best_estimated_1rm_kg"] == {"bench": 101.3} and s["targets"] is None
