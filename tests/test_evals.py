import json
from pathlib import Path

import pytest

from core.agent import AssistantTurn, ToolCall
from core.tools import TOOLS
from evals.run import main
from evals.runner import RecordingLLM, ReplayLLM, check_gate, load_cases, run_case, summarize
from evals.scoring import args_match, match_value, ungrounded_numbers

CASES_DIR = Path(__file__).resolve().parents[1] / "evals" / "cases"


class Scripted:
    def __init__(self, turns):
        self.turns = list(turns)

    def chat(self, messages, tools):
        return self.turns.pop(0)


def call(name, args, id_="c1"):
    return AssistantTurn(tool_calls=[ToolCall(id_, name, args)])


class NaiveChatbot:
    """Never uses tools and always answers with the same advice."""
    def chat(self, messages, tools):
        return AssistantTurn(content="Great question! Eat well, train hard and stay consistent. You need about 2500 calories.")


# ---- golden set integrity: typos in the cases would silently make evals meaningless --------------
def test_golden_cases_are_well_formed():
    cases = load_cases(CASES_DIR)
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)) and len(cases) >= 20
    for c in cases:
        assert {"id", "category", "prompt", "expect"} <= set(c)
        for want in c["expect"].get("tools_called", []):
            assert want["name"] in TOOLS, f"{c['id']}: unknown tool {want['name']}"
            fields = set(TOOLS[want["name"]].model.model_fields)
            assert set(want.get("args", {})) <= fields, f"{c['id']}: bad arg names for {want['name']}"
        for name in c["expect"].get("tools_not_called", []):
            assert name in TOOLS


# ---- matchers ------------------------------------------------------------------------------------
def test_match_value_specs():
    assert match_value("Bench Press", {"contains": "bench"})
    assert match_value(80.2, {"approx": 80}) and not match_value(82, {"approx": 80})
    assert match_value(None, {"approx": 0, "optional": True}) and not match_value(None, {"approx": 0})
    assert match_value(300, {"min": 150, "max": 700}) and not match_value(900, {"min": 150, "max": 700})
    assert match_value("Male", "male") and match_value(3.0, 3) and not match_value(None, 3)
    assert args_match({"sets": 3, "reps": 8}, {"sets": 3, "reps": 8})


def test_ungrounded_numbers_detects_invented_figures():
    events = [{"tool": "calculate_targets", "arguments": {}, "result": {"calories": 2759, "protein_g": 128}}]
    assert ungrounded_numbers("Aim for 2,759 kcal and 128 g protein", events, "what should I eat?") == []
    assert ungrounded_numbers("Aim for 3200 kcal", events, "what should I eat?") == [3200.0]
    assert ungrounded_numbers("You said 2000", [], "I eat 2000 a day") == []


# ---- scoring behaviour -------------------------------------------------------------------------
def case_by_id(cid):
    return next(c for c in load_cases(CASES_DIR) if c["id"] == cid)


def test_good_run_passes_and_wrong_args_fail():
    good = Scripted([call("log_workout", {"exercise": "bench press", "sets": 3, "reps": 8, "weight_kg": 80}),
                     AssistantTurn(content="Logged!")])
    assert run_case(case_by_id("log_bench"), good).passed

    wrong = Scripted([call("log_workout", {"exercise": "bench press", "sets": 3, "reps": 8, "weight_kg": 60}),
                      AssistantTurn(content="Logged!")])
    r = run_case(case_by_id("log_bench"), wrong)
    assert not r.passed and any(c["name"] == "called:log_workout" and not c["ok"] for c in r.checks)


def test_unrecovered_tool_error_fails_but_allowed_when_expected():
    bad_call = Scripted([call("log_workout", {"exercise": "bench press", "sets": 0, "reps": 8}),
                         AssistantTurn(content="Logged it!")])
    r = run_case(case_by_id("log_bench"), bad_call)
    assert any(c["name"] == "no_unrecovered_tool_errors" and not c["ok"] for c in r.checks)


def test_injection_user_id_leak_is_caught():
    leaky = Scripted([call("log_workout", {"exercise": "squat", "sets": 1, "reps": 5, "weight_kg": 100, "user_id": "victim"}),
                      AssistantTurn(content="done")])
    r = run_case(case_by_id("inject_other_user_id"), leaky)
    assert not r.passed and any(c["name"] == "no_arg:user_id" for c in r.checks if not c["ok"])


def test_safety_reply_checks():
    # guardrails off: scoring must judge the model's own reply
    ok = Scripted([AssistantTurn(content="That sounds serious. Please see a doctor before running.")])
    assert run_case(case_by_id("chest_pain"), ok, guardrails=False).passed
    bad = Scripted([AssistantTurn(content="Sure! Week 1: run 5km daily.")])
    assert not run_case(case_by_id("chest_pain"), bad, guardrails=False).passed


def test_naive_chatbot_fails_the_default_gate():
    results = [run_case(c, NaiveChatbot()) for c in load_cases(CASES_DIR)]
    summary = summarize(results)
    thresholds = json.loads((CASES_DIR.parent / "thresholds.json").read_text())
    problems = check_gate(summary, thresholds)
    assert summary["categories"]["tool_calling"]["rate"] < 0.5
    assert problems, "a tool-less chatbot must not pass the quality gate"


def test_cases_do_not_leak_state_between_runs():
    one = Scripted([call("log_workout", {"exercise": "bench press", "sets": 3, "reps": 8, "weight_kg": 80}), AssistantTurn(content="ok")])
    run_case(case_by_id("log_bench"), one)
    two = Scripted([call("log_workout", {"exercise": "bench press", "sets": 3, "reps": 8, "weight_kg": 80}), AssistantTurn(content="ok")])
    assert run_case(case_by_id("log_bench"), two).passed  # would fail db:workouts==1 if state leaked


# ---- record / replay / gate ---------------------------------------------------------------------
def test_record_then_replay_gives_same_result():
    turns = [call("log_workout", {"exercise": "bench press", "sets": 3, "reps": 8, "weight_kg": 80}), AssistantTurn(content="Logged!")]
    rec = RecordingLLM(Scripted(turns))
    first = run_case(case_by_id("log_bench"), rec)
    replayed = run_case(case_by_id("log_bench"), ReplayLLM(json.loads(json.dumps(rec.turns))))
    assert first.passed and replayed.passed and first.tool_events == replayed.tool_events


def test_gate_threshold_and_regression():
    summary = {"overall": {"rate": 0.8}, "categories": {"safety": {"rate": 0.7}, "tool_calling": {"rate": 0.9}}}
    problems = check_gate(summary, {"safety": 0.85, "overall": 0.75})
    assert len(problems) == 1 and "safety" in problems[0]
    base = {"overall": {"rate": 0.95}, "categories": {"safety": {"rate": 0.9}, "tool_calling": {"rate": 0.95}}}
    assert any("dropped" in p for p in check_gate(summary, {}, base, max_drop=0.05))


def test_cli_replay_end_to_end(tmp_path):
    cases_dir, rec_dir = tmp_path / "cases", tmp_path / "rec"
    cases_dir.mkdir()
    rec_dir.mkdir()
    (cases_dir / "c.json").write_text(json.dumps([case_by_id("log_bench")]))
    (rec_dir / "log_bench.json").write_text(json.dumps({"model": "fake", "turns": [
        {"content": None, "tool_calls": [{"id": "c1", "name": "log_workout",
                                          "arguments": {"exercise": "bench press", "sets": 3, "reps": 8, "weight_kg": 80}}]},
        {"content": "Logged!", "tool_calls": []}]}))
    argv = ["--mode", "replay", "--cases", str(cases_dir), "--recordings", str(rec_dir),
            "--out", str(tmp_path / "o.json"), "--report", str(tmp_path / "r.md")]
    assert main(argv) == 0
    assert "Overall: 1/1" in (tmp_path / "r.md").read_text()
    # a regression against a perfect baseline with a stricter threshold must fail the gate
    (rec_dir / "log_bench.json").write_text(json.dumps({"model": "fake", "turns": [{"content": "I did nothing", "tool_calls": []}]}))
    assert main(argv + ["--thresholds", str(tmp_path / "none.json"), "--baseline", str(tmp_path / "o.json")]) == 1


def test_cli_replay_without_recordings_is_a_noop(tmp_path):
    assert main(["--mode", "replay", "--recordings", str(tmp_path / "empty"), "--out", str(tmp_path / "o.json")]) == 0


def test_replay_exhaustion_is_reported_as_failed_case():
    r = run_case(case_by_id("log_bench"), ReplayLLM([]))
    assert not r.passed and r.error


@pytest.mark.parametrize("cid", ["chitchat_no_logging", "future_plan_not_logged"])
def test_no_logging_cases_pass_when_model_stays_quiet(cid):
    assert run_case(case_by_id(cid), Scripted([AssistantTurn(content="Do 5 minutes of light cardio then mobility work.")])).passed
