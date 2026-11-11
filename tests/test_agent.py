import json

from core import db
from core.agent import AssistantTurn, ToolCall, run_agent


class FakeLLM:
    """Scripted model: returns the queued turns in order and records what it was sent."""
    def __init__(self, turns):
        self.turns, self.calls = list(turns), []

    def chat(self, messages, tools):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        return self.turns.pop(0)


def test_agent_logs_then_answers():
    llm = FakeLLM([
        AssistantTurn(tool_calls=[ToolCall("c1", "log_workout", {"exercise": "bench press", "sets": 3, "reps": 8, "weight_kg": 80})]),
        AssistantTurn(content="Logged your bench press. Nice work!"),
    ])
    res = run_agent("u1", "bench 80kg 3x8", llm)
    assert res.reply.startswith("Logged")
    assert res.tool_events[0]["result"]["saved"] is True
    assert len(db.workouts_since("u1", "2000-01-01")) == 1
    tool_msg = llm.calls[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "c1"


def test_model_cannot_override_user_id():
    llm = FakeLLM([
        AssistantTurn(tool_calls=[ToolCall("c1", "log_workout", {"exercise": "squat", "sets": 1, "reps": 5, "user_id": "victim"})]),
        AssistantTurn(content="done"),
    ])
    run_agent("real_user", "squat", llm)
    assert db.workouts_since("victim", "2000-01-01") == []
    assert len(db.workouts_since("real_user", "2000-01-01")) == 1


def test_bad_arguments_feed_back_error_and_model_recovers():
    llm = FakeLLM([
        AssistantTurn(tool_calls=[ToolCall("c1", "log_meal", "{broken json")]),
        AssistantTurn(tool_calls=[ToolCall("c2", "log_meal", json.dumps({"description": "oats", "calories": 350}))]),
        AssistantTurn(content="Saved your oats."),
    ])
    res = run_agent("u1", "oats", llm)
    assert "error" in res.tool_events[0]["result"] and res.tool_events[1]["result"]["saved"]
    assert res.reply == "Saved your oats."


def test_step_limit_forces_plain_answer_without_tools():
    looping = AssistantTurn(tool_calls=[ToolCall("c", "estimate_1rm", {"weight_kg": 50, "reps": 5})])
    llm = FakeLLM([looping, looping, AssistantTurn(content="Final answer")])
    res = run_agent("u1", "loop", llm, max_steps=3)
    assert res.reply == "Final answer"
    assert llm.calls[-1]["tools"] is None  # last step offered no tools
