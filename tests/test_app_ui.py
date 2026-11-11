"""Drives the real Streamlit app headlessly (Streamlit's AppTest, same version as the Space)."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from core import auth, usage  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture
def fresh(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("ALLOW_GUEST", "1")
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "600")
    monkeypatch.setenv("RATE_LIMIT_BURST", "100")
    # the rate limiter is cached per process by st.cache_resource: start every test from a clean cache
    import streamlit as st

    st.cache_resource.clear()


def run_app():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def replies(at):
    # AppTest keeps the empty spinner container from the run that st.rerun() interrupted; a browser clears it
    return [m.markdown[0].value for m in at.chat_message if m.name == "assistant" and m.markdown]


def test_app_loads_as_a_guest_without_errors(fresh):
    at = run_app()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title[0].value.endswith("LLM Fitness Assistant")
    assert any("Guest session" in i.value for i in at.sidebar.info)
    captions = " ".join(c.value for c in at.sidebar.caption)
    assert "USDA FoodData Central" in captions  # attribution requested by the data licence


def test_chest_pain_is_blocked_by_the_guardrail_with_no_model_configured(fresh):
    at = run_app()
    at.chat_input[0].set_value("I get chest pain when I run").run()
    assert not at.exception, [e.value for e in at.exception]
    assert "medical help" in replies(at)[-1]
    assert any("Safety guardrail" in c.value for m in at.chat_message for c in m.caption)


def test_invalid_api_key_stops_the_app_with_an_error(fresh):
    at = run_app()
    at.sidebar.text_input[0].set_value("fa_not_a_real_key").run()
    assert any("Invalid or revoked" in e.value for e in at.sidebar.error)
    assert len(at.chat_input) == 0  # st.stop(): the chat box is not rendered


def test_valid_api_key_signs_in_and_revoked_key_is_rejected(fresh):
    key = auth.create_key("alice")
    at = run_app()
    at.sidebar.text_input[0].set_value(key).run()
    assert any("Signed in as alice" in s.value for s in at.sidebar.success) and len(at.chat_input) == 1
    auth.revoke_key(key[:8])
    at.sidebar.text_input[0].set_value(key).run()
    assert any("Invalid or revoked" in e.value for e in at.sidebar.error)


def test_guests_can_be_disabled(fresh, monkeypatch):
    monkeypatch.setenv("ALLOW_GUEST", "0")
    at = run_app()
    assert any("Enter an API key" in w.value for w in at.sidebar.warning) and len(at.chat_input) == 0


def test_daily_budget_stops_the_model_from_being_called(fresh, monkeypatch):
    monkeypatch.setenv("GUEST_DAILY_TOKEN_BUDGET", "100")
    usage.record(usage.GUEST_BUCKET, {"requests": 1, "prompt_tokens": 100, "completion_tokens": 0})
    at = run_app()
    at.chat_input[0].set_value("What is a good warm up?").run()
    assert "daily token budget" in replies(at)[-1]


def test_rate_limit_message_after_a_burst(fresh, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "1")
    monkeypatch.setenv("RATE_LIMIT_BURST", "1")
    import streamlit as st

    st.cache_resource.clear()
    at = run_app()
    at.chat_input[0].set_value("I get chest pain when I run").run()
    at.chat_input[0].set_value("I get chest pain when I run").run()
    assert "too quickly" in replies(at)[-1]


def test_profile_and_progress_appear_in_the_sidebar_for_a_signed_in_user(fresh):
    from core import db

    key = auth.create_key("bob")
    db.save_profile("bob", {"age": 30, "sex": "male"})
    db.add_workout("bob", {"exercise": "bench press", "sets": 3, "reps": 8, "weight_kg": 80})
    at = run_app()
    at.sidebar.text_input[0].set_value(key).run()
    assert not at.exception, [e.value for e in at.exception]
    labels = {m.label: m.value for m in at.sidebar.metric}
    assert labels["Sessions"] == "1" and labels["Volume (kg)"] in ("1920", "1,920")  # 3 x 8 x 80 kg
    assert any(j.value for j in at.sidebar.json)  # the profile renders (this crashed the app before the fix)
