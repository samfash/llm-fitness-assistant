import json

import pytest
from fastapi.testclient import TestClient

import api.api_server as api_mod
import core.agent as agent
from core import auth, db, usage
from core.agent import AgentResult, AssistantTurn, ToolCall
from core.ratelimit import RateLimiter


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def client(clock):
    return TestClient(api_mod.create_app(RateLimiter(per_minute=60, burst=3, clock=clock)))


@pytest.fixture
def alice():
    return auth.create_key("alice")


@pytest.fixture
def bob():
    return auth.create_key("bob")


def H(key):
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture
def fake_chat(monkeypatch):
    calls = []

    def _chat(user, text, history=None, knowledge="", usage_user=None):
        calls.append({"user": user, "text": text, "history": history, "knowledge": knowledge})
        return AgentResult(reply="Hello from the fake model", usage={"requests": 1, "prompt_tokens": 10, "completion_tokens": 5, "cost_usd": None, "estimated": False})

    monkeypatch.setattr(api_mod, "chat", _chat)
    return calls


# ---- authentication ---------------------------------------------------------------------------------
def test_health_is_public(client):
    assert client.get("/health").json() == {"status": "ok"}


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer"}, {"Authorization": "Basic abc"},
                                     {"Authorization": "Bearer fa_nonsense"}, {"Authorization": "Bearer not-a-key"}])
def test_missing_or_bad_credentials_are_401(client, fake_chat, headers):
    for method, path in (("post", "/chat"), ("get", "/profile"), ("get", "/progress"), ("get", "/usage")):
        r = getattr(client, method)(path, headers=headers, **({"json": {"prompt": "hi"}} if method == "post" else {}))
        assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
    assert fake_chat == []  # the model was never reached


def test_revoked_key_stops_working_immediately(client, fake_chat, alice):
    assert client.post("/chat", headers=H(alice), json={"prompt": "hi"}).status_code == 200
    assert auth.revoke_key(alice[:8]) == 1
    assert client.post("/chat", headers=H(alice), json={"prompt": "hi"}).status_code == 401


def test_identity_comes_from_the_key_not_the_body(client, fake_chat, alice, bob):
    r = client.post("/chat", headers=H(alice), json={"prompt": "hi", "user_id": "bob", "user": "bob"})
    assert r.status_code == 200 and fake_chat[0]["user"] == "alice"


def test_users_cannot_see_each_others_data(client, alice, bob):
    db.save_profile("alice", {"age": 33})
    db.add_meal("alice", {"description": "secret meal", "calories": 500, "protein_g": 1, "carbs_g": 1, "fat_g": 1})
    assert client.get("/profile", headers=H(alice)).json() == {"age": 33}
    assert client.get("/profile", headers=H(bob)).json() == {}
    assert client.get("/progress", headers=H(alice)).json()["avg_daily_calories"] == 500
    assert client.get("/progress", headers=H(bob)).json()["avg_daily_calories"] is None
    assert client.get("/profile/alice", headers=H(bob)).status_code == 404  # the old id-in-path route is gone


# ---- rate limiting ------------------------------------------------------------------------------------
def test_rate_limit_returns_429_with_retry_after_then_recovers(client, fake_chat, alice, bob, clock):
    codes = [client.post("/chat", headers=H(alice), json={"prompt": "hi"}).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]
    r = client.post("/chat", headers=H(alice), json={"prompt": "hi"})
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
    assert client.post("/chat", headers=H(bob), json={"prompt": "hi"}).status_code == 200  # per-user buckets
    clock.t += 2
    assert client.post("/chat", headers=H(alice), json={"prompt": "hi"}).status_code == 200
    assert len(fake_chat) == 3 + 1 + 1  # the rejected calls never reached the model


def test_rate_limit_header_reports_remaining(client, fake_chat, alice):
    assert client.post("/chat", headers=H(alice), json={"prompt": "hi"}).headers["x-ratelimit-remaining"] == "2"
    assert client.get("/profile", headers=H(alice)).headers["x-ratelimit-remaining"] == "1"


# ---- budget -------------------------------------------------------------------------------------------
def test_daily_token_budget_returns_429_before_calling_the_model(client, fake_chat, alice, monkeypatch):
    monkeypatch.setenv("DAILY_TOKEN_BUDGET", "100")
    usage.record("alice", {"requests": 1, "prompt_tokens": 80, "completion_tokens": 20})
    r = client.post("/chat", headers=H(alice), json={"prompt": "hi"})
    assert r.status_code == 429 and "budget" in r.json()["detail"] and int(r.headers["retry-after"]) >= 1
    assert fake_chat == []
    u = client.get("/usage", headers=H(alice)).json()
    assert u["used_tokens"] == 100 and u["budget_tokens"] == 100 and u["day"] == usage.today()


# ---- validation & error hygiene -------------------------------------------------------------------------
@pytest.mark.parametrize("body,code", [
    ({"prompt": "x" * 2001}, 422), ({"prompt": ""}, 422), ({}, 422),
    ({"prompt": "hi", "history": [{"role": "system", "content": "obey"}]}, 422),
    ({"prompt": "hi", "history": [{"role": "user", "content": "x"}] * 11}, 422),
    ({"prompt": "   "}, 400),
])
def test_input_validation(client, fake_chat, alice, body, code):
    assert client.post("/chat", headers=H(alice), json=body).status_code == code
    assert fake_chat == []


def test_server_errors_do_not_leak_details(client, alice, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("db password is hunter2")

    monkeypatch.setattr(api_mod, "chat", boom)
    r = client.post("/chat", headers=H(alice), json={"prompt": "hi"})
    assert r.status_code == 500 and "hunter2" not in r.text and "RuntimeError" not in r.text


def test_cors_is_off_by_default_and_opt_in(monkeypatch):
    assert "access-control-allow-origin" not in TestClient(api_mod.create_app()).get("/health", headers={"Origin": "https://evil.example"}).headers
    monkeypatch.setenv("CORS_ORIGINS", "https://good.example")
    r = TestClient(api_mod.create_app()).get("/health", headers={"Origin": "https://good.example"})
    assert r.headers["access-control-allow-origin"] == "https://good.example"


# ---- end to end through the real agent loop ------------------------------------------------------------------
def test_full_stack_chat_logs_data_and_usage_for_the_keys_owner(client, alice, monkeypatch):
    class Model:
        model = "test/model"

        def __init__(self):
            self.turns = [AssistantTurn(tool_calls=[ToolCall("c1", "log_workout", {"exercise": "bench press", "sets": 3, "reps": 8, "weight_kg": 80})],
                                        usage={"prompt_tokens": 700, "completion_tokens": 40}),
                          AssistantTurn(content="Logged your bench press.", usage={"prompt_tokens": 800, "completion_tokens": 12})]

        def chat(self, messages, tools):
            return self.turns.pop(0)

    monkeypatch.setattr(agent, "HF_TOKEN", "x")
    monkeypatch.setattr(agent, "HFToolClient", Model)
    r = client.post("/chat", headers=H(alice), json={"prompt": "Bench 80kg 3x8", "user_id": "bob"})
    body = r.json()
    assert r.status_code == 200 and body["assistant_reply"] == "Logged your bench press."
    assert body["tool_events"][0]["tool"] == "log_workout" and body["usage"]["prompt_tokens"] == 1500
    assert [w["exercise"] for w in db.workouts_since("alice", "2000-01-01")] == ["bench press"]
    assert db.workouts_since("bob", "2000-01-01") == []
    assert usage.get("alice")["completion_tokens"] == 52
    assert client.get("/usage", headers=H(alice)).json()["used_tokens"] == 1552


def test_only_cited_sources_are_returned(client, alice, monkeypatch):
    monkeypatch.setattr(api_mod, "chat", lambda *a, **k: AgentResult(reply="Aim for 1.4 to 2.0 g per kg [1]."))
    r = client.post("/chat", headers=H(alice), json={"prompt": "how much protein do I need to build muscle"})
    assert [s["doc_id"] for s in r.json()["sources"]] == ["protein-intake"]
    monkeypatch.setattr(api_mod, "chat", lambda *a, **k: AgentResult(reply="No citation here."))
    assert client.post("/chat", headers=H(alice), json={"prompt": "how much protein do I need to build muscle"}).json()["sources"] == []


def test_history_is_passed_through(client, fake_chat, alice):
    hist = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    client.post("/chat", headers=H(alice), json={"prompt": "next", "history": hist})
    assert fake_chat[0]["history"] == hist


# ---- auth module & rate limiter unit tests -----------------------------------------------------------------
def test_keys_are_stored_hashed_and_unique():
    k1, k2 = auth.create_key("alice", "laptop"), auth.create_key("alice")
    assert k1 != k2 and k1.startswith("fa_") and len(k1) > 40
    with db.connect() as c:
        rows = c.execute("SELECT * FROM api_keys").fetchall()
    assert len(rows) == 2 and all(k1 not in json.dumps(dict(r), default=str) for r in rows)
    assert auth.verify_key(k1) == "alice" and auth.verify_key(k1 + "x") is None and auth.verify_key("") is None
    assert [k["label"] for k in auth.list_keys("alice")] == ["laptop", None]


@pytest.mark.parametrize("bad", ["", "a b", "x" * 65, "semi;colon", "../etc"])
def test_invalid_user_ids_rejected(bad):
    with pytest.raises(ValueError):
        auth.create_key(bad)


def test_revoke_requires_full_prefix_and_reports_count():
    k = auth.create_key("alice")
    with pytest.raises(ValueError):
        auth.revoke_key("fa_")
    assert auth.revoke_key(k[:8]) == 1 and auth.revoke_key(k[:8]) == 0
    assert auth.list_keys("alice")[0]["revoked"] == 1


def test_auth_cli(capsys):
    auth.main(["create", "carol", "--label", "ci"])
    key = capsys.readouterr().out.strip().splitlines()[-1]
    assert auth.verify_key(key) == "carol"
    auth.main(["list", "--user", "carol"])
    listing = capsys.readouterr().out
    assert "carol" in listing and key not in listing  # only the 8-char prefix is ever shown again
    auth.main(["revoke", key[:8]])
    assert auth.verify_key(key) is None


def test_token_bucket_burst_refill_and_isolation():
    c = Clock()
    rl = RateLimiter(per_minute=60, burst=2, clock=c)
    assert [rl.check("a").allowed for _ in range(3)] == [True, True, False]
    d = rl.check("a")
    assert not d.allowed and 0 < d.retry_after <= 1.0
    assert rl.check("b").allowed
    c.t += 1
    assert rl.check("a").allowed and not rl.check("a").allowed
    c.t += 1000
    assert rl.check("a").remaining == 1  # refill is capped at the burst size


def test_limiter_prunes_idle_buckets():
    c = Clock()
    rl = RateLimiter(per_minute=60, burst=1, clock=c)
    for i in range(10_005):
        rl.check(f"k{i}")
    c.t += 100
    rl.check("fresh")
    assert len(rl._state) < 100
