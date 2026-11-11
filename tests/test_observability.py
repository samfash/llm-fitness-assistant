import json

import pytest

from core import tracing, usage
from core.agent import AssistantTurn, ToolCall, run_agent
from core.pricing import cost_usd
from core.rag_with_semantic_retriever import retrieve_with_sources


@pytest.fixture(autouse=True)
def spans():
    tracing.reset()
    ex = tracing.InMemoryExporter()
    tracing.add_exporter(ex)
    yield ex.spans
    tracing.reset()


class Scripted:
    model = "test/model"

    def __init__(self, turns):
        self.turns = list(turns)

    def chat(self, messages, tools):
        return self.turns.pop(0)


def by_name(spans, name):
    return [s for s in spans if s.name == name]


# ---- tracer ---------------------------------------------------------------------------------------
def test_nested_spans_link_parent_and_share_trace_id(spans):
    with tracing.span("root") as r:
        with tracing.span("child", k=1) as c:
            c.event("something", n=2)
    child, root = spans
    assert child.parent_id == root.span_id == r.span_id and child.trace_id == root.trace_id
    assert child.attrs == {"k": 1} and child.events[0]["name"] == "something" and root.parent_id is None
    assert child.duration_ms >= 0 and root.end_ns >= child.end_ns


def test_error_status_and_reraise(spans):
    with pytest.raises(ValueError), tracing.span("boom"):
        raise ValueError("bad thing")
    assert spans[0].status == "error" and "ValueError: bad thing" in spans[0].error


def test_exporter_failure_never_breaks_the_request(spans):
    class Broken:
        def on_end(self, s):
            raise RuntimeError("exporter down")

    tracing.add_exporter(Broken())
    with tracing.span("ok"):
        pass
    assert [s.name for s in spans] == ["ok"]


def test_jsonl_exporter_writes_one_line_per_span(tmp_path, monkeypatch):
    path = tmp_path / "sub" / "t.jsonl"
    monkeypatch.setenv("TRACE_FILE", str(path))
    tracing.reset()
    with tracing.span("a"), tracing.span("b"):
        pass
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    assert [x["name"] for x in lines] == ["b", "a"] and lines[0]["parent_id"] == lines[1]["span_id"]


def test_hash_id_is_stable_and_hides_the_value():
    assert tracing.hash_id("alice") == tracing.hash_id("alice") != tracing.hash_id("bob")
    assert "alice" not in tracing.hash_id("alice") and len(tracing.hash_id("alice")) == 12


# ---- OpenTelemetry bridge (real SDK, in-memory exporter) -------------------------------------------
def test_otel_exporter_preserves_hierarchy_attributes_events_and_errors():
    pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    mem = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(mem))
    tracing.add_exporter(tracing.OTelExporter(provider))
    with pytest.raises(RuntimeError), tracing.span("parent", user="u1", n=3, listy=[1, 2]):
        with tracing.span("child") as c:
            c.event("hit", doc="x")
        raise RuntimeError("fail")
    done = {s.name: s for s in mem.get_finished_spans()}
    assert set(done) == {"parent", "child"}
    assert done["child"].parent.span_id == done["parent"].context.span_id
    assert done["child"].context.trace_id == done["parent"].context.trace_id
    assert done["parent"].attributes["n"] == 3 and done["parent"].attributes["listy"] == "[1, 2]"
    assert done["parent"].status.status_code.name == "ERROR" and done["child"].events[0].name == "hit"
    assert done["child"].end_time >= done["child"].start_time


# ---- pricing ------------------------------------------------------------------------------------------
def test_cost_is_none_without_a_configured_price(monkeypatch):
    assert cost_usd("test/model", 1000, 1000) is None


def test_cost_from_env_json_and_file(monkeypatch, tmp_path):
    prices = {"test/model": {"input_per_mtok": 2.0, "output_per_mtok": 6.0}}
    monkeypatch.setenv("MODEL_PRICES", json.dumps(prices))
    assert cost_usd("test/model", 500_000, 100_000) == pytest.approx(1.0 + 0.6)
    assert cost_usd("other/model", 1, 1) is None
    f = tmp_path / "prices.json"
    f.write_text(json.dumps(prices))
    monkeypatch.setenv("MODEL_PRICES", str(f))
    assert cost_usd("test/model", 1_000_000, 0) == 2.0


def test_bad_price_config_degrades_to_unknown(monkeypatch):
    monkeypatch.setenv("MODEL_PRICES", "{not json")
    assert cost_usd("test/model", 10, 10) is None
    monkeypatch.setenv("MODEL_PRICES", json.dumps({"test/model": {"input_per_mtok": "free"}}))
    assert cost_usd("test/model", 10, 10) is None


# ---- agent instrumentation ----------------------------------------------------------------------------
def _log_turns():
    return [AssistantTurn(tool_calls=[ToolCall("c1", "log_workout", {"exercise": "bench press", "sets": 3, "reps": 8, "weight_kg": 80})],
                          usage={"prompt_tokens": 1000, "completion_tokens": 50}),
            AssistantTurn(content="Logged!", usage={"prompt_tokens": 1200, "completion_tokens": 10})]


def test_agent_run_produces_a_span_tree_with_token_counts(spans, monkeypatch):
    monkeypatch.setenv("MODEL_PRICES", json.dumps({"test/model": {"input_per_mtok": 1.0, "output_per_mtok": 2.0}}))
    res = run_agent("alice", "bench 80kg 3x8", Scripted(_log_turns()))
    root = by_name(spans, "agent.run")[0]
    llm, tool = by_name(spans, "llm.chat"), by_name(spans, "tool.log_workout")[0]
    assert len(llm) == 2 and all(s.parent_id == root.span_id for s in llm + [tool])
    assert tool.attrs["ok"] is True and tool.attrs["tool"] == "log_workout"
    assert root.attrs["prompt_tokens"] == 2200 and root.attrs["completion_tokens"] == 60 and root.attrs["steps"] == 2
    assert res.usage == {"requests": 2, "prompt_tokens": 2200, "completion_tokens": 60,
                         "cost_usd": pytest.approx((2200 * 1 + 60 * 2) / 1e6), "estimated": False}
    assert root.attrs["cost_usd"] == pytest.approx(res.usage["cost_usd"])


def test_missing_provider_usage_is_estimated_and_flagged(spans):
    res = run_agent("u", "hello there", Scripted([AssistantTurn(content="Hi! How can I help you today?")]))
    assert res.usage["estimated"] is True and res.usage["prompt_tokens"] > 0 and res.usage["cost_usd"] is None
    assert by_name(spans, "llm.chat")[0].attrs["estimated"] is True


def test_blocked_input_costs_nothing_and_makes_no_llm_span(spans):
    res = run_agent("u", "I get chest pain when I run", Scripted([]))
    assert res.usage["requests"] == 0 and by_name(spans, "llm.chat") == []
    root = by_name(spans, "agent.run")[0]
    assert root.attrs["blocked_before_model"] is True and "input:emergency:block" in root.attrs["guardrail_events"]


def test_failed_tool_call_is_marked_not_ok(spans):
    run_agent("u", "x", Scripted([AssistantTurn(tool_calls=[ToolCall("c", "log_workout", {"exercise": "sq", "sets": 0})]),
                                  AssistantTurn(content="sorry")]))
    assert by_name(spans, "tool.log_workout")[0].attrs["ok"] is False


def test_privacy_defaults_no_content_no_raw_user_id(spans, monkeypatch):
    run_agent("alice@example.com", "my secret message", Scripted([AssistantTurn(content="a secret reply")]))
    blob = json.dumps([s.to_dict() for s in spans])
    assert "alice@example.com" not in blob and "secret" not in blob
    spans.clear()
    monkeypatch.setenv("TRACE_CONTENT", "1")
    run_agent("alice@example.com", "my secret message", Scripted([AssistantTurn(content="a secret reply")]))
    root = by_name(spans, "agent.run")[0]
    assert root.attrs["input"] == "my secret message" and "alice@example.com" not in json.dumps(root.attrs)


def test_retrieval_span(spans):
    retrieve_with_sources("how much protein do I need to build muscle")
    sp = by_name(spans, "rag.retrieve")[0]
    assert sp.attrs["mode"] == "bm25" and sp.attrs["hits"] >= 1 and sp.attrs["top_doc"] == "protein-intake"


# ---- usage store & budgets (runs on both database backends) ----------------------------------------------
def test_usage_accumulates_per_user_and_day():
    usage.record("u", {"requests": 1, "prompt_tokens": 100, "completion_tokens": 20, "cost_usd": 0.5}, "2026-01-01")
    usage.record("u", {"requests": 2, "prompt_tokens": 10, "completion_tokens": 5, "cost_usd": 0.25}, "2026-01-01")
    usage.record("u", {"requests": 1, "prompt_tokens": 1, "completion_tokens": 1, "cost_usd": None}, "2026-01-02")
    assert usage.get("u", "2026-01-01") == {"requests": 3, "prompt_tokens": 110, "completion_tokens": 25, "cost_usd": 0.75}
    assert usage.get("u", "2026-01-02")["cost_usd"] is None  # unknown stays unknown, not 0
    assert usage.get("other", "2026-01-01")["requests"] == 0


def test_repeated_unknown_costs_stay_unknown_not_zero():
    for _ in range(3):
        usage.record("u", {"requests": 1, "prompt_tokens": 5, "completion_tokens": 5, "cost_usd": None}, "2026-01-01")
    got = usage.get("u", "2026-01-01")
    assert got["requests"] == 3 and got["prompt_tokens"] == 15 and got["cost_usd"] is None


def test_unknown_cost_does_not_erase_known_cost():
    usage.record("u", {"requests": 1, "prompt_tokens": 1, "completion_tokens": 1, "cost_usd": 0.1}, "2026-01-01")
    usage.record("u", {"requests": 1, "prompt_tokens": 1, "completion_tokens": 1, "cost_usd": None}, "2026-01-01")
    assert usage.get("u", "2026-01-01")["cost_usd"] == pytest.approx(0.1)


def test_budget_enforcement_and_guest_bucket(monkeypatch):
    monkeypatch.setenv("DAILY_TOKEN_BUDGET", "100")
    monkeypatch.setenv("GUEST_DAILY_TOKEN_BUDGET", "50")
    assert usage.within_budget("u")[0]
    usage.record("u", {"requests": 1, "prompt_tokens": 90, "completion_tokens": 10})
    ok, info = usage.within_budget("u")
    assert not ok and info["used_tokens"] == 100 and info["budget_tokens"] == 100
    usage.record(usage.GUEST_BUCKET, {"requests": 1, "prompt_tokens": 60, "completion_tokens": 0})
    assert not usage.within_budget("guest-abc", guest=True)[0]  # guests share one bucket
    assert usage.within_budget("someone-else")[0]
    monkeypatch.setenv("DAILY_TOKEN_BUDGET", "0")  # 0 = unlimited
    assert usage.within_budget("u")[0]


def test_chat_records_usage_for_the_right_bucket(monkeypatch):
    import core.agent as agent

    monkeypatch.setattr(agent, "HF_TOKEN", "x")
    monkeypatch.setattr(agent, "HFToolClient", lambda: Scripted(_log_turns()))
    res = agent.chat("alice", "bench 80kg 3x8", usage_user=usage.GUEST_BUCKET)
    assert res.reply == "Logged!"
    assert usage.get(usage.GUEST_BUCKET)["prompt_tokens"] == 2200 and usage.get("alice")["requests"] == 0


def test_fallback_path_without_a_token_is_traced_too(spans, monkeypatch):
    import core.agent as agent

    monkeypatch.setattr(agent, "HF_TOKEN", None)
    res = agent.chat("alice", "I get chest pain when I run")
    sp = by_name(spans, "agent.fallback")[0]
    assert res.guardrails[0]["rule"] == "emergency" and sp.attrs["blocked_before_model"] is True
    assert "input:emergency:block" in sp.attrs["guardrail_events"] and "alice" not in json.dumps(sp.attrs)
    monkeypatch.setattr(agent, "generate_response", lambda *a, **k: "Warm up for ten minutes.")
    ok = agent.chat("alice", "what is a good warm up")
    assert "Warm up" in ok.reply and by_name(spans, "agent.fallback")[-1].attrs["blocked_before_model"] is False
