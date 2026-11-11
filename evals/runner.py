"""Run golden cases through the agent, aggregate metrics, render a report, apply a quality gate."""
from __future__ import annotations

import json
import os
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from core import db
from core.agent import AssistantTurn, ToolCall, run_agent
from core.schemas import MealEntry, WorkoutEntry

from .scoring import score_case

EVAL_USER = "eval-user"


# ---- LLM wrappers: record a live run once, replay it deterministically in CI -------------------
class RecordingLLM:
    def __init__(self, inner):
        self.inner, self.turns = inner, []

    def chat(self, messages, tools):
        turn = self.inner.chat(messages, tools)
        self.turns.append({
            "content": turn.content,
            "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in turn.tool_calls],
        })
        return turn


class ReplayLLM:
    def __init__(self, turns):
        self.turns = list(turns)

    def chat(self, messages, tools):
        if not self.turns:
            raise RuntimeError("Recording exhausted: agent made more model calls than were recorded")
        t = self.turns.pop(0)
        return AssistantTurn(t["content"], [ToolCall(c["id"], c["name"], c["arguments"]) for c in t["tool_calls"]])


# ---- cases ---------------------------------------------------------------------------------------
@dataclass
class CaseResult:
    id: str
    category: str
    passed: bool
    checks: list[dict]
    reply: str = ""
    tool_events: list[dict] = field(default_factory=list)
    latency_s: float = 0.0
    error: str | None = None


def load_cases(path: Path) -> list[dict]:
    cases = []
    for f in sorted(Path(path).glob("*.json")):
        cases.extend(json.loads(f.read_text()))
    return cases


def _seed(setup: dict) -> None:
    if setup.get("profile"):
        db.save_profile(EVAL_USER, setup["profile"])
    for w in setup.get("workouts", []):
        db.add_workout(EVAL_USER, WorkoutEntry(**w).model_dump())
    for m in setup.get("meals", []):
        db.add_meal(EVAL_USER, MealEntry(**m).model_dump())


def run_case(case: dict, llm) -> CaseResult:
    """Fresh temporary database per case so cases cannot contaminate each other."""
    old = os.environ.get("FITNESS_DB_PATH")
    with tempfile.TemporaryDirectory() as td:
        os.environ["FITNESS_DB_PATH"] = str(Path(td) / "eval.db")
        try:
            _seed(case.get("setup", {}))
            t0 = time.perf_counter()
            try:
                res = run_agent(EVAL_USER, case["prompt"], llm)
            except Exception as e:  # provider/network errors count as a failed case, not a crashed run
                return CaseResult(case["id"], case["category"], False,
                                  [{"name": "agent_ran", "ok": False, "detail": repr(e)}], error=repr(e),
                                  latency_s=time.perf_counter() - t0)
            latency = time.perf_counter() - t0
            score = score_case(case, res, EVAL_USER)
            return CaseResult(case["id"], case["category"], score.passed, [asdict(c) for c in score.checks],
                              res.reply, res.tool_events, latency)
        finally:
            if old is None:
                os.environ.pop("FITNESS_DB_PATH", None)
            else:
                os.environ["FITNESS_DB_PATH"] = old


# ---- aggregation / report / gate -----------------------------------------------------------------
def summarize(results: list[CaseResult]) -> dict:
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r.category].append(r)
    cats = {c: {"n": len(rs), "passed": sum(r.passed for r in rs), "rate": sum(r.passed for r in rs) / len(rs)}
            for c, rs in by_cat.items()}
    n = len(results)
    lat = sorted(r.latency_s for r in results)
    failed_checks = Counter(c["name"].split(":")[0] for r in results for c in r.checks if not c["ok"])
    return {
        "overall": {"n": n, "passed": sum(r.passed for r in results), "rate": (sum(r.passed for r in results) / n) if n else 0.0},
        "categories": cats,
        "failed_check_types": dict(failed_checks),
        "avg_tool_calls": round(statistics.mean(len(r.tool_events) for r in results), 2) if n else 0,
        "latency_p50_s": round(statistics.median(lat), 2) if lat else 0,
        "latency_p95_s": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 2) if lat else 0,
    }


def render_report(summary: dict, results: list[CaseResult], meta: dict) -> str:
    o = summary["overall"]
    lines = [
        "# Eval report", "",
        f"- Mode: `{meta.get('mode')}`  |  Model: `{meta.get('model', 'n/a')}`  |  Date: {meta.get('date', '')}",
        f"- **Overall: {o['passed']}/{o['n']} ({o['rate']:.0%})**  |  avg tool calls/case: {summary['avg_tool_calls']}"
        f"  |  latency p50/p95: {summary['latency_p50_s']}s / {summary['latency_p95_s']}s", "",
        "| Category | Passed | Rate |", "|---|---|---|",
    ]
    for c, s in sorted(summary["categories"].items()):
        lines.append(f"| {c} | {s['passed']}/{s['n']} | {s['rate']:.0%} |")
    failures = [r for r in results if not r.passed]
    if failures:
        lines += ["", "## Failures", ""]
        for r in failures:
            lines.append(f"### `{r.id}` ({r.category})")
            for c in r.checks:
                if not c["ok"]:
                    lines.append(f"- ❌ **{c['name']}** {c['detail']}")
            lines.append(f"- reply: {r.reply[:300]!r}")
            lines.append("")
    return "\n".join(lines)


def check_gate(summary: dict, thresholds: dict, baseline: dict | None = None, max_drop: float = 0.05) -> list[str]:
    problems = []
    rates = {c: s["rate"] for c, s in summary["categories"].items()}
    rates["overall"] = summary["overall"]["rate"]
    for name, minimum in thresholds.items():
        if name in rates and rates[name] < minimum:
            problems.append(f"{name}: {rates[name]:.0%} is below the required {minimum:.0%}")
    if baseline:
        base = {c: s["rate"] for c, s in baseline["categories"].items()}
        base["overall"] = baseline["overall"]["rate"]
        for name, prev in base.items():
            if name in rates and rates[name] < prev - max_drop:
                problems.append(f"{name}: dropped {prev:.0%} -> {rates[name]:.0%} (max allowed drop {max_drop:.0%})")
    return problems
