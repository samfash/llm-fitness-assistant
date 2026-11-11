"""Deterministic scoring of one agent run against a golden case's `expect` block."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date

from core import db


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class CaseScore:
    checks: list[Check]

    @property
    def passed(self) -> bool:
        return all(c.ok for c in self.checks)


def match_value(value, spec) -> bool:
    """spec is a literal (case-insensitive equality) or {contains|approx+tol|min/max|any_of, optional}."""
    if isinstance(spec, dict):
        if value is None:
            return bool(spec.get("optional"))
        if "contains" in spec:
            return str(spec["contains"]).lower() in str(value).lower()
        if "any_of" in spec:
            return str(value).lower() in [str(x).lower() for x in spec["any_of"]]
        try:
            num = float(value)
        except (TypeError, ValueError):
            return False
        if "approx" in spec:
            return abs(num - float(spec["approx"])) <= spec.get("tol", 0.5)
        if "min" in spec or "max" in spec:
            return spec.get("min", float("-inf")) <= num <= spec.get("max", float("inf"))
        return False
    if value is None:
        return False
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        try:
            return float(value) == float(spec)
        except (TypeError, ValueError):
            return False
    return str(value).lower() == str(spec).lower()


def args_match(args: dict, spec: dict) -> bool:
    return all(match_value(args.get(k), v) for k, v in spec.items())


def _args(event: dict) -> dict:
    a = event.get("arguments")
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except json.JSONDecodeError:
            return {}
    return a if isinstance(a, dict) else {}


def _numeric_leaves(obj):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield float(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _numeric_leaves(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _numeric_leaves(v)


_NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")


def ungrounded_numbers(reply: str, events: list[dict], prompt: str, min_value: float = 100) -> list[float]:
    """Numbers >= min_value in the reply that appear in neither the prompt nor any tool input/output.

    Catches the model doing its own calorie/macro arithmetic or inventing figures.
    """
    allowed = {float(x.replace(",", "")) for x in _NUM.findall(prompt)}
    for ev in events:
        allowed.update(_numeric_leaves(ev.get("result")))
        allowed.update(_numeric_leaves(_args(ev)))
    today = date.today()
    allowed.update({float(today.year), float(today.month), float(today.day)})

    bad = []
    for raw in _NUM.findall(reply):
        try:
            n = float(raw.replace(",", ""))
        except ValueError:
            continue
        if n < min_value:
            continue
        if not any(abs(n - a) <= max(1.0, 0.01 * a) for a in allowed):
            bad.append(n)
    return bad


def score_case(case: dict, result, user_id: str) -> CaseScore:
    expect = case.get("expect", {})
    events = result.tool_events
    reply = result.reply or ""
    low = reply.lower()
    calls = [(e["tool"], _args(e)) for e in events]
    checks = [Check("non_empty_reply", bool(reply.strip()))]

    for want in expect.get("tools_called", []):
        hits = [a for n, a in calls if n == want["name"] and args_match(a, want.get("args", {}))]
        need = want.get("count")
        ok = len(hits) == need if need is not None else len(hits) >= 1
        checks.append(Check(f"called:{want['name']}", ok, "" if ok else f"{len(hits)} matching call(s); calls={calls}"))

    for name in expect.get("tools_not_called", []):
        n = sum(1 for t, _ in calls if t == name)
        checks.append(Check(f"not_called:{name}", n == 0, f"called {n}x" if n else ""))

    for key in expect.get("no_arg_keys", []):
        leaked = [t for t, a in calls if key in a]
        checks.append(Check(f"no_arg:{key}", not leaked, f"model passed '{key}' to {leaked}" if leaked else ""))

    if not expect.get("allow_tool_errors"):
        last = {}
        for e in events:
            last[e["tool"]] = e["result"]
        bad = [t for t, r in last.items() if isinstance(r, dict) and "error" in r]
        checks.append(Check("no_unrecovered_tool_errors", not bad, f"final result of {bad} was an error" if bad else ""))

    if "reply_contains_any" in expect:
        ok = any(s.lower() in low for s in expect["reply_contains_any"])
        checks.append(Check("reply_contains_any", ok, "" if ok else f"none of {expect['reply_contains_any']}"))
    for s in expect.get("reply_contains_all", []):
        checks.append(Check(f"reply_contains:{s}", s.lower() in low))
    for s in expect.get("reply_not_contains", []):
        checks.append(Check(f"reply_not_contains:{s}", s.lower() not in low))

    if expect.get("grounded_numbers"):
        bad = ungrounded_numbers(reply, events, case["prompt"])
        checks.append(Check("grounded_numbers", not bad, f"ungrounded: {bad}" if bad else ""))

    if "db" in expect:
        counts = {"workouts": len(db.workouts_since(user_id, "2000-01-01")),
                  "meals": len(db.meals_since(user_id, "2000-01-01"))}
        for table, n in expect["db"].items():
            checks.append(Check(f"db:{table}=={n}", counts[table] == n, f"got {counts[table]}"))
    return CaseScore(checks)
