"""Per-user daily usage (requests, tokens, cost) and the daily token budget. Stored via core.db (SQLite or Postgres)."""
from __future__ import annotations

import os
from datetime import UTC, datetime

from . import db

GUEST_BUCKET = "__guests__"  # all anonymous sessions share one budget so they cannot multiply it


def today() -> str:
    return datetime.now(UTC).date().isoformat()  # budgets reset at 00:00 UTC


def daily_budget(guest: bool = False) -> int:
    """Tokens per user per day; 0 means unlimited."""
    name, default = ("GUEST_DAILY_TOKEN_BUDGET", "100000") if guest else ("DAILY_TOKEN_BUDGET", "200000")
    try:
        return max(0, int(os.getenv(name, default)))
    except ValueError:
        return int(default)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)  # rough rule of thumb, only used when the provider reports no usage


def record(user_id: str, usage: dict, day: str | None = None) -> None:
    with db.connect() as c:
        c.execute(
            "INSERT INTO usage(user_id, day, requests, prompt_tokens, completion_tokens, cost_usd) VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(user_id, day) DO UPDATE SET requests = usage.requests + excluded.requests, "
            "prompt_tokens = usage.prompt_tokens + excluded.prompt_tokens, "
            "completion_tokens = usage.completion_tokens + excluded.completion_tokens, "
            "cost_usd = CASE WHEN usage.cost_usd IS NULL AND excluded.cost_usd IS NULL THEN NULL "
            "ELSE COALESCE(usage.cost_usd, 0) + COALESCE(excluded.cost_usd, 0) END",
            (user_id, day or today(), usage.get("requests", 0), usage.get("prompt_tokens", 0),
             usage.get("completion_tokens", 0), usage.get("cost_usd")),
        )


def get(user_id: str, day: str | None = None) -> dict:
    with db.connect() as c:
        row = c.execute("SELECT requests, prompt_tokens, completion_tokens, cost_usd FROM usage "
                        "WHERE user_id=? AND day=?", (user_id, day or today())).fetchone()
    base = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": None}
    return {**base, **dict(row)} if row else base


def within_budget(user_id: str, guest: bool = False) -> tuple[bool, dict]:
    """Return (allowed, details). Checked before a request; one request may overshoot the cap slightly."""
    cap = daily_budget(guest)
    used = get(GUEST_BUCKET if guest else user_id)
    total = used["prompt_tokens"] + used["completion_tokens"]
    return (cap == 0 or total < cap), {"used_tokens": total, "budget_tokens": cap, **used}
