"""Pre-flight check for local runs and for going live.

    python -m core.doctor                 # check configuration, database, data files
    python -m core.doctor --production    # stricter: missing token / SQLite / unlimited budget become failures
    python -m core.doctor --live          # also call the Hugging Face API once (checks token permission + tool calling)

Exit code is 1 if any check FAILS. Secrets are never printed (database passwords are masked).
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

from . import db
from .env import PLACEHOLDER, last_result, load_env

OK, WARN, FAIL, INFO = "ok", "warn", "fail", "info"
REQUIRED_FOR_LIVE = ("HF_TOKEN", "DATABASE_URL")
LOCAL_HOSTS = {None, "", "localhost", "127.0.0.1", "::1"}


@dataclass
class Check:
    status: str
    name: str
    detail: str
    fix: str = ""


def redact_url(url: str) -> str:
    """postgresql://user:secret@host/db -> postgresql://user:***@host/db"""
    p = urlparse(url)
    if p.password:
        return url.replace(f":{p.password}@", ":***@", 1)
    return url


def _strict(production: bool, status_if_prod: str = FAIL) -> str:
    return status_if_prod if production else WARN


# ----------------------------------------------------------------------------------------------- env file
def check_env_file(production: bool) -> list[Check]:
    r = last_result()
    out = [Check(INFO, ".env file", f"loaded {r.path} ({len(r.applied)} value(s) applied)") if r.path else
           Check(INFO, ".env file", "none found: using real environment variables only (normal on a Hugging Face Space)")]
    for name in r.placeholders:
        if os.environ.get(name):  # a real value comes from the environment, which wins over the file
            out.append(Check(INFO, f"{name} placeholder", f"{name} in .env is a placeholder, but a real value is set in the environment (it wins)"))
            continue
        required = name in REQUIRED_FOR_LIVE
        out.append(Check(_strict(production) if required else INFO, f"{name} placeholder",
                         f"{name} in .env still says {PLACEHOLDER}, so it is treated as not set",
                         "replace it with your real value, or delete the line" if required else "optional: replace or delete the line"))
    return out


# ----------------------------------------------------------------------------------------------- database
def assess_url(url: str) -> list[Check]:
    """Static checks on a DATABASE_URL (no connection made)."""
    out: list[Check] = []
    if not url.startswith(("postgres://", "postgresql://")):
        return [Check(FAIL, "DATABASE_URL format", "must start with postgresql://",
                      "copy the connection string from Neon: Console, Connect")]
    p = urlparse(url)
    ssl = parse_qs(p.query).get("sslmode", [None])[0]
    if p.hostname not in LOCAL_HOSTS and ssl not in ("require", "verify-ca", "verify-full"):
        out.append(Check(WARN, "database TLS", "no sslmode=require in the connection string",
                         "add ?sslmode=require to DATABASE_URL (Neon requires encrypted connections)"))
    if p.hostname and "-pooler" in p.hostname:
        out.append(Check(INFO, "database endpoint", "pooled endpoint (PgBouncer, transaction mode): supported"))
    elif p.hostname and p.hostname not in LOCAL_HOSTS:
        out.append(Check(INFO, "database endpoint", "direct endpoint: fine for one app instance; use the -pooler host if you run many"))
    return out


def check_database(production: bool) -> list[Check]:
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        path = os.getenv("FITNESS_DB_PATH", "data/fitness.db")
        out = [Check(_strict(production), "database", f"using SQLite at {path}: data is lost when a free Space restarts",
                     "set DATABASE_URL to a Neon connection string to keep data")]
    else:
        out = assess_url(url)
        if any(c.status == FAIL for c in out):
            return out
        try:  # quick reachability test first, so a wrong URL fails in seconds
            import psycopg

            t0 = time.perf_counter()
            with psycopg.connect(url, connect_timeout=10) as probe:
                version = probe.execute("SHOW server_version").fetchone()[0]
            first = time.perf_counter() - t0
        except Exception as e:
            return [*out, Check(FAIL, "database connection", f"cannot connect to {redact_url(url)}: {type(e).__name__}: {str(e)[:160]}",
                                "check the host, user, password and that the project is not deleted; copy a fresh string from Neon")]
        out.append(Check(OK, "database connection", f"PostgreSQL {version} reachable in {first:.2f}s"))
        if first > 1.5:
            out.append(Check(INFO, "database wake-up", "slow first connection: a suspended Neon compute was waking up (normal after 5 idle minutes)"))
    try:
        with db.connect() as c:
            applied = [r["version"] for r in c.execute("SELECT version FROM schema_migrations ORDER BY version").fetchall()]
        out.append(Check(OK, "database schema", f"migrations applied: {applied}"))
    except Exception as e:
        out.append(Check(FAIL, "database schema", f"could not run migrations: {type(e).__name__}: {str(e)[:160]}"))
    return out


# ----------------------------------------------------------------------------------------------- model token
def _default_client(model: str, token: str):
    from huggingface_hub import InferenceClient

    return InferenceClient(model=model, token=token, timeout=30)


def check_model(production: bool, live: bool, client_factory=None) -> list[Check]:
    from .llm_engine import REMOTE_MODEL

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
    if not token:
        return [Check(_strict(production), "HF_TOKEN", "not set: the app runs in basic chat mode (no meal/workout logging)",
                      "create a fine-grained token with 'Make calls to Inference Providers' at huggingface.co/settings/tokens")]
    out = [Check(OK if token.startswith("hf_") else WARN, "HF_TOKEN", f"set ({len(token)} characters)",
                 "" if token.startswith("hf_") else "Hugging Face tokens normally start with hf_")]
    out.append(Check(INFO, "model", f"HF_MODEL = {REMOTE_MODEL}"))
    if not live:
        out.append(Check(INFO, "live model test", "skipped (add --live to call the API once)"))
        return out
    client = (client_factory or _default_client)(REMOTE_MODEL, token)
    try:
        r = client.chat_completion(messages=[{"role": "user", "content": "Reply with the single word OK."}], max_tokens=8)
        used = getattr(getattr(r, "usage", None), "total_tokens", None)
        out.append(Check(OK, "model call", f"{REMOTE_MODEL} answered" + (f" ({used} tokens)" if used else "")))
    except Exception as e:
        msg = str(e)
        if "403" in msg or "permission" in msg.lower():
            fix = "the token needs the 'Make calls to Inference Providers' permission (create a new fine-grained token)"
        elif "404" in msg or "not found" in msg.lower() or "model" in msg.lower():
            fix = "this model is not available on Inference Providers: choose another HF_MODEL"
        else:
            fix = "check the token and your network"
        return [*out, Check(FAIL, "model call", f"{type(e).__name__}: {msg[:160]}", fix)]
    try:
        tools = [{"type": "function", "function": {"name": "ping", "description": "Call this tool to confirm it works.",
                                                   "parameters": {"type": "object", "properties": {}}}}]
        r = client.chat_completion(messages=[{"role": "user", "content": "Call the ping tool now."}], tools=tools,
                                   tool_choice="auto", max_tokens=40)
        called = bool(r.choices[0].message.tool_calls)
        out.append(Check(OK if called else WARN, "tool calling",
                         "the model returned a tool call" if called else "the model did not call the test tool",
                         "" if called else "this model may not support tool calling well: try another HF_MODEL, then re-run"))
    except Exception as e:
        out.append(Check(WARN, "tool calling", f"tools request failed: {type(e).__name__}: {str(e)[:120]}",
                         "pick a model that supports tool calling on Inference Providers"))
    return out


# ----------------------------------------------------------------------------------------------- data / policy
def check_data() -> list[Check]:
    from .foods import get_food_db
    from .rag.corpus import load_documents

    out = []
    n = len(get_food_db())
    out.append(Check(OK if n > 10_000 else FAIL, "food database", f"{n:,} USDA foods loaded",
                     "" if n > 10_000 else "fooddata/foods.csv and portions.csv are missing: restore them or re-run core.food_import"))
    try:
        out.append(Check(OK, "knowledge base", f"{len(load_documents())} documents"))
    except Exception as e:
        out.append(Check(FAIL, "knowledge base", str(e)[:160], "fix the front matter of the file named above"))
    return out


def _int_env(name: str, default: str) -> int | None:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return None


def check_access(production: bool, db_ok: bool = True) -> list[Check]:
    out = []
    guests = os.getenv("ALLOW_GUEST", "1") == "1"
    if guests:
        out.append(Check(_strict(production, WARN), "guest access", "ALLOW_GUEST=1: anyone with the link can chat (they share one token budget)",
                         "set ALLOW_GUEST=0 for a private deployment"))
    if not db_ok:  # do not wait on a database we already know is unreachable
        out.append(Check(INFO, "API keys", "skipped: the database is not available"))
        guests = True  # suppress the "nobody can sign in" conclusion below
    try:
        if not db_ok:
            raise RuntimeError("skipped")
        from . import auth

        active = [k for k in auth.list_keys() if not k["revoked"]]
        if not active and not guests:
            out.append(Check(FAIL, "API keys", "guests are disabled and no API key exists: nobody can sign in",
                             "python -m core.auth create <your-name>"))
        else:
            out.append(Check(OK if active else INFO, "API keys", f"{len(active)} active key(s)"))
    except Exception as e:
        if db_ok:
            out.append(Check(WARN, "API keys", f"could not read keys: {type(e).__name__}"))
    for name, default in (("DAILY_TOKEN_BUDGET", "200000"), ("GUEST_DAILY_TOKEN_BUDGET", "100000"), ("RATE_LIMIT_PER_MIN", "20")):
        v = _int_env(name, default)
        if v is None:
            out.append(Check(FAIL, name, f"'{os.getenv(name)}' is not a whole number", "use digits only, e.g. 200000"))
        elif v == 0 and name != "RATE_LIMIT_PER_MIN":
            out.append(Check(_strict(production, WARN), name, "0 = unlimited: model costs are uncapped", "set a daily limit"))
        else:
            out.append(Check(INFO, name, str(v)))
    return out


def check_observability() -> list[Check]:
    from .pricing import cost_usd

    out = []
    from .llm_engine import REMOTE_MODEL

    priced = cost_usd(REMOTE_MODEL, 1_000_000, 0) is not None
    out.append(Check(OK if priced else INFO, "cost tracking", "model price configured" if priced else
                     "MODEL_PRICES not set: tokens are tracked but dollar cost is reported as unknown"))
    trace = os.getenv("TRACE_FILE")
    if trace:
        folder = os.path.dirname(os.path.abspath(trace))
        out.append(Check(OK if os.access(folder, os.W_OK) else FAIL, "TRACE_FILE", f"{trace}",
                         "" if os.access(folder, os.W_OK) else "the folder is not writable"))
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        have = all(importlib.util.find_spec(m) for m in ("opentelemetry.sdk", "opentelemetry.exporter.otlp.proto.http"))
        out.append(Check(OK if have else FAIL, "OpenTelemetry", "exporter packages installed" if have else "packages missing",
                         "" if have else "pip install -r requirements-otel.txt"))
    return out


def run_checks(production: bool = False, live: bool = False, client_factory=None) -> list[Check]:
    database = check_database(production)
    db_ok = not any(c.status == FAIL for c in database)
    return [*check_env_file(production), *database, *check_model(production, live, client_factory),
            *check_data(), *check_access(production, db_ok), *check_observability()]


def render(checks: list[Check]) -> str:
    tag = {OK: "[ OK ]", WARN: "[WARN]", FAIL: "[FAIL]", INFO: "[info]"}
    lines = []
    for c in checks:
        lines.append(f"{tag[c.status]} {c.name}: {c.detail}")
        if c.fix and c.status in (WARN, FAIL):
            lines.append(f"        fix: {c.fix}")
    n = {s: sum(c.status == s for c in checks) for s in (OK, WARN, FAIL)}
    lines.append(f"\n{n[OK]} ok, {n[WARN]} warning(s), {n[FAIL]} failure(s)")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m core.doctor")
    p.add_argument("--production", action="store_true", help="treat missing token / SQLite / unlimited budget as failures")
    p.add_argument("--live", action="store_true", help="call the Hugging Face API once")
    a = p.parse_args(argv)
    load_env()
    checks = run_checks(a.production, a.live)
    print(render(checks))
    return 1 if any(c.status == FAIL for c in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
