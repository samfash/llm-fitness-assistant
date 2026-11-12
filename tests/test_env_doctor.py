import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import db, env
from core.doctor import FAIL, INFO, OK, WARN, assess_url, main, redact_url, render, run_checks

ROOT = Path(__file__).resolve().parents[1]


def by_name(checks, name):
    return [c for c in checks if c.name == name]


def statuses(checks, name):
    return [c.status for c in by_name(checks, name)]


# ---- .env parsing and loading ----------------------------------------------------------------------
def test_parse_handles_comments_export_quotes_and_hash_in_values():
    text = '# comment\n\nA=1\nexport B="two words"\nC=\'single\'\nD=pass#word\nnot a line\nE = spaced \n=nokey\n1BAD=x\n'
    assert env.parse(text) == {"A": "1", "B": "two words", "C": "single", "D": "pass#word", "E": "spaced"}


def test_load_applies_values_but_real_environment_wins(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("T_NEW=from_file\nT_EXISTING=from_file\n")
    monkeypatch.delenv("T_NEW", raising=False)
    monkeypatch.setenv("T_EXISTING", "from_env")
    r = env.load_env(f)
    assert os.environ["T_NEW"] == "from_file" and os.environ["T_EXISTING"] == "from_env"
    assert r.applied == ["T_NEW"] and r.overridden == ["T_EXISTING"]
    monkeypatch.delenv("T_NEW")


def test_placeholders_and_empty_values_are_treated_as_unset(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("T_TOKEN=hf_REPLACE_ME_token\nT_URL=postgresql://REPLACE_ME_USER:x@h/db\nT_EMPTY=\nT_OK=1\n")
    for k in ("T_TOKEN", "T_URL", "T_EMPTY", "T_OK"):
        monkeypatch.delenv(k, raising=False)
    r = env.load_env(f)
    assert r.placeholders == ["T_TOKEN", "T_URL"] and r.applied == ["T_OK"]
    assert "T_TOKEN" not in os.environ and "T_URL" not in os.environ and "T_EMPTY" not in os.environ
    monkeypatch.delenv("T_OK")


def test_automatic_loading_can_be_disabled_but_explicit_path_still_works(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("T_AUTO=1\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("T_AUTO", raising=False)
    monkeypatch.setenv("FITNESS_NO_DOTENV", "1")
    env.load_env()
    assert "T_AUTO" not in os.environ
    env.load_env(tmp_path / ".env")
    assert os.environ["T_AUTO"] == "1"
    monkeypatch.delenv("T_AUTO")


def test_shipped_env_template_contains_only_placeholders_for_secrets():
    """Reads the committed .env.example only: the real .env is gitignored and must never be read by tests."""
    parsed = env.parse((ROOT / ".env.example").read_text())
    assert {"HF_TOKEN", "DATABASE_URL"} <= set(parsed)
    assert all(env.PLACEHOLDER in parsed[k] for k in ("HF_TOKEN", "DATABASE_URL"))
    assert not any(v.startswith("hf_") and env.PLACEHOLDER not in v for v in parsed.values())
    assert not any("neon.tech" in v and env.PLACEHOLDER not in v for v in parsed.values())
    gitignore = (ROOT / ".gitignore").read_text().splitlines()
    assert ".env" in gitignore and ".env.example" not in gitignore


def test_every_documented_variable_is_actually_read_by_the_code():
    documented = set(env.parse((ROOT / ".env.example").read_text().replace("\n# ", "\n")).keys())
    code = " ".join(p.read_text() for p in list((ROOT / "core").rglob("*.py")) + [ROOT / "app.py", ROOT / "api" / "api_server.py",
                                                                                   ROOT / "ui" / "ui_streamlit.py"])
    external = {"OTEL_EXPORTER_OTLP_HEADERS", "OTEL_SERVICE_NAME"}  # consumed by the OpenTelemetry library itself
    for name in sorted(documented - external):
        assert name in code, f"{name} is documented in .env but no code reads it"


# ---- doctor: static URL assessment ------------------------------------------------------------------
NEON = "postgresql://u:p@ep-cool-123456-pooler.us-east-2.aws.neon.tech/db?sslmode=require&channel_binding=require"


def test_assess_url_recognises_pooled_neon_string():
    out = assess_url(NEON)
    assert [c.status for c in out] == [INFO] and "pooled" in out[0].detail


def test_assess_url_flags_missing_tls_and_direct_endpoint_and_bad_scheme():
    out = assess_url("postgresql://u:p@ep-cool-123456.us-east-2.aws.neon.tech/db")
    assert [c.name for c in out] == ["database TLS", "database endpoint"] and out[0].status == WARN
    assert "sslmode=require" in out[0].fix
    assert assess_url("mysql://u:p@h/db")[0].status == FAIL
    assert assess_url("postgresql://u:p@localhost/db") == []  # local development needs no TLS


def test_passwords_are_masked():
    assert redact_url("postgresql://user:s3cret@host/db?sslmode=require") == "postgresql://user:***@host/db?sslmode=require"
    assert redact_url("postgresql://user@host/db") == "postgresql://user@host/db"


# ---- doctor: full runs ---------------------------------------------------------------------------------
def test_doctor_on_the_active_backend_passes_without_failures(backend):
    checks = run_checks()
    assert not [c for c in checks if c.status == FAIL], render(checks)
    assert statuses(checks, "database schema") == [OK] and statuses(checks, "food database") == [OK]
    if backend == "postgres":
        assert statuses(checks, "database connection") == [OK] and "PostgreSQL" in by_name(checks, "database connection")[0].detail
    else:
        assert statuses(checks, "database") == [WARN]  # SQLite works but is ephemeral on a free Space


def test_production_mode_makes_sqlite_and_missing_token_failures(backend, monkeypatch):
    if backend != "sqlite":
        pytest.skip("sqlite-specific")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    checks = run_checks(production=True)
    assert statuses(checks, "database") == [FAIL] and statuses(checks, "HF_TOKEN") == [FAIL]
    assert "Neon" in by_name(checks, "database")[0].fix and main(["--production"]) == 1


def test_unfilled_placeholders_are_reported(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("HF_TOKEN=REPLACE_ME_x\nDATABASE_URL=postgresql://REPLACE_ME@h/d\nOTEL_SERVICE_NAME=REPLACE_ME\n")
    for k in ("HF_TOKEN", "DATABASE_URL", "OTEL_SERVICE_NAME"):
        monkeypatch.delenv(k, raising=False)
    env.load_env(f)
    loose, strict = run_checks(), run_checks(production=True)
    assert statuses(loose, "HF_TOKEN placeholder") == [WARN] and statuses(strict, "HF_TOKEN placeholder") == [FAIL]
    assert statuses(strict, "DATABASE_URL placeholder") == [FAIL] and statuses(strict, "OTEL_SERVICE_NAME placeholder") == [INFO]


def test_a_placeholder_is_harmless_when_the_real_value_comes_from_the_environment(tmp_path, monkeypatch):
    """Typical on a Space: secrets are real environment variables while a stale .env still has placeholders."""
    f = tmp_path / ".env"
    f.write_text("HF_TOKEN=REPLACE_ME_x\n")
    monkeypatch.setenv("HF_TOKEN", "hf_realtokenfromtheenvironment0000000000")
    env.load_env(f)
    checks = run_checks(production=True)
    assert statuses(checks, "HF_TOKEN placeholder") == [INFO] and statuses(checks, "HF_TOKEN") == [OK]
    assert "wins" in by_name(checks, "HF_TOKEN placeholder")[0].detail


def test_unreachable_database_fails_fast_and_never_prints_the_password(monkeypatch):
    import time

    monkeypatch.setenv("DATABASE_URL", "postgresql://alice:hunter2@127.0.0.1:1/db")
    t0 = time.perf_counter()
    checks = run_checks()
    assert time.perf_counter() - t0 < 8, "doctor must not wait on a database it knows is down"
    assert statuses(checks, "API keys") == [INFO]
    failed = by_name(checks, "database connection")[0]
    assert failed.status == FAIL and "hunter2" not in render(checks) and "alice:***@" in failed.detail
    assert "Neon" in failed.fix


def test_requests_fail_within_the_connect_timeout_when_the_database_is_down(backend, monkeypatch):
    if backend != "postgres":
        pytest.skip("postgres-specific")
    import time

    import psycopg

    monkeypatch.setattr(db, "_CONNECT_TIMEOUT", 2)  # the pool reads this when it is created
    monkeypatch.setenv("DATABASE_URL", "postgresql://alice:x@127.0.0.1:1/db")
    t0 = time.perf_counter()
    with pytest.raises(psycopg.OperationalError), db.connect():
        pass
    assert time.perf_counter() - t0 < 2 + 3
    db.close_pools()


def test_non_postgres_database_url_is_a_failure(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "mysql://u:p@h/db")
    assert any(c.status == FAIL and c.name == "DATABASE_URL format" for c in run_checks())


class FakeClient:
    """Stands in for huggingface_hub.InferenceClient."""

    def __init__(self, error=None, tool_call=True):
        self.error, self.tool_call, self.calls = error, tool_call, []

    def chat_completion(self, **kw):
        self.calls.append(kw)
        if self.error:
            raise self.error
        msg = SimpleNamespace(content="OK", tool_calls=[SimpleNamespace()] if kw.get("tools") and self.tool_call else None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=SimpleNamespace(total_tokens=9))


def live(client, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_abc123")
    return run_checks(live=True, client_factory=lambda model, token: client)


def test_live_check_success_with_tool_calling(monkeypatch):
    client = FakeClient()
    checks = live(client, monkeypatch)
    assert statuses(checks, "model call") == [OK] and statuses(checks, "tool calling") == [OK]
    assert client.calls[1]["tools"][0]["function"]["name"] == "ping"


def test_live_check_explains_a_403_permission_error(monkeypatch):
    checks = live(FakeClient(error=RuntimeError("403 Forbidden: does not have sufficient permissions to call Inference Providers")), monkeypatch)
    failed = by_name(checks, "model call")[0]
    assert failed.status == FAIL and "Make calls to Inference Providers" in failed.fix and by_name(checks, "tool calling") == []


def test_live_check_explains_an_unavailable_model_and_warns_when_tools_are_ignored(monkeypatch):
    checks = live(FakeClient(error=RuntimeError("404 model not found")), monkeypatch)
    assert "HF_MODEL" in by_name(checks, "model call")[0].fix
    assert statuses(live(FakeClient(tool_call=False), monkeypatch), "tool calling") == [WARN]


def test_token_shape_warning_and_no_live_call_by_default(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "not-an-hf-token")
    checks = run_checks()
    assert statuses(checks, "HF_TOKEN") == [WARN] and statuses(checks, "live model test") == [INFO]


def test_access_checks(monkeypatch, backend):
    from core import auth

    monkeypatch.setenv("ALLOW_GUEST", "0")
    assert any(c.status == FAIL and c.name == "API keys" for c in run_checks())  # nobody could sign in
    auth.create_key("alice")
    assert statuses(run_checks(), "API keys") == [OK]
    monkeypatch.setenv("DAILY_TOKEN_BUDGET", "lots")
    assert statuses(run_checks(), "DAILY_TOKEN_BUDGET") == [FAIL]
    monkeypatch.setenv("DAILY_TOKEN_BUDGET", "0")
    assert statuses(run_checks(production=True), "DAILY_TOKEN_BUDGET") == [WARN]


def test_doctor_cli_output_and_exit_code(capsys, backend):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "[ OK ] food database" in out and "ok," in out and "failure(s)" in out


# ---- resilience claims for managed Postgres (Neon) -------------------------------------------------
def test_pool_recovers_after_the_server_drops_every_connection(backend):
    """Neon suspends idle compute and drops connections; the next request must still succeed."""
    if backend != "postgres":
        pytest.skip("postgres-specific")
    import psycopg

    db.save_profile("u", {"age": 30})  # the pool now holds an idle connection
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as admin:
        killed = admin.execute("SELECT count(*) FROM (SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                               "WHERE datname = current_database() AND pid <> pg_backend_pid()) t").fetchone()[0]
    assert killed >= 1
    assert db.get_profile("u") == {"age": 30}  # transparently reconnected
    db.save_profile("u", {"sex": "male"})
    assert db.get_profile("u") == {"age": 30, "sex": "male"}


def test_no_server_side_prepared_statements_are_used(backend):
    """Safe behind PgBouncer in transaction mode regardless of the pooler's version."""
    if backend != "postgres":
        pytest.skip("postgres-specific")
    with db.connect() as c:
        for _ in range(12):  # psycopg would auto-prepare after 5 executions by default
            c.execute("SELECT 1 AS x").fetchone()
        n = c.execute("SELECT COUNT(*) AS n FROM pg_prepared_statements").fetchone()["n"]
    assert n == 0


def test_migration_lock_is_transaction_scoped_so_it_works_behind_a_transaction_pooler():
    src = (ROOT / "core" / "db.py").read_text()
    assert "pg_advisory_xact_lock" in src and "pg_advisory_lock(" not in src


def test_docker_image_never_contains_secrets_or_local_state():
    ignored = (ROOT / ".dockerignore").read_text().splitlines()
    for pattern in (".env", ".git", "data", "__pycache__"):
        assert pattern in ignored, pattern
    assert "COPY . ." in (ROOT / "Dockerfile.api").read_text()  # which is exactly why .dockerignore matters


def test_no_dead_modules_shadow_the_real_ones():
    assert not (ROOT / "core" / "rag.py").exists()  # a stale toy module next to the core/rag/ package
    assert not (ROOT / "core" / "memory.py").exists()


# ---- the README must describe the system that actually exists --------------------------------------------
def _readme():
    return (ROOT / "README.md").read_text(encoding="utf-8")


def test_readme_has_no_stale_claims_and_keeps_the_space_front_matter():
    text = _readme()
    for stale in ("LangChain", "Chroma", "CHROMA_PATH", "HUGGINGFACE_API_KEY"):
        assert stale not in text, f"README still mentions {stale}"
    assert text.startswith("---\ntitle:") and "sdk: streamlit" in text and 'sdk_version: "1.38.0"' in text
    assert "app_file: app.py" in text and 'python_version: "3.11"' in text


def test_every_setting_in_the_readme_table_is_read_by_the_code():
    import re

    text = _readme()
    block = text.split("## ⚙️ Configuration reference")[1].split("\n## ")[0]
    names = set(re.findall(r"`([A-Z][A-Z0-9_]{2,})`", block))
    code = " ".join(p.read_text() for p in list((ROOT / "core").rglob("*.py")) + [ROOT / "app.py", ROOT / "api" / "api_server.py",
                                                                                   ROOT / "ui" / "ui_streamlit.py"])
    external = {"OTEL_EXPORTER_OTLP_ENDPOINT"}  # read by core.tracing; also documented by the OpenTelemetry project
    assert len(names) >= 15
    for name in sorted(names):
        assert name in code or name in external, f"{name} is in the README table but no code reads it"


def test_readme_numbers_match_the_data():
    import json

    cases = sum(len(json.loads(p.read_text())) for p in (ROOT / "evals" / "cases").glob("*.json"))
    assert f"golden set of {cases} cases" in _readme()
    with open(ROOT / "fooddata" / "foods.csv", encoding="utf-8") as f:
        foods = sum(1 for _ in f) - 1
    assert f"{foods:,} foods" in _readme()


# ---- .gitignore must hide only what it is meant to hide (a bare `data/` once hid evals/data/) ---------------
def test_gitignore_hides_only_local_state_and_never_a_project_file(tmp_path):
    import shutil
    import subprocess

    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    shutil.copy(ROOT / ".gitignore", tmp_path / ".gitignore")
    skip_dirs = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "venv", "node_modules"}
    files = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*")
             if p.is_file() and not (set(p.relative_to(ROOT).parts) & skip_dirs) and not p.name.endswith(".pyc")]
    assert "evals/data/food_qa.json" in files and "evals/data/retrieval_qa.json" in files
    result = subprocess.run(["git", "-C", str(tmp_path), "check-ignore", "--stdin"], input="\n".join(files),
                            capture_output=True, text=True)
    ignored = set(result.stdout.split())
    intended = lambda f: f == ".env" or f.startswith(("data/", "evals/results/")) or f.startswith("traces")  # noqa: E731
    wrongly = sorted(f for f in ignored if not intended(f))
    assert not wrongly, f".gitignore would keep these project files out of git: {wrongly}"
    # and the local state it is meant to hide really is hidden
    hidden = subprocess.run(["git", "-C", str(tmp_path), "check-ignore", "--stdin"], capture_output=True, text=True,
                            input=".env\ndata/fitness.db\nevals/results/latest.json\ntraces.jsonl\ncore/__pycache__/x.pyc\n").stdout.split()
    assert len(hidden) == 5
