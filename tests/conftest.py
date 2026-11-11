import os
import sys
from pathlib import Path

import pytest

os.environ["FITNESS_NO_DOTENV"] = "1"  # never load a developer's real .env during tests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import db  # noqa: E402


@pytest.fixture(scope="session")
def _pg_url(tmp_path_factory):
    """One real PostgreSQL server for the whole session (embedded via the `pgserver` package)."""
    pgserver = pytest.importorskip("pgserver")
    pytest.importorskip("psycopg")
    server = pgserver.get_server(tmp_path_factory.mktemp("pg"), cleanup_mode="delete")
    yield server.get_uri()
    db.close_pools()
    server.cleanup()


@pytest.fixture(autouse=True, params=["sqlite", "postgres"])
def backend(request, tmp_path, monkeypatch):
    """Every test runs against both storage backends."""
    monkeypatch.setenv("FITNESS_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("RAG_DENSE", "off")  # never download embedding models in tests
    for var in ("MODEL_PRICES", "TRACE_FILE", "TRACE_CONTENT", "DAILY_TOKEN_BUDGET", "GUEST_DAILY_TOKEN_BUDGET"):
        monkeypatch.delenv(var, raising=False)
    if request.param == "postgres":
        url = request.getfixturevalue("_pg_url")
        monkeypatch.setenv("DATABASE_URL", url)
        db.reset_for_tests()
        yield request.param
        monkeypatch.setenv("DATABASE_URL", url)  # a test may have changed it; clean up against the real server
        db.reset_for_tests()
    else:
        monkeypatch.delenv("DATABASE_URL", raising=False)
        yield request.param
