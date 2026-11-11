"""Tiny .env loader (no dependency).

Rules:
  * Real environment variables always win, so Hugging Face Space secrets and CI settings override a local .env.
  * A value containing REPLACE_ME (or an empty value) is treated as NOT SET. An unfilled placeholder can never
    be mistaken for a real credential; `python -m core.doctor` reports which ones are still unfilled.
  * Comments must be on their own line (a '#' inside a value, e.g. in a password, is kept as is).
  * Set FITNESS_NO_DOTENV=1 to disable automatic loading (the test suite does this).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER = "REPLACE_ME"


@dataclass
class LoadResult:
    path: str | None = None
    applied: list[str] = field(default_factory=list)       # variables taken from the file
    placeholders: list[str] = field(default_factory=list)  # variables skipped because still REPLACE_ME
    overridden: list[str] = field(default_factory=list)    # variables ignored because the environment already set them


_last = LoadResult()
_loaded_paths: set[str] = set()


def parse(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key.isidentifier():
            out[key] = value
    return out


def load_env(path: str | os.PathLike | None = None) -> LoadResult:
    """Load a .env file into os.environ (without overriding). Safe to call repeatedly."""
    global _last
    if path is None and os.getenv("FITNESS_NO_DOTENV") == "1":
        return _last
    candidates = [Path(path)] if path else [Path.cwd() / ".env", ROOT / ".env"]
    for p in candidates:
        if not p.is_file():
            continue
        key = str(p.resolve())
        if key in _loaded_paths and path is None:
            return _last
        _loaded_paths.add(key)
        result = LoadResult(path=key)
        for name, value in parse(p.read_text(encoding="utf-8")).items():
            if PLACEHOLDER in value:
                result.placeholders.append(name)
            elif value == "":
                continue
            elif name in os.environ:
                result.overridden.append(name)
            else:
                os.environ[name] = value
                result.applied.append(name)
        _last = result
        return result
    return _last


def last_result() -> LoadResult:
    return _last
