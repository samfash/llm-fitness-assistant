"""Lightweight tracing: nested spans exported to a JSONL file and/or OpenTelemetry.

    TRACE_FILE=traces.jsonl                 append one JSON line per finished span
    OTEL_EXPORTER_OTLP_ENDPOINT=...         also export via OpenTelemetry (needs opentelemetry-sdk and
                                            opentelemetry-exporter-otlp-proto-http); works with any OTLP backend
    TRACE_CONTENT=1                         also record message text (OFF by default: it may contain personal data)

User ids are hashed. Exporter failures are logged and never affect the request.
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field

log = logging.getLogger(__name__)
_current: contextvars.ContextVar[Span | None] = contextvars.ContextVar("current_span", default=None)
_exporters: list = []
_lock = threading.Lock()
_defaults_loaded = False


def hash_id(value: str) -> str:
    return hashlib.sha256(str(value).encode()).hexdigest()[:12]


def content_enabled() -> bool:
    return os.getenv("TRACE_CONTENT", "0") == "1"


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_id: str | None
    start_ns: int
    end_ns: int | None = None
    attrs: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    status: str = "ok"
    error: str | None = None

    def set(self, **kv) -> None:
        self.attrs.update(kv)

    def event(self, name: str, **kv) -> None:
        self.events.append({"name": name, "ts_ns": time.time_ns(), **kv})

    @property
    def duration_ms(self) -> float:
        return ((self.end_ns or time.time_ns()) - self.start_ns) / 1e6

    def to_dict(self) -> dict:
        return {"name": self.name, "trace_id": self.trace_id, "span_id": self.span_id, "parent_id": self.parent_id,
                "start_ns": self.start_ns, "duration_ms": round(self.duration_ms, 2), "status": self.status,
                "error": self.error, "attrs": self.attrs, "events": self.events}


# ------------------------------------------------------------------------------------------ exporters
class InMemoryExporter:
    def __init__(self):
        self.spans: list[Span] = []

    def on_end(self, s: Span) -> None:
        self.spans.append(s)


class JsonlExporter:
    def __init__(self, path: str):
        self.path, self._lock = path, threading.Lock()
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)

    def on_end(self, s: Span) -> None:
        line = json.dumps(s.to_dict(), default=str)
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


class OTelExporter:
    """Mirrors our spans into OpenTelemetry (parent/child links, times, attributes, events, errors)."""

    def __init__(self, tracer_provider=None):
        from opentelemetry import trace

        provider = tracer_provider or trace.get_tracer_provider()
        self._tracer = provider.get_tracer("fitness-assistant")
        self._live: dict = {}

    def on_start(self, s: Span) -> None:
        from opentelemetry.trace import set_span_in_context

        parent = self._live.get(s.parent_id)
        ctx = set_span_in_context(parent) if parent is not None else None
        self._live[s.span_id] = self._tracer.start_span(s.name, context=ctx, start_time=s.start_ns)

    def on_end(self, s: Span) -> None:
        from opentelemetry.trace import Status, StatusCode

        o = self._live.pop(s.span_id, None)
        if o is None:
            return
        for k, v in s.attrs.items():
            if v is not None:
                o.set_attribute(k, v if isinstance(v, (str, bool, int, float)) else json.dumps(v, default=str))
        for ev in s.events:
            o.add_event(ev["name"], {k: str(v) for k, v in ev.items() if k not in ("name", "ts_ns")}, ev["ts_ns"])
        if s.status == "error":
            o.set_status(Status(StatusCode.ERROR, s.error or ""))
        o.end(end_time=s.end_ns)


def configure_otlp() -> OTelExporter | None:
    """Set up an OTLP/HTTP exporter from the standard OTEL_* environment variables."""
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(resource=Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "fitness-assistant")}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)
        return OTelExporter(provider)
    except Exception as e:  # missing packages or bad config must not stop the app
        log.warning("OpenTelemetry export disabled: %s", e)
        return None


def add_exporter(exporter) -> None:
    with _lock:
        _exporters.append(exporter)


def reset() -> None:
    """Remove all exporters and re-read the environment on next use (tests)."""
    global _defaults_loaded
    with _lock:
        _exporters.clear()
        _defaults_loaded = False


def _load_defaults() -> None:
    global _defaults_loaded
    with _lock:
        if _defaults_loaded:
            return
        _defaults_loaded = True
        if os.getenv("TRACE_FILE"):
            _exporters.append(JsonlExporter(os.environ["TRACE_FILE"]))
        if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
            ex = configure_otlp()
            if ex:
                _exporters.append(ex)


def _notify(hook: str, s: Span) -> None:
    with _lock:
        targets = list(_exporters)
    for e in targets:
        fn = getattr(e, hook, None)
        if fn:
            try:
                fn(s)
            except Exception as err:
                log.warning("trace exporter %s failed: %s", type(e).__name__, err)


@contextmanager
def span(name: str, **attrs):
    _load_defaults()
    parent = _current.get()
    s = Span(name, parent.trace_id if parent else uuid.uuid4().hex, uuid.uuid4().hex[:16],
             parent.span_id if parent else None, time.time_ns(), attrs=dict(attrs))
    token = _current.set(s)
    _notify("on_start", s)
    try:
        yield s
    except BaseException as exc:
        s.status, s.error = "error", f"{type(exc).__name__}: {exc}"[:300]
        raise
    finally:
        s.end_ns = time.time_ns()
        _current.reset(token)
        _notify("on_end", s)
