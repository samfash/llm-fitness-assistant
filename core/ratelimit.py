"""Token-bucket rate limiter (in process).

Limitation: state lives in this process, so with several workers/instances each has its own bucket and the
effective limit is multiplied. Behind multiple instances, put a shared store (Redis) or a gateway in front.
"""
from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    allowed: bool
    remaining: int
    retry_after: float  # seconds until one more request is allowed (0 when allowed)


class RateLimiter:
    def __init__(self, per_minute: float = 20, burst: int | None = None, clock=time.monotonic):
        self.rate = per_minute / 60.0  # tokens per second
        self.capacity = float(burst if burst is not None else max(1, int(per_minute)))
        self.clock = clock
        self._state: dict[str, tuple[float, float]] = {}  # key -> (tokens, last_seen)
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> RateLimiter:
        return cls(per_minute=float(os.getenv("RATE_LIMIT_PER_MIN", "20")), burst=int(os.getenv("RATE_LIMIT_BURST", "0")) or None)

    def check(self, key: str) -> Decision:
        now = self.clock()
        with self._lock:
            tokens, last = self._state.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            if tokens >= 1.0:
                self._state[key] = (tokens - 1.0, now)
                self._prune(now)
                return Decision(True, int(tokens - 1.0), 0.0)
            self._state[key] = (tokens, now)
            wait = (1.0 - tokens) / self.rate if self.rate > 0 else math.inf
            return Decision(False, 0, wait)

    def _prune(self, now: float) -> None:
        if len(self._state) > 10_000:  # drop idle, fully refilled buckets so memory stays bounded
            self._state = {k: (t, ts) for k, (t, ts) in self._state.items()
                           if min(self.capacity, t + (now - ts) * self.rate) < self.capacity}
