"""ratelimit.py — a minimal in-memory rate limiter.

No Redis, no distributed state: matches the project's deployment model (one
Function Compute instance serving the demo/API), the same "process-local is
fine here" call already made for autopilot.py's check counters. A sliding
window of request timestamps per client id, thread-safe (FastAPI runs sync
endpoints in a thread pool — see autopilot.py / llm.py for the same reason
their globals are lock-guarded).

Not a defense against a distributed attack or a determined adversary who
rotates client ids — a rail against one client (accidental script loop,
runaway retry, or a single abusive caller) overwhelming a single-instance
demo deployment that has no other throttling in front of it.
"""
from __future__ import annotations

import os
import threading
import time

LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "60"))
_WINDOW_SECONDS = 60.0

_LOCK = threading.Lock()
_HITS: dict[str, list[float]] = {}


def allow(client_id: str) -> bool:
    """True and records a hit if client_id is still under the limit for the
    trailing 60s; False (and does not record) if it would exceed it."""
    now = time.time()
    cutoff = now - _WINDOW_SECONDS
    with _LOCK:
        hits = [t for t in _HITS.get(client_id, []) if t >= cutoff]
        if len(hits) >= LIMIT_PER_MINUTE:
            _HITS[client_id] = hits          # still trim, even on rejection
            return False
        hits.append(now)
        _HITS[client_id] = hits
        return True


def reset() -> None:
    """Test/dev only — wipe all tracked clients."""
    with _LOCK:
        _HITS.clear()
