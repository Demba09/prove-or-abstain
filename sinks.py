"""
sinks.py — where an EXECUTE action actually goes.

The actuator produces a typed Action; on ASSERT with autopilot and high
confidence that Action is an EXECUTE. This module is the one place that turns
an EXECUTE into an outward effect: a POST to a configured webhook (Slack, an
ops endpoint, a queue). Everything else is a no-op here.

Safety, restated in code:
  - only kind == "EXECUTE" is ever dispatched. RECOMMEND / ESCALATE / NONE
    (every ABSTAIN path) return {"dispatched": false} without touching the
    network. The actuator already guarantees ABSTAIN never yields EXECUTE;
    this is the second line of the same property.
  - with no ACTION_WEBHOOK_URL set, an EXECUTE is a DRY RUN: it reports what
    it *would* send and dispatches nothing. So the default configuration —
    and the whole test suite — makes no network call and stays deterministic.
  - any transport error is caught and reported as dispatched=false with the
    reason; the agent never crashes because a sink was unreachable.
"""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import asdict

WEBHOOK_ENV = "ACTION_WEBHOOK_URL"
DEFAULT_TIMEOUT = 5.0


def dispatch_action(action, *, timeout: float = DEFAULT_TIMEOUT) -> dict:
    """Dispatch an EXECUTE action to the configured webhook. Returns a small
    receipt dict: {dispatched, target, detail}. Never raises."""
    payload = asdict(action) if not isinstance(action, dict) else dict(action)
    kind = payload.get("kind")

    if kind != "EXECUTE":
        return {"dispatched": False, "target": None,
                "detail": f"{kind} is not an executable action; nothing dispatched."}

    url = os.environ.get(WEBHOOK_ENV, "").strip()
    if not url:
        return {"dispatched": False, "target": None,
                "detail": f"dry run — set {WEBHOOK_ENV} to dispatch this EXECUTE "
                          f"to a real endpoint."}

    try:
        body = json.dumps({"event": "prove_or_abstain.execute",
                           "action": payload}).encode()
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            code = resp.getcode()
        return {"dispatched": True, "target": url,
                "detail": f"POST {url} -> HTTP {code}."}
    except Exception as exc:  # unreachable sink must not crash the agent
        return {"dispatched": False, "target": url,
                "detail": f"sink unreachable ({type(exc).__name__}: {exc})."}
