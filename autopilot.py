"""autopilot.py — execution tracker and autonomous monitor for Track 4.

Keeps an in-memory log of every EXECUTE action the agent takes, plus a status
flag that the autonomous monitor uses to avoid action-storms (same metric ×
same cause fired repeatedly without a resolve).

This is the "autopilot" half of the Track 4 story:
  - an execution record is created on every ASSERT+EXECUTE
  - the record stays "active" until a human resolves it (or a timeout)
  - the autonomous check endpoint skips an already-active alert
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field


@dataclass
class Execution:
    id: str
    metric: str
    dimension: str | None
    segment: str | None
    confidence: float
    action_kind: str
    action_detail: str
    report: str
    trace: list[str]
    created_at: float
    resolved_at: float | None = None
    resolved_by: str | None = None


@dataclass
class Dashboard:
    """Snapshot returned by GET /dashboard."""
    version: str = "0.5.0"
    last_check_at: float | None = None
    last_check_verdict: str | None = None
    active_alerts: list = field(default_factory=list)
    total_executions: int = 0
    total_checks: int = 0
    uptime_seconds: float = 0.0


# --- in-memory state (for a hackathon; a real deployment uses a DB) ---
_EXECUTIONS: dict[str, Execution] = {}
_START_TIME = time.time()
_LAST_CHECK: float | None = None
_LAST_VERDICT: str | None = None
_TOTAL_CHECKS: int = 0


def record_execution(metric: str, dimension: str | None, segment: str | None,
                     confidence: float, action_kind: str, action_detail: str,
                     report: str, trace: list[str]) -> Execution:
    """Record a new EXECUTE action. Deduplicates: if the same metric +
    dimension + segment is already active, skip (avoid action-storms)."""
    key = f"{metric}:{dimension}={segment}"
    existing = _EXECUTIONS.get(key)
    if existing is not None and existing.resolved_at is None:
        return existing  # already alerted, skip

    entry = Execution(
        id=key,
        metric=metric,
        dimension=dimension,
        segment=segment,
        confidence=confidence,
        action_kind=action_kind,
        action_detail=action_detail,
        report=report,
        trace=trace,
        created_at=time.time(),
    )
    _EXECUTIONS[key] = entry
    return entry


def resolve_execution(key: str, resolved_by: str = "human") -> Execution | None:
    entry = _EXECUTIONS.get(key)
    if entry is None or entry.resolved_at is not None:
        return None
    entry.resolved_at = time.time()
    entry.resolved_by = resolved_by
    return entry


def record_check(verdict: str | None = None) -> None:
    global _LAST_CHECK, _LAST_VERDICT, _TOTAL_CHECKS
    _LAST_CHECK = time.time()
    _LAST_VERDICT = verdict
    _TOTAL_CHECKS += 1


def get_dashboard() -> Dashboard:
    active = [{
        "id": e.id,
        "metric": e.metric,
        "cause": f"{e.dimension}={e.segment}" if e.dimension else None,
        "confidence": e.confidence,
        "action": e.action_kind,
        "detail": e.action_detail,
        "report": e.report,
        "created_at": e.created_at,
        "resolved": e.resolved_at is not None,
    } for e in _EXECUTIONS.values()]

    return Dashboard(
        last_check_at=_LAST_CHECK,
        last_check_verdict=_LAST_VERDICT,
        active_alerts=active,
        total_executions=len(_EXECUTIONS),
        total_checks=_TOTAL_CHECKS,
        uptime_seconds=time.time() - _START_TIME,
    )
