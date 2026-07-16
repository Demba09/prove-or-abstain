"""autopilot.py — execution tracker and autonomous monitor for Track 4.

Thin adapter over memory.py: every EXECUTE becomes an investigation row plus a
deduplicated active alert (same metric × cause is not re-fired until resolved).
The Execution/Dashboard dataclasses and every function signature are unchanged,
so api/app.py and mcp_server.py keep working verbatim — only the storage moved
from an in-memory dict to SQLite.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field

from prove_or_abstain import memory


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
    version: str = "0.4.0"
    last_check_at: float | None = None
    last_check_verdict: str | None = None
    active_alerts: list = field(default_factory=list)
    total_executions: int = 0
    total_checks: int = 0
    uptime_seconds: float = 0.0


# The autonomous-check counters stay process-local (no table in the spec).
_START_TIME = time.time()
_LAST_CHECK: float | None = None
_LAST_VERDICT: str | None = None
_TOTAL_CHECKS: int = 0


def _iso_to_epoch(s: str | None) -> float | None:
    if not s:
        return None
    from datetime import datetime
    return datetime.fromisoformat(s).timestamp()


def _execution_from_alert(alert: dict) -> Execution:
    """Rebuild the legacy Execution view from an alert row (+ its
    investigation) so the API shape is byte-for-byte identical."""
    inv = memory.get_investigation(alert["investigation_id"]) or {}
    return Execution(
        id=alert["id"],
        metric=alert["metric"],
        dimension=(alert["cause"].split("=", 1)[0] if alert.get("cause") else None),
        segment=(alert["cause"].split("=", 1)[1]
                 if alert.get("cause") and "=" in alert["cause"] else None),
        confidence=inv.get("confidence", 0.0),
        action_kind=alert["action"],
        action_detail=alert.get("detail", ""),
        report=inv.get("report", ""),
        trace=inv.get("trace", []),
        created_at=_iso_to_epoch(alert["created_at"]) or time.time(),
        resolved_at=_iso_to_epoch(alert.get("resolved_at")),
        resolved_by=alert.get("resolved_by"),
    )


def record_execution(metric: str, dimension: str | None, segment: str | None,
                     confidence: float, action_kind: str, action_detail: str,
                     report: str, trace: list[str]) -> Execution:
    """Record a new EXECUTE action. Deduplicates on metric × cause."""
    cause = f"{dimension}={segment}"
    inv_id = memory.record_investigation(
        metric, [dimension] if dimension else [], "ASSERT", confidence,
        cause, report, trace, mode="autopilot", model="",
    )
    aid = memory.create_alert(inv_id, metric, cause, action_kind, action_detail)
    alert = next((a for a in memory.get_all_alerts() if a["id"] == aid), None)
    return _execution_from_alert(alert) if alert else Execution(
        aid, metric, dimension, segment, confidence, action_kind,
        action_detail, report, trace, time.time())


def resolve_execution(key: str, resolved_by: str = "human") -> Execution | None:
    if not memory.resolve_alert(key, resolved_by):
        return None
    alert = next((a for a in memory.get_all_alerts() if a["id"] == key), None)
    return _execution_from_alert(alert) if alert else None


def get_executions() -> dict[str, Execution]:
    return {a["id"]: _execution_from_alert(a) for a in memory.get_all_alerts()}


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
    } for e in get_executions().values()]

    return Dashboard(
        last_check_at=_LAST_CHECK,
        last_check_verdict=_LAST_VERDICT,
        active_alerts=active,
        total_executions=len(active),
        total_checks=_TOTAL_CHECKS,
        uptime_seconds=time.time() - _START_TIME,
    )
