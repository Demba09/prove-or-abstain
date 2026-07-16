"""
memory.py — lightweight SQLite persistence for investigation history & alerts.

Two tables:
  investigations — one row per completed investigation (verdict, confidence,
                   cause, report, trace, mode/model, token/cost accounting).
  active_alerts  — one row per fired action, deduplicated by metric×cause,
                   resolvable by a human (the Track-4 human-in-the-loop).

The DB path comes from PROBATIO_DB (default ":memory:", so tests and the
offline demo never touch disk; a real deployment or the monitor sets it to a
file for persistence). A single shared connection is guarded by a lock because
FastAPI runs sync endpoints in a threadpool.

active_alerts carries two columns beyond the minimal schema — `detail` and
`resolved_by` — so autopilot.py can rebuild its Execution/Dashboard view
byte-for-byte and nothing downstream (api/app.py, mcp_server.py) changes.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone

_DB_PATH = os.environ.get("PROBATIO_DB", ":memory:")
_LOCK = threading.Lock()
_CONN: sqlite3.Connection | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    global _CONN
    if _CONN is None:
        _CONN = sqlite3.connect(_DB_PATH, check_same_thread=False)
        _CONN.row_factory = sqlite3.Row
        _init(_CONN)
    return _CONN


def _init(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS investigations (
            id TEXT PRIMARY KEY,
            metric TEXT, dims TEXT, verdict TEXT, confidence REAL,
            cause TEXT, report TEXT, trace TEXT, mode TEXT, model TEXT,
            tokens_used INTEGER, cost_usd REAL, created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS active_alerts (
            id TEXT PRIMARY KEY,
            investigation_id TEXT,
            metric TEXT, cause TEXT, action TEXT, detail TEXT,
            created_at TEXT, resolved_at TEXT, resolved_by TEXT
        );
        """
    )
    conn.commit()


# ------------------------------------------------------------ investigations
def record_investigation(metric: str, dims, verdict: str, confidence: float,
                         cause: str | None, report: str, trace,
                         mode: str, model: str,
                         tokens_used: int = 0, cost_usd: float = 0.0) -> str:
    inv_id = uuid.uuid4().hex
    with _LOCK:
        conn = _connect()
        conn.execute(
            "INSERT INTO investigations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (inv_id, metric, json.dumps(dims), verdict, float(confidence),
             cause, report, json.dumps(trace), mode, model,
             int(tokens_used), float(cost_usd), _now_iso()),
        )
        conn.commit()
    return inv_id


def _row_to_investigation(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["dims"] = json.loads(d["dims"]) if d["dims"] else []
    d["trace"] = json.loads(d["trace"]) if d["trace"] else []
    return d


def get_history(limit: int = 50) -> list[dict]:
    with _LOCK:
        rows = _connect().execute(
            "SELECT * FROM investigations ORDER BY created_at DESC LIMIT ?",
            (limit,)).fetchall()
    return [_row_to_investigation(r) for r in rows]


def get_investigation(inv_id: str) -> dict | None:
    with _LOCK:
        r = _connect().execute(
            "SELECT * FROM investigations WHERE id = ?", (inv_id,)).fetchone()
    return _row_to_investigation(r) if r else None


# ------------------------------------------------------------- active alerts
def create_alert(investigation_id: str, metric: str, cause: str,
                 action: str, detail: str = "") -> str:
    """Create an alert, deduplicated by metric×cause: if one is already active
    (unresolved) it is returned untouched; a previously-resolved one for the
    same metric×cause is reactivated."""
    aid = f"{metric}:{cause}"
    with _LOCK:
        conn = _connect()
        existing = conn.execute(
            "SELECT resolved_at FROM active_alerts WHERE id = ?", (aid,)
        ).fetchone()
        if existing is not None and existing["resolved_at"] is None:
            return aid  # already active — skip (anti action-storm)
        conn.execute(
            "INSERT OR REPLACE INTO active_alerts "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (aid, investigation_id, metric, cause, action, detail,
             _now_iso(), None, None),
        )
        conn.commit()
    return aid


def resolve_alert(alert_id: str, resolved_by: str = "human") -> bool:
    with _LOCK:
        conn = _connect()
        cur = conn.execute(
            "UPDATE active_alerts SET resolved_at = ?, resolved_by = ? "
            "WHERE id = ? AND resolved_at IS NULL",
            (_now_iso(), resolved_by, alert_id))
        conn.commit()
        return cur.rowcount > 0


def get_active_alerts() -> list[dict]:
    with _LOCK:
        rows = _connect().execute(
            "SELECT * FROM active_alerts WHERE resolved_at IS NULL "
            "ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


def get_all_alerts() -> list[dict]:
    with _LOCK:
        rows = _connect().execute(
            "SELECT * FROM active_alerts ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


def get_stats() -> dict:
    with _LOCK:
        conn = _connect()
        total = conn.execute("SELECT COUNT(*) FROM investigations").fetchone()[0]
        asserts = conn.execute(
            "SELECT COUNT(*) FROM investigations WHERE verdict='ASSERT'"
        ).fetchone()[0]
        abstains = conn.execute(
            "SELECT COUNT(*) FROM investigations WHERE verdict='ABSTAIN'"
        ).fetchone()[0]
        avg = conn.execute(
            "SELECT AVG(confidence) FROM investigations").fetchone()[0]
    return {"total_investigations": total, "asserts": asserts,
            "abstains": abstains, "avg_confidence": float(avg or 0.0)}


def reset() -> None:
    """Wipe both tables — for tests and a clean monitor restart."""
    with _LOCK:
        conn = _connect()
        conn.executescript("DELETE FROM investigations; DELETE FROM active_alerts;")
        conn.commit()


if __name__ == "__main__":
    reset()
    iid = record_investigation("conversion", ["segment"], "ASSERT", 0.79,
                               "segment=paid", "report...", ["t1", "t2"],
                               "agent", "qwen-plus", 1234, 0.0021)
    create_alert(iid, "conversion", "segment=paid", "EXECUTE", "pause campaign")
    create_alert(iid, "conversion", "segment=paid", "EXECUTE", "dup")  # deduped
    print("history:", len(get_history()), "active:", len(get_active_alerts()))
    print("stats:", get_stats())
    print("resolve:", resolve_alert("conversion:segment=paid"))
    print("active after resolve:", len(get_active_alerts()))
