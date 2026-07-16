"""
audit.py — reproducible, verifiable audit trails.

Every investigation can be frozen into a trail: the SHA256 of its inputs, the
tool calls Qwen made, the four gate decisions, the verdict/confidence/action
and the cost. Because the pipeline is deterministic, verify_replay can re-run
the same inputs and confirm the verdict is bit-for-bit reproducible — the
guarantee an auditor actually wants.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from prove_or_abstain.gates import (MATERIAL_REL, CONCENTRATION_MIN,
                                    SIGNIFICANCE_ALPHA, INTERACTION_MAX)


def _input_hash(state: dict) -> str:
    """A stable fingerprint of the investigation inputs."""
    payload = {
        "baseline": state["baseline"].to_json(orient="records"),
        "current": state["current"].to_json(orient="records"),
        "dims": list(state.get("dims", [])),
        "metrics": list(state.get("metrics", [])),
        "metric_kinds": dict(state.get("metric_kinds", {})),
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def _gate_flags(rep) -> dict:
    """Recover the four structural gate decisions from a GateReport, using the
    same thresholds gates.py applied."""
    import math
    p = rep.leading_p
    return {
        "material": abs(rep.delta_R_relative) >= MATERIAL_REL,
        "localized": rep.concentration >= CONCENTRATION_MIN,
        "significant": (not math.isnan(p)) and p <= SIGNIFICANCE_ALPHA,
        "clean": rep.interaction_share <= INTERACTION_MAX,
    }


def create_audit_trail(state: dict, reports: dict, agent_trace: list,
                       model: str, mode: str, cost: dict | None = None) -> dict:
    ih = _input_hash(state)
    win = state.get("winning_report")
    an = (state.get("anomalies") or [None])[0]
    action = (state.get("actions") or [None])[0]
    cause = f"{state.get('winning_dim')}={win.leading_segment}" if win is not None else None
    return {
        "investigation_id": ih[:16],
        "input_hash": ih,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "model": model,
        "metric": state.get("target_metric"),
        "dims": list(state.get("dims", [])),
        "delta_rel": round(an.delta_rel, 6) if an else None,
        "tool_calls": list(agent_trace),
        "winning_dim": state.get("winning_dim"),
        "verdict": state.get("verdict"),
        "confidence": state.get("confidence"),
        "cause": cause,
        "gates": _gate_flags(win) if win is not None else None,
        "gates_by_dim": {d: _gate_flags(r) for d, r in reports.items()},
        "report": state.get("report"),
        "action": ({"kind": action.kind, "detail": action.detail}
                   if action is not None else None),
        "cost": cost or {},
    }


def verify_replay(original_trail: dict, replayed_state: dict) -> bool:
    """True iff a fresh run reproduces the trail's verdict, confidence and
    cause. Prints a diff on mismatch."""
    win = replayed_state.get("winning_report")
    cause = f"{replayed_state.get('winning_dim')}={win.leading_segment}" if win else None
    checks = {
        "verdict": (original_trail.get("verdict"), replayed_state.get("verdict")),
        "confidence": (round(original_trail.get("confidence") or 0.0, 6),
                       round(replayed_state.get("confidence") or 0.0, 6)),
        "cause": (original_trail.get("cause"), cause),
        "input_hash": (original_trail.get("input_hash"), _input_hash(replayed_state)),
    }
    ok = all(a == b for a, b in checks.values())
    if not ok:
        diffs = {k: v for k, v in checks.items() if v[0] != v[1]}
        print(f"[audit] replay mismatch: {diffs}")
    return ok


def export_audit(investigation_id: str) -> str:
    """Pretty-JSON export of a persisted investigation (from memory.py)."""
    from prove_or_abstain import memory
    inv = memory.get_investigation(investigation_id)
    return json.dumps(inv, indent=2, ensure_ascii=False, default=str)


if __name__ == "__main__":
    import os
    os.environ.setdefault("QWEN_MOCK", "1")
    os.environ.setdefault("PROBATIO_DB", ":memory:")
    from prove_or_abstain.panels import BASELINE, CLEAN
    from prove_or_abstain.agent_loop import investigate_agentic

    st = {"baseline": BASELINE, "current": CLEAN,
          "metrics": ["conversion", "activation"], "metric_kinds": {},
          "dims": ["device", "segment"], "autopilot_enabled": True, "trace": []}
    final = investigate_agentic(dict(st))
    trail = create_audit_trail(final, final["reports_by_dim"],
                               final["agent_trace"], final["llm"]["model"], "agent")
    print(json.dumps({k: trail[k] for k in
                      ("investigation_id", "verdict", "confidence", "gates")}, indent=2))
    replay = investigate_agentic(dict(st))
    print("reproducible:", verify_replay(trail, replay))
