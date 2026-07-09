"""Reporter node (J10). Renders the sourced report + audit trail: the anomaly, the
ranked evidence, the verdict (assertion or abstention), and every number with its
provenance. This is what becomes the LinkedIn before/after.

STUB: assembles a minimal report dict from current state.
"""
from __future__ import annotations


def report(state: dict) -> dict:
    state["report"] = {
        "anomaly": state.get("anomaly"),
        "verdict": state.get("verdict"),
        "action": state.get("action"),
        "evidence": state.get("evidence", {}),
        "iterations": state.get("iteration", 0),
    }
    return state
