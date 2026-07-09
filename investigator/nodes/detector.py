"""Detector node (J3). Statistical, deterministic, explainable. The LLM does not
detect — it only narrates later. Compares current-period rate to baseline; emits
an Anomaly if the relative deviation exceeds the threshold, else None.

STUB: returns a placeholder anomaly so the skeleton flows end-to-end.
Real logic is timeboxed to ~0.5 day on J3.
"""
from __future__ import annotations

from ..config import ANOMALY_REL_THRESHOLD  # noqa: F401  (used by real impl)


def detect(state: dict) -> dict:
    # TODO J3: load df from state["dataset_ref"], compute aggregate metric per grain,
    # compare to baseline_window, set anomaly or None.
    state["anomaly"] = {
        "metric": state["metric_config"]["name"],
        "period": "current",
        "baseline": 0.07,
        "observed": 0.094,
        "delta": 0.024,
        "direction": "up",
    }  # placeholder
    return state
