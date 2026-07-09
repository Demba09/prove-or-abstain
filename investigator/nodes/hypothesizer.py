"""Hypothesizer node (J4). The LLM PRIORITISES which dimensions are plausible
given the metric semantics; each hypothesis is "the anomaly is concentrated in
segment X of dimension Y" and must be falsifiable by a precise computation.

STUB: emits one hypothesis per configured dimension (top segment TBD by the
Investigator). Real prioritisation is timeboxed to ~0.5 day on J4.
"""
from __future__ import annotations


def hypothesize(state: dict) -> dict:
    dims = state["metric_config"]["dimensions"]
    # TODO J4: ask Qwen to rank dimensions by plausibility for this metric.
    state["hypotheses"] = [
        {"id": f"h{i}", "dimension": d, "segment": "*", "rationale": "stub", "status": "pending"}
        for i, d in enumerate(dims)
    ]
    return state
