"""Investigator node (J5-J7, the densest). For each hypothesis, quantify its real
contribution to the anomaly via the rate decomposition. ALL numbers from
pandas/numpy. The analytic core already lives in investigator.attribution.

STUB: leaves evidence empty; wiring to attribution.decompose_rate_metric comes J5.
"""
from __future__ import annotations

from ..attribution import decompose_rate_metric  # noqa: F401  (used by real impl)


def investigate(state: dict) -> dict:
    # TODO J5-J7: for each hypothesis dimension, run decompose_rate_metric on the
    # loaded df, pick the top segment, write contribution/share/rate_before/after
    # into state["evidence"][hypothesis_id].
    state["evidence"] = {}  # placeholder
    return state
