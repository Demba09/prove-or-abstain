"""
agent_state.py — state contract of the Probatio graph.

AgentState is a TypedDict: the schema LangGraph passes between the nodes.
Each node receives the current state and returns a partial dict of the keys
it changes; LangGraph merges.

Design rule running through the whole architecture:
    Numbers come from pandas/numpy. Never from the LLM.
    The LLM prioritizes, phrases, and speculates (labelled). It does not compute.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, TypedDict

import pandas as pd

from prove_or_abstain.gates import GateReport


@dataclass
class MetricAnomaly:
    """A metric whose aggregate move exceeds the materiality threshold."""
    metric: str
    R0: float
    R1: float
    delta_rel: float        # |ΔR| / R0


@dataclass
class Action:
    """Typed decision produced by the actuator. ABSTAIN never yields EXECUTE."""
    kind: str               # "EXECUTE" | "RECOMMEND" | "ESCALATE" | "NONE"
    metric: str
    dim: Optional[str] = None
    segment: Any = None
    detail: str = ""


class AgentState(TypedDict, total=False):
    # --- inputs ---
    baseline: pd.DataFrame          # long panel: [metric, *dims, n, c]
    current: pd.DataFrame
    metrics: list[str]
    metric_kinds: dict              # metric -> "rate" (default) | "sum"
    dims: list[str]                 # candidate dimensions, in trial order
    autopilot_enabled: bool

    # --- detection ---
    anomalies: list[MetricAnomaly]
    target_metric: str              # the metric under investigation

    # --- investigation loop ---
    dims_to_try: list[str]          # queue of remaining dimensions
    iteration: int
    max_iterations: int
    current_dim: str
    hypothesis: str                 # plain-text hypothesis for the current dim
    dims_planned: bool              # has the LLM already ordered the dimensions?
    investigation: dict             # {out, agg, kind} for the current dim
    gate_report: GateReport         # current dimension's report
    reports_by_dim: dict            # dim -> GateReport (every dim tried)

    # --- final verdict ---
    verdict: str                    # "ASSERT" | "ABSTAIN" | "NO_ANOMALY"
    confidence: float
    winning_dim: Optional[str]
    winning_report: Optional[GateReport]

    # --- drill-down (after ASSERT: refine within the winning segment) ---
    drilldown: Optional[dict]       # {parent, reports_by_dim, refined}

    # --- output ---
    actions: list[Action]
    report: str
    speculations: list[str]         # LLM business hypotheses, labelled speculation
    llm: dict                       # {"model": ..., "mode": "mock"|"real"|"fallback"}
    trace: list[str]                # readable investigation log
    agent_trace: list[dict]         # Qwen's tool-calling steps (agent mode only)
