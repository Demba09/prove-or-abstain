"""Shared state contract for the investigation graph.

This is the spine of the system. Every node reads the state, fills its slice,
and returns it. Freeze this first — everything else plugs into it.

Design rule that runs through the whole architecture:
    Numbers come from pandas/numpy. Never from the LLM.
    The LLM prioritises, orients, and narrates. It does not compute.
"""

from __future__ import annotations

from typing import Optional, TypedDict


class MetricConfig(TypedDict):
    name: str                 # e.g. "churn_30d"
    kind: str                 # "rate" | "sum"  -> selects the attribution math
    grain: str                # period column granularity, e.g. "week"
    baseline_window: int      # how many prior periods define the baseline
    dimensions: list[str]     # candidate explanatory dimensions
    churn_col: str            # 0/1 outcome column for rate metrics
    period_col: str           # column holding period index (0 = baseline, 1 = current)


class Anomaly(TypedDict):
    metric: str
    period: str
    baseline: float
    observed: float
    delta: float              # observed - baseline (signed)
    direction: str            # "up" | "down"


class Hypothesis(TypedDict):
    id: str
    dimension: str            # which dimension the cause lives in
    segment: str              # which segment of that dimension
    rationale: str            # LLM-written, why this is plausible
    status: str               # "pending" | "tested" | "supported" | "rejected"


class Verdict(TypedDict):
    top_cause: Optional[str]
    confidence: str           # "high" | "medium" | "low"
    explained_share: float    # share of delta explained by the top hypothesis
    separation: float         # gap between top-1 and top-2 explained share
    supporting: list[str]     # hypothesis ids that hold up
    rejected: list[str]
    abstained: bool           # THE differentiator: proved or abstained


class Action(TypedDict):
    # The terminal decision. The agent acts only on proof, and disengages otherwise.
    type: str                 # "recommend" | "escalate_to_human" | "no_op"
    target: Optional[str]     # e.g. "channel=paid_partner_x" for a scoped action
    rationale: str            # sourced, plain-language justification
    payload: dict             # structured details for a downstream system
    requires_human: bool      # True when abstaining -> hand back the wheel
    executed: bool            # only ever True under high confidence + autopilot flag


class AgentState(TypedDict):
    # --- Input ---
    dataset_ref: str          # handle to the dataframe / table (here: a parquet path)
    metric_config: MetricConfig

    # --- Filled by the Detector ---
    anomaly: Optional[Anomaly]

    # --- Filled by the Hypothesizer ---
    hypotheses: list[Hypothesis]

    # --- Filled by the Investigator ---
    # {hypothesis_id: {contribution, contribution_share, segment_size,
    #                  rate_before, rate_after, rate_effect, mix_effect, interaction}}
    evidence: dict

    # --- Filled by the Verifier ---
    verdict: Optional[Verdict]

    # --- Filled by the Actuator (the terminal decision) ---
    action: Optional[Action]

    # --- Loop control ---
    iteration: int            # incremented on each loop-back, bounded

    # --- Filled by the Reporter ---
    report: Optional[dict]
