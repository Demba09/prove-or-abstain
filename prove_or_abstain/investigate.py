"""
investigate.py — the shared tail every entry point funnels through: build
the state, invoke the graph or the agent loop, serialize the result.

Moved out of api/app.py so prove_or_abstain/ingest.py (the "Watch a source"
path) can call it without api/app.py importing ingest.py and ingest.py
importing api/app.py back — a circular import. Pure relocation: same code,
same behaviour, only the import path changed at the call sites.
"""
from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np
import pandas as pd

from prove_or_abstain.autopilot import record_execution
from prove_or_abstain.webhook import notify
from prove_or_abstain.agent_loop import investigate_agentic
from prove_or_abstain.graph import APP as INVESTIGATION_GRAPH
from prove_or_abstain.llm import get_client


def _jsonable(v):
    """Flatten numpy scalars / NaN in a final state down to strict JSON —
    necessary as soon as the panels come from a user CSV."""
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def _run_investigation(baseline: pd.DataFrame, current: pd.DataFrame,
                       metrics: list[str], dims: list[str],
                       autopilot: bool,
                       metric_kinds: dict | None = None,
                       mode: str = "graph") -> dict:
    state = {
        "baseline": baseline,
        "current": current,
        "metrics": metrics,
        "metric_kinds": metric_kinds or {},
        "dims": dims,
        "autopilot_enabled": autopilot,
        "trace": [],
    }
    # Snapshot the cost tracker so `cost` reports THIS request's spend, not the
    # process-cumulative total (0 in mock mode).
    _client = get_client()
    _tok0, _usd0 = _client.tracker.total_tokens, _client.tracker.cost_usd

    # Both paths produce the same verdict; "agent" adds Qwen's tool-call trace.
    final = investigate_agentic(state) if mode == "agent" \
        else INVESTIGATION_GRAPH.invoke(state)

    win = final.get("winning_report")
    drill = final.get("drilldown")

    actions = final.get("actions")
    if actions and actions[0].kind == "EXECUTE":
        a = actions[0]
        cause = f"{a.dim}={a.segment}" if a.dim else None
        record_execution(
            a.metric, a.dim, a.segment, final.get("confidence", 0.0),
            a.kind, a.detail,
            final.get("report", ""), final.get("trace", []),
        )
        notify(a.metric, final.get("verdict", "ASSERT"),
               final.get("confidence", 0.0),
               cause, a.kind, a.detail)

    return _jsonable({
        "verdict": final.get("verdict"),
        "confidence": final.get("confidence"),
        "root_cause": (
            {"dimension": final.get("winning_dim"), "segment": win.leading_segment}
            if win is not None
            else None
        ),
        "gates": {dim: asdict(rep) for dim, rep in final.get("reports_by_dim", {}).items()},
        "drilldown": (
            {"parent": drill["parent"],
             "refined": drill["refined"],
             "gates": {d: asdict(r) for d, r in drill["reports_by_dim"].items()}}
            if drill
            else None
        ),
        "action": asdict(final["actions"][0]) if final.get("actions") else None,
        "report": final.get("report"),
        "speculations": final.get("speculations", []),
        "llm": final.get("llm"),
        "cost": {
            "model": _client.tracker.model,
            "tokens": _client.tracker.total_tokens - _tok0,
            "usd": round(_client.tracker.cost_usd - _usd0, 6),
        },
        "trace": final.get("trace", []),
        "agent_trace": final.get("agent_trace", []),
    })
