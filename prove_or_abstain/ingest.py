"""
ingest.py — "Watch a source": the single entry point for the continuous-
ingestion capability (as opposed to "Compare two snapshots", the existing
explicit baseline+current endpoints, which are untouched by this module).

One observation in, and the pipeline either seeds (cold start, nothing to
compare yet) or investigates it against the pooled reference window of every
PRIOR observation persisted for that source_id (prove_or_abstain/memory.py +
reference.py). Never duplicates the state-building/graph-invocation logic —
that's still api/app.py's _run_investigation (moved to
prove_or_abstain/investigate.py so this module can import it without a
circular dependency on api/app.py).
"""
from __future__ import annotations

import pandas as pd

from prove_or_abstain import memory, reference
from prove_or_abstain.investigate import _run_investigation


def ingest_and_investigate(source_id: str, panel: pd.DataFrame,
                           metrics: list[str], dims: list[str],
                           metric_kinds: dict | None = None,
                           autopilot: bool = False, mode: str = "graph",
                           window: int | None = None) -> dict:
    """Record one observation for source_id, then either seed (first
    sighting of this source_id — cold start) or investigate it against the
    pooled reference window of everything persisted so far for it."""
    prior_count = memory.count_observations(source_id)

    if prior_count == 0:
        memory.record_observation(source_id, panel, dims, metrics)
        return {"cold_start": True, "verdict": "BASELINE_SET", "source_id": source_id}

    # Build the reference window from PRIOR observations only, before this
    # one is recorded — see reference.py's ordering contract.
    baseline = reference.build_reference_window(source_id, dims, window=window)
    memory.record_observation(source_id, panel, dims, metrics)

    result = _run_investigation(baseline, panel, metrics=metrics, dims=dims,
                                autopilot=autopilot, metric_kinds=metric_kinds,
                                mode=mode)
    return {"cold_start": False, "source_id": source_id, **result}
