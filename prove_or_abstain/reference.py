"""
reference.py — pool a watched source's PRIOR observations into one baseline.

Reuses the pooling algebra already proven in panels.py::split_series
(sum raw counts cell-by-cell across several periods) — that's what keeps the
significance gate's two-proportion z-test valid, since it needs real
(c0, n0, c1, n1) counts, not an average of already-computed rates. The
algebra is duplicated here (4 lines), not called from split_series, because
the pooling axis is different: split_series pools PERIODS found inside one
caller-supplied panel; this pools OBSERVATIONS persisted over time for a
source_id (prove_or_abstain/memory.py) — no period column exists in a
stored observation, so the exclusion set is (n, c) alone, not
(period_col, n, c) — see panels.py:127 for the one-line comparison.

Computes nothing about materiality/significance — pure data pooling. The
gates (gates.py) decide everything downstream, unchanged.
"""
from __future__ import annotations

import pandas as pd

from prove_or_abstain import memory


def build_reference_window(source_id: str, dims: list[str],
                           window: int | None = None) -> pd.DataFrame:
    """Pool the last `window` PRIOR observations for source_id (all of them
    if window is None) into a single long-panel baseline [metric,
    <dims...>, n, c] — the exact shape detector()/gates.py already expect.

    Raises ValueError if there are zero prior observations — callers must
    have already handled the cold-start case (see ingest.py) before calling
    this; there is nothing to pool on the very first sighting of a source.

    Ordering contract: call this BEFORE recording the new/current
    observation via memory.record_observation() — this function pools
    whatever is already in memory.observations for source_id, so recording
    the current one first would fold it into its own baseline.
    """
    observations = memory.get_observations(source_id)
    if not observations:
        raise ValueError(f"no prior observations for source_id={source_id!r} "
                         f"— cold start must be handled before pooling a reference")
    if window is not None:
        observations = observations[-window:]

    panels = [o["panel"] for o in observations]
    concat = pd.concat(panels, ignore_index=True)
    keys = [c for c in concat.columns if c not in ("n", "c")]
    return concat.groupby(keys, as_index=False)[["n", "c"]].sum()
