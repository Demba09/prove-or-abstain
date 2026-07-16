"""
prove_or_abstain/panels.py — multimetric/multidim demo data + projection.

Long panel: one row per atomic cell [metric, segment, device, n, c]
(raw counts; rates are ALWAYS derived downstream).

project() projects the panel onto ONE metric + ONE dimension via groupby:
that is the "per-hypothesis projection" layer. decompose() then operates
on the projection.

Four calibrated scenarios (opposite structures on 'conversion'):
  CLEAN    : only the 'paid' segment breaks. Localizes along 'segment' but
             looks diffuse along 'device' (paid is ~50/50 mobile/desktop).
             -> the loop tries 'device' (ABSTAIN) then 'segment' (ASSERT).
  DIFFUSE  : every rate drops uniformly. No dimension localizes.
             -> dimensions exhausted -> final ABSTAIN.
  MIXSHIFT : composition AND rates move at once (rate + mix + interaction
             all non-zero). Same per-segment totals as the MIXSHIFT
             validated in gate_check_gates.py -> ABSTAIN, entangled mechanism.
  DEEP     : only paid × mobile collapses -> ASSERT device=mobile, then the
             drill-down refines: segment=paid within mobile.
"""
from __future__ import annotations
import pandas as pd

SEGMENTS = ["organic", "paid", "referral", "email"]
DEVICES = ["mobile", "desktop"]

# n per (segment, device). 'paid' deliberately 50/50 mobile/desktop.
_N = {
    ("organic",  "mobile"): 5000, ("organic",  "desktop"): 5000,
    ("paid",     "mobile"): 3000, ("paid",     "desktop"): 3000,
    ("referral", "mobile"): 1500, ("referral", "desktop"): 1500,
    ("email",    "mobile"): 500,  ("email",    "desktop"): 500,
}
# baseline conversion rate per segment (device-independent)
_RATE0 = {"organic": 0.05, "paid": 0.07, "referral": 0.08, "email": 0.12}
# baseline activation rate (stable metric, serves as a decoy for the detector)
_ACT0 = {"organic": 0.30, "paid": 0.32, "referral": 0.28, "email": 0.35}


def _rows(rate_conv, rate_act, n_map=None):
    """Build a long panel from (segment, device) -> rate functions."""
    n_map = _N if n_map is None else n_map
    rows = []
    for seg in SEGMENTS:
        for dev in DEVICES:
            n = n_map[(seg, dev)]
            rows.append({"metric": "conversion", "segment": seg, "device": dev,
                         "n": n, "c": round(n * rate_conv(seg, dev))})
            rows.append({"metric": "activation", "segment": seg, "device": dev,
                         "n": n, "c": round(n * rate_act(seg, dev))})
    return pd.DataFrame(rows)


BASELINE = _rows(lambda s, d: _RATE0[s], lambda s, d: _ACT0[s])

# CLEAN: 'paid' conversion drops 7.0% -> 5.0%; everything else unchanged.
CLEAN = _rows(
    lambda s, d: 0.05 if s == "paid" else _RATE0[s],
    lambda s, d: _ACT0[s],
)

# DIFFUSE: EVERY segment's conversion drops by 0.6pp; same aggregate ΔR.
DIFFUSE = _rows(
    lambda s, d: _RATE0[s] - 0.006,
    lambda s, d: _ACT0[s],
)

# DEEP: only the paid × mobile cell collapses (7.0% -> 3.0%).
# Localizes first on ONE dimension (device=mobile), then the drill-down
# refines within mobile: segment=paid. Demonstrates the refinement.
DEEP = _rows(
    lambda s, d: 0.03 if (s, d) == ("paid", "mobile") else _RATE0[s],
    lambda s, d: _ACT0[s],
)

# MIXSHIFT: composition moves (organic grows, paid shrinks) AND rates move
# (organic drops, paid/email...) -> rate, mix and interaction all non-zero.
# Same per-segment totals as gate_check_gates.py's MIXSHIFT, split 50/50
# mobile/desktop so it stays diffuse along 'device'.
_N_MIXSHIFT = {
    ("organic",  "mobile"): 7000, ("organic",  "desktop"): 7000,
    ("paid",     "mobile"): 2000, ("paid",     "desktop"): 2000,
    ("referral", "mobile"): 1500, ("referral", "desktop"): 1500,
    ("email",    "mobile"): 500,  ("email",    "desktop"): 500,
}
_RATE_MIXSHIFT = {"organic": 0.045, "paid": 0.08, "referral": 0.08, "email": 0.10}

MIXSHIFT = _rows(
    lambda s, d: _RATE_MIXSHIFT[s],
    lambda s, d: _ACT0[s],
    n_map=_N_MIXSHIFT,
)


def metric_totals(panel: pd.DataFrame, metric: str) -> tuple[float, float]:
    """(n_total, c_total) for one metric, across all dimensions."""
    sub = panel[panel["metric"] == metric]
    return float(sub["n"].sum()), float(sub["c"].sum())


def project(panel: pd.DataFrame, metric: str, dim: str) -> pd.DataFrame:
    """Project the panel onto (metric, dim): groupby dim, sum the raw counts.
    Returns [dim, n, c] — ready for decompose(base, curr, dims=dim)."""
    sub = panel[panel["metric"] == metric]
    return sub.groupby(dim, as_index=False)[["n", "c"]].sum()


# --------------------------------------------------------------- time series
def split_series(panel: pd.DataFrame, window: int | None = None,
                 period_col: str = "period") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a multi-period long panel into (baseline, current).

    current  = the last period.
    baseline = the `window` preceding periods (all of them if None), POOLED by
               summing n and c per cell — a rolling baseline, more robust than
               a single reference period.
    """
    if window is not None and window < 1:
        raise ValueError("window must be >= 1")
    periods = sorted(panel[period_col].unique())
    if len(periods) < 2:
        raise ValueError("a series panel needs at least 2 periods")
    base_periods = periods[:-1] if window is None else periods[-1 - window:-1]

    keys = [col for col in panel.columns if col not in (period_col, "n", "c")]
    base = (panel[panel[period_col].isin(base_periods)]
            .groupby(keys, as_index=False)[["n", "c"]].sum())
    curr = (panel[panel[period_col] == periods[-1]]
            .groupby(keys, as_index=False)[["n", "c"]].sum())
    return base, curr


def make_series(n_periods: int = 8) -> pd.DataFrame:
    """Multi-period demo panel: n_periods-1 stable weeks (BASELINE), then the
    last week breaks (CLEAN). Deterministic."""
    frames = []
    for p in range(n_periods - 1):
        f = BASELINE.copy()
        f["period"] = p
        frames.append(f)
    last = CLEAN.copy()
    last["period"] = n_periods - 1
    frames.append(last)
    return pd.concat(frames, ignore_index=True)
