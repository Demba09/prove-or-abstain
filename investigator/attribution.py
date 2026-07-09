"""Contribution attribution — the analytic core.

Pure pandas/numpy. No LLM anywhere near a number. This module is used in two
places: the Investigator node (to test hypotheses) and the dataset generator's
validation (to prove scenario 1 is localizable and scenario 2 is diffuse).

Rate-metric decomposition
-------------------------
Aggregate rate R = sum_s w_s * r_s  (segment weight x segment rate).

    dR = R1 - R0 = sum_s (w1_s * r1_s - w0_s * r0_s)

Per-segment decomposition:
    rate_effect   = w0_s * (r1_s - r0_s)          # the segment's rate moved
    mix_effect    = r0_s * (w1_s - w0_s)          # the composition moved
    interaction   = (w1_s - w0_s) * (r1_s - r0_s)
    contribution  = w1_s*r1_s - w0_s*r0_s         # = rate + mix + interaction

The contributions sum to dR exactly. explained_share = contribution / dR.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def decompose_rate_metric(
    df: pd.DataFrame,
    dimension: str,
    churn_col: str = "churned",
    period_col: str = "period",
    baseline_period: int = 0,
    current_period: int = 1,
) -> pd.DataFrame:
    """Decompose the change in an aggregate rate along one dimension.

    Returns one row per segment with rate/mix/interaction effects, the total
    contribution, and explained_share (contribution / dR). Rows are sorted by
    descending |contribution|. The sum of `contribution` equals dR exactly
    (up to float error), which is the property that makes the result auditable.
    """
    base = df[df[period_col] == baseline_period]
    curr = df[df[period_col] == current_period]

    n0_total = len(base)
    n1_total = len(curr)
    if n0_total == 0 or n1_total == 0:
        raise ValueError("Both baseline and current periods must be non-empty.")

    segments = sorted(set(df[dimension].dropna().unique()))
    rows = []
    for s in segments:
        b = base[base[dimension] == s]
        c = curr[curr[dimension] == s]

        n0, n1 = len(b), len(c)
        w0 = n0 / n0_total
        w1 = n1 / n1_total
        r0 = b[churn_col].mean() if n0 else 0.0
        r1 = c[churn_col].mean() if n1 else 0.0

        rate_effect = w0 * (r1 - r0)
        mix_effect = r0 * (w1 - w0)
        interaction = (w1 - w0) * (r1 - r0)
        contribution = w1 * r1 - w0 * r0  # == rate + mix + interaction

        rows.append(
            {
                "dimension": dimension,
                "segment": s,
                "segment_size": int(n1),
                "weight_before": w0,
                "weight_after": w1,
                "rate_before": float(r0),
                "rate_after": float(r1),
                "rate_effect": rate_effect,
                "mix_effect": mix_effect,
                "interaction": interaction,
                "contribution": contribution,
            }
        )

    out = pd.DataFrame(rows)
    delta_r = float(out["contribution"].sum())  # = R1 - R0
    out["explained_share"] = out["contribution"] / delta_r if delta_r != 0 else np.nan
    out = out.reindex(out["contribution"].abs().sort_values(ascending=False).index)
    out.attrs["delta_r"] = delta_r
    return out.reset_index(drop=True)


def aggregate_rate(df: pd.DataFrame, churn_col: str, period_col: str, period: int) -> float:
    """Aggregate rate for a period — the thing the Detector compares to baseline."""
    sub = df[df[period_col] == period]
    return float(sub[churn_col].mean()) if len(sub) else 0.0
