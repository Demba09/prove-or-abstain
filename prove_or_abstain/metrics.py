"""
metrics.py — aggregation of a decomposition.

aggregate() recomposes the R0/R1 totals, sums the contributions and exposes
the residual (must be ~0). Kept out of the 'reference' (oracle) file so that
production code does not depend on it.
"""
from __future__ import annotations
import pandas as pd


def aggregate(out: pd.DataFrame) -> dict:
    R0 = float((out.w0 * out.r0).sum())
    R1 = float((out.w1 * out.r1).sum())
    return {
        "R0": R0, "R1": R1, "delta_R": R1 - R0,
        "rate": float(out.rate.sum()),
        "mix": float(out.mix.sum()),
        "interaction": float(out.interaction.sum()),
        "sum_contrib": float(out.contribution.sum()),
        "residual": (R1 - R0) - float(out.contribution.sum()),
    }
