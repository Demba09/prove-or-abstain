"""
attribution.py — the rate/mix/interaction decomposition (production version).

Same signature as attribution_reference.decompose(), which serves as the
independently written oracle it is validated against.
"""
import numpy as np
import pandas as pd


def decompose(base: pd.DataFrame, curr: pd.DataFrame, dims, n_col="n", c_col="c") -> pd.DataFrame:
    """
    base, curr: DataFrames with columns [*dims, n_col, c_col].

    Returns a DataFrame indexed by segment with:
    - n0, n1, c0, c1: raw counts (needed by the gates' z-test)
    - w0, w1: mix shares (baseline, current)
    - r0, r1: rates (baseline, current)
    - rate, mix, interaction: exact decomposition
    - contribution: their sum

    Identity: w1*r1 - w0*r0 = w0*dr + r0*dw + dw*dr
    """
    # STEP 1: normalize dims to a list
    if isinstance(dims, str):
        dims = [dims]

    # STEP 2: index by segment(s)
    b = base.set_index(dims)
    c = curr.set_index(dims)

    # STEP 3: merge the indexes (segments present in base OR curr)
    idx = b.index.union(c.index)
    b = b.reindex(idx, fill_value=0)
    c = c.reindex(idx, fill_value=0)

    # STEP 4: totals
    N0 = b[n_col].sum()   # baseline total
    N1 = c[n_col].sum()   # current total

    # STEP 5: mix shares w0, w1
    w0 = b[n_col] / N0
    w1 = c[n_col] / N1

    # STEP 6: rates r0, r1 (guarding against division by zero)
    r0 = (b[c_col] / b[n_col]).replace([np.inf, -np.inf], 0).fillna(0)
    r1 = (c[c_col] / c[n_col]).replace([np.inf, -np.inf], 0).fillna(0)

    # STEP 7: differential changes
    dw = w1 - w0  # mix change
    dr = r1 - r0  # rate change

    # STEP 8: exact decomposition
    rate = w0 * dr          # mix frozen at baseline
    mix = r0 * dw           # rate frozen at baseline
    interaction = dw * dr   # combined effect

    # STEP 9: build the output DataFrame
    out = pd.DataFrame({
        "n0": b[n_col], "n1": c[n_col],
        "c0": b[c_col], "c1": c[c_col],
        "w0": w0, "w1": w1,
        "r0": r0, "r1": r1,
        "rate": rate,
        "mix": mix,
        "interaction": interaction,
    })

    # STEP 10: total contribution (sum of the three)
    out["contribution"] = out[["rate", "mix", "interaction"]].sum(axis=1)

    return out


def decompose_sum(base: pd.DataFrame, curr: pd.DataFrame, dims,
                  n_col="n", c_col="c") -> pd.DataFrame:
    """
    Volume/rate decomposition for a SUM metric V = Σₛ nₛ·rₛ
    (e.g. revenue: n = customers, c = total amount, r = average basket).

    Same algebra as decompose(), but with raw counts in place of mix shares:
        ΔVₛ = n₀·Δr (rate) + r₀·Δn (mix/volume) + Δn·Δr (interaction)
    contributionₛ = c₁ₛ - c₀ₛ, and Σ contributions = ΔV exactly.

    w0/w1 are set to the raw counts so that metrics.aggregate()
    (R = Σ w·r) recomposes V0/V1 unchanged.
    """
    if isinstance(dims, str):
        dims = [dims]

    b = base.set_index(dims)
    c = curr.set_index(dims)
    idx = b.index.union(c.index)
    b = b.reindex(idx, fill_value=0)
    c = c.reindex(idx, fill_value=0)

    r0 = (b[c_col] / b[n_col]).replace([np.inf, -np.inf], 0).fillna(0)
    r1 = (c[c_col] / c[n_col]).replace([np.inf, -np.inf], 0).fillna(0)
    dn = c[n_col] - b[n_col]
    dr = r1 - r0

    out = pd.DataFrame({
        "n0": b[n_col], "n1": c[n_col],
        "c0": b[c_col], "c1": c[c_col],
        "w0": b[n_col], "w1": c[n_col],
        "r0": r0, "r1": r1,
        "rate": b[n_col] * dr,
        "mix": r0 * dn,
        "interaction": dn * dr,
    })
    out["contribution"] = out[["rate", "mix", "interaction"]].sum(axis=1)
    return out
