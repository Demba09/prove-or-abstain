"""
attribution_reference.py — reference ORACLE for Probatio.

Purpose: an independently written, verified implementation of the
rate/mix/interaction decomposition. The production version (attribution.py,
identical signature) is diffed against this one in gate_check.py and in the
test suite. As long as the residual is not ~0 and the contributions do not
match the oracle, the math layer is not trusted.

Convention: raw counts are primary, rates are ALWAYS derived. No LLM here.
"""
import numpy as np
import pandas as pd


def decompose(base: pd.DataFrame, curr: pd.DataFrame, dims, n_col="n", c_col="c") -> pd.DataFrame:
    """
    base, curr: panels with columns [*dims, n_col, c_col] (raw counts).
    Returns a DataFrame indexed by segment with w0,w1,r0,r1 and the
    rate/mix/interaction contributions + total contribution.

    Exact identity:  w1*r1 - w0*r0 = w0*dr + r0*dw + dw*dr
    """
    b = base.set_index(dims if isinstance(dims, list) else [dims])
    c = curr.set_index(dims if isinstance(dims, list) else [dims])
    idx = b.index.union(c.index)                      # segments in either panel
    b = b.reindex(idx, fill_value=0)
    c = c.reindex(idx, fill_value=0)

    N0, N1 = b[n_col].sum(), c[n_col].sum()
    w0, w1 = b[n_col] / N0, c[n_col] / N1             # mix shares
    r0 = (b[c_col] / b[n_col]).replace([np.inf, -np.inf], 0).fillna(0)   # derived rates
    r1 = (c[c_col] / c[n_col]).replace([np.inf, -np.inf], 0).fillna(0)

    dw, dr = w1 - w0, r1 - r0
    rate = w0 * dr                                    # mix frozen at baseline
    mix = r0 * dw                                     # rate frozen at baseline
    interaction = dw * dr                             # entangled share

    out = pd.DataFrame({"w0": w0, "w1": w1, "r0": r0, "r1": r1,
                        "rate": rate, "mix": mix, "interaction": interaction})
    out["contribution"] = out[["rate", "mix", "interaction"]].sum(axis=1)
    return out


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


def assert_matches(candidate_out: pd.DataFrame, dims, tol=1e-10):
    """Plug a candidate implementation here: checks zero residual + column
    consistency."""
    agg = aggregate(candidate_out)
    assert abs(agg["residual"]) < tol, f"non-zero residual: {agg['residual']:.3e}"
    recomposed = candidate_out["rate"] + candidate_out["mix"] + candidate_out["interaction"]
    assert np.allclose(recomposed, candidate_out["contribution"], atol=tol), "inconsistent contributions"
    return agg


# ----------------------------------------------------------------------
# Demo data: SAME aggregate ΔR, OPPOSITE attribution structure.
# ----------------------------------------------------------------------
def _panel(rows):
    return pd.DataFrame(rows)


BASELINE = _panel([
    {"segment": "organic",  "n": 10000, "c": 500},   # 5.0%
    {"segment": "paid",     "n": 6000,  "c": 420},   # 7.0%
    {"segment": "referral", "n": 3000,  "c": 240},   # 8.0%
    {"segment": "email",    "n": 1000,  "c": 120},   # 12.0%
])

# CLEAN: a single segment breaks (paid 7.0% -> 5.0%), stable mix.
CLEAN = _panel([
    {"segment": "organic",  "n": 10000, "c": 500},
    {"segment": "paid",     "n": 6000,  "c": 300},
    {"segment": "referral", "n": 3000,  "c": 240},
    {"segment": "email",    "n": 1000,  "c": 120},
])

# DIFFUSE: every rate drops ~0.6pp, stable mix. SAME aggregate ΔR.
DIFFUSE = _panel([
    {"segment": "organic",  "n": 10000, "c": 440},   # 5.0 -> 4.4
    {"segment": "paid",     "n": 6000,  "c": 384},   # 7.0 -> 6.4
    {"segment": "referral", "n": 3000,  "c": 222},   # 8.0 -> 7.4
    {"segment": "email",    "n": 1000,  "c": 114},   # 12.0 -> 11.4
])


def _abstain_signal(agg, out):
    """Illustrative heuristic: who dominates the attribution?"""
    top = out.contribution.abs().max()
    total = out.contribution.abs().sum()
    concentration = top / total if total else 0          # 1 = a single culprit
    inter_share = abs(agg["interaction"]) / (abs(agg["rate"]) + abs(agg["mix"]) + 1e-12)
    return concentration, inter_share


if __name__ == "__main__":
    pd.set_option("display.float_format", lambda x: f"{x:.5f}")
    for name, curr in [("CLEAN", CLEAN), ("DIFFUSE", DIFFUSE)]:
        out = decompose(BASELINE, curr, dims="segment")
        agg = aggregate(out)
        conc, inter = _abstain_signal(agg, out)
        print(f"\n=== {name} ===")
        print(out[["r0", "r1", "rate", "mix", "interaction", "contribution"]])
        print(f"\n  R0={agg['R0']:.4%}  R1={agg['R1']:.4%}  ΔR={agg['delta_R']:+.4%}")
        print(f"  rate={agg['rate']:+.5f}  mix={agg['mix']:+.5f}  interaction={agg['interaction']:+.5f}")
        print(f"  Σcontrib={agg['sum_contrib']:+.5f}   RESIDUAL={agg['residual']:+.2e}")
        print(f"  top-segment concentration={conc:.2f}   interaction weight={inter:.3f}")
        verdict = "ASSERT (localized cause)" if conc >= 0.55 else "ABSTAIN (diffuse cause)"
        print(f"  => {verdict}")
