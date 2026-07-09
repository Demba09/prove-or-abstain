"""
attribution_reference.py — ORACLE de référence pour Probatio (Phase 0).

But : implémentation vérifiée de la décomposition rate/mix/interaction.
Tu écris TA version (signature identique) et tu la compares à celle-ci avec
`assert_matches`. Tant que le résidu n'est pas ~0 et que tes contributions
ne collent pas à l'oracle, Phase 0 n'est pas franchie.

Convention : counts bruts primaires, taux TOUJOURS dérivés. Le LLM ne calcule rien ici.
"""
import numpy as np
import pandas as pd


def decompose(base: pd.DataFrame, curr: pd.DataFrame, dims, n_col="n", c_col="c") -> pd.DataFrame:
    """
    base, curr : panels avec colonnes [*dims, n_col, c_col] (counts bruts).
    Retourne un DataFrame indexé par segment avec w0,w1,r0,r1 et
    les contributions rate/mix/interaction + contribution totale.

    Identité exacte :  w1*r1 - w0*r0 = w0*dr + r0*dw + dw*dr
    """
    b = base.set_index(dims if isinstance(dims, list) else [dims])
    c = curr.set_index(dims if isinstance(dims, list) else [dims])
    idx = b.index.union(c.index)                      # segments présents dans l'un OU l'autre
    b = b.reindex(idx, fill_value=0)
    c = c.reindex(idx, fill_value=0)

    N0, N1 = b[n_col].sum(), c[n_col].sum()
    w0, w1 = b[n_col] / N0, c[n_col] / N1             # parts de mix
    r0 = (b[c_col] / b[n_col]).replace([np.inf, -np.inf], 0).fillna(0)   # taux dérivés
    r1 = (c[c_col] / c[n_col]).replace([np.inf, -np.inf], 0).fillna(0)

    dw, dr = w1 - w0, r1 - r0
    rate = w0 * dr                                    # mix figé au baseline
    mix = r0 * dw                                     # taux figé au baseline
    interaction = dw * dr                             # part entremêlée

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
    """Branche TA version ici : vérifie résidu nul + cohérence colonne par colonne."""
    agg = aggregate(candidate_out)
    assert abs(agg["residual"]) < tol, f"résidu non nul : {agg['residual']:.3e}"
    recomposed = candidate_out["rate"] + candidate_out["mix"] + candidate_out["interaction"]
    assert np.allclose(recomposed, candidate_out["contribution"], atol=tol), "contributions incohérentes"
    return agg


# ----------------------------------------------------------------------
# Données de démo : MÊME ΔR agrégé, structure d'attribution OPPOSÉE.
# ----------------------------------------------------------------------
def _panel(rows, dims=("segment",)):
    return pd.DataFrame(rows)


BASELINE = _panel([
    {"segment": "organic",  "n": 10000, "c": 500},   # 5.0%
    {"segment": "paid",     "n": 6000,  "c": 420},   # 7.0%
    {"segment": "referral", "n": 3000,  "c": 240},   # 8.0%
    {"segment": "email",    "n": 1000,  "c": 120},   # 12.0%
])

# CLEAN : un seul segment décroche (paid 7.0% -> 5.0%), mix stable.
CLEAN = _panel([
    {"segment": "organic",  "n": 10000, "c": 500},
    {"segment": "paid",     "n": 6000,  "c": 300},
    {"segment": "referral", "n": 3000,  "c": 240},
    {"segment": "email",    "n": 1000,  "c": 120},
])

# DIFFUSE : tous les taux baissent de ~0.6pp, mix stable. MÊME ΔR global.
DIFFUSE = _panel([
    {"segment": "organic",  "n": 10000, "c": 440},   # 5.0 -> 4.4
    {"segment": "paid",     "n": 6000,  "c": 384},   # 7.0 -> 6.4
    {"segment": "referral", "n": 3000,  "c": 222},   # 8.0 -> 7.4
    {"segment": "email",    "n": 1000,  "c": 114},   # 12.0 -> 11.4
])


def _abstain_signal(agg, out):
    """Heuristique illustrative : qui domine l'attribution ?"""
    top = out.contribution.abs().max()
    total = out.contribution.abs().sum()
    concentration = top / total if total else 0          # 1 = un seul coupable
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
        print(f"  Σcontrib={agg['sum_contrib']:+.5f}   RÉSIDU={agg['residual']:+.2e}")
        print(f"  concentration top-segment={conc:.2f}   poids interaction={inter:.3f}")
        verdict = "ASSERT (cause localisée)" if conc >= 0.55 else "ABSTAIN (cause diffuse)"
        print(f"  => {verdict}")
