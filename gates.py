"""
gates.py — Probatio's ASSERT / ABSTAIN decision.

The computation (attribution.py) is ALWAYS exact (zero residual). The gates
recompute nothing: they decide whether that computation DESIGNATES a
localizable cause, or whether the agent should abstain.

Principle: ASSERT requires EVERY gate to pass. If a single one fails, we
ABSTAIN — and the reason is the failed gate(s). A justified refusal, not a
shrug. This is the autopilot's safety property: "an agent that can act must
be able to refuse to act".

The four gates:
  material    : the anomaly is large enough to deserve an explanation
  localized   : one segment genuinely dominates the contribution
  significant : the leading segment's move is not sampling noise
                (two-proportion z-test; for sum metrics, where per-unit
                variance is not observable, fall back to a sample floor)
  clean       : rate and mix effects are separable (bounded interaction)
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math

import numpy as np
import pandas as pd


# --- named thresholds (easy to justify in the README / the demo) ---
MATERIAL_REL = 0.02        # |ΔR|/R0: below this the anomaly is negligible
CONCENTRATION_MIN = 0.55   # top contributor's share: THE central gate
SIGNIFICANCE_ALPHA = 0.01  # max (two-sided) p-value of the leader's z-test
Z_CRIT = 2.576             # critical |z| matching alpha=0.01 (two-sided)
SAMPLE_FLOOR = 1000        # sum-metric fallback: min baseline n of the leader
INTERACTION_MAX = 0.50     # |interaction|/(|rate|+|mix|): above, mechanism entangled


@dataclass
class GateReport:
    verdict: str                  # "ASSERT" | "ABSTAIN"
    confidence: float             # 0..1, only meaningful on ASSERT
    leading_segment: object       # leading segment (index value)
    concentration: float
    interaction_share: float
    leading_sample_n: float
    leading_z: float              # leader's z-test statistic (NaN if not applicable)
    leading_p: float              # two-sided p-value (NaN if not applicable)
    delta_R_relative: float
    reasons: list = field(default_factory=list)    # why this verdict
    subscores: dict = field(default_factory=dict)  # confidence components


def _ramp(x, lo, hi):
    """0 below lo, 1 above hi, linear in between. Bounds a factor to 0..1."""
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def two_proportion_z(c0: float, n0: float, c1: float, n1: float) -> tuple[float, float]:
    """Two-proportion z-test (H0: r0 = r1). Returns (z, two-sided p)."""
    if min(n0, n1) <= 0:
        return 0.0, 1.0
    p_pool = (c0 + c1) / (n0 + n1)
    se = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n0 + 1.0 / n1))
    if se == 0.0:
        return 0.0, 1.0
    z = (c1 / n1 - c0 / n0) / se
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return float(z), float(p)


def evaluate_gates(agg: dict, out: pd.DataFrame, baseline_n: pd.Series | None = None,
                   material_rel: float = MATERIAL_REL,
                   concentration_min: float = CONCENTRATION_MIN,
                   sample_floor: float = SAMPLE_FLOOR,
                   interaction_max: float = INTERACTION_MAX,
                   alpha: float = SIGNIFICANCE_ALPHA,
                   kind: str = "rate") -> GateReport:
    """
    agg  : output of aggregate() (R0, R1, delta_R, rate, mix, interaction, residual).
    out  : output of decompose()/decompose_sum() (indexed by segment).
    baseline_n : Series segment -> baseline n (fallback when out lacks n0/n1/c0/c1).
    kind : "rate" -> significance gate via z-test;
           "sum"  -> no observable variance, fall back to the sample floor.
    """
    contrib_abs = out["contribution"].abs()
    total_abs = float(contrib_abs.sum())

    # leading segment = largest absolute contribution
    leading = contrib_abs.idxmax() if total_abs > 0 else None
    concentration = float(contrib_abs.max() / total_abs) if total_abs > 0 else 0.0

    # mechanism: relative weight of the interaction term
    denom = abs(agg["rate"]) + abs(agg["mix"]) + 1e-12
    inter_share = abs(agg["interaction"]) / denom

    # material effect relative to the baseline
    delta_rel = abs(agg["delta_R"]) / (abs(agg["R0"]) + 1e-12)

    # baseline sample size of the leading segment
    has_counts = leading is not None and {"n0", "n1", "c0", "c1"} <= set(out.columns)
    if has_counts:
        leading_n = float(out.loc[leading, "n0"])
    elif leading is not None and baseline_n is not None and leading in baseline_n.index:
        leading_n = float(baseline_n.loc[leading])
    else:
        leading_n = float("nan")

    # --- the four gates ---
    reasons = []

    g_material = delta_rel >= material_rel
    if not g_material:
        reasons.append(f"negligible anomaly (ΔR/R0={delta_rel:.1%} < {material_rel:.0%})")

    g_localized = concentration >= concentration_min
    if not g_localized:
        reasons.append(f"diffuse cause (concentration={concentration:.2f} < {concentration_min})")

    # significance: z-test (rates) or sample floor (sums)
    z, p = float("nan"), float("nan")
    if kind == "rate" and has_counts:
        row = out.loc[leading]
        z, p = two_proportion_z(row["c0"], row["n0"], row["c1"], row["n1"])
        g_signif = p <= alpha
        if not g_signif:
            reasons.append(
                f"leading segment's move not significant (p={p:.3f} > {alpha}, z={z:+.2f})")
        f_signif = _ramp(abs(z), lo=Z_CRIT, hi=2 * Z_CRIT)
    else:
        g_signif = (not np.isnan(leading_n)) and (leading_n >= sample_floor)
        if not g_signif:
            n_txt = "unknown" if np.isnan(leading_n) else f"{leading_n:.0f}"
            reasons.append(f"leading segment too small (n={n_txt} < {sample_floor})")
        f_signif = _ramp(0.0 if np.isnan(leading_n) else leading_n,
                         lo=sample_floor, hi=sample_floor * 5)

    g_clean = inter_share <= interaction_max
    if not g_clean:
        reasons.append(f"entangled mechanism (interaction={inter_share:.2f} > {interaction_max})")

    passed = g_material and g_localized and g_signif and g_clean
    verdict = "ASSERT" if passed else "ABSTAIN"

    # interpretable confidence: product of factors bounded to 0..1
    f_conc = _ramp(concentration, lo=concentration_min, hi=1.0)
    f_clean = 1.0 - min(inter_share, 1.0)
    confidence = float(f_conc * f_signif * f_clean) if passed else 0.0

    return GateReport(
        verdict=verdict, confidence=confidence, leading_segment=leading,
        concentration=concentration, interaction_share=float(inter_share),
        leading_sample_n=leading_n, leading_z=z, leading_p=p,
        delta_R_relative=float(delta_rel),
        reasons=reasons or ["all gates pass"],
        subscores={"f_concentration": f_conc, "f_significance": f_signif, "f_clean": f_clean},
    )
