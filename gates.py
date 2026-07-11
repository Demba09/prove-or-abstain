"""
gates.py — la décision ASSERT / ABSTAIN de Probatio.

Le calcul (attribution.py) est TOUJOURS exact (résidu nul). Les gates ne
recalculent rien : elles décident si ce calcul DÉSIGNE une cause localisable,
ou s'il faut s'abstenir.

Principe : ASSERT exige que TOUTES les gates passent. Si une seule échoue,
on ABSTAIN — et la raison = la (les) gate(s) échouée(s). Un refus motivé,
pas un haussement d'épaules. C'est la propriété de sûreté de l'autopilot :
"un agent qui peut agir doit pouvoir refuser d'agir".

Les quatre gates :
  material    : l'anomalie est assez grosse pour mériter une explication
  localized   : un segment domine réellement la contribution
  significant : le mouvement du segment meneur n'est pas du bruit
                (test z de deux proportions ; pour les métriques somme, où la
                variance unitaire n'est pas observable, repli sur un plancher
                d'effectif)
  clean       : effets taux et mix séparables (interaction bornée)
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math

import numpy as np
import pandas as pd


# --- seuils nommés (faciles à justifier dans le README / la démo) ---
MATERIAL_REL = 0.02        # |ΔR|/R0 : en-dessous, anomalie négligeable
CONCENTRATION_MIN = 0.55   # part du top contributeur : LA gate centrale
SIGNIFICANCE_ALPHA = 0.01  # p-value max (bilatérale) du test z du segment meneur
Z_CRIT = 2.576             # |z| critique correspondant à alpha=0.01 (bilatéral)
SAMPLE_FLOOR = 1000        # repli métriques somme : n baseline minimal du meneur
INTERACTION_MAX = 0.50     # |interaction|/(|rate|+|mix|) : au-delà, mécanisme entremêlé


@dataclass
class GateReport:
    verdict: str                  # "ASSERT" | "ABSTAIN"
    confidence: float             # 0..1, n'a de sens que si ASSERT
    leading_segment: object       # segment meneur (valeur d'index)
    concentration: float
    interaction_share: float
    leading_sample_n: float
    leading_z: float              # test z du segment meneur (NaN si non applicable)
    leading_p: float              # p-value bilatérale (NaN si non applicable)
    delta_R_relative: float
    reasons: list = field(default_factory=list)    # pourquoi ce verdict
    subscores: dict = field(default_factory=dict)  # composants de la confiance


def _ramp(x, lo, hi):
    """0 sous lo, 1 au-dessus de hi, linéaire entre. Borne un facteur dans 0..1."""
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def two_proportion_z(c0: float, n0: float, c1: float, n1: float) -> tuple[float, float]:
    """Test z de deux proportions (H0 : r0 = r1). Renvoie (z, p bilatérale)."""
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
    agg  : sortie de aggregate() (R0, R1, delta_R, rate, mix, interaction, residual).
    out  : sortie de decompose()/decompose_sum() (indexé par segment).
    baseline_n : Series segment -> n baseline (repli si out n'a pas n0/n1/c0/c1).
    kind : "rate" -> gate de significativité par test z ;
           "sum"  -> pas de variance observable, repli plancher d'effectif.
    """
    contrib_abs = out["contribution"].abs()
    total_abs = float(contrib_abs.sum())

    # segment meneur = plus grosse contribution absolue
    leading = contrib_abs.idxmax() if total_abs > 0 else None
    concentration = float(contrib_abs.max() / total_abs) if total_abs > 0 else 0.0

    # mécanisme : poids relatif de l'interaction
    denom = abs(agg["rate"]) + abs(agg["mix"]) + 1e-12
    inter_share = abs(agg["interaction"]) / denom

    # effet matériel relatif au baseline
    delta_rel = abs(agg["delta_R"]) / (abs(agg["R0"]) + 1e-12)

    # effectif baseline du segment meneur
    has_counts = leading is not None and {"n0", "n1", "c0", "c1"} <= set(out.columns)
    if has_counts:
        leading_n = float(out.loc[leading, "n0"])
    elif leading is not None and baseline_n is not None and leading in baseline_n.index:
        leading_n = float(baseline_n.loc[leading])
    else:
        leading_n = float("nan")

    # --- les quatre gates ---
    reasons = []

    g_material = delta_rel >= material_rel
    if not g_material:
        reasons.append(f"anomalie négligeable (ΔR/R0={delta_rel:.1%} < {material_rel:.0%})")

    g_localized = concentration >= concentration_min
    if not g_localized:
        reasons.append(f"cause diffuse (concentration={concentration:.2f} < {concentration_min})")

    # significativité : test z (taux) ou plancher d'effectif (sommes)
    z, p = float("nan"), float("nan")
    if kind == "rate" and has_counts:
        row = out.loc[leading]
        z, p = two_proportion_z(row["c0"], row["n0"], row["c1"], row["n1"])
        g_signif = p <= alpha
        if not g_signif:
            reasons.append(
                f"mouvement du segment meneur non significatif (p={p:.3f} > {alpha}, z={z:+.2f})")
        f_signif = _ramp(abs(z), lo=Z_CRIT, hi=2 * Z_CRIT)
    else:
        g_signif = (not np.isnan(leading_n)) and (leading_n >= sample_floor)
        if not g_signif:
            n_txt = "inconnu" if np.isnan(leading_n) else f"{leading_n:.0f}"
            reasons.append(f"segment meneur trop petit (n={n_txt} < {sample_floor})")
        f_signif = _ramp(0.0 if np.isnan(leading_n) else leading_n,
                         lo=sample_floor, hi=sample_floor * 5)

    g_clean = inter_share <= interaction_max
    if not g_clean:
        reasons.append(f"mécanisme entremêlé (interaction={inter_share:.2f} > {interaction_max})")

    passed = g_material and g_localized and g_signif and g_clean
    verdict = "ASSERT" if passed else "ABSTAIN"

    # confiance interprétable : produit de facteurs bornés 0..1
    f_conc = _ramp(concentration, lo=concentration_min, hi=1.0)
    f_clean = 1.0 - min(inter_share, 1.0)
    confidence = float(f_conc * f_signif * f_clean) if passed else 0.0

    return GateReport(
        verdict=verdict, confidence=confidence, leading_segment=leading,
        concentration=concentration, interaction_share=float(inter_share),
        leading_sample_n=leading_n, leading_z=z, leading_p=p,
        delta_R_relative=float(delta_rel),
        reasons=reasons or ["toutes les gates passent"],
        subscores={"f_concentration": f_conc, "f_significance": f_signif, "f_clean": f_clean},
    )
