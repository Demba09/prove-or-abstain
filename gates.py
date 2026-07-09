"""
gates.py — la décision ASSERT / ABSTAIN de Probatio.

Le calcul (attribution.py) est TOUJOURS exact (résidu nul). Les gates ne
recalculent rien : elles décident si ce calcul DÉSIGNE une cause localisable,
ou s'il faut s'abstenir.

Principe : ASSERT exige que TOUTES les gates passent. Si une seule échoue,
on ABSTAIN — et la raison = la (les) gate(s) échouée(s). Un refus motivé,
pas un haussement d'épaules. C'est la propriété de sûreté de l'autopilot :
"un agent qui peut agir doit pouvoir refuser d'agir".
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


# --- seuils nommés (faciles à justifier dans le README / la démo) ---
MATERIAL_REL = 0.02       # |ΔR|/R0 : en-dessous, anomalie négligeable
CONCENTRATION_MIN = 0.55  # part du top contributeur : LA gate centrale
SAMPLE_FLOOR = 1000       # n baseline minimal du segment meneur pour être fiable
INTERACTION_MAX = 0.50    # |interaction|/(|rate|+|mix|) : au-delà, mécanisme entremêlé


@dataclass
class GateReport:
    verdict: str                  # "ASSERT" | "ABSTAIN"
    confidence: float             # 0..1, n'a de sens que si ASSERT
    leading_segment: object       # segment meneur (valeur d'index)
    concentration: float
    interaction_share: float
    leading_sample_n: float
    delta_R_relative: float
    reasons: list = field(default_factory=list)    # pourquoi ce verdict
    subscores: dict = field(default_factory=dict)  # composants de la confiance


def _ramp(x, lo, hi):
    """0 sous lo, 1 au-dessus de hi, linéaire entre. Borne un facteur dans 0..1."""
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def evaluate_gates(agg: dict, out: pd.DataFrame, baseline_n: pd.Series | None = None,
                   material_rel: float = MATERIAL_REL,
                   concentration_min: float = CONCENTRATION_MIN,
                   sample_floor: float = SAMPLE_FLOOR,
                   interaction_max: float = INTERACTION_MAX) -> GateReport:
    """
    agg  : sortie de aggregate() (R0, R1, delta_R, rate, mix, interaction, residual).
    out  : sortie de decompose() (indexé par segment, colonne 'contribution').
    baseline_n : Series segment -> n baseline (pour la gate d'échantillon).
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

    # échantillon du segment meneur
    if leading is not None and baseline_n is not None and leading in baseline_n.index:
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

    g_sample = (not np.isnan(leading_n)) and (leading_n >= sample_floor)
    if not g_sample:
        n_txt = "inconnu" if np.isnan(leading_n) else f"{leading_n:.0f}"
        reasons.append(f"segment meneur trop petit (n={n_txt} < {sample_floor})")

    g_clean = inter_share <= interaction_max
    if not g_clean:
        reasons.append(f"mécanisme entremêlé (interaction={inter_share:.2f} > {interaction_max})")

    passed = g_material and g_localized and g_sample and g_clean
    verdict = "ASSERT" if passed else "ABSTAIN"

    # confiance interprétable : produit de facteurs bornés 0..1
    f_conc = _ramp(concentration, lo=concentration_min, hi=1.0)
    f_sample = _ramp(0.0 if np.isnan(leading_n) else leading_n,
                     lo=sample_floor, hi=sample_floor * 5)
    f_clean = 1.0 - min(inter_share, 1.0)
    confidence = float(f_conc * f_sample * f_clean) if passed else 0.0

    return GateReport(
        verdict=verdict, confidence=confidence, leading_segment=leading,
        concentration=concentration, interaction_share=float(inter_share),
        leading_sample_n=leading_n, delta_R_relative=float(delta_rel),
        reasons=reasons or ["toutes les gates passent"],
        subscores={"f_concentration": f_conc, "f_sample": f_sample, "f_clean": f_clean},
    )
