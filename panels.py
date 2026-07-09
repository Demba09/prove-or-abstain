"""
panels.py — données de démo multimetric/multidim + projection.

Panel long : une ligne par cellule atomique [metric, segment, device, n, c]
(counts bruts ; les taux sont TOUJOURS dérivés en aval).

project() projette ce panel sur UNE métrique + UNE dimension via groupby :
c'est la couche "per-hypothesis projection". decompose() opère ensuite dessus.

Trois scénarios calibrés (structure opposée sur 'conversion') :
  CLEAN    : seul le segment 'paid' décroche. Localise le long de 'segment',
             mais paraît diffus le long de 'device' (paid ~50/50 mobile/desktop).
             -> la boucle essaie 'device' (ABSTAIN) puis 'segment' (ASSERT).
  DIFFUSE  : tous les taux baissent uniformément. Aucune dimension ne localise.
             -> on épuise les dimensions -> ABSTAIN final.
  MIXSHIFT : composition ET taux bougent en même temps (mix + rate + interaction
             tous non nuls). Mêmes totaux par segment que le MIXSHIFT validé dans
             gate_check_gates.py -> ABSTAIN, mécanisme entremêlé.
"""
from __future__ import annotations
import pandas as pd

SEGMENTS = ["organic", "paid", "referral", "email"]
DEVICES = ["mobile", "desktop"]

# n par (segment, device). 'paid' volontairement 50/50 mobile/desktop.
_N = {
    ("organic",  "mobile"): 5000, ("organic",  "desktop"): 5000,
    ("paid",     "mobile"): 3000, ("paid",     "desktop"): 3000,
    ("referral", "mobile"): 1500, ("referral", "desktop"): 1500,
    ("email",    "mobile"): 500,  ("email",    "desktop"): 500,
}
# taux conversion baseline par segment (indépendant du device)
_RATE0 = {"organic": 0.05, "paid": 0.07, "referral": 0.08, "email": 0.12}
# taux activation baseline (métrique stable, sert de leurre au détecteur)
_ACT0 = {"organic": 0.30, "paid": 0.32, "referral": 0.28, "email": 0.35}


def _rows(rate_conv, rate_act, n_map=None):
    """Construit un panel long à partir de fonctions (segment->taux)."""
    n_map = _N if n_map is None else n_map
    rows = []
    for seg in SEGMENTS:
        for dev in DEVICES:
            n = n_map[(seg, dev)]
            rows.append({"metric": "conversion", "segment": seg, "device": dev,
                         "n": n, "c": round(n * rate_conv(seg))})
            rows.append({"metric": "activation", "segment": seg, "device": dev,
                         "n": n, "c": round(n * rate_act(seg))})
    return pd.DataFrame(rows)


BASELINE = _rows(lambda s: _RATE0[s], lambda s: _ACT0[s])

# CLEAN : conversion de 'paid' chute 7.0% -> 5.0% ; le reste inchangé.
CLEAN = _rows(
    lambda s: 0.05 if s == "paid" else _RATE0[s],
    lambda s: _ACT0[s],
)

# DIFFUSE : conversion de TOUS les segments baisse de 0.6pp ; même ΔR global.
DIFFUSE = _rows(
    lambda s: _RATE0[s] - 0.006,
    lambda s: _ACT0[s],
)

# MIXSHIFT : la composition bouge (organic gonfle, paid rétrécit) ET les taux
# bougent (organic baisse, paid/email…) -> rate, mix et interaction tous non nuls.
# Totaux par segment identiques au MIXSHIFT de gate_check_gates.py, répartis
# 50/50 mobile/desktop pour rester diffus le long de 'device'.
_N_MIXSHIFT = {
    ("organic",  "mobile"): 7000, ("organic",  "desktop"): 7000,
    ("paid",     "mobile"): 2000, ("paid",     "desktop"): 2000,
    ("referral", "mobile"): 1500, ("referral", "desktop"): 1500,
    ("email",    "mobile"): 500,  ("email",    "desktop"): 500,
}
_RATE_MIXSHIFT = {"organic": 0.045, "paid": 0.08, "referral": 0.08, "email": 0.10}

MIXSHIFT = _rows(
    lambda s: _RATE_MIXSHIFT[s],
    lambda s: _ACT0[s],
    n_map=_N_MIXSHIFT,
)


def metric_totals(panel: pd.DataFrame, metric: str) -> tuple[float, float]:
    """(n_total, c_total) pour une métrique, toutes dimensions confondues."""
    sub = panel[panel["metric"] == metric]
    return float(sub["n"].sum()), float(sub["c"].sum())


def project(panel: pd.DataFrame, metric: str, dim: str) -> pd.DataFrame:
    """Projette le panel sur (metric, dim) : groupby dim, somme des counts bruts.
    Retourne [dim, n, c] — prêt pour decompose(base, curr, dims=dim)."""
    sub = panel[panel["metric"] == metric]
    return sub.groupby(dim, as_index=False)[["n", "c"]].sum()
