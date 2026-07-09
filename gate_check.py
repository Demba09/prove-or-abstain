"""
gate_check.py — ferme Phase 0.

Compare TA decompose à l'oracle, colonne par colonne, sur 3 scénarios.
CLEAN/DIFFUSE laissent mix=interaction=0 : insuffisant seuls.
MIXSHIFT allume les trois termes -> c'est lui qui teste vraiment.

Pose ce fichier à côté de attribution_reference.py ET de ton attribution.py.
"""
import numpy as np
import pandas as pd
from attribution_reference import BASELINE, CLEAN, DIFFUSE, decompose as oracle

from attribution import decompose as mine   # <-- TA version

MIXSHIFT = pd.DataFrame([
    {"segment": "organic",  "n": 14000, "c": 630},   # n ET r bougent
    {"segment": "paid",     "n": 4000,  "c": 320},
    {"segment": "referral", "n": 3000,  "c": 240},
    {"segment": "email",    "n": 1000,  "c": 100},
])

COLS = ["rate", "mix", "interaction", "contribution"]
all_ok = True
for name, curr in [("CLEAN", CLEAN), ("DIFFUSE", DIFFUSE), ("MIXSHIFT", MIXSHIFT)]:
    a = oracle(BASELINE, curr, dims="segment")
    b = mine(BASELINE, curr, dims="segment").reindex(a.index)
    ok = np.allclose(a[COLS].values, b[COLS].values, atol=1e-10)
    touches_mix = abs(a["mix"]).sum() > 1e-9
    print(f"{name:9s} match oracle: {ok}   (exerce mix/interaction ? {touches_mix})")
    if not ok:
        all_ok = False
        diff = (a[COLS] - b[COLS]).abs()
        print(diff[diff > 1e-10].dropna(how="all"))

print("\nPhase 0 CLOSE \u2713" if all_ok else "\nPhase 0 PAS ENCORE \u2014 corrige et relance.")
