"""
scripts/gate_check.py — validates the math layer against the oracle.

Compares the production decompose() to the reference oracle, column by
column, on 3 scenarios. CLEAN/DIFFUSE leave mix=interaction=0, which is not
enough on its own; MIXSHIFT lights up all three terms — it is the scenario
that really tests the decomposition.
"""
import sys
from pathlib import Path

# Runnable both as `python scripts/x.py` and `python -m scripts.x`:
# put the repo root on sys.path so `prove_or_abstain` and `scripts` resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from scripts.attribution_reference import BASELINE, CLEAN, DIFFUSE, decompose as oracle

from prove_or_abstain.attribution import decompose as mine   # <-- production version

MIXSHIFT = pd.DataFrame([
    {"segment": "organic",  "n": 14000, "c": 630},   # n AND r both move
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
    print(f"{name:9s} matches oracle: {ok}   (exercises mix/interaction? {touches_mix})")
    if not ok:
        all_ok = False
        diff = (a[COLS] - b[COLS]).abs()
        print(diff[diff > 1e-10].dropna(how="all"))

print("\nMath layer VALIDATED ✓" if all_ok else "\nNOT YET — fix and rerun.")
