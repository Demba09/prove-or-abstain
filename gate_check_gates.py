"""
gate_check_gates.py — ferme la sous-étape "décision" de Phase 1.

  CLEAN    -> ASSERT  (une cause localisée : paid décroche seul)
  DIFFUSE  -> ABSTAIN (baisse uniforme, aucun coupable)
  MIXSHIFT -> ABSTAIN (mix + taux bougent partout, pas de coupable net)
"""
import pandas as pd
from attribution_reference import BASELINE, CLEAN, DIFFUSE, aggregate
from attribution import decompose
from gates import evaluate_gates

MIXSHIFT = pd.DataFrame([
    {"segment": "organic",  "n": 14000, "c": 630},
    {"segment": "paid",     "n": 4000,  "c": 320},
    {"segment": "referral", "n": 3000,  "c": 240},
    {"segment": "email",    "n": 1000,  "c": 100},
])

baseline_n = BASELINE.set_index("segment")["n"]
expected = {"CLEAN": "ASSERT", "DIFFUSE": "ABSTAIN", "MIXSHIFT": "ABSTAIN"}

all_ok = True
for name, curr in [("CLEAN", CLEAN), ("DIFFUSE", DIFFUSE), ("MIXSHIFT", MIXSHIFT)]:
    out = decompose(BASELINE, curr, dims="segment")
    agg = aggregate(out)
    rep = evaluate_gates(agg, out, baseline_n=baseline_n)
    ok = rep.verdict == expected[name]
    all_ok &= ok
    print(f"\n=== {name} ===  attendu {expected[name]} -> {rep.verdict}  {'OK' if ok else 'XX'}")
    print(f"  segment meneur   : {rep.leading_segment}")
    print(f"  concentration    : {rep.concentration:.3f}")
    print(f"  interaction_share: {rep.interaction_share:.3f}")
    print(f"  n meneur         : {rep.leading_sample_n:.0f}")
    print(f"  ΔR relatif       : {rep.delta_R_relative:.1%}")
    print(f"  confiance        : {rep.confidence:.3f}")
    print(f"  raisons          : {rep.reasons}")

print("\nSous-étape décision CLOSE \u2713" if all_ok else "\nPAS ENCORE \u2014 ajuste et relance.")
