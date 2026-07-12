"""
gate_check_gates.py — validates the decision layer (the gates).

  CLEAN    -> ASSERT  (one localized cause: paid breaks alone)
  DIFFUSE  -> ABSTAIN (uniform drop, no culprit)
  MIXSHIFT -> ABSTAIN (mix + rates both move, no clean culprit)
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
    print(f"\n=== {name} ===  expected {expected[name]} -> {rep.verdict}  {'OK' if ok else 'XX'}")
    print(f"  leading segment  : {rep.leading_segment}")
    print(f"  concentration    : {rep.concentration:.3f}")
    print(f"  interaction_share: {rep.interaction_share:.3f}")
    print(f"  leader n         : {rep.leading_sample_n:.0f}")
    print(f"  leader z-test    : z={rep.leading_z:+.2f}, p={rep.leading_p:.4f}")
    print(f"  relative ΔR      : {rep.delta_R_relative:.1%}")
    print(f"  confidence       : {rep.confidence:.3f}")
    print(f"  reasons          : {rep.reasons}")

print("\nDecision layer VALIDATED ✓" if all_ok else "\nNOT YET — tune and rerun.")
