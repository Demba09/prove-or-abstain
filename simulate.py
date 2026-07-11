"""
simulate.py — exécute le flux SANS langgraph, en répliquant le routing,
pour valider la logique end-to-end. Force le mode mock : toujours offline
et déterministe, même si une clé Qwen est configurée.
"""
import os
os.environ["QWEN_MOCK"] = "1"   # avant tout import qui instancie le client

from panels import BASELINE, CLEAN, DIFFUSE
from nodes import (detector, route_after_detect, hypothesizer, investigator,
                   verifier, route_after_verify, driller, actuator, reporter)


def run(state):
    state = {**state, **detector(state)}
    if route_after_detect(state) == "report":
        return {**state, **reporter(state)}
    while True:
        state = {**state, **hypothesizer(state)}
        state = {**state, **investigator(state)}
        state = {**state, **verifier(state)}
        if route_after_verify(state) == "actuate":
            break
    state = {**state, **driller(state)}
    state = {**state, **actuator(state)}
    state = {**state, **reporter(state)}
    return state


def base_state(current, autopilot=False):
    return {
        "baseline": BASELINE, "current": current,
        "metrics": ["conversion", "activation"],
        "dims": ["device", "segment"],   # device d'abord -> force la boucle
        "autopilot_enabled": autopilot,
    }


if __name__ == "__main__":
    for name, curr in [("CLEAN", CLEAN), ("DIFFUSE", DIFFUSE)]:
        print("\n" + "=" * 70)
        print(f"SCÉNARIO {name}")
        print("=" * 70)
        s = run(base_state(curr, autopilot=(name == "CLEAN")))
        for line in s["trace"]:
            print("  ·", line)
        print(f"\n  VERDICT : {s['verdict']}   confiance={s.get('confidence', 0):.2f}")
        print(f"  ACTION  : {s['actions'][0].kind} — {s['actions'][0].detail}")
        print(f"  RAPPORT : {s['report']}")
