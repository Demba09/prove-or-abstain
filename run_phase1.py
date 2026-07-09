"""
run_phase1.py — exécute le graphe compilé sur les deux scénarios de démo.

  CLEAN  (autopilot ON)  -> ASSERT segment=paid -> EXECUTE
  DIFFUSE (autopilot OFF) -> ABSTAIN -> ESCALATE

Si DASHSCOPE_API_KEY est configurée (via .env ou export), Qwen est appelé
pour le plan d'exploration et la rédaction ; sinon, fallback mock automatique.

Lance :  python3.12 run_phase1.py
(nécessite : pip install langgraph openai python-dotenv)
"""
try:
    from dotenv import load_dotenv
    load_dotenv()                    # lit .env -> os.environ (clé jamais en dur)
except ModuleNotFoundError:
    pass

from graph import APP
from panels import BASELINE, CLEAN, DIFFUSE


def initial_state(current, autopilot=False):
    return {
        "baseline": BASELINE,
        "current": current,
        "metrics": ["conversion", "activation"],
        "dims": ["device", "segment"],   # device d'abord -> exerce la boucle
        "autopilot_enabled": autopilot,
        "trace": [],
    }


if __name__ == "__main__":
    for name, curr, autopilot in [("CLEAN", CLEAN, True), ("DIFFUSE", DIFFUSE, False)]:
        print("\n" + "=" * 72)
        print(f"SCÉNARIO {name}   (autopilot={'ON' if autopilot else 'OFF'})")
        print("=" * 72)
        final = APP.invoke(initial_state(curr, autopilot))
        for line in final.get("trace", []):
            print("  ·", line)
        action = final["actions"][0]
        print(f"\n  VERDICT  : {final['verdict']}   confiance={final.get('confidence', 0):.2f}")
        print(f"  ACTION   : {action.kind} — {action.detail}")
        print(f"  RAPPORT  : {final['report']}")
