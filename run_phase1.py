"""
run_phase1.py — runs the compiled graph on the two headline demo scenarios.

  CLEAN   (autopilot ON)  -> ASSERT segment=paid -> EXECUTE
  DIFFUSE (autopilot OFF) -> ABSTAIN -> ESCALATE

If DASHSCOPE_API_KEY is configured (via .env or export), Qwen is called for
the exploration plan and the wording; otherwise the deterministic mock is
used automatically.

Run:  python run_phase1.py
"""
try:
    from dotenv import load_dotenv
    load_dotenv()                    # reads .env -> os.environ (key never hardcoded)
except ModuleNotFoundError:
    pass

from graph import APP
from panels import BASELINE, CLEAN, DIFFUSE


def initial_state(current, autopilot=False):
    return {
        "baseline": BASELINE,
        "current": current,
        "metrics": ["conversion", "activation"],
        "dims": ["device", "segment"],   # device first -> exercises the loop
        "autopilot_enabled": autopilot,
        "trace": [],
    }


if __name__ == "__main__":
    for name, curr, autopilot in [("CLEAN", CLEAN, True), ("DIFFUSE", DIFFUSE, False)]:
        print("\n" + "=" * 72)
        print(f"SCENARIO {name}   (autopilot={'ON' if autopilot else 'OFF'})")
        print("=" * 72)
        final = APP.invoke(initial_state(curr, autopilot))
        for line in final.get("trace", []):
            print("  ·", line)
        action = final["actions"][0]
        print(f"\n  VERDICT : {final['verdict']}   confidence={final.get('confidence', 0):.2f}")
        print(f"  ACTION  : {action.kind} — {action.detail}")
        print(f"  REPORT  : {final['report']}")
