# Demo video — recording script

I can't record screen video myself (no capture tool in my toolset), so this is
a shot-by-shot script: narration + exact commands + expected output, timed for
a **~2 minute** Devpost demo. Record with QuickTime (macOS: Cmd+Shift+5) or
any screen recorder while running these commands in a terminal, ideally next
to the architecture diagram (`docs/architecture.svg`) open in an image viewer
or browser tab.

Two takes are scripted: the **ASSERT** scenario (`clean`) and the **ABSTAIN**
scenario (`diffuse`, with `mixshift` as an optional third beat if you have
time to spare).

## Setup (before hitting record)

```bash
cd prove-or-abstain
source .venv/bin/activate
export QWEN_MOCK=1        # deterministic, no network dependency during the take
uvicorn api.app:app --reload
```

Open a second terminal pane for the `curl` calls, and have
`docs/architecture.svg` visible in a browser tab to cut to.

---

## Beat 1 — the hook (0:00–0:15)

**Show:** the README hero line, or just talk to camera / voiceover over the
architecture diagram.

**Narration:**

> "Most AI agents that monitor metrics will confidently blame something even
> when the data doesn't support it. This one is different: it only acts when
> it can prove a cause with math, and it abstains — on purpose — when it
> can't. Watch."

---

## Beat 2 — ASSERT scenario (0:15–0:55)

**Show:** terminal, run:

```bash
curl -s -X POST localhost:8000/investigate \
  -H 'content-type: application/json' \
  -d '{"panel":"clean"}' | python3 -m json.tool
```

**Expected output (verdict block):**

```json
{
  "panel": "clean",
  "verdict": "ASSERT",
  "confidence": 1.0,
  "root_cause": { "dimension": "segment", "segment": "paid" },
  "gates": {
    "device": "ABSTAIN",
    "segment": "ASSERT"
  },
  "action": { "kind": "RECOMMEND", "...": "..." }
}
```

**Narration (read while output is on screen):**

> "Conversion dropped. The agent's first guess — 'device' — doesn't localize
> the cause, so it doesn't stop there: it loops to the next candidate
> dimension. 'segment' does localize: the 'paid' segment alone collapsed from
> 7% to 5% conversion, everything else untouched. Every one of the four gates
> passes — material, localized, sample size, clean mechanism — so it ASSERTs
> with full confidence and recommends a scoped action."

**Optional cut:** scroll to the `"trace"` array in the JSON and highlight the
line showing the `device` → `segment` loop-back — this is the visual proof
the bounded loop actually ran, not just a lucky first guess.

**Bonus 10s — autopilot:**

```bash
curl -s -X POST localhost:8000/investigate \
  -H 'content-type: application/json' \
  -d '{"panel":"clean","autopilot":true}' | python3 -c \
  "import json,sys; print(json.load(sys.stdin)['action'])"
```

> "With autopilot on and confidence above 0.70, the same evidence now
> triggers EXECUTE instead of just a recommendation."

---

## Beat 3 — ABSTAIN scenario (0:55–1:35)

**Show:** terminal, run:

```bash
curl -s -X POST localhost:8000/investigate \
  -H 'content-type: application/json' \
  -d '{"panel":"diffuse"}' | python3 -m json.tool
```

**Expected output (verdict block):**

```json
{
  "panel": "diffuse",
  "verdict": "ABSTAIN",
  "confidence": 0.0,
  "root_cause": null,
  "gates": { "device": "ABSTAIN", "segment": "ABSTAIN" },
  "action": { "kind": "ESCALATE", "...": "aucune dimension testée ne localise..." }
}
```

**Narration:**

> "Same size of anomaly, same aggregate drop in conversion — but this time
> every segment fell by roughly the same amount. There's no single culprit
> to point to. The agent tries both dimensions, neither one concentrates the
> contribution above the 0.55 threshold, so instead of guessing, it
> abstains, escalates to a human, and says exactly why: no dimension
> localizes a cause. This is the safety property — an autopilot that can act
> must be able to refuse to act."

**Optional third beat (if time allows), `mixshift`:**

```bash
curl -s -X POST localhost:8000/investigate \
  -H 'content-type: application/json' -d '{"panel":"mixshift"}' \
  | python3 -c "import json,sys; b=json.load(sys.stdin); print(b['verdict'], b['gates'])"
```

> "This one abstains for a *different* reason — composition and rate are
> both shifting at once, so the interaction term is too large to trust a
> single explanation. Two different failure modes, both caught by name."

---

## Beat 4 — close (1:35–2:00)

**Show:** `docs/architecture.svg` full screen, or the gates table in the
README.

**Narration:**

> "Every number here comes from pandas — never from the LLM. The model only
> orders which dimension to try first and writes the final sentence; the
> verdict itself is deterministic and auditable. It's built as a bounded
> LangGraph loop, ships as one Docker image, and deploys to Alibaba Cloud
> Function Compute with no code changes — just a container. Prove or
> abstain: that's the whole idea."

---

## Recording checklist

- [ ] `QWEN_MOCK=1` exported before starting `uvicorn` — output is reproducible take after take
- [ ] Terminal font size large enough to read on a recorded video (18pt+)
- [ ] Pipe every `curl` through `python3 -m json.tool` (or `jq` if installed) — raw JSON is unwatchable
- [ ] Do the ASSERT take and the ABSTAIN take as two separate clips if easier to re-record cleanly
- [ ] Cut in `docs/architecture.svg` at the open and the close
