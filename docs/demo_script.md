# Demo video — recording script

Shot-by-shot script for a **~2 minute** Devpost demo, narration + exact
actions + expected results. Record with QuickTime (macOS: Cmd+Shift+5) or any
screen recorder. The whole demo happens in the browser on the demo page —
no terminal on screen except optionally at the very start.

## Setup (before hitting record)

```bash
cd prove-or-abstain
source .venv/bin/activate
export QWEN_MOCK=1        # deterministic output, identical take after take
uvicorn api.app:app
```

Open http://localhost:8000 in a clean browser window (hide bookmarks bar).
If you want the real Qwen prose instead of the deterministic mock, unset
`QWEN_MOCK` and make sure `DASHSCOPE_API_KEY` is set — the verdicts and every
number stay identical either way; only the wording of the report and the
speculations change.

---

## Beat 1 — the hook (0:00–0:15)

**Show:** the demo page, idle.

**Narration:**

> "Most AI agents that monitor metrics will confidently blame something even
> when the data doesn't support it. This one is different: it only asserts a
> root cause when the math proves it — and it abstains, on purpose, when it
> can't. Watch."

---

## Beat 2 — ASSERT (0:15–0:50)

**Action:** click **clean**.

**Expected on screen:** green **ASSERT** badge, confidence 0.79, root cause
`segment = paid`; two gate cards — `device` red (rejected), `segment` green
(all gates pass); a RECOMMEND action; the report; a speculation box.

**Narration:**

> "Conversion dropped. The agent's first hypothesis — the device dimension —
> doesn't localize the cause, so it loops and tries the next one. Segment
> does localize it: paid alone collapsed from 7 to 5 percent. Look at the
> gates: the effect is material, concentrated on one segment, statistically
> significant — that's a real two-proportion z-test, p below 10⁻⁵, not a
> magic threshold — and the mechanism is clean. So it asserts, and recommends
> a scoped action. Below, the LLM suggests *why* this might have happened —
> explicitly labelled as speculation, never mixed with the proven numbers."

**Optional:** tick **autopilot**, click **clean** again — the action becomes
EXECUTE. "With autopilot on and confidence above the bar, it acts on its own."

---

## Beat 3 — ABSTAIN (0:50–1:20)

**Action:** click **diffuse**.

**Expected on screen:** red **ABSTAIN** badge, no root cause, both gate cards
red with named reasons (diffuse cause, not significant), ESCALATE action.

**Narration:**

> "Same size of drop — but this time every segment fell by the same amount.
> There is no culprit to point to. Both dimensions fail the gates, and the
> agent says exactly why: the cause is diffuse and the leading segment's move
> isn't even statistically significant. So instead of guessing, it abstains
> and escalates to a human. An autopilot that can act must be able to refuse
> to act."

---

## Beat 4 — drill-down (1:20–1:45)

**Action:** click **deep**.

**Expected on screen:** ASSERT `device = mobile`, plus the drill-down line:
*within device = mobile, the cause narrows to segment = paid*.

**Narration:**

> "One more: here a single cell of the data broke — paid users on mobile.
> The agent proves the anomaly lives on mobile, then drills down inside
> mobile and narrows it to the paid segment. The recommended action targets
> exactly that intersection — not the whole metric."

---

## Beat 5 — close (1:45–2:00)

**Show:** the architecture diagram (`docs/architecture.svg`).

**Narration:**

> "Every number comes from pandas — the LLM only orders the search and writes
> the words. It handles rates and revenue-style sums, CSV uploads, and time
> series against a rolling baseline. One Docker image, deployed on Alibaba
> Cloud Function Compute. Prove or abstain — that's the whole idea."

---

## Recording checklist

- [ ] `QWEN_MOCK=1` exported before `uvicorn` — reproducible take after take
- [ ] Browser zoom ~125% so badges and gate cards read well on video
- [ ] One beat per clip if that's easier to re-record cleanly
- [ ] Cut in `docs/architecture.svg` at the close
