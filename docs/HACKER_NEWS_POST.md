# Show HN: Prove-or-Abstain — an AI agent that refuses to act without proof

## Titre
**Prove-or-Abstain** — an AI agent that ABSTAINS when data doesn't support a conclusion

## Post (copy-paste pour Hacker News)

---

**Show HN: [Prove-or-Abstain](https://github.com/Demba09/prove-or-abstain)** — an AI agent that refuses to act without proof

Most AI analytics tools will always give you an answer, even when the data doesn't support one. They hallucinate root causes.

Prove-or-Abstain is different. It's a causal investigation agent built on LangGraph that:

1. **Detects** when a metric moved materially
2. **Hypothesizes** which dimension (segment, device, geography...) might explain it
3. **Investigates** by decomposing the change (rate vs mix vs interaction)
4. **Verifies** against 4 statistical gates (material, localized, significant, clean)
5. **Either ASSERTs** a proven cause OR **ABSTAINS** and escalates to a human

The ABSTAIN verdict is the key design choice. An agent that can act on data needs a principled way to refuse to act when evidence is insufficient.

**Tech stack:** Python 3.12+, LangGraph, FastAPI, pandas/numpy (all math is exact — zero residual decomposition)

**Features:**
- 4 built-in demo scenarios (clean ASSERT, diffuse ABSTAIN, mixshift ABSTAIN, deep drill-down)
- SQL connector (Postgres/MySQL/SQLite)
- Google Sheets connector
- Natural language query routing via Qwen
- Full audit trail + statistical z-test significance gate

**Demo:** deploy on Docker or run locally in 3 commands.

Source: https://github.com/Demba09/prove-or-abstain

Would love feedback on the causal attribution math and the ABSTAIN design pattern.
