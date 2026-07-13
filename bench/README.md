# Benchmark

Does the agent invent a cause when there isn't one? This measures it, against
a plain-prompt baseline on the **same** Qwen model.

## What's here

- `../scenarios.py` — 45 seeded test cases with known answers (9 with a real
  single cause, 36 with none: diffuse, mix-shift, noise, too-small).
- `agents.py` — the two agents behind one interface: our full pipeline
  (offline, deterministic) and `NaiveAgent`, a direct honest prompt with
  structured output that is explicitly allowed to answer "no single cause".
- `metrics.py` — the four numbers: cause-invention rate, precision, recall,
  abstention-reason accuracy.
- `run.py` — runs the suite, writes `results.json` + `metrics.json`, prints a table.
- `figure.py` — renders `../docs/benchmark.svg`.

## Run it

```bash
# our agent only (offline, deterministic):
QWEN_MOCK=1 python -m bench.run

# both columns (needs a DashScope key, unset the mock):
DASHSCOPE_API_KEY=sk-... QWEN_MOCK= python -m bench.run

python -m bench.figure        # regenerate the figure from metrics.json
```

Same seed in, same numbers out. `results.json` / `metrics.json` are written
here; commit them if you want the numbers versioned alongside the figure.

## The baseline is honest on purpose

`NaiveAgent`'s prompt hands the model the exact per-segment numbers and asks it
to name the responsible segment *or say there isn't one*. If it invents causes
anyway, that is the model's disposition under a reasonable prompt — not a
strawman. The whole claim is that the discipline lives in the scaffolding, so
the baseline must be one a competent engineer would actually ship.
