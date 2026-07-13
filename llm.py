"""
llm.py — Qwen/DashScope client for Probatio.

Strict boundary (this is the project's thesis): the LLM computes nothing and
decides no verdict. It does exactly three things:
  - plan_dimensions() : PROPOSE an exploration order for the dimensions,
    through a Qwen function call (`rank_dimensions`) whose schema constrains
    the answer to a permutation of the given dimensions. The math tests every
    dimension anyway, so the order never changes the final verdict — only how
    fast it is found.
  - write_report()    : PHRASE a conclusion from ALREADY COMPUTED numbers.
    It is explicitly forbidden from inventing a figure or a cause.
  - speculate_causes(): offer business hypotheses about the WHY, clearly
    labelled as unverified speculation, kept apart from the proven verdict.

Mock mode: if DASHSCOPE_API_KEY is absent or QWEN_MOCK=1, no network call is
made and deterministic outputs are returned — the pipeline runs offline.
Any failed real call falls back to the deterministic output: the agent
never crashes because of the LLM.
"""
from __future__ import annotations
import json
import os

DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1]
        if s.endswith("```"):
            s = s.rsplit("```", 1)[0]
    return s.strip()


class QwenClient:
    def __init__(self, mock: bool | None = None, model: str | None = None,
                 base_url: str | None = None):
        self.model = model or os.environ.get("QWEN_MODEL", DEFAULT_MODEL)
        self.base_url = base_url or os.environ.get("QWEN_BASE_URL", DEFAULT_BASE_URL)
        self.api_key = os.environ.get("DASHSCOPE_API_KEY")
        if mock is None:
            mock = os.environ.get("QWEN_MOCK") == "1" or not self.api_key
        self.mock = mock
        self._client = None

    # --- low level: one raw chat call (lazy openai import) ---
    def complete(self, system: str, user: str,
                 temperature: float = 0.2, max_tokens: int = 400) -> str:
        if self.mock:
            raise RuntimeError("complete() called in mock mode")
        if self._client is None:
            from openai import OpenAI  # imported only when actually calling
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        resp = self._client.chat.completions.create(
            model=self.model, temperature=temperature, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return (resp.choices[0].message.content or "").strip()

    # --- low level: a tool/function call that forces a typed answer ---
    def complete_tool(self, system: str, user: str, tool: dict,
                      temperature: float = 0.2, max_tokens: int = 200) -> dict:
        """One chat call constrained to emit a single tool call, returning the
        parsed arguments dict. Qwen (DashScope compatible-mode) speaks the
        OpenAI tools protocol, so the model answers through a schema instead of
        free-form prose — no fence-stripping, no ad-hoc JSON parsing."""
        if self.mock:
            raise RuntimeError("complete_tool() called in mock mode")
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        resp = self._client.chat.completions.create(
            model=self.model, temperature=temperature, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            tools=[tool],
            tool_choice={"type": "function",
                         "function": {"name": tool["function"]["name"]}},
        )
        call = resp.choices[0].message.tool_calls[0]
        return json.loads(call.function.arguments)

    # --- use 1: propose an exploration order (decides nothing) ---
    def plan_dimensions(self, metric: str, delta_rel: float, dims: list[str]) -> list[str]:
        if self.mock:
            return list(dims)
        system = (
            "You are planning a causal investigation on a business metric. "
            "You are given an anomalous metric and a list of analysis dimensions. "
            "Call `rank_dimensions` with the SAME dimensions reordered from the "
            "most likely to localize the cause to the least likely. "
            "Do not invent any dimension, do not drop any, do not compute anything."
        )
        user = json.dumps({"metric": metric, "delta_rel": round(delta_rel, 4),
                           "dimensions": list(dims)}, ensure_ascii=False)
        tool = {
            "type": "function",
            "function": {
                "name": "rank_dimensions",
                "description": "Return the dimensions in exploration order.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order": {
                            "type": "array",
                            "items": {"type": "string", "enum": list(dims)},
                            "description": "the dimensions, most to least likely",
                        }
                    },
                    "required": ["order"],
                },
            },
        }
        # 1) structured function call; 2) free-text JSON; 3) original order.
        try:
            order = self.complete_tool(system, user, tool, max_tokens=120)["order"]
        except Exception:
            try:
                order = json.loads(_strip_fences(self.complete(system, user, max_tokens=120)))
            except Exception:
                return list(dims)
        order = [d for d in order if d in dims]          # guard: subset only
        order += [d for d in dims if d not in order]     # re-add anything dropped
        return order or list(dims)

    # --- setup aid: suggest how to frame an unknown CSV (decides nothing) ---
    def suggest_setup(self, columns: list[str], sample_rows: list[dict]) -> dict | None:
        """Given the columns and a few sample rows of an uploaded CSV, PROPOSE
        which columns are analysis dimensions and which metrics are sums
        (revenue-like) rather than rates. This is a framing aid for the user to
        confirm — it never drives a run on its own, and the verdict is
        deterministic once the user has ratified the inputs. Returns None in
        mock mode or on any error (the deterministic default then stands)."""
        if self.mock:
            return None
        reserved = {"metric", "n", "c", "period"}
        candidate_dims = [c for c in columns if c not in reserved]
        metrics = sorted({r.get("metric") for r in sample_rows if r.get("metric") is not None})
        system = (
            "You help set up a causal analysis on an uploaded metrics table. "
            "Columns are: metric, n (population), c (numerator), optional period, "
            "and one or more dimension columns. Call `suggest_analysis_setup` "
            "with: the subset of columns that are analysis dimensions, and the "
            "metrics that are SUMS (a total such as revenue where c can exceed n) "
            "rather than rates (where c is a count bounded by n). Decide nothing "
            "else; the user confirms your suggestion before anything runs."
        )
        user = json.dumps({"columns": list(columns), "candidate_dimensions": candidate_dims,
                           "metrics": metrics, "sample_rows": sample_rows[:8]},
                          ensure_ascii=False)
        tool = {
            "type": "function",
            "function": {
                "name": "suggest_analysis_setup",
                "description": "Propose dimensions and sum-type metrics for the table.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dimensions": {"type": "array",
                                       "items": {"type": "string", "enum": candidate_dims}},
                        "sum_metrics": {"type": "array",
                                        "items": {"type": "string", "enum": metrics}},
                        "rationale": {"type": "string",
                                      "description": "one short sentence, plain language"},
                    },
                    "required": ["dimensions", "sum_metrics", "rationale"],
                },
            },
        }
        try:
            out = self.complete_tool(system, user, tool, max_tokens=250)
            dims = [d for d in out.get("dimensions", []) if d in candidate_dims]
            sums = [m for m in out.get("sum_metrics", []) if m in metrics]
            return {"dimensions": dims or candidate_dims, "sum_metrics": sums,
                    "rationale": str(out.get("rationale", ""))[:400]}
        except Exception:
            return None

    # --- use 3: speculate about the business WHY (labelled, never mixed
    # with the proven verdict; computes nothing, decides nothing) ---
    def speculate_causes(self, payload: dict) -> list[str]:
        if self.mock:
            return template_speculations(payload)
        system = (
            "A causal investigation has PROVEN that a metric move localizes to a "
            "given segment. Suggest 2 plausible BUSINESS hypotheses for the root "
            "cause (campaign, pricing, product, technical...). These are "
            "speculations for a human to verify: do not invent any figure, do not "
            "restate the verdict. Return ONLY a JSON array of 2 short strings in "
            "English, each phrased as a question."
        )
        try:
            out = json.loads(_strip_fences(self.complete(system,
                    json.dumps(payload, ensure_ascii=False), max_tokens=200)))
            out = [s for s in out if isinstance(s, str)][:3]
            return out or template_speculations(payload)
        except Exception:
            return template_speculations(payload)

    # --- use 2: write the conclusion (computes nothing) ---
    def write_report(self, payload: dict) -> str:
        if self.mock:
            return template_report(payload)
        system = (
            "Write the conclusion of a causal investigation in 2 to 3 sentences, "
            "in English, factual and sober. You are given a verdict and ALREADY "
            "COMPUTED numbers. You must not invent ANY figure or ANY cause: you "
            "only rephrase. If the verdict is ABSTAIN, state clearly that no "
            "localized cause could be proven and that the agent refrains from acting."
        )
        try:
            return self.complete(system, json.dumps(payload, ensure_ascii=False))
        except Exception:
            return template_report(payload)


def template_report(p: dict) -> str:
    """Deterministic wording (mock mode / fallback). Same facts, no LLM."""
    verdict = p.get("verdict")
    metric = p.get("metric", "—")
    if verdict == "NO_ANOMALY":
        return "No material anomaly detected. Nothing to explain."
    if verdict == "ASSERT":
        refined = f" The drill-down refines the cause to {p['refined']}." if p.get("refined") else ""
        return (f"PROVEN — the move in '{metric}' localizes to "
                f"{p['winning_dim']}={p['leading_segment']} "
                f"(concentration {p['concentration']:.0%}, confidence {p['confidence']:.2f})."
                f"{refined} "
                f"Action: {p['action_kind']}. {p['action_detail']}")
    return (f"ABSTAINED — the move in '{metric}' is real but does not localize "
            f"on any tested dimension ({', '.join(p.get('dims_tried', []))}). "
            f"Likely a systemic cause. Action: {p['action_kind']}. {p['action_detail']}")


def template_speculations(p: dict) -> list[str]:
    """Deterministic speculations (mock mode / fallback)."""
    tgt = p.get("refined") or f"{p.get('winning_dim')}={p.get('leading_segment')}"
    return [
        f"Did anything change recently on the {tgt} side — campaign, pricing, "
        f"landing page, tracking?",
        f"Does a technical incident limited to {tgt} (integration, payment, "
        f"latency) coincide with the period?",
    ]


# --- lazy singleton shared by the nodes ---
_CLIENT: QwenClient | None = None


def get_client() -> QwenClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = QwenClient()
    return _CLIENT
