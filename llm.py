"""
llm.py — Qwen/DashScope client for Probatio.

Strict boundary (this is the project's thesis): the LLM computes nothing and
decides no verdict. It does exactly three things:
  - plan_dimensions() : PROPOSE an exploration order for the dimensions.
    The math tests every dimension anyway, so the order never changes the
    final verdict — only how fast it is found.
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

    # --- use 1: propose an exploration order (decides nothing) ---
    def plan_dimensions(self, metric: str, delta_rel: float, dims: list[str]) -> list[str]:
        if self.mock:
            return list(dims)
        system = (
            "You are planning a causal investigation on a business metric. "
            "You are given an anomalous metric and a list of analysis dimensions. "
            "Return ONLY a JSON array of the same dimensions, reordered from the "
            "most likely to localize the cause to the least likely. "
            "Do not invent any dimension, do not drop any, do not compute anything."
        )
        user = json.dumps({"metric": metric, "delta_rel": round(delta_rel, 4),
                           "dimensions": list(dims)}, ensure_ascii=False)
        try:
            order = json.loads(_strip_fences(self.complete(system, user, max_tokens=120)))
            order = [d for d in order if d in dims]          # guard: subset only
            order += [d for d in dims if d not in order]     # re-add anything dropped
            return order or list(dims)
        except Exception:
            return list(dims)                                # fallback: original order

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
