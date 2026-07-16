"""
llm.py — Qwen/DashScope client for Probatio.

Strict boundary (this is the project's thesis): the LLM computes nothing and
decides no verdict. It does exactly four things:
  - plan_dimensions() : PROPOSE an exploration order for the dimensions.
    The math tests every dimension anyway, so the order never changes the
    final verdict — only how fast it is found.
  - write_report()    : PHRASE a conclusion from ALREADY COMPUTED numbers.
    It is explicitly forbidden from inventing a figure or a cause.
  - speculate_causes(): offer business hypotheses about the WHY, clearly
    labelled as unverified speculation, kept apart from the proven verdict.
  - route_query()     : map a free-text question to one of the built-in
    panels/metrics. It only SELECTS among options the caller supplies —
    it cannot invent a panel or metric that doesn't exist.
  - chat_with_tools() : ORCHESTRATE an investigation via tool calls. Qwen
    chooses which dimensions to test and when to stop; the tools run the
    same deterministic math, and the verdict is recomputed from the gate
    reports afterwards — so Qwen drives the path, never the outcome.

Mock mode: if DASHSCOPE_API_KEY is absent or QWEN_MOCK=1, no network call is
made and deterministic outputs are returned — the pipeline runs offline.
Any failed real call falls back to the deterministic output: the agent
never crashes because of the LLM.

Visibility: every call updates `last_mode` ("mock" | "real" | "fallback")
and `last_error` so callers can show, in the trace and in the API response,
whether Qwen was actually invoked — instead of silently looking identical
to the offline demo.

Region note: DASHSCOPE_API_KEY is tied to the account's region. International
accounts use the default base_url below; mainland China accounts must set
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1. A key/URL
mismatch fails with 401/404 and silently falls back to mock — run
check_qwen.py before a live demo to confirm which mode is actually active.
"""
from __future__ import annotations
import json
import os
import re

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
        # Visibility for callers/trace/API: what actually happened on the
        # last call, not just what was requested.
        self.last_mode = "mock" if self.mock else "real"
        self.last_error: str | None = None
        # Token/cost accounting — stays at 0 in mock mode (no round trips).
        from prove_or_abstain.cost_tracker import CostTracker
        self.tracker = CostTracker(self.model)

    def _account(self, resp) -> None:
        """Add a response's token usage to the cost tracker, if present."""
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self.tracker.add_usage(getattr(usage, "prompt_tokens", 0),
                                   getattr(usage, "completion_tokens", 0))

    # --- low level: one raw chat call (lazy openai import) ---
    def complete(self, system: str, user: str,
                 temperature: float = 0.2, max_tokens: int = 400,
                 response_format: dict | None = None) -> str:
        if self.mock:
            raise RuntimeError("complete() called in mock mode")
        if self._client is None:
            from openai import OpenAI  # imported only when actually calling
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        kwargs = {}
        if response_format is not None:
            kwargs["response_format"] = response_format
        resp = self._client.chat.completions.create(
            model=self.model, temperature=temperature, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            **kwargs,
        )
        self._account(resp)
        return (resp.choices[0].message.content or "").strip()

    # --- use 5: drive an agentic tool-calling loop (orchestrates, decides
    # nothing). Qwen picks which tools to call; the tools run the exact same
    # deterministic math as the graph, so the verdict stays LLM-independent. ---
    def chat_with_tools(self, messages: list[dict], tools: list[dict],
                        temperature: float = 0.1, max_tokens: int = 500) -> dict:
        """One assistant turn with OpenAI-style function calling.

        Returns a normalized dict:
          {"message": <assistant message dict to append to history>,
           "tool_calls": [{"id", "name", "arguments": <parsed dict>}, ...],
           "content": <assistant text or "">}

        Raises in mock mode — the agent loop drives deterministically offline
        and never reaches here (see prove_or_abstain/agent_loop.py).
        """
        if self.mock:
            raise RuntimeError("chat_with_tools() called in mock mode")
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        resp = self._client.chat.completions.create(
            model=self.model, temperature=temperature, max_tokens=max_tokens,
            messages=messages, tools=tools, tool_choice="auto",
        )
        self._account(resp)
        msg = resp.choices[0].message
        calls = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append({"id": tc.id, "name": tc.function.name, "arguments": args})
        # Re-serialize the assistant turn so it can be appended to history.
        assistant_msg: dict = {"role": "assistant", "content": msg.content or ""}
        if calls:
            assistant_msg["tool_calls"] = [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"],
                              "arguments": json.dumps(c["arguments"])}}
                for c in calls
            ]
        return {"message": assistant_msg, "tool_calls": calls,
                "content": msg.content or ""}

    # --- resilience: recover JSON from a messy model reply, else fall back ---
    def _robust_parse_json(self, text: str, fallback):
        """Best-effort JSON recovery: (1) parse the fenced text; (2) parse the
        substring between the outermost braces/brackets; (3) regex-extract the
        known scalar fields; (4) give up and return `fallback`."""
        s = _strip_fences(text or "")
        try:
            return json.loads(s)
        except Exception:
            pass
        for open_c, close_c in (("{", "}"), ("[", "]")):
            i, j = s.find(open_c), s.rfind(close_c)
            if 0 <= i < j:
                try:
                    return json.loads(s[i:j + 1])
                except Exception:
                    pass
        fields: dict = {}
        m = re.search(r'verdict"?\s*[:=]\s*"?(ASSERT|ABSTAIN|NO_ANOMALY)', s, re.I)
        if m:
            fields["verdict"] = m.group(1).upper()
        m = re.search(r'confidence"?\s*[:=]\s*([0-9]*\.?[0-9]+)', s, re.I)
        if m:
            try:
                fields["confidence"] = float(m.group(1))
            except ValueError:
                pass
        m = re.search(r'panel"?\s*[:=]\s*"?([a-zA-Z_]+)', s)
        if m:
            fields["panel"] = m.group(1)
        m = re.search(r'metric"?\s*[:=]\s*"?([a-zA-Z_]+)', s)
        if m:
            fields["metric"] = m.group(1)
        return fields or fallback

    # --- use 1: propose an exploration order (decides nothing) ---
    def plan_dimensions(self, metric: str, delta_rel: float, dims: list[str]) -> list[str]:
        if self.mock:
            self.last_mode, self.last_error = "mock", None
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
            parsed = self._robust_parse_json(
                self.complete(system, user, max_tokens=120), list(dims))
            order = parsed if isinstance(parsed, list) else list(dims)
            order = [d for d in order if d in dims]          # guard: subset only
            order += [d for d in dims if d not in order]     # re-add anything dropped
            self.last_mode, self.last_error = "real", None
            return order or list(dims)
        except Exception as exc:
            self.last_mode, self.last_error = "fallback", str(exc)
            return list(dims)                                # fallback: original order

    # --- use 3: speculate about the business WHY (labelled, never mixed
    # with the proven verdict; computes nothing, decides nothing) ---
    def speculate_causes(self, payload: dict) -> list[str]:
        if self.mock:
            self.last_mode, self.last_error = "mock", None
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
            parsed = self._robust_parse_json(self.complete(system,
                    json.dumps(payload, ensure_ascii=False), max_tokens=200), [])
            out = [s for s in parsed if isinstance(s, str)][:3] \
                if isinstance(parsed, list) else []
            self.last_mode, self.last_error = "real", None
            return out or template_speculations(payload)
        except Exception as exc:
            self.last_mode, self.last_error = "fallback", str(exc)
            return template_speculations(payload)

    # --- use 2: write the conclusion (computes nothing) ---
    def write_report(self, payload: dict) -> str:
        if self.mock:
            self.last_mode, self.last_error = "mock", None
            return template_report(payload)
        system = (
            "Write the conclusion of a causal investigation in 2 to 3 sentences, "
            "in English, factual and sober. You are given a verdict and ALREADY "
            "COMPUTED numbers. You must not invent ANY figure or ANY cause: you "
            "only rephrase. If the verdict is ABSTAIN, state clearly that no "
            "localized cause could be proven and that the agent refrains from acting."
        )
        try:
            text = self.complete(system, json.dumps(payload, ensure_ascii=False))
            self.last_mode, self.last_error = "real", None
            return text
        except Exception as exc:
            self.last_mode, self.last_error = "fallback", str(exc)
            return template_report(payload)

    # --- use 4: route a free-text question to a built-in panel/metric ---
    # (selects among caller-supplied options only — invents neither) ---
    def route_query(self, query: str, panels: dict[str, str],
                    metrics: list[str]) -> dict:
        """panels: {panel_name: short description}. Returns
        {"panel": <one of panels>, "metric": <one of metrics or None>,
         "reason": <short text>}. Falls back to a deterministic keyword
        match in mock mode or on any error — never raises, never picks a
        value outside the supplied options."""
        if self.mock:
            self.last_mode, self.last_error = "mock", None
            return template_route_query(query, panels, metrics)
        system = (
            "You route a user's free-text question to ONE of the given analysis "
            "panels and (optionally) ONE of the given metrics. You do not compute "
            "or verify anything — the panel already contains the data; a "
            "deterministic pipeline will investigate it. Return ONLY a JSON "
            "object: {\"panel\": <one of the panel names>, "
            "\"metric\": <one of the metric names, or null>, "
            "\"reason\": <one short sentence>}. If unsure, pick the closest panel "
            "by description and set metric to null."
        )
        user = json.dumps({"query": query, "panels": panels, "metrics": metrics},
                          ensure_ascii=False)
        try:
            raw = self._robust_parse_json(self.complete(
                system, user, max_tokens=150,
                response_format={"type": "json_object"}), {})
            if not isinstance(raw, dict):
                raise ValueError("router did not return an object")
            panel = raw.get("panel")
            metric = raw.get("metric")
            if panel not in panels:
                raise ValueError(f"model picked an unknown panel: {panel!r}")
            if metric is not None and metric not in metrics:
                metric = None
            self.last_mode, self.last_error = "real", None
            return {"panel": panel, "metric": metric,
                   "reason": raw.get("reason", "")}
        except Exception as exc:
            self.last_mode, self.last_error = "fallback", str(exc)
            return template_route_query(query, panels, metrics)


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


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z][a-z'-]*", text.lower()))


def template_route_query(query: str, panels: dict[str, str],
                         metrics: list[str]) -> dict:
    """Deterministic keyword routing (mock mode / fallback). Picks the
    panel whose name or description shares the most whole words with the
    query; defaults to the first panel if nothing matches."""
    q_words = _words(query)
    best_panel, best_score = next(iter(panels)), -1
    for name, desc in panels.items():
        score = len(_words(name + " " + desc) & q_words)
        if score > best_score:
            best_panel, best_score = name, score
    metric = next((m for m in metrics if m.lower() in q_words), None)
    return {"panel": best_panel, "metric": metric,
           "reason": "keyword match (mock/fallback routing, no LLM call)"}


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
