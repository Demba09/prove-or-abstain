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
    it cannot invent a panel or metric that doesn't exist. On a follow-up
    turn it may also SELECT a (dim, segment) filter mentioned in the text,
    from the values the caller supplies — same rule, no invention.
  - extract_filter()  : the same (dim, segment) selection as route_query's
    follow-up, but standalone — for "ask" against a dataset that isn't one
    of the named built-in panels (a watched source has exactly one active
    dataset, so there's nothing to route to, only a filter to extract).
  - chat_with_tools() : ORCHESTRATE an investigation via tool calls. Qwen
    chooses which dimensions to test and when to stop; the tools run the
    same deterministic math, and the verdict is recomputed from the gate
    reports afterwards — so Qwen drives the path, never the outcome.
  - suggest_setup()   : CLASSIFY each metric name found in an unfamiliar
    upload as "rate" or "sum", from the name alone — the one setup step
    that is actually a text-understanding task (dimensions are inferred
    exactly, with no ambiguity, from the CSV's own columns; see
    api/app.py:_investigate_pair). Never invents a metric name outside
    the ones supplied.
  - map_schema()      : MAP an unfamiliar RAW source's columns onto the
    long-panel contract [metric, dims..., n, c] — dimensions, metric name,
    n/c counts, OR (when no n/c pair exists as columns at all — one row per
    observation, e.g. raw weekly sales) a value column to deterministically
    aggregate into n=count/c=sum, grouped by Qwen-selected dim_columns. The
    ONE exception to "the verdict never depends on Qwen": interpreting
    ambiguous column names has no single deducible answer, so mock and real
    mode can genuinely disagree here. The only place with a 2nd,
    self-verification pass in real mode (Qwen re-checks its own first
    answer before it's acted on) — see its docstring for why. Never invents
    a column name outside the ones supplied, at either pass — and the
    aggregation arithmetic itself is plain pandas, not Qwen.

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
import threading

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
            "restate the verdict. If 'events' lists operational events already "
            "logged for this exact segment, GROUND at least one hypothesis in the "
            "most relevant one (name it and its date) instead of guessing blindly; "
            "still phrase it as something for a human to confirm, not a fact. "
            "Do not invent an event outside the supplied list. Return ONLY a JSON "
            "array of 2 short strings in English, each phrased as a question."
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

    # --- use 4: route a free-text question to a built-in panel/metric, and
    # (on a follow-up) SELECT a segment filter mentioned in the text ---
    # (selects among caller-supplied options only — invents neither) ---
    def route_query(self, query: str, panels: dict[str, str],
                    metrics: list[str], dims: dict[str, list] | None = None,
                    previous_panel: str | None = None) -> dict:
        """panels: {panel_name: short description}.
        dims: {dim_name: [known segment values]} of the ACTIVE panel — pass
        this on a follow-up turn ("and on mobile only?") so the router can
        also extract a filter. Returns {"panel": <one of panels>,
        "metric": <one of metrics or None>, "filter": {"dim":.., "segment":..}
        or None, "reason": <short text>}. Falls back to a deterministic
        keyword match in mock mode or on any error — never raises, never
        picks a value outside the supplied options."""
        if self.mock:
            self.last_mode, self.last_error = "mock", None
            return template_route_query(query, panels, metrics, dims, previous_panel)
        system = (
            "You route a user's free-text question to ONE of the given analysis "
            "panels and (optionally) ONE of the given metrics. You do not compute "
            "or verify anything — the panel already contains the data; a "
            "deterministic pipeline will investigate it. If 'dims' is given, the "
            "user may be following up on a previous result (e.g. 'and on mobile "
            "only?'): if the text clearly names ONE known segment value from "
            "'dims', return it as a filter; otherwise filter is null. If "
            "'previous_panel' is given and the text does not name a different "
            "scenario, keep using previous_panel. Return ONLY a JSON object: "
            "{\"panel\": <one of the panel names>, "
            "\"metric\": <one of the metric names, or null>, "
            "\"filter\": {\"dim\": <one of dims' keys>, \"segment\": <one of "
            "that dim's values>} or null, "
            "\"reason\": <one short sentence>}. If unsure, pick the closest "
            "panel by description and set metric/filter to null."
        )
        user = json.dumps({"query": query, "panels": panels, "metrics": metrics,
                           "dims": dims or {}, "previous_panel": previous_panel},
                          ensure_ascii=False)
        try:
            raw = self._robust_parse_json(self.complete(
                system, user, max_tokens=200,
                response_format={"type": "json_object"}), {})
            if not isinstance(raw, dict):
                raise ValueError("router did not return an object")
            panel = raw.get("panel") or previous_panel
            metric = raw.get("metric")
            if panel not in panels:
                raise ValueError(f"model picked an unknown panel: {panel!r}")
            if metric is not None and metric not in metrics:
                metric = None
            self.last_mode, self.last_error = "real", None
            return {"panel": panel, "metric": metric,
                   "filter": _guard_filter(raw.get("filter"), dims),
                   "reason": raw.get("reason", "")}
        except Exception as exc:
            self.last_mode, self.last_error = "fallback", str(exc)
            return template_route_query(query, panels, metrics, dims, previous_panel)

    # --- use 5: SELECT a (dim, segment) filter from free text against a
    # single already-active dataset (no panel to pick, unlike route_query) ---
    def extract_filter(self, query: str, dims: dict[str, list]) -> dict | None:
        """"Ask" against a watched source: there's exactly one dataset in
        play, so this only needs the filter half of route_query(). Returns
        {"dim":.., "segment":..} or None — never a value outside `dims`."""
        if self.mock:
            self.last_mode, self.last_error = "mock", None
            return _scan_filter(query, dims)
        system = (
            "The user may be asking to narrow an analysis to one segment "
            "(e.g. 'and on mobile only?'). If the text clearly names ONE "
            "known segment value from 'dims', return it as a filter; "
            "otherwise return null. Return ONLY a JSON object: "
            "{\"dim\": <one of dims' keys> or null, \"segment\": <one of "
            "that dim's values> or null}."
        )
        user = json.dumps({"query": query, "dims": dims}, ensure_ascii=False)
        try:
            raw = self._robust_parse_json(self.complete(
                system, user, max_tokens=100,
                response_format={"type": "json_object"}), {})
            self.last_mode, self.last_error = "real", None
            return _guard_filter(raw, dims)
        except Exception as exc:
            self.last_mode, self.last_error = "fallback", str(exc)
            return _scan_filter(query, dims)

    # --- use 6: classify each metric NAME as "rate" or "sum" — the only
    # setup step where the caller doesn't already know the answer exactly.
    # Dimensions need no classification: every non-reserved CSV column IS a
    # dimension, with no ambiguity (see api/app.py:_investigate_pair). ---
    def suggest_setup(self, metrics: list[str]) -> dict:
        """Returns {"sum_metrics": <subset of metrics>, "reason": <text>}.
        Never returns a name outside `metrics`."""
        if self.mock:
            self.last_mode, self.last_error = "mock", None
            return template_suggest_setup(metrics)
        system = (
            "You are given the metric names found in an uploaded dataset (a "
            "long panel of [metric, dimensions..., n, c] rows: n = population, "
            "c = a count). For each metric name, decide whether it is a RATE "
            "(c is a count of successes out of n — e.g. conversion, activation, "
            "click-through) or a SUM (c is a total quantity, not bounded by n — "
            "e.g. revenue, spend, refunds). Judge from the name alone. Return "
            "ONLY a JSON object: {\"sum_metrics\": [<names from the supplied "
            "list that are SUM-kind>], \"reason\": <one short sentence>}. Do "
            "not invent a metric name outside the supplied list."
        )
        user = json.dumps({"metrics": metrics}, ensure_ascii=False)
        try:
            raw = self._robust_parse_json(self.complete(
                system, user, max_tokens=150,
                response_format={"type": "json_object"}), {})
            if not isinstance(raw, dict):
                raise ValueError("suggest_setup did not return an object")
            sum_metrics = [m for m in raw.get("sum_metrics", []) if m in metrics]
            self.last_mode, self.last_error = "real", None
            return {"sum_metrics": sum_metrics, "reason": raw.get("reason", "")}
        except Exception as exc:
            self.last_mode, self.last_error = "fallback", str(exc)
            return template_suggest_setup(metrics)

    # --- use 7: map an unfamiliar RAW source's columns onto the long-panel
    # contract [metric, dims..., n, c] — the one place Qwen decides the
    # SHAPE of the data feeding the calculation, not just its order, its
    # wording, or a choice among already-known values. Unlike every other
    # use above, this means the final result CAN differ between mock and
    # real mode: interpreting ambiguous column names has no single
    # mechanically-deducible answer (that's the whole reason Qwen is asked
    # at all). The trade for that honesty: Qwen's decision is used directly,
    # not gated behind a human confirmation — but it self-verifies in a 2nd
    # pass before acting, catching its own misreads the same way a careful
    # analyst re-checks their own work. Never invents a column name outside
    # the ones supplied, at either pass. ---
    def map_schema(self, columns: list[str], sample_rows: list[dict]) -> dict:
        """Returns {"dim_columns": [...], "metric_column": <name or None>,
        "n_column": <name or None>, "c_column": <name or None>,
        "value_column": <name or None>, "self_verified": bool,
        "reason": <text>}.

        Two mappings a caller can get back:
        - n_column/c_column set: the raw file already has a genuine
          population/success count PAIR as two of its columns — the
          existing rename-only path (n0 -> n, c0 -> c).
        - value_column set instead: the file has no such pair — one row
          per observation, one numeric column to total (sales, revenue,
          amount...). Nothing here computes anything: the caller
          deterministically aggregates n=count(rows), c=sum(value_column)
          grouped by dim_columns (api/app.py:_apply_schema_mapping) — same
          discipline as everywhere else in this codebase, Qwen SELECTS
          which column plays which role, the arithmetic is plain pandas.
          dim_columns here is a POSITIVE selection (which columns are
          meaningful breakdowns), not "everything left over" — a raw
          per-row file usually also has covariates (weather, price index,
          a date) that would fragment the aggregation into meaningless
          near-singleton groups if swept in by default.

        In mock mode, a single deterministic pass (self_verified=True — the
        template has no doubt to re-examine). In real mode, a proposal pass
        followed by a self-verification pass that may correct it."""
        if self.mock:
            self.last_mode, self.last_error = "mock", None
            return template_map_schema(columns, sample_rows)

        propose_system = (
            "You are given the column names and a few sample rows of a raw "
            "dataset that must be reshaped into a long panel "
            "[metric, dimensions..., n, c]. First check: does this data "
            "already have a genuine population/denominator count column "
            "(n) AND a separate success/numerator-or-total count column "
            "(c)? If yes, identify those two, plus the ONE column holding "
            "the metric name (categorical, e.g. 'conversion'/'revenue'), "
            "and set value_column to null. If NOT — e.g. one row per "
            "observation with a single numeric quantity to total, like "
            "sales or revenue, and no separate count columns — instead "
            "identify value_column: the ONE column holding that numeric "
            "quantity to sum (never an id, a date, or an unrelated "
            "covariate), and leave n_column/c_column null; the caller "
            "will deterministically compute n=count of rows and c=sum of "
            "value_column per group. Either way, identify dim_columns: "
            "ONLY the columns that are meaningful categorical breakdowns "
            "to compare across (e.g. store/platform/country) — exclude "
            "dates, ids, and unrelated covariates even if no other role "
            "was assigned to them. Return ONLY a JSON object: "
            "{\"dim_columns\": [...], \"metric_column\": <name or null>, "
            "\"n_column\": <name or null>, \"c_column\": <name or null>, "
            "\"value_column\": <name or null>, \"reason\": <one short "
            "sentence>}. Do not invent a column name outside the supplied "
            "list."
        )
        user = json.dumps({"columns": columns, "sample_rows": sample_rows[:5]},
                          ensure_ascii=False)
        try:
            proposed = self._guarded_schema_mapping(
                self._robust_parse_json(self.complete(
                    propose_system, user, max_tokens=200,
                    response_format={"type": "json_object"}), {}),
                columns)

            verify_system = (
                "You just proposed this column mapping for a raw dataset. "
                "Re-examine it against the sample rows: does each column "
                "really hold what you assigned it (metric name / "
                "population count n / success-or-total count c / a value "
                "to sum / dimension)? If a value_column was proposed, "
                "double check it isn't an id, a date, or a covariate "
                "unrelated to the metric, and that dim_columns doesn't "
                "sweep in dates/ids/covariates as if they were breakdowns. "
                "If something looks wrong, correct it. Return ONLY a JSON "
                "object with the FINAL mapping (corrected or unchanged): "
                "{\"dim_columns\": [...], \"metric_column\": <name or null>, "
                "\"n_column\": <name or null>, \"c_column\": <name or null>, "
                "\"value_column\": <name or null>, \"self_verified\": <true "
                "if you kept your first answer, false if you corrected "
                "it>, \"reason\": <one short sentence>}. Do not invent a "
                "column name outside the supplied list."
            )
            verify_user = json.dumps({"columns": columns, "sample_rows": sample_rows[:5],
                                      "proposed_mapping": proposed}, ensure_ascii=False)
            final = self._guarded_schema_mapping(
                self._robust_parse_json(self.complete(
                    verify_system, verify_user, max_tokens=200,
                    response_format={"type": "json_object"}), {}),
                columns)
            final["self_verified"] = bool(final.get("self_verified", True))
            self.last_mode, self.last_error = "real", None
            return final
        except Exception as exc:
            self.last_mode, self.last_error = "fallback", str(exc)
            return template_map_schema(columns, sample_rows)

    @staticmethod
    def _guarded_schema_mapping(raw: dict, columns: list[str]) -> dict:
        """Anti-invention guard shared by both map_schema() passes: every
        returned column name must be one actually supplied, else dropped —
        same discipline as _guard_filter()/suggest_setup(). Also guards
        value_column (the aggregate-instead-of-rename path, see
        map_schema()'s docstring) the same way."""
        if not isinstance(raw, dict):
            raise ValueError("map_schema did not return an object")
        cols = set(columns)
        metric_col = raw.get("metric_column")
        n_col = raw.get("n_column")
        c_col = raw.get("c_column")
        value_col = raw.get("value_column")
        metric_col = metric_col if metric_col in cols else None
        n_col = n_col if n_col in cols else None
        c_col = c_col if c_col in cols else None
        value_col = value_col if value_col in cols else None
        dim_cols = [c for c in raw.get("dim_columns", [])
                   if c in cols and c not in (metric_col, n_col, c_col, value_col)]
        out = {"dim_columns": dim_cols, "metric_column": metric_col,
              "n_column": n_col, "c_column": c_col, "value_column": value_col,
              "reason": raw.get("reason", "")}
        if "self_verified" in raw:
            out["self_verified"] = raw["self_verified"]
        return out


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


def _guard_filter(raw_filter, dims: dict[str, list] | None) -> dict | None:
    """A filter is only accepted if it names a dim AND a segment value the
    caller actually supplied — same anti-invention rule as panel/metric."""
    if not isinstance(raw_filter, dict) or not dims:
        return None
    dim, seg = raw_filter.get("dim"), raw_filter.get("segment")
    if dim in dims and seg in dims[dim]:
        return {"dim": dim, "segment": seg}
    return None


def _scan_filter(query: str, dims: dict[str, list] | None) -> dict | None:
    """Deterministic keyword scan (mock mode / fallback, shared by
    template_route_query and extract_filter): the first known segment value
    of any dim that's literally named in the query, else None. Segment
    values are matched as a whole word OR a raw substring (not just a
    `_words` token) so multi-word/underscored values ("majority_women")
    still match — `_words` would otherwise split them into pieces that
    never equal the full value."""
    q_words, q_lower = _words(query), query.lower()
    for dim, segments in (dims or {}).items():
        hit = next((s for s in segments
                    if str(s).lower() in q_words or str(s).lower() in q_lower), None)
        if hit is not None:
            return {"dim": dim, "segment": hit}
    return None


def template_route_query(query: str, panels: dict[str, str],
                         metrics: list[str], dims: dict[str, list] | None = None,
                         previous_panel: str | None = None) -> dict:
    """Deterministic keyword routing (mock mode / fallback). Picks the
    panel whose name or description shares the most whole words with the
    query; defaults to the first panel if nothing matches. If `dims` is
    given, also looks for one known segment value named in the query and
    returns it as a filter (a follow-up like "and on mobile only?")."""
    q_words = _words(query)

    panel = previous_panel
    if panel not in panels:
        best_panel, best_score = next(iter(panels)), -1
        for name, desc in panels.items():
            score = len(_words(name + " " + desc) & q_words)
            if score > best_score:
                best_panel, best_score = name, score
        panel = best_panel
    metric = next((m for m in metrics if m.lower() in q_words), None)

    return {"panel": panel, "metric": metric, "filter": _scan_filter(query, dims),
           "reason": "keyword match (mock/fallback routing, no LLM call)"}


def template_suggest_setup(metrics: list[str]) -> dict:
    """Deterministic sum/rate classification (mock mode / fallback): a
    metric name is treated as SUM-kind if it contains a keyword typical of
    an unbounded total; everything else defaults to RATE."""
    _SUM_HINTS = ("revenue", "spend", "cost", "amount", "total", "price",
                 "value", "refund", "payout", "gmv")
    sum_metrics = [m for m in metrics if any(h in m.lower() for h in _SUM_HINTS)]
    return {"sum_metrics": sum_metrics,
           "reason": "keyword heuristic (mock/fallback, no LLM call)"}


def template_map_schema(columns: list[str], sample_rows: list[dict]) -> dict:
    """Deterministic heuristic column matching (mock mode / fallback), one
    pass, self_verified=True — no LLM call, so there is nothing for a
    template to doubt about its own answer."""
    # Multi-character substrings only: a bare "n" or "c" hint would match as
    # a substring of almost any word ("category" contains "c") — those are
    # handled separately below as an exact (case-insensitive) name match.
    _METRIC_HINTS = ("metric", "event", "kpi")
    _N_HINTS = ("total", "count", "population", "impressions", "sent", "attempts")
    _C_HINTS = ("success", "conversion", "converted", "purchase", "click")

    def _first_match(hints, taken):
        for col in columns:
            if col in taken:
                continue
            if col.lower() in ("n", "c"):     # exact bare name, not a substring guess
                continue
            if any(h in col.lower() for h in hints):
                return col
        return None

    def _first_exact(name: str, taken):
        for col in columns:
            if col not in taken and col.lower() == name:
                return col
        return None

    metric_col = _first_match(_METRIC_HINTS, set())
    n_col = (_first_exact("n", {metric_col})
            or _first_match(_N_HINTS, {metric_col}))
    c_col = (_first_exact("c", {metric_col, n_col})
            or _first_match(_C_HINTS, {metric_col, n_col}))

    if n_col and c_col:
        dim_cols = [c for c in columns if c not in (metric_col, n_col, c_col)]
        return {"dim_columns": dim_cols, "metric_column": metric_col,
               "n_column": n_col, "c_column": c_col, "value_column": None,
               "self_verified": True,
               "reason": "keyword heuristic (mock/fallback, no LLM call)"}

    # No n/c pair found by keyword — try the aggregate-instead path (see
    # map_schema()'s docstring). Only commits to a value_column when exactly
    # ONE candidate remains: a column that isn't an id/date/flag/code by
    # name and whose sample values are all numeric. More than one candidate
    # (e.g. a metric column sitting alongside several numeric covariates)
    # is a real judgment call a keyword scan can't make — this deliberately
    # returns None rather than guess, same as it does for n/c.
    #
    # Two DIFFERENT exclusion lists on purpose: "store_id"/"product_id" are
    # everyday, perfectly good DIMENSION names (an id column is usually the
    # meaningful category to group by) — they're only wrong as a VALUE to
    # sum. Only a raw date is wrong on both counts (a time axis, not a
    # thing to total or a breakdown to compare — "Watch a source" already
    # handles the time dimension via its own persisted history).
    _NON_VALUE_NAME_HINTS = ("id", "date", "flag", "code")
    _NON_DIM_NAME_HINTS = ("date",)

    def _is_numeric_column(col):
        values = [row[col] for row in sample_rows if col in row]
        return bool(values) and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)

    numeric_candidates = [
        c for c in columns
        if c != metric_col
        and not any(h in c.lower() for h in _NON_VALUE_NAME_HINTS)
        and _is_numeric_column(c)
    ]
    # When exactly one numeric column remains, it's the unambiguous value to
    # sum. When there are several (e.g. Weekly_Sales alongside Temperature,
    # Fuel_Price, CPI), prefer the one whose name hints at a value/sum/total
    # rather than a covariate. Only applies when NO n or c was found by
    # keyword — if the data has Total/Employed kind of columns, the caller
    # needs Qwen's judgment, not this heuristic.
    if not n_col and not c_col:
        if len(numeric_candidates) == 1:
            value_col = numeric_candidates[0]
        elif len(numeric_candidates) > 1:
            _VALUE_HINTS = ("sales", "revenue", "amount", "price", "value",
                           "sum", "total", "cost", "spend", "income", "profit",
                           "refund", "gmv", "volume")
            value_hits = [c for c in numeric_candidates
                         if any(h in c.lower() for h in _VALUE_HINTS)]
            value_col = value_hits[0] if len(value_hits) == 1 else None
        else:
            value_col = None
    else:
        value_col = None

    if value_col:
        dim_cols = [c for c in columns
                   if c not in (metric_col, value_col)
                   and not any(h in c.lower() for h in _NON_DIM_NAME_HINTS)
                   and not _is_numeric_column(c)]
        return {"dim_columns": dim_cols, "metric_column": metric_col,
               "n_column": None, "c_column": None, "value_column": value_col,
               "self_verified": True,
               "reason": "keyword heuristic (mock/fallback, no LLM call)"}

    dim_cols = [c for c in columns if c not in (metric_col, n_col, c_col)]
    return {"dim_columns": dim_cols, "metric_column": metric_col,
           "n_column": n_col, "c_column": c_col, "value_column": None,
           "self_verified": True,
           "reason": "keyword heuristic (mock/fallback, no LLM call)"}


def template_speculations(p: dict) -> list[str]:
    """Deterministic speculations (mock mode / fallback). Grounds the first
    hypothesis in a logged event when one is available for this exact
    segment (see evidence.py), instead of guessing blindly."""
    tgt = p.get("refined") or f"{p.get('winning_dim')}={p.get('leading_segment')}"
    events = p.get("events") or []
    out = []
    if events:
        e = events[0]
        out.append(f"A '{e['event']}' event was logged for {tgt} on {e['date']} "
                   f"({e['source']}) — is this the cause?")
    else:
        out.append(f"Did anything change recently on the {tgt} side — campaign, "
                   f"pricing, landing page, tracking?")
    out.append(f"Does a technical incident limited to {tgt} (integration, "
              f"payment, latency) coincide with the period?")
    return out


# --- lazy singleton shared by the nodes ---
_CLIENT: QwenClient | None = None
# FastAPI runs sync `def` endpoints in a thread pool, so concurrent requests
# can call get_client() on the very first request at once — without a lock,
# two threads can both see _CLIENT is None and each build a QwenClient (whose
# cost tracker starts at zero), silently orphaning one and undercounting cost.
_CLIENT_LOCK = threading.Lock()


def get_client() -> QwenClient:
    global _CLIENT
    if _CLIENT is None:
        with _CLIENT_LOCK:
            if _CLIENT is None:               # re-check: lost the race, don't rebuild
                _CLIENT = QwenClient()
    return _CLIENT
