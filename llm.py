"""
llm.py — client Qwen/DashScope pour Probatio.

Frontière stricte (c'est la thèse du projet) : le LLM ne calcule rien et ne
décide aucun verdict. Il fait exactement deux choses :
  - plan_dimensions() : PROPOSER un ordre d'exploration des dimensions.
    La math teste quand même toutes les dimensions ; l'ordre ne change donc
    jamais le verdict final, seulement la vitesse à laquelle on le trouve.
  - write_report() : RÉDIGER une conclusion à partir de chiffres DÉJÀ calculés.
    On lui interdit explicitement d'inventer un chiffre ou une cause.

Mode mock : si DASHSCOPE_API_KEY est absente ou QWEN_MOCK=1, on n'appelle pas
le réseau et on renvoie des sorties déterministes — le pipeline tourne offline.
Tout appel réel qui échoue retombe sur le fallback : l'agent ne crashe jamais.
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

    # --- bas niveau : un appel chat brut (import openai paresseux) ---
    def complete(self, system: str, user: str,
                 temperature: float = 0.2, max_tokens: int = 400) -> str:
        if self.mock:
            raise RuntimeError("complete() appelé en mode mock")
        if self._client is None:
            from openai import OpenAI  # importé seulement si on appelle vraiment
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        resp = self._client.chat.completions.create(
            model=self.model, temperature=temperature, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return (resp.choices[0].message.content or "").strip()

    # --- usage 1 : proposer un ordre d'exploration (ne décide rien) ---
    def plan_dimensions(self, metric: str, delta_rel: float, dims: list[str]) -> list[str]:
        if self.mock:
            return list(dims)
        system = (
            "Tu planifies une investigation causale sur une métrique business. "
            "On te donne une métrique anormale et une liste de dimensions d'analyse. "
            "Renvoie UNIQUEMENT un tableau JSON des mêmes dimensions, ré-ordonnées de "
            "la plus susceptible de localiser la cause à la moins susceptible. "
            "N'invente aucune dimension, n'en retire aucune, ne calcule rien."
        )
        user = json.dumps({"metric": metric, "delta_rel": round(delta_rel, 4),
                           "dimensions": list(dims)}, ensure_ascii=False)
        try:
            order = json.loads(_strip_fences(self.complete(system, user, max_tokens=120)))
            order = [d for d in order if d in dims]          # garde-fou : sous-ensemble
            order += [d for d in dims if d not in order]     # complète les oubliées
            return order or list(dims)
        except Exception:
            return list(dims)                                # fallback : ordre d'origine

    # --- usage 2 : rédiger la conclusion (ne calcule rien) ---
    def write_report(self, payload: dict) -> str:
        if self.mock:
            return template_report(payload)
        system = (
            "Tu rédiges la conclusion d'une investigation causale en 2 à 3 phrases, "
            "en français, ton factuel et sobre. On te fournit un verdict et des CHIFFRES "
            "DÉJÀ CALCULÉS. Tu ne dois inventer AUCUN chiffre ni AUCUNE cause : tu te "
            "contentes de reformuler. Si le verdict est ABSTAIN, explique clairement "
            "qu'aucune cause localisée n'a pu être prouvée et qu'on s'abstient d'agir."
        )
        try:
            return self.complete(system, json.dumps(payload, ensure_ascii=False))
        except Exception:
            return template_report(payload)


def template_report(p: dict) -> str:
    """Rédaction déterministe (mode mock / fallback). Mêmes faits, sans LLM."""
    verdict = p.get("verdict")
    metric = p.get("metric", "—")
    if verdict == "NO_ANOMALY":
        return "Aucune anomalie matérielle détectée. Rien à expliquer."
    if verdict == "ASSERT":
        return (f"PROUVÉ — la variation de '{metric}' se localise sur "
                f"{p['winning_dim']}={p['leading_segment']} "
                f"(concentration {p['concentration']:.0%}, confiance {p['confidence']:.2f}). "
                f"Action : {p['action_kind']}. {p['action_detail']}")
    return (f"ABSTENTION — la variation de '{metric}' est réelle mais ne se localise "
            f"sur aucune dimension testée ({', '.join(p.get('dims_tried', []))}). "
            f"Cause vraisemblablement systémique. Action : {p['action_kind']}. {p['action_detail']}")


# --- singleton paresseux partagé par les nodes ---
_CLIENT: QwenClient | None = None


def get_client() -> QwenClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = QwenClient()
    return _CLIENT
