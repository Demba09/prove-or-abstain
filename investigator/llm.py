"""Qwen client via DashScope (OpenAI-compatible endpoint).

Thin wrapper. Contract for the whole system: the LLM prioritises, orients, and
narrates. It NEVER returns a number that ends up in the analysis. If you find
yourself asking it for a figure, that is a design bug.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from .config import DASHSCOPE_BASE_URL, QWEN_MODEL


class QwenClient:
    def __init__(self, model: str = QWEN_MODEL, base_url: str = DASHSCOPE_BASE_URL):
        # Imported lazily so the rest of the package runs without the SDK installed.
        from openai import OpenAI

        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("Set DASHSCOPE_API_KEY in the environment.")
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def complete(
        self,
        system: str,
        user: str,
        json_mode: bool = False,
        temperature: float = 0.2,
    ) -> str:
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )
        return resp.choices[0].message.content

    def complete_json(self, system: str, user: str, **kwargs) -> dict:
        raw = self.complete(system, user, json_mode=True, **kwargs)
        return json.loads(raw)


# Optional singleton so nodes don't each construct a client.
_client: Optional[QwenClient] = None


def get_client() -> QwenClient:
    global _client
    if _client is None:
        _client = QwenClient()
    return _client
