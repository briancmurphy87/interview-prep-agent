from __future__ import annotations

import os

# ----------------------------
# 3) LLM interface
# ----------------------------
class LLM:
    """
    Minimal wrapper. Implemented for OpenAI Responses API style.
    If you use another provider, replace only this class.
    """

    def __init__(self, model: str = "gpt-4.1-mini"):
        self.model = model

    def complete(self, system: str, user: str) -> str:
        """
        Returns raw text response.
        Requires: pip install openai
        Env var: OPENAI_API_KEY
        """
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("Missing dependency. Run: pip install openai") from e

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY in your environment.")

        client = OpenAI()
        resp = client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.output_text
