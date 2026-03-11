from __future__ import annotations

import os

# ----------------------------
# 3) LLM interface
# ----------------------------
class LLM:
    """
    Thin wrapper around the OpenAI Responses API.

    This class should stay small. Its job is to:
    - validate local environment
    - submit prompt input
    - return raw text
    """

    def __init__(self, model: str = "gpt-4.1-mini"):
        self.model = model

    def complete(self, system: str, user: str) -> str:
        """
        Returns raw text response.

        Requirements:
        - pip install openai
        - OPENAI_API_KEY must be set
        """
        try:
            from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError
        except ImportError as e:
            raise RuntimeError("Missing dependency. Run: pip install openai") from e

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. "
                "Set it in your environment or in a .env file."
            )

        client = OpenAI(api_key=api_key)

        try:
            resp = client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except RateLimitError as e:
            message = str(e)
            if "insufficient_quota" in message:
                raise RuntimeError(
                    "OpenAI API request failed: insufficient quota. "
                    "Enable billing / add credits in the OpenAI Platform, "
                    "or verify that this API key belongs to the correct billed project."
                ) from e
            raise RuntimeError(
                "OpenAI API request failed due to rate limiting. Please retry shortly."
            ) from e
        except APITimeoutError as e:
            raise RuntimeError(
                "OpenAI API request timed out. Please retry."
            ) from e
        except APIConnectionError as e:
            raise RuntimeError(
                "OpenAI API connection failed. Check your network connection."
            ) from e

        return resp.output_text