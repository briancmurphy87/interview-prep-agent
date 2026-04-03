# ----------------------------
# 3) LLM interface
# ----------------------------
from __future__ import annotations

import os
import time
from typing import Any

# ---------------------------------------------------------------------------
# Cost estimation
# Approximate prices per million tokens.  Update as pricing changes.
# ---------------------------------------------------------------------------
_COST_PER_M_TOKENS: dict[str, tuple[float, float]] = {
    # model_name: (input_usd_per_M, output_usd_per_M)
    "gpt-4.1-mini":  (0.40,  1.60),
    "gpt-4.1-nano":  (0.10,  0.40),
    "gpt-4.1":       (2.00,  8.00),
    "gpt-4o":        (2.50, 10.00),
    "gpt-4o-mini":   (0.15,  0.60),
}


def _estimate_cost(
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
) -> float | None:
    """Return estimated USD cost for a call, or None if pricing is unknown."""
    rates = _COST_PER_M_TOKENS.get(model)
    if rates is None or input_tokens is None or output_tokens is None:
        return None
    input_cost  = (input_tokens  / 1_000_000) * rates[0]
    output_cost = (output_tokens / 1_000_000) * rates[1]
    return round(input_cost + output_cost, 6)


class LLM:
    """
    Thin wrapper around the OpenAI Responses API.

    This class should stay small. Its job is to:
    - validate local environment
    - submit prompt input
    - return raw text

    Observability: every call is appended to self.call_log as a dict with:
      ok, duration_ms, model, input_chars, output_chars,
      input_tokens, output_tokens, estimated_cost_usd, error (on failure)
    """

    def __init__(self, model: str = "gpt-4.1-mini"):
        self.model = model
        # Accumulates one entry per complete() call for the run artifact.
        self.call_log: list[dict[str, Any]] = []

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

        input_chars = len(system) + len(user)
        t0 = time.monotonic()

        try:
            resp = client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except RateLimitError as e:
            self.call_log.append({
                "ok": False,
                "duration_ms": round((time.monotonic() - t0) * 1000),
                "model": self.model,
                "input_chars": input_chars,
                "output_chars": None,
                "input_tokens": None,
                "output_tokens": None,
                "estimated_cost_usd": None,
                "error": "RateLimitError",
            })
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
            self.call_log.append({
                "ok": False,
                "duration_ms": round((time.monotonic() - t0) * 1000),
                "model": self.model,
                "input_chars": input_chars,
                "output_chars": None,
                "input_tokens": None,
                "output_tokens": None,
                "estimated_cost_usd": None,
                "error": "APITimeoutError",
            })
            raise RuntimeError("OpenAI API request timed out. Please retry.") from e
        except APIConnectionError as e:
            self.call_log.append({
                "ok": False,
                "duration_ms": round((time.monotonic() - t0) * 1000),
                "model": self.model,
                "input_chars": input_chars,
                "output_chars": None,
                "input_tokens": None,
                "output_tokens": None,
                "estimated_cost_usd": None,
                "error": "APIConnectionError",
            })
            raise RuntimeError("OpenAI API connection failed. Check your network connection.") from e

        duration_ms = round((time.monotonic() - t0) * 1000)
        output_text = resp.output_text
        output_chars = len(output_text)

        # Extract token usage if the response provides it (safely)
        usage = getattr(resp, "usage", None)
        input_tokens: int | None = getattr(usage, "input_tokens", None)
        output_tokens: int | None = getattr(usage, "output_tokens", None)

        self.call_log.append({
            "ok": True,
            "duration_ms": duration_ms,
            "model": self.model,
            "input_chars": input_chars,
            "output_chars": output_chars,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": _estimate_cost(self.model, input_tokens, output_tokens),
        })

        return output_text