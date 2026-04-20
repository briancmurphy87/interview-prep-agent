# ----------------------------
# 3) LLM interface
# ----------------------------
from __future__ import annotations

import json
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
    - return raw text (complete) or parsed JSON (complete_json)

    Retry behaviour:
      Transient network errors (APITimeoutError, APIConnectionError) are
      retried up to MAX_RETRIES times with exponential back-off starting at
      RETRY_BASE_DELAY_S.  Rate-limit errors are not retried because they
      either indicate a billing issue (insufficient_quota) or require
      respecting a Retry-After header that we do not yet parse.

    Observability: every API attempt is appended to self.call_log as a dict:
      ok, duration_ms, model, input_chars, output_chars,
      input_tokens, output_tokens, estimated_cost_usd, error (on failure),
      attempt (1-indexed, present only when > 1)
    """

    MAX_RETRIES: int = 2           # up to 3 total attempts (initial + 2 retries)
    RETRY_BASE_DELAY_S: float = 1.0  # back-off delays: 1 s, 2 s

    def __init__(self, model: str = "gpt-4.1-mini"):
        self.model = model
        # Accumulates one entry per API attempt for the run artifact.
        self.call_log: list[dict[str, Any]] = []

    def complete(self, system: str, user: str) -> str:
        """
        Submit a prompt and return the raw text response.

        Retries automatically on transient network errors (timeout, connection
        refused) with exponential back-off.  Raises RuntimeError on permanent
        failures (billing, quota, exhausted retries).

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

        last_transient_exc: Exception | None = None

        for attempt in range(self.MAX_RETRIES + 1):
            t0 = time.monotonic()
            attempt_label = attempt + 1  # 1-indexed for readability in logs

            try:
                resp = client.responses.create(
                    model=self.model,
                    input=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )

            except RateLimitError as e:
                # Not retried — either a billing issue or needs Retry-After handling.
                duration_ms = round((time.monotonic() - t0) * 1000)
                self.call_log.append({
                    "ok": False,
                    "duration_ms": duration_ms,
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

            except (APITimeoutError, APIConnectionError) as e:
                # Transient — retry with back-off.
                duration_ms = round((time.monotonic() - t0) * 1000)
                error_type = type(e).__name__
                log_entry: dict[str, Any] = {
                    "ok": False,
                    "duration_ms": duration_ms,
                    "model": self.model,
                    "input_chars": input_chars,
                    "output_chars": None,
                    "input_tokens": None,
                    "output_tokens": None,
                    "estimated_cost_usd": None,
                    "error": error_type,
                }
                if attempt_label > 1:
                    log_entry["attempt"] = attempt_label
                self.call_log.append(log_entry)

                if attempt < self.MAX_RETRIES:
                    delay = self.RETRY_BASE_DELAY_S * (2 ** attempt)
                    time.sleep(delay)
                    last_transient_exc = e
                    continue

                # Exhausted retries.
                if isinstance(e, APITimeoutError):
                    raise RuntimeError(
                        f"OpenAI API request timed out after {attempt_label} attempt(s). "
                        "Please retry."
                    ) from e
                raise RuntimeError(
                    f"OpenAI API connection failed after {attempt_label} attempt(s). "
                    "Check your network connection."
                ) from e

            # --- Success path ---
            duration_ms = round((time.monotonic() - t0) * 1000)
            output_text = resp.output_text
            output_chars = len(output_text)

            # Extract token usage if the response provides it (safely)
            usage = getattr(resp, "usage", None)
            input_tokens: int | None = getattr(usage, "input_tokens", None)
            output_tokens: int | None = getattr(usage, "output_tokens", None)

            log_entry = {
                "ok": True,
                "duration_ms": duration_ms,
                "model": self.model,
                "input_chars": input_chars,
                "output_chars": output_chars,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "estimated_cost_usd": _estimate_cost(self.model, input_tokens, output_tokens),
            }
            if attempt_label > 1:
                log_entry["attempt"] = attempt_label
            self.call_log.append(log_entry)

            return output_text

        # Should be unreachable; satisfy the type checker.
        raise RuntimeError("Unexpected retry exhaustion") from last_transient_exc

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        """
        Like complete(), but parses the response as JSON and returns a dict.

        The system and user prompts must instruct the model to return JSON only.
        Raises ValueError if the response cannot be parsed as JSON.

        Use this for structured tool calls (evaluation, requirement extraction)
        where downstream code depends on a specific schema.
        """
        raw = self.complete(system=system, user=user)
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid JSON: {raw!r}") from e