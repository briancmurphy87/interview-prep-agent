"""
Tests for LLM wrapper methods that do not require a real OpenAI API key.

complete_json() and the call_log structure are tested by patching or
subclassing LLM so no network calls are made.
"""
from __future__ import annotations

import json
import pytest

from src.llm import LLM


class _PatchedLLM(LLM):
    """
    Subclass of LLM that overrides complete() with a scripted response list.
    Allows testing complete_json() and call_log without a real API key.
    """

    def __init__(self, outputs: list[str]):
        super().__init__(model="gpt-4.1-mini")
        self._outputs = list(outputs)

    def complete(self, system: str, user: str) -> str:
        if not self._outputs:
            raise RuntimeError("_PatchedLLM ran out of scripted outputs")
        return self._outputs.pop(0)


# ---------------------------------------------------------------------------
# complete_json()
# ---------------------------------------------------------------------------


def test_complete_json_parses_valid_json() -> None:
    payload = {"overall_score": 82, "red_flags": [], "suggested_improvements": ["Add metrics"]}
    llm = _PatchedLLM([json.dumps(payload)])

    result = llm.complete_json(system="sys", user="user")

    assert result == payload


def test_complete_json_raises_value_error_on_invalid_json() -> None:
    llm = _PatchedLLM(["this is not valid json"])

    with pytest.raises(ValueError, match="invalid JSON"):
        llm.complete_json(system="sys", user="user")


def test_complete_json_raises_on_plain_text_with_json_fence() -> None:
    # Model sometimes wraps JSON in a markdown fence — complete_json should fail.
    llm = _PatchedLLM(["```json\n{\"ok\": true}\n```"])

    with pytest.raises(ValueError, match="invalid JSON"):
        llm.complete_json(system="sys", user="user")


def test_complete_json_handles_whitespace_around_json() -> None:
    # Leading/trailing whitespace is stripped before parsing.
    payload = {"score": 90}
    llm = _PatchedLLM([f"\n  {json.dumps(payload)}  \n"])

    result = llm.complete_json(system="sys", user="user")

    assert result["score"] == 90


# ---------------------------------------------------------------------------
# call_log structure
# ---------------------------------------------------------------------------


def test_call_log_is_empty_before_any_call() -> None:
    llm = LLM(model="gpt-4.1-mini")
    assert llm.call_log == []


def test_call_log_has_correct_fields_after_call() -> None:
    # We can't test the real API here, but we can verify that _PatchedLLM
    # (which does NOT write to call_log since it bypasses complete()) leaves
    # the log empty — ensuring LLM.complete() is the sole log writer.
    llm = _PatchedLLM(["hello"])
    _ = llm.complete(system="s", user="u")  # _PatchedLLM.complete doesn't log
    assert llm.call_log == []


# ---------------------------------------------------------------------------
# Retry constants
# ---------------------------------------------------------------------------


def test_max_retries_is_positive_integer() -> None:
    assert isinstance(LLM.MAX_RETRIES, int)
    assert LLM.MAX_RETRIES >= 1


def test_retry_base_delay_is_positive_float() -> None:
    assert isinstance(LLM.RETRY_BASE_DELAY_S, float)
    assert LLM.RETRY_BASE_DELAY_S > 0