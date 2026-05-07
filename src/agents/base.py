from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from src.agent_state import AgentState


@dataclass
class ToolAction:
    tool: str
    args: dict[str, Any]


@dataclass
class FinalAction:
    final: str


def parse_action(raw: str) -> ToolAction | FinalAction:
    try:
        payload = json.loads(raw.strip())
    except json.JSONDecodeError as e:
        raise ValueError(f"Model returned invalid JSON: {raw}") from e

    if not isinstance(payload, dict):
        raise ValueError(f"Model returned non-object JSON: {payload!r}")

    has_tool = "tool" in payload
    has_final = "final" in payload

    if has_tool == has_final:
        raise ValueError(
            f"Model output must contain exactly one of 'tool' or 'final': {payload!r}"
        )

    if has_tool:
        tool = payload["tool"]
        args = payload.get("args", {})
        if not isinstance(tool, str):
            raise ValueError(f"Tool name must be a string: {payload!r}")
        if not isinstance(args, dict):
            raise ValueError(f"Tool args must be an object: {payload!r}")
        return ToolAction(tool=tool, args=args)

    final = payload["final"]
    if not isinstance(final, str):
        raise ValueError(f"Final value must be a string: {payload!r}")
    return FinalAction(final=final)


def run_tool(
    state: AgentState,
    tool_name: str,
    tool_args: dict[str, Any],
    tool_fn: Callable[..., dict[str, Any]],
) -> dict[str, Any] | None:
    """Dispatch a single tool call: record timing, handle errors, update history."""
    t0 = time.monotonic()
    input_chars = len(json.dumps(tool_args, default=str))
    try:
        result = tool_fn(state, **tool_args)
        state.add_tool_history(
            tool_name=tool_name,
            args=tool_args,
            result=result,
            duration_ms=round((time.monotonic() - t0) * 1000),
            input_chars=input_chars,
            output_chars=len(json.dumps(result, default=str)),
        )
        state.add_note(f"Ran {tool_name}")
        return result
    except TypeError as e:
        state.add_note(f"Tool argument mismatch for {tool_name}: {e}")
        state.add_tool_history(
            tool_name=tool_name,
            args=tool_args,
            error=f"argument_mismatch: {e}",
            duration_ms=round((time.monotonic() - t0) * 1000),
            input_chars=input_chars,
        )
        return None
    except Exception as e:
        state.add_note(f"Tool execution failed for {tool_name}: {e}")
        state.add_tool_history(
            tool_name=tool_name,
            args=tool_args,
            error=f"execution_failed: {e}",
            duration_ms=round((time.monotonic() - t0) * 1000),
            input_chars=input_chars,
        )
        return None
