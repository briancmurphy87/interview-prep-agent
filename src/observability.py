# ----------------------------
# 5) Observability
# ----------------------------
"""
Run artifact assembly.

build_run_artifact() collects timing and cost data from two sources:

  state.tool_history  — tool spans, augmented with duration_ms / input_chars /
                        output_chars by agent_loop.py at dispatch time.

  llm.call_log        — LLM call spans recorded by LLM.complete(), each
                        containing duration_ms, token counts, and estimated cost.

The returned dict matches the run artifact schema:

  {
    "run_id":    "<timestamp>_<slug>",
    "model":     "gpt-4.1-mini",
    "tools": [
      {"tool": "load_resume_corpus",   "ok": true,  "duration_ms": 12, ...},
      {"tool": "generate_target_resume","ok": true,  "duration_ms": 8421,
       "input_chars": 15423, "output_chars": 6881},
      ...
    ],
    "llm_calls": [
      {"ok": true, "duration_ms": 2140, "input_tokens": 3200,
       "output_tokens": 820, "estimated_cost_usd": 0.0045},
      ...
    ],
    "totals": {
      "duration_ms":        10433,
      "llm_call_count":     3,
      "estimated_cost_usd": 0.0184
    }
  }

The totals.duration_ms is the sum of all tool and LLM span durations.  It
reflects measured wall time per call — not end-to-end elapsed time, which also
includes LLM prompt-building, JSON parsing, and other overhead.
"""

from __future__ import annotations

from typing import Any

from src.agent_state import AgentState
from src.llm import LLM


def build_run_artifact(
    run_id: str,
    llm: LLM,
    state: AgentState,
) -> dict[str, Any]:
    """
    Assemble the observability artifact for a completed agent run.

    Args:
        run_id: Unique identifier for this run (e.g. "20260403T142219Z_netflix_swe_n_tech").
        llm:    The LLM instance used during the run (carries call_log).
        state:  The final AgentState (carries tool_history).

    Returns:
        A dict ready for JSON serialisation.
    """
    # --- Tool spans ---
    # Each entry comes from state.tool_history, which agent_loop augments with
    # duration_ms, input_chars, and output_chars at dispatch time.
    tool_spans: list[dict[str, Any]] = []
    for entry in state.tool_history:
        span: dict[str, Any] = {
            "tool": entry["tool"],
            "ok": "error" not in entry,
        }
        if "duration_ms" in entry:
            span["duration_ms"] = entry["duration_ms"]
        if "input_chars" in entry:
            span["input_chars"] = entry["input_chars"]
        if "output_chars" in entry:
            span["output_chars"] = entry["output_chars"]
        if "error" in entry:
            span["error"] = entry["error"]
        tool_spans.append(span)

    # --- LLM call spans ---
    # Populated by LLM.complete() — one entry per API call including
    # agent-loop orchestration calls and tool-internal calls (e.g.
    # generate_target_resume, tool_evaluate_target_resume).
    llm_spans = llm.call_log

    # --- Totals ---
    tool_duration_total = sum(s.get("duration_ms") or 0 for s in tool_spans)
    llm_duration_total = sum(c.get("duration_ms") or 0 for c in llm_spans)

    known_costs = [
        c["estimated_cost_usd"]
        for c in llm_spans
        if c.get("estimated_cost_usd") is not None
    ]
    estimated_cost: float | None = (
        round(sum(known_costs), 6) if known_costs else None
    )

    return {
        "run_id": run_id,
        "model": llm.model,
        "tools": tool_spans,
        "llm_calls": llm_spans,
        "totals": {
            # Sum of individually measured spans; excludes inter-call overhead.
            "duration_ms": tool_duration_total + llm_duration_total,
            "llm_call_count": len(llm_spans),
            "estimated_cost_usd": estimated_cost,
        },
    }
