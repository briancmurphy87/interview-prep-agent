from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from src.agent_state import AgentState
from src.llm import LLM
from src.tools import TOOLS

# ----------------------------
# 4) Agent loop
# ----------------------------
ToolName = Literal[
    "extract_jd_requirements",
    "find_resume_evidence",
    "score_resume_fit",
    "render_report",
    "dump_state_summary",
]


@dataclass
class ToolAction:
    tool: str
    args: dict[str, Any]


@dataclass
class FinalAction:
    final: str


SYSTEM = """You are an Interview Prep Agent.

Your job is to analyze a job description and a resume, then produce a concise markdown interview-prep report.

You have these tools:

1) extract_jd_requirements(top_k:int) -> {"requirements":[...]}
2) find_resume_evidence(requirement:str, max_snippets:int) -> {"requirement":"...", "snippets":[...]}
3) score_resume_fit(requirements:list[str], evidence_per_requirement:int) -> {"score":int, "matched":[...], "gaps":[...]}
4) render_report() -> {"ok":true, "bytes":int}
5) dump_state_summary() -> {"num_notes":int, "artifact_keys":[...], "num_tool_calls":int}

Rules:
- Use tools instead of guessing when a tool can provide the needed information.
- Prefer this sequence:
  1. extract_jd_requirements
  2. score_resume_fit
  3. render_report
  4. final
- Do not call render_report until requirements and fit analysis exist.
- Do not return final until report_md exists in artifacts.
- Return JSON only. No markdown fences. No extra prose.

Valid JSON formats:

Tool call:
{"tool":"TOOL_NAME","args":{...}}

Final:
{"final":"done"}
"""


def _build_user_prompt(state: AgentState) -> str:
    notes = "\n".join(f"- {n}" for n in state.notes[-15:])
    tool_hist = json.dumps(state.tool_history[-8:], indent=2)
    artifacts_summary = json.dumps(
        {
            "artifact_keys": sorted(state.artifacts.keys()),
            "has_requirements": "requirements_json" in state.artifacts,
            "has_fit_analysis": "fit_analysis_json" in state.artifacts,
            "has_report": "report_md" in state.artifacts,
        },
        indent=2,
    )

    return f"""
JOB DESCRIPTION:
{state.jd_text}

RESUME:
{state.resume_text}

NOTES:
{notes if notes else "(none)"}

ARTIFACT SUMMARY:
{artifacts_summary}

RECENT TOOL HISTORY:
{tool_hist if state.tool_history else "(none)"}

Decide next action.
""".strip()


def _parse_action(raw: str) -> ToolAction | FinalAction:
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


def agent_step(llm: LLM, state: AgentState) -> ToolAction | FinalAction:
    raw = llm.complete(SYSTEM, _build_user_prompt(state))
    return _parse_action(raw)


def run_agent(llm: LLM, state: AgentState, max_iters: int = 8) -> AgentState:
    for _ in range(max_iters):
        try:
            action = agent_step(llm, state)
        except Exception as e:
            state.add_note(f"Agent step failed: {e}")
            break

        if isinstance(action, ToolAction):
            if action.tool not in TOOLS:
                state.add_note(f"Unknown tool requested: {action.tool}")
                state.add_tool_history(
                    tool_name=action.tool,
                    args=action.args,
                    error="unknown_tool",
                )
                continue

            try:
                result = TOOLS[action.tool](state, **action.args)
                state.add_tool_history(
                    tool_name=action.tool,
                    args=action.args,
                    result=result,
                )
                state.add_note(f"Ran {action.tool}")
            except TypeError as e:
                state.add_note(f"Tool argument mismatch for {action.tool}: {e}")
                state.add_tool_history(
                    tool_name=action.tool,
                    args=action.args,
                    error=f"argument_mismatch: {e}",
                )
            except Exception as e:
                state.add_note(f"Tool execution failed for {action.tool}: {e}")
                state.add_tool_history(
                    tool_name=action.tool,
                    args=action.args,
                    error=f"execution_failed: {e}",
                )
            continue

        if isinstance(action, FinalAction):
            if "report_md" not in state.artifacts:
                state.add_note(
                    "Model tried to finish before report existed. Rendering fallback report."
                )
                try:
                    result = TOOLS["render_report"](state)
                    state.add_tool_history(
                        tool_name="render_report",
                        args={},
                        result=result,
                    )
                except Exception as e:
                    state.add_note(f"Fallback render_report failed: {e}")
            return state

    state.add_note("Max iterations reached. Attempting final render.")
    if "report_md" not in state.artifacts:
        try:
            result = TOOLS["render_report"](state)
            state.add_tool_history(
                tool_name="render_report",
                args={},
                result=result,
            )
        except Exception as e:
            state.add_note(f"Final fallback render failed: {e}")
            state.artifacts["report_md"] = (
                "# Interview Prep Report\n\n"
                "The agent did not complete successfully.\n"
            )
    return state