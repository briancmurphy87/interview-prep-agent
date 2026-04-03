# ----------------------------
# 4) Agent loop
# ----------------------------
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Literal

from src.agent_state import AgentState
from src.llm import LLM
from src.tools import TOOLS


ToolName = Literal[
    "load_resume_corpus",
    "extract_jd_requirements",
    "retrieve_similar_resume_examples",
    "generate_target_resume",
    "dump_state_summary",
]


@dataclass
class ToolAction:
    tool: str
    args: dict[str, Any]


@dataclass
class FinalAction:
    final: str


SYSTEM = """You are a Resume Tailoring Agent.

Your only workflow is:

1. load_resume_corpus
2. extract_jd_requirements
3. retrieve_similar_resume_examples
4. generate_target_resume
5. final

Available tools:

1) load_resume_corpus(corpus_dir:str)
2) extract_jd_requirements(top_k:int)
3) retrieve_similar_resume_examples(top_k:int)
4) generate_target_resume(top_k:int)
5) dump_state_summary()

Rules:
- Your goal is to produce a targeted resume for the target job description.
- Use tools instead of guessing when tools can provide the information.
- Prefer the exact sequence shown above.
- Do not return final until target_resume_txt exists in artifacts.
- Return JSON only. No markdown fences. No extra prose.

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
            "has_retrieved_examples": "retrieved_examples_json" in state.artifacts,
            "has_target_resume": "target_resume_txt" in state.artifacts,
            "corpus_dir": state.artifacts.get("corpus_dir"),
        },
        indent=2,
    )

    return f"""
TARGET JOB DESCRIPTION:
{state.jd_text}

RAW RESUME:
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
    """
    Run the resume tailoring agent loop until completion or max_iters.

    Fallback chain (in order):
    1. Normal exit: LLM emits {"final": "done"} after producing target_resume_txt.
    2. Premature final: LLM emits {"final": "done"} before target_resume_txt exists —
       tool_generate_target_resume is called directly (top_k=2), then the loop exits.
    3. Max iterations reached: same forced generate call; if that also fails, a
       placeholder error string is written to target_resume_txt so downstream code
       never sees a missing artifact.
    """
    for _ in range(max_iters):
        try:
            action = agent_step(llm, state)
        except Exception as e:
            state.add_note(f"Agent step failed: {e}")
            break

        if isinstance(action, ToolAction):
            tool_name = action.tool
            tool_args = action.args

            if tool_name == "generate_target_resume":
                _t0 = time.monotonic()
                _input_chars = len(json.dumps(tool_args, default=str))
                try:
                    from src.tools import tool_generate_target_resume

                    result = tool_generate_target_resume(
                        state=state,
                        llm=llm,
                        **tool_args,
                    )
                    state.add_tool_history(
                        tool_name=tool_name,
                        args=tool_args,
                        result=result,
                        duration_ms=round((time.monotonic() - _t0) * 1000),
                        input_chars=_input_chars,
                        output_chars=len(json.dumps(result, default=str)),
                    )
                    state.add_note(f"Ran {tool_name}")
                except TypeError as e:
                    state.add_note(f"Tool argument mismatch for {tool_name}: {e}")
                    state.add_tool_history(
                        tool_name=tool_name,
                        args=tool_args,
                        error=f"argument_mismatch: {e}",
                        duration_ms=round((time.monotonic() - _t0) * 1000),
                        input_chars=_input_chars,
                    )
                except Exception as e:
                    state.add_note(f"Tool execution failed for {tool_name}: {e}")
                    state.add_tool_history(
                        tool_name=tool_name,
                        args=tool_args,
                        error=f"execution_failed: {e}",
                        duration_ms=round((time.monotonic() - _t0) * 1000),
                        input_chars=_input_chars,
                    )
                continue

            if tool_name not in TOOLS:
                state.add_note(f"Unknown tool requested: {tool_name}")
                state.add_tool_history(
                    tool_name=tool_name,
                    args=tool_args,
                    error="unknown_tool",
                )
                continue

            _t0 = time.monotonic()
            _input_chars = len(json.dumps(tool_args, default=str))
            try:
                if tool_name == "load_resume_corpus" and "corpus_dir" not in tool_args:
                    corpus_dir = state.artifacts.get("corpus_dir")
                    if corpus_dir:
                        tool_args = {
                            **tool_args,
                            "corpus_dir": corpus_dir,
                        }

                result = TOOLS[tool_name](state, **tool_args)
                state.add_tool_history(
                    tool_name=tool_name,
                    args=tool_args,
                    result=result,
                    duration_ms=round((time.monotonic() - _t0) * 1000),
                    input_chars=_input_chars,
                    output_chars=len(json.dumps(result, default=str)),
                )
                state.add_note(f"Ran {tool_name}")
            except TypeError as e:
                state.add_note(f"Tool argument mismatch for {tool_name}: {e}")
                state.add_tool_history(
                    tool_name=tool_name,
                    args=tool_args,
                    error=f"argument_mismatch: {e}",
                    duration_ms=round((time.monotonic() - _t0) * 1000),
                    input_chars=_input_chars,
                )
            except Exception as e:
                state.add_note(f"Tool execution failed for {tool_name}: {e}")
                state.add_tool_history(
                    tool_name=tool_name,
                    args=tool_args,
                    error=f"execution_failed: {e}",
                    duration_ms=round((time.monotonic() - _t0) * 1000),
                    input_chars=_input_chars,
                )
            continue

        if isinstance(action, FinalAction):
            if "target_resume_txt" not in state.artifacts:
                state.add_note(
                    "Model tried to finish before producing target_resume_txt."
                )
                _t0 = time.monotonic()
                try:
                    from src.tools import tool_generate_target_resume

                    result = tool_generate_target_resume(
                        state=state,
                        llm=llm,
                        top_k=2,
                    )
                    state.add_tool_history(
                        tool_name="generate_target_resume",
                        args={"top_k": 2},
                        result=result,
                        duration_ms=round((time.monotonic() - _t0) * 1000),
                        input_chars=len(json.dumps({"top_k": 2})),
                        output_chars=len(json.dumps(result, default=str)),
                    )
                except Exception as e:
                    state.add_note(f"Fallback generate_target_resume failed: {e}")
                    state.add_tool_history(
                        tool_name="generate_target_resume",
                        args={"top_k": 2},
                        error=f"execution_failed: {e}",
                        duration_ms=round((time.monotonic() - _t0) * 1000),
                        input_chars=len(json.dumps({"top_k": 2})),
                    )

            return state

    state.add_note("Max iterations reached. Attempting final fallback.")

    if "target_resume_txt" not in state.artifacts:
        _t0 = time.monotonic()
        try:
            from src.tools import tool_generate_target_resume

            result = tool_generate_target_resume(
                state=state,
                llm=llm,
                top_k=2,
            )
            state.add_tool_history(
                tool_name="generate_target_resume",
                args={"top_k": 2},
                result=result,
                duration_ms=round((time.monotonic() - _t0) * 1000),
                input_chars=len(json.dumps({"top_k": 2})),
                output_chars=len(json.dumps(result, default=str)),
            )
            return state
        except Exception as e:
            state.add_note(f"Final fallback generate_target_resume failed: {e}")
            state.add_tool_history(
                tool_name="generate_target_resume",
                args={"top_k": 2},
                error=f"execution_failed: {e}",
                duration_ms=round((time.monotonic() - _t0) * 1000),
                input_chars=len(json.dumps({"top_k": 2})),
            )

        state.artifacts["target_resume_txt"] = (
            "Targeted resume generation did not complete successfully.\n"
        )

    return state
