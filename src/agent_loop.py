# ----------------------------
# 4) Agent loop
# ----------------------------
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from src.agent_state import AgentState
from src.llm import LLM
from src.tools import TOOLS

ToolName = Literal[
    "extract_jd_requirements",
    "find_resume_evidence",
    "score_resume_fit",
    "render_report",
    "dump_state_summary",
    "load_resume_corpus",
    "retrieve_similar_resume_examples",
    "generate_target_resume",
]


@dataclass
class ToolAction:
    tool: str
    args: dict[str, Any]


@dataclass
class FinalAction:
    final: str


SYSTEM = """You are an Interview Prep Agent.

You support two workflows:

1) Analysis workflow:
- analyze job description vs resume
- produce a report

2) Resume synthesis workflow:
- load prior resume/JD examples from a corpus
- retrieve the most relevant prior examples
- generate a tailored resume for the target JD

Available tools:

1) extract_jd_requirements(top_k:int)
2) find_resume_evidence(requirement:str, max_snippets:int)
3) score_resume_fit(requirements:list[str], evidence_per_requirement:int)
4) render_report()
5) dump_state_summary()
6) load_resume_corpus(corpus_dir:str)
7) retrieve_similar_resume_examples(top_k:int)
8) generate_target_resume(top_k:int)

Rules:
- Use tools instead of guessing when tools can provide the information.
- If desired_output == "target_resume", you MUST prioritize the resume synthesis workflow.
- If desired_output == "target_resume", do not call render_report unless resume generation fails.
- If desired_output == "target_resume", prefer this sequence:
  1. load_resume_corpus
  2. extract_jd_requirements
  3. retrieve_similar_resume_examples
  4. generate_target_resume
  5. final
- If desired_output == "report", prefer this sequence:
  1. extract_jd_requirements
  2. score_resume_fit
  3. render_report
  4. final
- Do not return final until the requested output artifact exists.
- Return JSON only.

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
            "has_target_resume": "target_resume_txt" in state.artifacts,
            "corpus_dir": state.artifacts.get("corpus_dir"),
            "desired_output": state.artifacts.get("desired_output"),
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


def run_agent(llm: LLM, state: AgentState, max_iters: int = 10) -> AgentState:
    """
    Main agent loop.

    Supports two workflows:

    1) Analysis workflow
       - extract_jd_requirements
       - score_resume_fit
       - render_report
       - final

    2) Resume synthesis workflow
       - load_resume_corpus
       - extract_jd_requirements
       - retrieve_similar_resume_examples
       - generate_target_resume
       - final
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
                    )
                    state.add_note(f"Ran {tool_name}")
                except TypeError as e:
                    state.add_note(f"Tool argument mismatch for {tool_name}: {e}")
                    state.add_tool_history(
                        tool_name=tool_name,
                        args=tool_args,
                        error=f"argument_mismatch: {e}",
                    )
                except Exception as e:
                    state.add_note(f"Tool execution failed for {tool_name}: {e}")
                    state.add_tool_history(
                        tool_name=tool_name,
                        args=tool_args,
                        error=f"execution_failed: {e}",
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

            try:
                # Small convenience: if the model asks to load the corpus
                # without specifying a path, fall back to state.artifacts["corpus_dir"]
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
                )
                state.add_note(f"Ran {tool_name}")
            except TypeError as e:
                state.add_note(f"Tool argument mismatch for {tool_name}: {e}")
                state.add_tool_history(
                    tool_name=tool_name,
                    args=tool_args,
                    error=f"argument_mismatch: {e}",
                )
            except Exception as e:
                state.add_note(f"Tool execution failed for {tool_name}: {e}")
                state.add_tool_history(
                    tool_name=tool_name,
                    args=tool_args,
                    error=f"execution_failed: {e}",
                )
            continue

        if isinstance(action, FinalAction):
            desired_output = state.artifacts.get("desired_output", "report")
            has_report = "report_md" in state.artifacts
            has_target_resume = "target_resume_txt" in state.artifacts

            if desired_output == "target_resume" and not has_target_resume:
                state.add_note(
                    "Model tried to finish before producing target_resume_txt."
                )
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
                    )
                except Exception as e:
                    state.add_note(f"Fallback generate_target_resume failed: {e}")

            elif desired_output == "report" and not has_report:
                state.add_note(
                    "Model tried to finish before producing report_md."
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

    state.add_note("Max iterations reached. Attempting final fallback.")

    desired_output = state.artifacts.get("desired_output", "report")
    has_report = "report_md" in state.artifacts
    has_target_resume = "target_resume_txt" in state.artifacts

    if not has_report and not has_target_resume:
        if desired_output == "target_resume":
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
                )
                return state
            except Exception as e:
                state.add_note(f"Final fallback generate_target_resume failed: {e}")

        if desired_output == "report":
            if (
                "requirements_json" in state.artifacts
                or "fit_analysis_json" in state.artifacts
            ):
                try:
                    result = TOOLS["render_report"](state)
                    state.add_tool_history(
                        tool_name="render_report",
                        args={},
                        result=result,
                    )
                    return state
                except Exception as e:
                    state.add_note(f"Final fallback render failed: {e}")

        # Last resort fallback artifact
        if desired_output == "target_resume":
            state.artifacts["target_resume_txt"] = (
                "Targeted resume generation did not complete successfully.\n"
            )
        else:
            state.artifacts["report_md"] = (
                "# Interview Prep Report\n\n"
                "The agent did not complete successfully.\n"
            )

    return state