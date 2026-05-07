from __future__ import annotations

import json

from src.agent_state import AgentState
from src.agents.base import FinalAction, parse_action, run_tool
from src.llm import LLM
from src.tools import TOOLS

ALLOWED_TOOLS = {
    "load_resume_corpus",
    "extract_jd_requirements",
    "retrieve_similar_resume_examples",
    "score_resume_fit",
}

SYSTEM = """You are a Resume Analyst Agent.

Your job is to build the evidence foundation for resume generation:

1. load_resume_corpus(corpus_dir: str)          — load reference examples
2. extract_jd_requirements(top_k: int)          — extract key requirements from the JD
3. retrieve_similar_resume_examples(top_k: int) — rank corpus examples by JD similarity
4. score_resume_fit()                           — match requirements to resume evidence

Rules:
- Call tools in the order above.
- Do not emit final until fit_analysis_json exists in artifacts.
- Only call the 4 tools listed; others will be rejected.
- Return JSON only. No markdown fences. No prose.

Tool call: {"tool":"TOOL_NAME","args":{...}}
Final:     {"final":"done"}
"""


def _build_prompt(state: AgentState) -> str:
    notes = "\n".join(f"- {n}" for n in state.notes[-10:])
    tool_hist = json.dumps(state.tool_history[-6:], indent=2)
    summary = json.dumps(
        {
            "artifact_keys": sorted(state.artifacts.keys()),
            "corpus_dir": state.artifacts.get("corpus_dir"),
            "has_requirements": "requirements_json" in state.artifacts,
            "has_retrieved_examples": "retrieved_examples_json" in state.artifacts,
            "has_fit_analysis": "fit_analysis_json" in state.artifacts,
        },
        indent=2,
    )
    return f"""TARGET JOB DESCRIPTION:
{state.jd_text}

RAW RESUME:
{state.resume_text}

NOTES:
{notes or "(none)"}

ARTIFACT SUMMARY:
{summary}

RECENT TOOL HISTORY:
{tool_hist if state.tool_history else "(none)"}

Decide next action.""".strip()


class AnalystAgent:
    """
    LLM-driven loop that loads the corpus, extracts JD requirements,
    retrieves similar examples, and scores resume fit.

    Contract:
      Input:  state with jd_text, resume_text, and corpus_dir in artifacts
      Output: state with fit_analysis_json (and requirements/retrieval as side effects)
    """

    def run(self, state: AgentState, llm: LLM, max_iters: int = 6) -> AgentState:
        for _ in range(max_iters):
            if "fit_analysis_json" in state.artifacts:
                break

            try:
                raw = llm.complete(SYSTEM, _build_prompt(state))
                action = parse_action(raw)
            except Exception as e:
                state.add_note(f"AnalystAgent step failed: {e}")
                break

            if isinstance(action, FinalAction):
                break

            tool_name = action.tool
            tool_args = action.args

            if tool_name not in ALLOWED_TOOLS:
                state.add_note(f"AnalystAgent rejected disallowed tool: {tool_name}")
                state.add_tool_history(
                    tool_name=tool_name, args=tool_args, error="disallowed_tool"
                )
                continue

            if tool_name == "load_resume_corpus" and "corpus_dir" not in tool_args:
                corpus_dir = state.artifacts.get("corpus_dir")
                if corpus_dir:
                    tool_args = {**tool_args, "corpus_dir": corpus_dir}

            run_tool(state, tool_name, tool_args, TOOLS[tool_name])

        # Fallback: guarantee the downstream agents always have what they need.
        if "requirements_json" not in state.artifacts:
            run_tool(
                state,
                "extract_jd_requirements",
                {"top_k": 10},
                TOOLS["extract_jd_requirements"],
            )
        if "fit_analysis_json" not in state.artifacts:
            run_tool(state, "score_resume_fit", {}, TOOLS["score_resume_fit"])

        return state
