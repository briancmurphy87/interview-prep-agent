from __future__ import annotations

from src.agent_state import AgentState
from src.llm import LLM
from src.tools import (
    tool_evaluate_target_resume,
    tool_render_report,
    tool_revise_target_resume,
)


class CriticAgent:
    """
    Deterministic pipeline that evaluates the draft, triggers a revision pass
    when the score falls below the threshold, and renders the final report.

    Separating evaluation and revision into a dedicated agent makes the
    quality-gate boundary explicit and independently testable.

    Contract:
      Input:  state with target_resume_txt
      Output: state with resume_evaluation_json, revision_metadata_json, report_md
    """

    def run(self, state: AgentState, llm: LLM) -> AgentState:
        if "resume_evaluation_json" not in state.artifacts:
            try:
                tool_evaluate_target_resume(state=state, llm=llm)
                state.add_note("CriticAgent: evaluated initial draft")
            except Exception as e:
                state.add_note(f"CriticAgent evaluation failed: {e}")

        if "revision_metadata_json" not in state.artifacts:
            try:
                tool_revise_target_resume(state=state, llm=llm)
                state.add_note("CriticAgent: ran revision pass")
            except Exception as e:
                state.add_note(f"CriticAgent revision failed: {e}")

        if "report_md" not in state.artifacts:
            try:
                tool_render_report(state)
                state.add_note("CriticAgent: rendered report")
            except Exception as e:
                state.add_note(f"CriticAgent report render failed: {e}")

        return state
