from __future__ import annotations

import json
import time

from src.agent_state import AgentState
from src.llm import LLM
from src.tools import tool_generate_target_resume


class WriterAgent:
    """
    Generates a targeted resume using the evidence grounding produced by AnalystAgent.

    This agent has a single responsibility and no decision loop — it calls
    tool_generate_target_resume exactly once and handles failure with a
    placeholder so downstream agents are never blocked on a missing artifact.

    Contract:
      Input:  state with fit_analysis_json and retrieved_examples_json
      Output: state with target_resume_txt
    """

    def run(self, state: AgentState, llm: LLM, top_k: int = 2) -> AgentState:
        if "target_resume_txt" in state.artifacts:
            return state

        t0 = time.monotonic()
        args = {"top_k": top_k}
        input_chars = len(json.dumps(args))

        try:
            result = tool_generate_target_resume(state=state, llm=llm, **args)
            state.add_tool_history(
                tool_name="generate_target_resume",
                args=args,
                result=result,
                duration_ms=round((time.monotonic() - t0) * 1000),
                input_chars=input_chars,
                output_chars=len(json.dumps(result, default=str)),
            )
            state.add_note("WriterAgent generated target resume")
        except Exception as e:
            state.add_note(f"WriterAgent generation failed: {e}")
            state.add_tool_history(
                tool_name="generate_target_resume",
                args=args,
                error=f"execution_failed: {e}",
                duration_ms=round((time.monotonic() - t0) * 1000),
                input_chars=input_chars,
            )
            state.artifacts["target_resume_txt"] = (
                "Resume generation did not complete successfully.\n"
            )

        return state
