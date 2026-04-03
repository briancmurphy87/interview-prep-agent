from __future__ import annotations

from src.agent_loop import run_agent
from src.agent_state import AgentState


class FakeLLM:
    """
    Deterministic fake LLM for testing the agent loop.
    It returns a scripted sequence of JSON actions.
    """

    def __init__(self, outputs: list[str]):
        self.outputs = outputs
        self.idx = 0

    def complete(self, system: str, user: str) -> str:
        if self.idx >= len(self.outputs):
            raise RuntimeError("FakeLLM ran out of scripted outputs")
        output = self.outputs[self.idx]
        self.idx += 1
        return output


def make_state() -> AgentState:
    return AgentState(
        jd_text=(
            "Need a sales engineer with cloud, backup, disaster recovery, "
            "customer demos, and enterprise experience."
        ),
        resume_text=(
            "Built enterprise systems.\n"
            "Led customer demos.\n"
            "Worked on cloud infrastructure.\n"
            "Improved reliability and resilience.\n"
        ),
    )


def test_run_agent_happy_path() -> None:
    llm = FakeLLM(
        outputs=[
            '{"tool":"extract_jd_requirements","args":{"top_k":5}}',
            '{"tool":"score_resume_fit","args":{"requirements":["cloud","backup","customer","enterprise"],"evidence_per_requirement":2}}',
            '{"tool":"render_report","args":{}}',
            '{"final":"done"}',
        ]
    )

    state = run_agent(llm=llm, state=make_state(), max_iters=8)

    assert "requirements_json" in state.artifacts
    assert "fit_analysis_json" in state.artifacts
    assert "report_md" in state.artifacts
    assert len(state.tool_history) == 3
    assert state.artifacts["report_md"].startswith("# Interview Prep Report")


def test_run_agent_unknown_tool_records_error_and_recovers() -> None:
    llm = FakeLLM(
        outputs=[
            '{"tool":"not_a_real_tool","args":{}}',
            '{"tool":"extract_jd_requirements","args":{"top_k":3}}',
            '{"tool":"score_resume_fit","args":{"requirements":["cloud","customer"],"evidence_per_requirement":1}}',
            '{"tool":"render_report","args":{}}',
            '{"final":"done"}',
        ]
    )

    state = run_agent(llm=llm, state=make_state(), max_iters=8)

    assert len(state.tool_history) >= 4
    assert state.tool_history[0]["error"] == "unknown_tool"
    assert "report_md" in state.artifacts


def test_run_agent_invalid_json_writes_error_placeholder() -> None:
    # When every LLM call returns invalid JSON the loop breaks early, exhausts
    # the max-iters fallback, and writes an error string to target_resume_txt.
    # report_md is NOT produced by run_agent — that happens in main() post-loop.
    llm = FakeLLM(
        outputs=[
            'this is not valid json',
        ]
    )

    state = run_agent(llm=llm, state=make_state(), max_iters=2)

    assert "target_resume_txt" in state.artifacts
    assert "did not complete successfully" in state.artifacts["target_resume_txt"].lower()
    assert any("Agent step failed" in note for note in state.notes)
    assert "report_md" not in state.artifacts


# ---------------------------------------------------------------------------
# Fallback: premature {"final":"done"} before target_resume_txt exists
# ---------------------------------------------------------------------------


def test_run_agent_premature_final_triggers_generate_fallback() -> None:
    # LLM says "done" immediately, before any resume was generated.
    # The loop should call tool_generate_target_resume directly (which itself
    # calls llm.complete for the actual resume text).
    llm = FakeLLM(
        outputs=[
            '{"final":"done"}',              # premature final
            "Generated resume text here.",   # consumed by tool_generate_target_resume
        ]
    )

    state = run_agent(llm=llm, state=make_state(), max_iters=8)

    assert "target_resume_txt" in state.artifacts
    assert any(
        "tried to finish" in note or "generate_target_resume" in note
        for note in state.notes
    )


# ---------------------------------------------------------------------------
# Tool argument mismatch is recorded and loop continues
# ---------------------------------------------------------------------------


def test_run_agent_tool_argument_mismatch_records_error_and_continues() -> None:
    # score_resume_fit requires `requirements` arg; omitting it causes TypeError.
    llm = FakeLLM(
        outputs=[
            '{"tool":"score_resume_fit","args":{}}',   # missing required arg
            '{"tool":"extract_jd_requirements","args":{"top_k":3}}',
            '{"tool":"render_report","args":{}}',
            '{"final":"done"}',
        ]
    )

    state = run_agent(llm=llm, state=make_state(), max_iters=8)

    # First entry should record the argument mismatch
    assert state.tool_history[0]["tool"] == "score_resume_fit"
    assert "argument_mismatch" in state.tool_history[0]["error"]
    # Loop recovered and continued to produce subsequent artifacts
    assert "requirements_json" in state.artifacts
    assert "report_md" in state.artifacts