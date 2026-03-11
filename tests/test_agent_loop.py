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


def test_run_agent_invalid_json_falls_back_to_final_render() -> None:
    llm = FakeLLM(
        outputs=[
            'this is not valid json',
        ]
    )

    state = run_agent(llm=llm, state=make_state(), max_iters=2)

    assert "report_md" in state.artifacts
    assert "did not complete successfully" in state.artifacts["report_md"].lower() or \
           state.artifacts["report_md"].startswith("# Interview Prep Report")
    assert any("Agent step failed" in note for note in state.notes)