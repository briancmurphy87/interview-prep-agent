from __future__ import annotations

from src.agent_state import AgentState
from src.tools import (
    tool_extract_jd_requirements,
    tool_find_resume_evidence,
    tool_score_resume_fit,
    tool_render_report,
)


def make_state() -> AgentState:
    return AgentState(
        jd_text=(
            "We are hiring a sales engineer with experience in cloud backup, "
            "disaster recovery, customer presentations, and enterprise software."
        ),
        resume_text=(
            "Brian Murphy\n"
            "Built low-latency enterprise software systems.\n"
            "Presented technical solutions to clients and internal stakeholders.\n"
            "Worked on cloud infrastructure and backup-adjacent resilience workflows.\n"
            "Led customer demos and technical architecture discussions.\n"
        ),
    )


def test_extract_jd_requirements_populates_artifacts() -> None:
    state = make_state()

    result = tool_extract_jd_requirements(state, top_k=10)

    assert "requirements" in result
    assert isinstance(result["requirements"], list)
    assert len(result["requirements"]) > 0
    assert "requirements_json" in state.artifacts
    assert state.artifacts["requirements_json"] == result


def test_find_resume_evidence_returns_matching_lines() -> None:
    state = make_state()

    result = tool_find_resume_evidence(
        state,
        requirement="customer presentations",
        max_snippets=3,
    )

    assert result["requirement"] == "customer presentations"
    assert isinstance(result["snippets"], list)
    assert len(result["snippets"]) >= 1
    assert any("Presented technical solutions" in s or "customer demos" in s for s in result["snippets"])


def test_score_resume_fit_creates_structured_fit_artifact() -> None:
    state = make_state()

    requirements = ["cloud", "backup", "customer", "sales", "enterprise"]
    result = tool_score_resume_fit(
        state,
        requirements=requirements,
        evidence_per_requirement=2,
    )

    assert "score" in result
    assert "matched" in result
    assert "gaps" in result
    assert isinstance(result["score"], int)
    assert "fit_analysis_json" in state.artifacts
    assert state.artifacts["fit_analysis_json"] == result


def test_render_report_writes_markdown_artifact() -> None:
    state = make_state()

    tool_extract_jd_requirements(state, top_k=5)
    tool_score_resume_fit(
        state,
        requirements=state.artifacts["requirements_json"]["requirements"],
        evidence_per_requirement=2,
    )

    result = tool_render_report(state)

    assert result["ok"] is True
    assert "report_md" in state.artifacts
    report = state.artifacts["report_md"]
    assert report.startswith("# Interview Prep Report")
    assert "## Overall Fit" in report
    assert "## Key Requirements" in report
    assert "## Resume Evidence" in report