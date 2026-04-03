from __future__ import annotations

import pytest

from src.agent_state import AgentState
from src.tools import (
    tool_extract_jd_requirements,
    tool_find_resume_evidence,
    tool_load_resume_corpus,
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


# ---------------------------------------------------------------------------
# tool_find_resume_evidence — edge cases
# ---------------------------------------------------------------------------


def test_find_resume_evidence_empty_requirement_returns_empty() -> None:
    state = make_state()
    result = tool_find_resume_evidence(state, requirement="", max_snippets=5)
    assert result["requirement"] == ""
    assert result["snippets"] == []


def test_find_resume_evidence_no_match_returns_empty_snippets() -> None:
    state = make_state()
    result = tool_find_resume_evidence(
        state,
        requirement="quantum cryptography blockchain",
        max_snippets=5,
    )
    assert result["snippets"] == []


# ---------------------------------------------------------------------------
# tool_score_resume_fit — edge cases
# ---------------------------------------------------------------------------


def test_score_resume_fit_empty_requirements_returns_zero_no_crash() -> None:
    state = make_state()
    result = tool_score_resume_fit(state, requirements=[], evidence_per_requirement=2)
    assert result["score"] == 0
    assert result["matched"] == []
    assert result["gaps"] == []
    assert "fit_analysis_json" in state.artifacts


# ---------------------------------------------------------------------------
# tool_render_report — with LLM evaluation payload present
# ---------------------------------------------------------------------------


def test_render_report_includes_llm_evaluation_section() -> None:
    state = make_state()
    tool_extract_jd_requirements(state, top_k=5)
    tool_score_resume_fit(
        state,
        requirements=state.artifacts["requirements_json"]["requirements"],
        evidence_per_requirement=2,
    )
    state.artifacts["resume_evaluation_json"] = {
        "overall_score": 82,
        "jd_alignment": {"score": 85, "reason": "Strong overlap with role requirements."},
        "keyword_coverage": {"score": 78, "reason": "Most keywords present."},
        "clarity": {"score": 90, "reason": "Concise and well-structured."},
        "exaggeration_risk": {"score": 80, "reason": "Claims are defensible."},
        "ats_compatibility": {"score": 88, "reason": "Standard formatting."},
        "red_flags": ["One vague leadership claim"],
        "suggested_improvements": ["Quantify impact metrics"],
    }

    result = tool_render_report(state)

    assert result["ok"] is True
    report = state.artifacts["report_md"]
    assert "## LLM Resume Evaluation" in report
    assert "82" in report
    assert "JD Alignment" in report
    assert "Keyword Coverage" in report
    assert "Red Flags" in report
    assert "One vague leadership claim" in report
    assert "Suggested Improvements" in report
    assert "Quantify impact metrics" in report


# ---------------------------------------------------------------------------
# tool_load_resume_corpus — filesystem behaviour
# ---------------------------------------------------------------------------


def test_load_resume_corpus_reads_expected_structure(tmp_path) -> None:
    slug_dir = tmp_path / "acme_swe"
    slug_dir.mkdir()
    (slug_dir / "jd.txt").write_text("Engineer with Python and cloud experience.", encoding="utf-8")
    (slug_dir / "resume_variant.txt").write_text("Led backend Python services.", encoding="utf-8")

    state = AgentState(jd_text="Python cloud engineer", resume_text="Python developer")
    result = tool_load_resume_corpus(state, corpus_dir=str(tmp_path))

    assert result["num_examples"] == 1
    assert "acme_swe" in result["slugs"]
    assert len(state.corpus_examples) == 1
    assert state.corpus_examples[0].slug == "acme_swe"
    assert "Python" in state.corpus_examples[0].jd_text


def test_load_resume_corpus_skips_dirs_missing_resume_variant(tmp_path) -> None:
    incomplete = tmp_path / "no_variant"
    incomplete.mkdir()
    (incomplete / "jd.txt").write_text("Some JD.", encoding="utf-8")
    # resume_variant.txt intentionally absent

    state = AgentState(jd_text="x", resume_text="y")
    result = tool_load_resume_corpus(state, corpus_dir=str(tmp_path))

    assert result["num_examples"] == 0
    assert state.corpus_examples == []


def test_load_resume_corpus_missing_directory_raises() -> None:
    state = AgentState(jd_text="x", resume_text="y")
    with pytest.raises(FileNotFoundError):
        tool_load_resume_corpus(state, corpus_dir="/nonexistent/path/abc123")