from __future__ import annotations

import pytest

from src.agent_state import AgentState
from src.tools import (
    REVISION_THRESHOLD,
    tool_extract_jd_requirements,
    tool_find_resume_evidence,
    tool_load_resume_corpus,
    tool_render_report,
    tool_revise_target_resume,
    tool_score_resume_fit,
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


def test_score_resume_fit_reads_requirements_from_artifacts_when_none_given() -> None:
    # After extract_jd_requirements runs, score_resume_fit() can be called with
    # no args; it should fall back to requirements_json in state artifacts.
    state = make_state()
    tool_extract_jd_requirements(state, top_k=5)
    assert "requirements_json" in state.artifacts

    result = tool_score_resume_fit(state)   # no requirements kwarg

    assert "score" in result
    assert "fit_analysis_json" in state.artifacts
    # requirements used should match what was extracted
    extracted = state.artifacts["requirements_json"]["requirements"]
    all_reqs = [m["requirement"] for m in result["matched"]] + result["gaps"]
    assert sorted(all_reqs) == sorted(extracted)


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

    assert result["num_examples_loaded"] == 1
    assert result["num_examples_skipped"] == 0
    assert "acme_swe" in result["slugs"]
    assert result["skipped"] == []
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

    assert result["num_examples_loaded"] == 0
    assert result["num_examples_skipped"] == 1
    assert state.corpus_examples == []


def test_load_resume_corpus_reports_skip_reason(tmp_path) -> None:
    # Directory missing resume_variant.txt should appear in skipped list with reason.
    no_variant = tmp_path / "no_variant"
    no_variant.mkdir()
    (no_variant / "jd.txt").write_text("Some JD.", encoding="utf-8")

    # Directory missing jd.* should also be skipped with a distinct reason.
    no_jd = tmp_path / "no_jd"
    no_jd.mkdir()
    (no_jd / "resume_variant.txt").write_text("Some resume.", encoding="utf-8")

    # A fully empty directory should be skipped too.
    empty = tmp_path / "empty_dir"
    empty.mkdir()

    state = AgentState(jd_text="x", resume_text="y")
    result = tool_load_resume_corpus(state, corpus_dir=str(tmp_path))

    assert result["num_examples_loaded"] == 0
    assert result["num_examples_skipped"] == 3
    skipped_slugs = {s["slug"] for s in result["skipped"]}
    assert "no_variant" in skipped_slugs
    assert "no_jd" in skipped_slugs
    assert "empty_dir" in skipped_slugs

    # Each skipped entry should have a non-empty reason string.
    for entry in result["skipped"]:
        assert isinstance(entry["reason"], str)
        assert len(entry["reason"]) > 0


def test_load_resume_corpus_missing_directory_raises() -> None:
    state = AgentState(jd_text="x", resume_text="y")
    with pytest.raises(FileNotFoundError):
        tool_load_resume_corpus(state, corpus_dir="/nonexistent/path/abc123")


# ---------------------------------------------------------------------------
# tool_revise_target_resume
# ---------------------------------------------------------------------------


class _FakeLLM:
    """Minimal fake LLM that returns scripted responses in sequence."""

    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)
        self.call_log: list[dict] = []

    def complete(self, system: str, user: str) -> str:
        if not self.outputs:
            raise RuntimeError("_FakeLLM ran out of scripted outputs")
        out = self.outputs.pop(0)
        self.call_log.append({"ok": True, "output": out})
        return out


_GOOD_EVAL_JSON = """{
  "overall_score": 82,
  "jd_alignment": {"score": 85, "reason": "Strong overlap."},
  "keyword_coverage": {"score": 78, "reason": "Most keywords present."},
  "clarity": {"score": 90, "reason": "Concise."},
  "exaggeration_risk": {"score": 80, "reason": "Claims defensible."},
  "ats_compatibility": {"score": 88, "reason": "Standard format."},
  "red_flags": [],
  "suggested_improvements": ["Quantify impact metrics"]
}"""

_LOW_EVAL_JSON = """{
  "overall_score": 55,
  "jd_alignment": {"score": 50, "reason": "Weak alignment."},
  "keyword_coverage": {"score": 60, "reason": "Missing keywords."},
  "clarity": {"score": 70, "reason": "Acceptable."},
  "exaggeration_risk": {"score": 65, "reason": "Some risk."},
  "ats_compatibility": {"score": 72, "reason": "Mostly ok."},
  "red_flags": ["Unsupported claim about ML experience"],
  "suggested_improvements": ["Add cloud keywords", "Quantify metrics"]
}"""

_REVISED_EVAL_JSON = """{
  "overall_score": 74,
  "jd_alignment": {"score": 72, "reason": "Improved."},
  "keyword_coverage": {"score": 75, "reason": "Better coverage."},
  "clarity": {"score": 78, "reason": "Cleaner."},
  "exaggeration_risk": {"score": 80, "reason": "Better grounded."},
  "ats_compatibility": {"score": 78, "reason": "Improved."},
  "red_flags": [],
  "suggested_improvements": []
}"""


def test_revise_target_resume_skipped_when_score_above_threshold() -> None:
    # When initial score >= REVISION_THRESHOLD, revision should not trigger.
    state = make_state()
    state.artifacts["target_resume_txt"] = "Initial resume text."
    state.artifacts["resume_evaluation_json"] = {
        "overall_score": REVISION_THRESHOLD,  # exactly at threshold → no revision
        "red_flags": [],
        "suggested_improvements": [],
    }

    llm = _FakeLLM([])  # no LLM calls should happen
    result = tool_revise_target_resume(state, llm=llm)

    assert result["triggered"] is False
    assert result["initial_score"] == REVISION_THRESHOLD
    assert "revised_resume_txt" not in state.artifacts
    assert "revision_evaluation_json" not in state.artifacts
    assert "revision_metadata_json" in state.artifacts
    assert len(llm.call_log) == 0  # LLM was never called


def test_revise_target_resume_triggered_when_score_below_threshold() -> None:
    # When initial score < REVISION_THRESHOLD, revision should be triggered.
    # The fake LLM returns: revised resume text, then revised evaluation JSON.
    state = make_state()
    state.artifacts["target_resume_txt"] = "Initial resume text."
    state.artifacts["resume_evaluation_json"] = {
        "overall_score": REVISION_THRESHOLD - 1,
        "jd_alignment": {"score": 50, "reason": "Weak."},
        "keyword_coverage": {"score": 60, "reason": "Missing keywords."},
        "red_flags": ["Unsupported claim"],
        "suggested_improvements": ["Add cloud keywords"],
    }

    llm = _FakeLLM(["Revised resume text.", _REVISED_EVAL_JSON])
    result = tool_revise_target_resume(state, llm=llm)

    assert result["triggered"] is True
    assert result["initial_score"] == REVISION_THRESHOLD - 1
    assert result["revised_score"] == 74  # from _REVISED_EVAL_JSON
    assert result["delta"] == 74 - (REVISION_THRESHOLD - 1)
    assert "revised_resume_txt" in state.artifacts
    assert state.artifacts["revised_resume_txt"] == "Revised resume text."
    assert "revision_evaluation_json" in state.artifacts
    assert state.artifacts["revision_evaluation_json"]["overall_score"] == 74
    assert len(llm.call_log) == 2  # revision prompt + re-evaluation prompt


def test_revise_target_resume_skipped_when_no_evaluation() -> None:
    # If resume_evaluation_json was never written, revision should be skipped gracefully.
    state = make_state()
    state.artifacts["target_resume_txt"] = "Some resume."

    llm = _FakeLLM([])
    result = tool_revise_target_resume(state, llm=llm)

    assert result["triggered"] is False
    assert result["initial_score"] is None
    assert "revised_resume_txt" not in state.artifacts
    assert len(llm.call_log) == 0


def test_revise_target_resume_preserves_original_evaluation() -> None:
    # The original resume_evaluation_json should not be overwritten by revision.
    state = make_state()
    state.artifacts["target_resume_txt"] = "Initial resume text."
    original_eval = {
        "overall_score": REVISION_THRESHOLD - 15,
        "jd_alignment": {"score": 45, "reason": "Poor."},
        "keyword_coverage": {"score": 55, "reason": "Missing many."},
        "red_flags": ["Fabricated metric"],
        "suggested_improvements": ["Ground claims in evidence"],
    }
    state.artifacts["resume_evaluation_json"] = original_eval

    llm = _FakeLLM(["Revised text.", _REVISED_EVAL_JSON])
    tool_revise_target_resume(state, llm=llm)

    # Original eval must be preserved; revision goes to a separate key.
    assert state.artifacts["resume_evaluation_json"] is original_eval
    assert "revision_evaluation_json" in state.artifacts


# ---------------------------------------------------------------------------
# tool_render_report — revision section
# ---------------------------------------------------------------------------


def test_render_report_includes_revision_section_when_triggered() -> None:
    state = make_state()
    tool_extract_jd_requirements(state, top_k=5)
    tool_score_resume_fit(
        state,
        requirements=state.artifacts["requirements_json"]["requirements"],
        evidence_per_requirement=2,
    )
    state.artifacts["resume_evaluation_json"] = {
        "overall_score": 55,
        "jd_alignment": {"score": 50, "reason": "Weak."},
        "keyword_coverage": {"score": 60, "reason": "Missing."},
        "clarity": {"score": 70, "reason": "Ok."},
        "exaggeration_risk": {"score": 65, "reason": "Some risk."},
        "ats_compatibility": {"score": 72, "reason": "Mostly ok."},
        "red_flags": [],
        "suggested_improvements": [],
    }
    state.artifacts["revision_metadata_json"] = {
        "triggered": True,
        "initial_score": 55,
        "revised_score": 74,
        "delta": 19,
        "threshold": REVISION_THRESHOLD,
        "reason": "score 55 < threshold 70; revision attempted",
    }
    state.artifacts["revision_evaluation_json"] = {
        "overall_score": 74,
        "jd_alignment": {"score": 72, "reason": "Improved."},
        "keyword_coverage": {"score": 75, "reason": "Better."},
    }

    tool_render_report(state)
    report = state.artifacts["report_md"]

    assert "## Revision Pass" in report
    assert "Revision triggered: **Yes**" in report
    assert "Initial score: **55/100**" in report
    assert "Revised score: **74/100**" in report
    assert "+19" in report


def test_render_report_revision_section_shows_not_triggered() -> None:
    state = make_state()
    tool_extract_jd_requirements(state, top_k=5)
    tool_score_resume_fit(
        state,
        requirements=state.artifacts["requirements_json"]["requirements"],
        evidence_per_requirement=2,
    )
    state.artifacts["resume_evaluation_json"] = {
        "overall_score": 85,
        "jd_alignment": {"score": 85, "reason": "Good."},
        "keyword_coverage": {"score": 80, "reason": "Good."},
        "clarity": {"score": 90, "reason": "Clear."},
        "exaggeration_risk": {"score": 88, "reason": "Safe."},
        "ats_compatibility": {"score": 85, "reason": "Standard."},
        "red_flags": [],
        "suggested_improvements": [],
    }
    state.artifacts["revision_metadata_json"] = {
        "triggered": False,
        "initial_score": 85,
        "threshold": REVISION_THRESHOLD,
        "reason": "score 85 >= threshold 70; revision not needed",
    }

    tool_render_report(state)
    report = state.artifacts["report_md"]

    assert "## Revision Pass" in report
    assert "Revision triggered: **No**" in report