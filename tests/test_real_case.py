from pathlib import Path
import json

from src.agent_state import AgentState
from src.tools import (
    tool_extract_jd_requirements,
    tool_score_resume_fit,
    tool_render_report,
)


FIXTURE_DIR = Path("tests/fixtures/netflix_swe5")


def load_fixture():
    jd = (FIXTURE_DIR / "jd.txt").read_text(encoding="utf-8")
    resume = (FIXTURE_DIR / "resume.txt").read_text(encoding="utf-8")
    expected = json.loads(
        (FIXTURE_DIR / "expected_sections.json").read_text(encoding="utf-8")
    )
    return jd, resume, expected


def test_real_netflix_swe5_pipeline():
    jd, resume, expected = load_fixture()

    state = AgentState(
        jd_text=jd,
        resume_text=resume,
    )

    reqs = tool_extract_jd_requirements(state, top_k=15)

    assert "requirements" in reqs
    assert isinstance(reqs["requirements"], list)
    assert len(reqs["requirements"]) > 0

    lowered_requirements = [r.lower() for r in reqs["requirements"]]

    matches = 0
    for keyword in expected["expected_keywords"]:
        if any(keyword in r for r in lowered_requirements):
            matches += 1

    assert matches >= 3, (
        f"Expected at least 3 keyword matches, but found {matches}. "
        f"Extracted requirements: {reqs['requirements']}"
    )

    fit = tool_score_resume_fit(
        state,
        requirements=reqs["requirements"],
        evidence_per_requirement=3,
    )

    assert "score" in fit
    assert "matched" in fit
    assert "gaps" in fit
    assert isinstance(fit["score"], int)
    assert 0 <= fit["score"] <= 100

    render_result = tool_render_report(state)
    assert render_result["ok"] is True

    markdown = state.artifacts["report_md"]

    for section in expected["required_markdown_sections"]:
        assert section in markdown

    assert len(markdown) > 500