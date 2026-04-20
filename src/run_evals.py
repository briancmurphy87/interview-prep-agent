"""
Batch evaluation runner for the resume tailoring agent.

Usage:
    python -m src.run_evals \\
        --eval-dir evals/cases \\
        --out-dir  evals/outputs \\
        --corpus   resume_corpus \\
        --model    gpt-4.1-mini

Directory layout expected under --eval-dir:

    evals/cases/
        <slug>/
            jd.txt                     # target job description
            raw_resume.txt             # base resume (unmodified)
            expected_requirements.json # optional; used for validation

For each valid case the runner will:
  1. Run the full agent loop (load corpus → extract reqs → retrieve → generate resume)
  2. Apply the same post-loop safety net as main(): score fit, evaluate with LLM, render report
  3. Save all artifacts to --out-dir/<slug>/
  4. If expected_requirements.json is present, validate that at least
     `min_keyword_matches` of the expected_keywords appear in extracted requirements.

After all cases, a summary JSON is written to evals/summaries/summary_<timestamp>.json
containing per-case scores and the following aggregate metrics:
  - average_overall_score        (from LLM evaluator)
  - average_jd_alignment         (from LLM evaluator)
  - average_keyword_coverage     (from LLM evaluator)
  - average_exaggeration_safety  (from LLM evaluator; higher = safer)
  - average_ats_compatibility    (from LLM evaluator)
  - average_fit_score            (from deterministic scorer)
  - retrieval_hit_rate           (fraction of cases with ≥1 corpus match)
  - num_failed                   (cases that raised an uncaught exception)
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.agent_loop import run_agent
from src.agent_state import AgentState
from src.llm import LLM
from src.observability import build_run_artifact
from src.tools import (
    tool_evaluate_target_resume,
    tool_render_report,
    tool_revise_target_resume,
    tool_score_resume_fit,
)

load_dotenv()


# ---------------------------------------------------------------------------
# Case discovery
# ---------------------------------------------------------------------------


def _discover_cases(eval_dir: Path) -> list[Path]:
    """
    Return subdirectories of eval_dir that contain both jd.txt and
    raw_resume.txt, sorted alphabetically for reproducible ordering.
    """
    cases = []
    for child in sorted(eval_dir.iterdir()):
        if (
            child.is_dir()
            and (child / "jd.txt").exists()
            and (child / "raw_resume.txt").exists()
        ):
            cases.append(child)
    return cases


# ---------------------------------------------------------------------------
# Per-case runner
# ---------------------------------------------------------------------------


def _run_case(case_dir: Path, corpus_dir: str, llm: LLM, run_id: str) -> dict[str, Any]:
    """
    Execute the full pipeline for a single eval case and return a result dict.

    The result dict contains:
      slug           — directory name used as the case identifier
      status         — "success" or "failed"
      elapsed_seconds
      scores         — flat dict of numeric scores (None if unavailable)
      retrieval_hit  — True if corpus retrieval returned ≥1 match
      tool_call_count
      requirement_validation — dict with pass/fail if expected_requirements.json exists
      error          — exception message on failure, else None
      state          — the AgentState object (used by _save_case_outputs, not serialised)
    """
    slug = case_dir.name
    jd_text = (case_dir / "jd.txt").read_text(encoding="utf-8").strip()
    resume_text = (case_dir / "raw_resume.txt").read_text(encoding="utf-8").strip()

    state = AgentState(jd_text=jd_text, resume_text=resume_text)
    state.artifacts["corpus_dir"] = corpus_dir

    start = time.monotonic()

    try:
        state = run_agent(llm=llm, state=state)

        # Post-loop pipeline — mirrors the logic in agent.py:main().
        # All four stages run unconditionally so every case produces complete
        # artifacts for comparison in the summary.
        requirements = state.artifacts.get("requirements_json", {}).get("requirements", [])

        if "fit_analysis_json" not in state.artifacts and requirements:
            tool_score_resume_fit(
                state,
                requirements=requirements,
                evidence_per_requirement=3,
            )

        if "resume_evaluation_json" not in state.artifacts:
            tool_evaluate_target_resume(state=state, llm=llm)

        # Revision pass: revises the draft if the initial score is below
        # REVISION_THRESHOLD and re-evaluates the revised version.
        if "revision_metadata_json" not in state.artifacts:
            tool_revise_target_resume(state=state, llm=llm)

        if "report_md" not in state.artifacts:
            tool_render_report(state)

        status = "success"
        error = None

    except Exception as exc:
        status = "failed"
        error = str(exc)

    elapsed = round(time.monotonic() - start, 2)

    # Extract scores into a flat dict for easy aggregation.
    # When revision was triggered, final_overall uses the revised score so
    # the summary reflects the best outcome for each case.
    evaluation = state.artifacts.get("resume_evaluation_json", {})
    revision_eval = state.artifacts.get("revision_evaluation_json", {})
    revision_meta = state.artifacts.get("revision_metadata_json", {})

    initial_overall: int | None = evaluation.get("overall_score")
    revised_overall: int | None = revision_eval.get("overall_score") if revision_meta.get("triggered") else None
    final_overall = revised_overall if revised_overall is not None else initial_overall

    scores: dict[str, int | None] = {
        "overall": final_overall,
        "initial_overall": initial_overall,
        "revised_overall": revised_overall,
        "jd_alignment": (evaluation.get("jd_alignment") or {}).get("score"),
        "keyword_coverage": (evaluation.get("keyword_coverage") or {}).get("score"),
        "exaggeration_safety": (evaluation.get("exaggeration_risk") or {}).get("score"),
        "ats_compatibility": (evaluation.get("ats_compatibility") or {}).get("score"),
        "fit_score": (state.artifacts.get("fit_analysis_json") or {}).get("score"),
    }

    # Retrieval hit: did the agent find at least one relevant corpus example?
    matches = (state.artifacts.get("retrieved_examples_json") or {}).get("matches", [])
    retrieval_hit = len(matches) > 0

    # Optional: validate extracted requirements against expected_requirements.json
    requirement_validation = _validate_requirements(case_dir, state)

    run_artifact = build_run_artifact(run_id=run_id, llm=llm, state=state)

    return {
        "slug": slug,
        "status": status,
        "elapsed_seconds": elapsed,
        "scores": scores,
        "retrieval_hit": retrieval_hit,
        "tool_call_count": len(state.tool_history),
        "requirement_validation": requirement_validation,
        "error": error,
        "state": state,
        "run_artifact": run_artifact,
    }


def _validate_requirements(
    case_dir: Path, state: AgentState
) -> dict[str, Any] | None:
    """
    If expected_requirements.json exists in the case directory, check that
    at least `min_keyword_matches` of the expected_keywords appear as
    substrings of the extracted requirements.

    Returns None if no expected file is present.
    Returns a dict with keys: passed, matched_count, min_required, matched_keywords.
    """
    expected_path = case_dir / "expected_requirements.json"
    if not expected_path.exists():
        return None

    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    expected_keywords: list[str] = expected.get("expected_keywords", [])
    min_required: int = expected.get("min_keyword_matches", 3)

    extracted = [
        r.lower()
        for r in state.artifacts.get("requirements_json", {}).get("requirements", [])
    ]

    matched = [kw for kw in expected_keywords if any(kw in req for req in extracted)]

    return {
        "passed": len(matched) >= min_required,
        "matched_count": len(matched),
        "min_required": min_required,
        "matched_keywords": matched,
    }


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------


def _save_case_outputs(result: dict[str, Any], out_dir: Path) -> None:
    """
    Write all available artifacts for a case to out_dir/<slug>/.

    Files written (when the artifact is present):
      target_resume.txt   — tailored resume text
      report.md           — full markdown report
      requirements.json   — extracted JD requirements
      retrieved_examples.json — corpus retrieval results
      evaluation.json     — LLM evaluator scores and feedback
      run_metadata.json   — timing, tool counts, scores, validation, notes
    """
    state: AgentState = result["state"]
    case_out = out_dir / result["slug"]
    case_out.mkdir(parents=True, exist_ok=True)

    def _write(name: str, text: str) -> None:
        (case_out / name).write_text(text, encoding="utf-8")

    def _write_json(name: str, data: Any) -> None:
        _write(name, json.dumps(data, indent=2))

    artifact_map = {
        "target_resume_txt": "target_resume.txt",
        "revised_resume_txt": "revised_resume.txt",  # present only when revision triggered
        "report_md": "report.md",
    }
    json_artifact_map = {
        "requirements_json": "requirements.json",
        "retrieved_examples_json": "retrieved_examples.json",
        "resume_evaluation_json": "evaluation.json",
        "revision_evaluation_json": "revision_evaluation.json",  # present only when revision triggered
        "revision_metadata_json": "revision_metadata.json",
    }

    for key, filename in artifact_map.items():
        if key in state.artifacts:
            _write(filename, state.artifacts[key])

    for key, filename in json_artifact_map.items():
        if key in state.artifacts:
            _write_json(filename, state.artifacts[key])

    # Observability: full timing + cost breakdown for this run
    if "run_artifact" in result:
        _write_json("run_artifact.json", result["run_artifact"])

    # Always write run metadata so failed cases are still observable
    metadata = {
        "slug": result["slug"],
        "status": result["status"],
        "elapsed_seconds": result["elapsed_seconds"],
        "tool_call_count": result["tool_call_count"],
        "retrieval_hit": result["retrieval_hit"],
        "scores": result["scores"],
        "requirement_validation": result["requirement_validation"],
        "error": result["error"],
        "notes": state.notes,
    }
    _write_json("run_metadata.json", metadata)


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------


def _avg(results: list[dict[str, Any]], key: str) -> float | None:
    """Average a score key across successful runs that have a non-None value."""
    vals = [
        r["scores"][key]
        for r in results
        if r["status"] == "success" and r["scores"].get(key) is not None
    ]
    return round(sum(vals) / len(vals), 1) if vals else None


def _compute_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate metrics across all eval cases.

    Averages are computed only over successful runs that produced the relevant
    artifact. A case is a retrieval hit if the corpus returned ≥1 match.
    """
    successful = [r for r in results if r["status"] == "success"]
    retrieval_hits = sum(1 for r in results if r.get("retrieval_hit"))
    validation_results = [
        r["requirement_validation"]
        for r in results
        if r["requirement_validation"] is not None
    ]
    validation_pass_count = sum(1 for v in validation_results if v["passed"])

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_cases": len(results),
        "num_successful": len(successful),
        "num_failed": len(results) - len(successful),
        "retrieval_hit_rate": (
            round(retrieval_hits / len(results), 2) if results else 0.0
        ),
        "requirement_validation": {
            "cases_with_expected": len(validation_results),
            "passed": validation_pass_count,
            "failed": len(validation_results) - validation_pass_count,
        },
        "averages": {
            "overall_score": _avg(successful, "overall"),
            "jd_alignment": _avg(successful, "jd_alignment"),
            "keyword_coverage": _avg(successful, "keyword_coverage"),
            "exaggeration_safety": _avg(successful, "exaggeration_safety"),
            "ats_compatibility": _avg(successful, "ats_compatibility"),
            "fit_score": _avg(successful, "fit_score"),
        },
        # Per-case snapshot for inspection (full artifacts live in --out-dir)
        "cases": [
            {
                "slug": r["slug"],
                "status": r["status"],
                "elapsed_seconds": r["elapsed_seconds"],
                "retrieval_hit": r["retrieval_hit"],
                "scores": r["scores"],
                "requirement_validation": r["requirement_validation"],
                "error": r["error"],
            }
            for r in results
        ],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch evaluation runner for the resume tailoring agent."
    )
    parser.add_argument(
        "--eval-dir",
        required=True,
        help="Directory containing eval case subdirectories (each with jd.txt + raw_resume.txt)",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory to write per-case output artifacts",
    )
    parser.add_argument(
        "--corpus",
        required=True,
        help="Path to resume corpus directory (shared across all eval cases)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="LLM model name (default: gpt-4.1-mini)",
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir)
    summaries_dir = Path("evals/summaries")

    if not eval_dir.exists():
        raise FileNotFoundError(f"Eval directory does not exist: {eval_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    llm = LLM(model=args.model)
    cases = _discover_cases(eval_dir)

    if not cases:
        print(f"No valid eval cases found in {eval_dir} (need jd.txt + raw_resume.txt)")
        return

    slugs = [c.name for c in cases]
    print(f"Found {len(cases)} eval case(s): {slugs}\n")

    results: list[dict[str, Any]] = []

    for case_dir in cases:
        slug = case_dir.name
        print(f"[{slug}] Running...")

        run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{slug}"
        result = _run_case(case_dir=case_dir, corpus_dir=args.corpus, llm=llm, run_id=run_id)
        _save_case_outputs(result, out_dir)

        if result["status"] == "success":
            s = result["scores"]
            hit = "HIT" if result["retrieval_hit"] else "MISS"
            val = result["requirement_validation"]
            val_str = ""
            if val is not None:
                val_str = f" | reqs={'PASS' if val['passed'] else 'FAIL'} ({val['matched_count']}/{val['min_required']})"
            # Show revision delta when it was triggered
            rev_str = ""
            if s.get("revised_overall") is not None:
                delta = (s["revised_overall"] or 0) - (s["initial_overall"] or 0)
                rev_str = f" | revised={s['revised_overall']} ({delta:+d})"
            print(
                f"[{slug}] OK ({result['elapsed_seconds']}s)"
                f" | overall={s['overall']}"
                f" | jd_align={s['jd_alignment']}"
                f" | kw_cov={s['keyword_coverage']}"
                f" | retrieval={hit}"
                f"{rev_str}"
                f"{val_str}"
            )
        else:
            print(f"[{slug}] FAILED ({result['elapsed_seconds']}s): {result['error']}")

        results.append(result)

    summary = _compute_summary(results)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_path = summaries_dir / f"summary_{timestamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    avgs = summary["averages"]
    print(f"\n{'='*60}")
    print(f"Cases: {summary['total_cases']}  |  Success: {summary['num_successful']}  |  Failed: {summary['num_failed']}")
    print(f"Retrieval hit rate : {summary['retrieval_hit_rate']:.0%}")
    print(f"Avg overall score  : {avgs['overall_score']}")
    print(f"Avg JD alignment   : {avgs['jd_alignment']}")
    print(f"Avg keyword cov.   : {avgs['keyword_coverage']}")
    print(f"Avg exag. safety   : {avgs['exaggeration_safety']}")
    print(f"Avg fit score      : {avgs['fit_score']}")
    print(f"Summary            : {summary_path}")


if __name__ == "__main__":
    main()
