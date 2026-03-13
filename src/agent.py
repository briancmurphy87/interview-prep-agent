from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from src.agent_loop import run_agent
from src.agent_state import AgentState
from src.llm import LLM
from src.tools import tool_score_resume_fit, tool_render_report


load_dotenv()


def read_text_file(path: str) -> str:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    return file_path.read_text(encoding="utf-8").strip()


def write_text_file(path: str, content: str) -> None:
    Path(path).write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jd", required=True, help="Path to target job description text file")
    parser.add_argument("--resume", required=True, help="Path to raw/base resume text file")
    parser.add_argument("--corpus", required=True, help="Path to resume corpus directory")
    parser.add_argument("--out-resume", required=True, help="Output path for targeted resume")
    parser.add_argument("--out-report", required=True, help="Output path for companion report")
    parser.add_argument("--model", default="gpt-4.1-mini", help="LLM model name")
    args = parser.parse_args()

    llm = LLM(model=args.model)

    initial_state = AgentState(
        jd_text=read_text_file(args.jd),
        resume_text=read_text_file(args.resume),
    )
    initial_state.artifacts["corpus_dir"] = args.corpus

    final_state = run_agent(
        llm=llm,
        state=initial_state,
    )

    target_resume = final_state.artifacts.get("target_resume_txt", "")
    write_text_file(args.out_resume, target_resume)

    requirements_payload = final_state.artifacts.get("requirements_json", {})
    requirements = requirements_payload.get("requirements", [])

    if "fit_analysis_json" not in final_state.artifacts and requirements:
        tool_score_resume_fit(
            final_state,
            requirements=requirements,
            evidence_per_requirement=3,
        )

    if "report_md" not in final_state.artifacts:
        tool_render_report(final_state)

    report_md = final_state.artifacts.get("report_md", "")
    write_text_file(args.out_report, report_md)

    print(f"Wrote {args.out_resume} ({len(target_resume.encode('utf-8'))} bytes)")
    print(f"Wrote {args.out_report} ({len(report_md.encode('utf-8'))} bytes)")
    print(f"Artifacts: {sorted(final_state.artifacts.keys())}")
    print(f"Tool calls: {len(final_state.tool_history)}")


if __name__ == "__main__":
    main()