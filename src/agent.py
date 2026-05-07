from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from src.agent_state import AgentState
from src.agents import AnalystAgent, CriticAgent, WriterAgent
from src.llm import LLM

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

    state = AgentState(
        jd_text=read_text_file(args.jd),
        resume_text=read_text_file(args.resume),
    )
    state.artifacts["corpus_dir"] = args.corpus

    # --- Three-agent pipeline ---
    state = AnalystAgent().run(state, llm)   # corpus → requirements → retrieval → fit
    state = WriterAgent().run(state, llm)    # evidence grounding → draft resume
    state = CriticAgent().run(state, llm)    # evaluate → revise if needed → report

    # Write the best available resume: revised draft if revision was triggered, else initial.
    revision_triggered = state.artifacts.get("revision_metadata_json", {}).get("triggered", False)
    if revision_triggered and "revised_resume_txt" in state.artifacts:
        best_resume = state.artifacts["revised_resume_txt"]
        resume_note = "(revised draft)"
    else:
        best_resume = state.artifacts.get("target_resume_txt", "")
        resume_note = "(initial draft)"

    report_md = state.artifacts.get("report_md", "")

    write_text_file(args.out_resume, best_resume)
    write_text_file(args.out_report, report_md)

    print(f"Wrote {args.out_resume} ({len(best_resume.encode('utf-8'))} bytes) {resume_note}")
    print(f"Wrote {args.out_report} ({len(report_md.encode('utf-8'))} bytes)")

    revision_meta = state.artifacts.get("revision_metadata_json", {})
    if revision_meta.get("triggered"):
        print(
            f"Revision: initial={revision_meta.get('initial_score')} → "
            f"revised={revision_meta.get('revised_score')} "
            f"(delta={revision_meta.get('delta'):+d})"
        )

    print(f"Artifacts: {sorted(state.artifacts.keys())}")
    print(f"Tool calls: {len(state.tool_history)}")


if __name__ == "__main__":
    main()
