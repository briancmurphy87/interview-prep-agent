from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from src.agent_loop import run_agent
from src.agent_state import AgentState
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
    parser.add_argument(
        "--corpus",
        default=None,
        help="Path to resume corpus directory",
    )
    parser.add_argument("--jd", required=True, help="Path to job description text file")
    parser.add_argument("--resume", required=True, help="Path to resume text file")
    parser.add_argument("--out", default="report.md", help="Output markdown path")
    parser.add_argument("--model", default="gpt-4.1-mini", help="LLM model name")
    args = parser.parse_args()

    llm = LLM(model=args.model)

    initial_state = AgentState(
        jd_text=read_text_file(args.jd),
        resume_text=read_text_file(args.resume),
    )
    if args.corpus:
        initial_state.artifacts["corpus_dir"] = args.corpus
        initial_state.artifacts["desired_output"] = "target_resume"
    else:
        initial_state.artifacts["desired_output"] = "report"

    final_state = run_agent(
        llm=llm,
        state=initial_state,
    )

    output_text = final_state.artifacts.get("target_resume_txt")
    if output_text is None:
        output_text = final_state.artifacts.get("report_md", "")
    write_text_file(args.out, output_text)

    print(f"Wrote {args.out} ({len(output_text.encode('utf-8'))} bytes)")
    print(f"Artifacts: {sorted(final_state.artifacts.keys())}")
    print(f"Tool calls: {len(final_state.tool_history)}")


if __name__ == "__main__":
    main()