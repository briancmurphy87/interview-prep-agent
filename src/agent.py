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

    final_state = run_agent(
        llm=llm,
        state=initial_state,
    )

    report = final_state.artifacts.get("report_md", "")
    write_text_file(args.out, report)

    print(f"Wrote {args.out} ({len(report.encode('utf-8'))} bytes)")
    print(f"Artifacts: {sorted(final_state.artifacts.keys())}")
    print(f"Tool calls: {len(final_state.tool_history)}")


if __name__ == "__main__":
    main()