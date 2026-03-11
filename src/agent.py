from __future__ import annotations

import argparse

# TODO: consider using this when you no longer want to use "export" command from terminal
from dotenv import load_dotenv

from src.agent_loop import run_agent

from src.agent_state import AgentState
from src.llm import LLM

load_dotenv()

def init_input_job_description(jd_file_path_txt: str) -> str:
    with open(jd_file_path_txt, "r", encoding="utf-8") as f:
        return f.read().strip()

def init_input_resume(resume_file_path_txt: str) -> str:
    with open(resume_file_path_txt, "r", encoding="utf-8") as f:
        return f.read().strip()

# ----------------------------
# 5) CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jd", required=True, help="Path to job description text file")
    ap.add_argument("--resume", required=True, help="Path to resume text file")
    ap.add_argument("--out", default="report.md", help="Output markdown path")
    ap.add_argument("--model", default="gpt-4.1-mini", help="LLM model name")
    args = ap.parse_args()

    # init model
    llm = LLM(model=args.model)

    # init agent state and its input args
    state = run_agent(
        llm=llm,
        state=AgentState(
            jd_text=init_input_job_description(args.jd),
            resume_text=init_input_resume(args.resume),
        ),
    )

    # output report
    report = state.artifacts.get("report_md", "")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Wrote {args.out} ({len(report.encode('utf-8'))} bytes)")


if __name__ == "__main__":
    main()