from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# TODO: consider using this when you no longer want to use "export" command from terminal
from dotenv import load_dotenv
load_dotenv()

# ----------------------------
# 1) Agent state
# ----------------------------

@dataclass
class AgentState:
    jd_text: str
    resume_text: str
    notes: List[str] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)  # e.g. {"report_md": "..."}
    tool_history: List[Dict[str, Any]] = field(default_factory=list)


# ----------------------------
# 2) Tools
# ----------------------------

def tool_extract_keywords(state: AgentState, top_k: int = 20) -> Dict[str, Any]:
    """
    Very lightweight keyword extraction. Good enough for an MVP.
    """
    text = (state.jd_text + "\n" + state.resume_text).lower()
    words = re.findall(r"[a-z0-9\+\#]{2,}", text)
    stop = {
        "the","and","for","with","you","your","our","are","will","this","that","from","to","in","of","on","a","an",
        "as","by","or","at","be","is","it","we","they","their","them","us","can","may","all","any","more","most",
        "years","year","experience","role","team","work","strong","skills"
    }
    freq: Dict[str, int] = {}
    for w in words:
        if w in stop:
            continue
        freq[w] = freq.get(w, 0) + 1
    top = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return {"keywords": [k for k, _ in top]}


def tool_find_evidence(state: AgentState, query: str, max_snippets: int = 5) -> Dict[str, Any]:
    """
    Finds short snippets in resume that match query terms.
    """
    hay = state.resume_text
    q = query.lower().strip()
    if not q:
        return {"snippets": []}

    # simple match: split query into tokens, find lines containing any token
    tokens = [t for t in re.findall(r"[a-z0-9\+\#]{2,}", q) if len(t) > 1]
    lines = [ln.strip() for ln in hay.splitlines() if ln.strip()]
    hits: List[str] = []
    for ln in lines:
        lnl = ln.lower()
        if any(t in lnl for t in tokens):
            hits.append(ln)
        if len(hits) >= max_snippets:
            break
    return {"snippets": hits}


def tool_write_report(state: AgentState, report_md: str) -> Dict[str, Any]:
    state.artifacts["report_md"] = report_md
    return {"ok": True, "bytes": len(report_md.encode("utf-8"))}


TOOLS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "extract_keywords": tool_extract_keywords,
    "find_evidence": tool_find_evidence,
    "write_report": tool_write_report,
}


# ----------------------------
# 3) LLM interface
# ----------------------------

class LLM:
    """
    Minimal wrapper. Implemented for OpenAI Responses API style.
    If you use another provider, replace only this class.
    """

    def __init__(self, model: str = "gpt-4.1-mini"):
        self.model = model

    def complete(self, system: str, user: str) -> str:
        """
        Returns raw text response.
        Requires: pip install openai
        Env var: OPENAI_API_KEY
        """
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("Missing dependency. Run: pip install openai") from e

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY in your environment.")

        client = OpenAI()
        resp = client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.output_text


# ----------------------------
# 4) Agent loop
# ----------------------------

SYSTEM = """You are an Interview Prep Agent.
You must decide whether to call tools. You have these tools:

1) extract_keywords(top_k:int) -> {keywords:[...]}
2) find_evidence(query:str, max_snippets:int) -> {snippets:[...]}
3) write_report(report_md:str) -> {ok:bool, bytes:int}

Rules:
- When you need structured facts (keywords, evidence), call tools instead of guessing.
- Keep outputs practical, tailored, and concise.
- Produce a final markdown report and call write_report(report_md) before finishing.

Tool calling format:
Return a JSON object with either:
A) {"tool":"TOOL_NAME","args":{...}}
or
B) {"final":"..."} when completely done (only after write_report was called).
"""

def agent_step(llm: LLM, state: AgentState) -> Dict[str, Any]:
    # Build a compact "working context" for the LLM
    notes = "\n".join(f"- {n}" for n in state.notes[-15:])
    tool_hist = json.dumps(state.tool_history[-8:], indent=2)

    user = f"""
JOB DESCRIPTION:
{state.jd_text}

RESUME:
{state.resume_text}

NOTES:
{notes if notes else "(none)"}

RECENT TOOL HISTORY:
{tool_hist if state.tool_history else "(none)"}

Decide next action.
"""
    raw = llm.complete(SYSTEM, user)

    # best-effort parse: model should return JSON
    raw_stripped = raw.strip()
    try:
        payload = json.loads(raw_stripped)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        # fallback: treat as final text
        return {"final": raw}

    return {"final": raw}


def run_agent(llm: LLM, state: AgentState, max_iters: int = 12) -> AgentState:
    for _ in range(max_iters):
        action = agent_step(llm, state)

        if "tool" in action:
            tool_name = action["tool"]
            args = action.get("args", {}) or {}

            if tool_name not in TOOLS:
                state.notes.append(f"Unknown tool requested: {tool_name}")
                state.tool_history.append({"tool": tool_name, "args": args, "error": "unknown_tool"})
                continue

            result = TOOLS[tool_name](state, **args)
            state.tool_history.append({"tool": tool_name, "args": args, "result": result})
            state.notes.append(f"Ran {tool_name} with {args} -> {result}")
            continue

        if "final" in action:
            # allow final only if report exists (agent should have called write_report)
            if "report_md" not in state.artifacts:
                state.notes.append("Model tried to finish without writing report. Forcing report write.")
                TOOLS["write_report"](state, report_md=str(action["final"]))
            return state

    state.notes.append("Max iterations reached. Returning best effort.")
    if "report_md" not in state.artifacts:
        TOOLS["write_report"](state, report_md="# Interview Prep Report\n\n(Max iterations reached.)\n")
    return state


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

    with open(args.jd, "r", encoding="utf-8") as f:
        jd_text = f.read().strip()

    with open(args.resume, "r", encoding="utf-8") as f:
        resume_text = f.read().strip()

    llm = LLM(model=args.model)
    state = AgentState(jd_text=jd_text, resume_text=resume_text)
    state = run_agent(llm, state)

    report = state.artifacts.get("report_md", "")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Wrote {args.out} ({len(report.encode('utf-8'))} bytes)")


if __name__ == "__main__":
    main()