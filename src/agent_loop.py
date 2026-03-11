from __future__ import annotations

import json
from typing import Any

from src.tools import TOOLS
from src.llm import LLM
from src.agent_state import AgentState

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


def agent_step(llm: LLM, state: AgentState) -> dict[str, Any]:
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
