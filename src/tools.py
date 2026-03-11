from __future__ import annotations

# import json
import re
from typing import Any, Callable

from src.agent_state import AgentState
#
# ----------------------------
# 2) Tools
# ----------------------------

ToolResult = dict[str, Any]
ToolFn = Callable[..., ToolResult]


_STOPWORDS = {
    "the", "and", "for", "with", "you", "your", "our", "are", "will", "this", "that",
    "from", "to", "in", "of", "on", "a", "an", "as", "by", "or", "at", "be", "is",
    "it", "we", "they", "their", "them", "us", "can", "may", "all", "any", "more",
    "most", "years", "year", "experience", "role", "team", "work", "strong", "skills",
    "job", "summary", "preferred", "qualifications", "what", "youll", "do",
}

_REQUIREMENT_HINTS = {
    "sales", "engineer", "engineering", "pre-sales", "presales", "customer", "customers",
    "technical", "leadership", "cloud", "backup", "disaster", "recovery", "analytics",
    "cxo", "management", "presentations", "architecture", "architecting", "saas",
    "enterprise", "software", "data", "security", "resilience", "poc", "demo", "demos",
    "stakeholders", "advisor", "trusted", "solutions", "business",
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9\+\#\-]{2,}", text.lower())


def _resume_lines(state: AgentState) -> list[str]:
    return [ln.strip() for ln in state.resume_text.splitlines() if ln.strip()]


def tool_extract_jd_requirements(state: AgentState, top_k: int = 15) -> ToolResult:
    """
    Extract rough requirement keywords from the JD only.
    Lightweight and deterministic by design.
    """
    words = _tokenize(state.jd_text)
    freq: dict[str, int] = {}

    for word in words:
        if word in _STOPWORDS:
            continue
        if word not in _REQUIREMENT_HINTS and len(word) < 4:
            continue
        freq[word] = freq.get(word, 0) + 1

    ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    requirements = [word for word, _ in ranked[:top_k]]

    payload = {"requirements": requirements}
    state.artifacts["requirements_json"] = payload
    return payload


def tool_find_resume_evidence(
    state: AgentState,
    requirement: str,
    max_snippets: int = 5,
) -> ToolResult:
    """
    Find resume lines that appear relevant to a given requirement.
    Deterministic lexical matching for transparency/debuggability.
    """
    req = requirement.lower().strip()
    if not req:
        return {"requirement": requirement, "snippets": []}

    tokens = [t for t in _tokenize(req) if t not in _STOPWORDS]
    if not tokens:
        return {"requirement": requirement, "snippets": []}

    lines = _resume_lines(state)
    scored_hits: list[tuple[int, str]] = []

    for line in lines:
        line_lower = line.lower()
        score = sum(1 for token in tokens if token in line_lower)
        if score > 0:
            scored_hits.append((score, line))

    scored_hits.sort(key=lambda pair: (-pair[0], pair[1]))
    snippets = [line for _, line in scored_hits[:max_snippets]]

    return {
        "requirement": requirement,
        "snippets": snippets,
    }


def tool_score_resume_fit(
    state: AgentState,
    requirements: list[str],
    evidence_per_requirement: int = 3,
) -> ToolResult:
    """
    Build a structured fit view from deterministic evidence lookups.
    """
    matched: list[dict[str, Any]] = []
    gaps: list[str] = []

    for requirement in requirements:
        result = tool_find_resume_evidence(
            state,
            requirement=requirement,
            max_snippets=evidence_per_requirement,
        )
        snippets = result["snippets"]
        if snippets:
            matched.append(
                {
                    "requirement": requirement,
                    "evidence": snippets,
                }
            )
        else:
            gaps.append(requirement)

    score = 0
    if requirements:
        score = round(100 * len(matched) / len(requirements))

    fit = {
        "score": score,
        "matched": matched,
        "gaps": gaps,
    }
    state.artifacts["fit_analysis_json"] = fit
    return fit


def tool_render_report(state: AgentState) -> ToolResult:
    """
    Deterministically render markdown from structured artifacts.
    """
    requirements_payload = state.artifacts.get("requirements_json", {})
    fit_payload = state.artifacts.get("fit_analysis_json", {})

    requirements = requirements_payload.get("requirements", [])
    score = fit_payload.get("score", 0)
    matched = fit_payload.get("matched", [])
    gaps = fit_payload.get("gaps", [])

    lines: list[str] = []
    lines.append("# Interview Prep Report")
    lines.append("")
    lines.append("## Overall Fit")
    lines.append("")
    lines.append(f"- Estimated fit score: **{score}/100**")
    lines.append(f"- Requirements analyzed: **{len(requirements)}**")
    lines.append(f"- Requirements with evidence: **{len(matched)}**")
    lines.append(f"- Gaps / weak evidence areas: **{len(gaps)}**")
    lines.append("")

    lines.append("## Key Requirements")
    lines.append("")
    if requirements:
        for req in requirements:
            lines.append(f"- {req}")
    else:
        lines.append("- No requirements extracted.")
    lines.append("")

    lines.append("## Resume Evidence")
    lines.append("")
    if matched:
        for item in matched:
            lines.append(f"### {item['requirement']}")
            lines.append("")
            for snippet in item["evidence"]:
                lines.append(f"- {snippet}")
            lines.append("")
    else:
        lines.append("- No matching evidence found.")
        lines.append("")

    lines.append("## Gaps / Weak Areas")
    lines.append("")
    if gaps:
        for gap in gaps:
            lines.append(f"- {gap}")
    else:
        lines.append("- No obvious gaps identified from deterministic matching.")
    lines.append("")

    lines.append("## Suggested Talking Points")
    lines.append("")
    lines.append("- Emphasize systems design, technical breadth, and customer-facing translation.")
    lines.append("- Tie low-latency and reliability experience to enterprise resilience and data protection.")
    lines.append("- Highlight communication with stakeholders, clients, executives, and technical partners.")
    lines.append("")

    report_md = "\n".join(lines).strip() + "\n"
    state.artifacts["report_md"] = report_md

    return {
        "ok": True,
        "bytes": len(report_md.encode("utf-8")),
    }


def tool_dump_state_summary(state: AgentState) -> ToolResult:
    """
    Small debug tool. Helpful when reasoning about state transitions.
    """
    return {
        "num_notes": len(state.notes),
        "artifact_keys": sorted(state.artifacts.keys()),
        "num_tool_calls": len(state.tool_history),
    }


TOOLS: dict[str, ToolFn] = {
    "extract_jd_requirements": tool_extract_jd_requirements,
    "find_resume_evidence": tool_find_resume_evidence,
    "score_resume_fit": tool_score_resume_fit,
    "render_report": tool_render_report,
    "dump_state_summary": tool_dump_state_summary,
}