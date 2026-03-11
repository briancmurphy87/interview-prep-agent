from __future__ import annotations

import re
from typing import Any, Callable

from src.agent_state import AgentState

# ----------------------------
# 2) Tools
# ----------------------------
def tool_extract_keywords(state: AgentState, top_k: int = 20) -> dict[str, Any]:
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
    freq: dict[str, int] = {}
    for w in words:
        if w in stop:
            continue
        freq[w] = freq.get(w, 0) + 1
    top = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return {"keywords": [k for k, _ in top]}


def tool_find_evidence(state: AgentState, query: str, max_snippets: int = 5) -> dict[str, Any]:
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
    hits: list[str] = []
    for ln in lines:
        lnl = ln.lower()
        if any(t in lnl for t in tokens):
            hits.append(ln)
        if len(hits) >= max_snippets:
            break
    return {"snippets": hits}


def tool_write_report(state: AgentState, report_md: str) -> dict[str, Any]:
    state.artifacts["report_md"] = report_md
    return {"ok": True, "bytes": len(report_md.encode("utf-8"))}


TOOLS: dict[str, Callable[..., dict[str, Any]]] = {
    "extract_keywords": tool_extract_keywords,
    "find_evidence": tool_find_evidence,
    "write_report": tool_write_report,
}