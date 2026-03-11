#
# ----------------------------
# 2) Tools
# ----------------------------
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from src.agent_state import AgentState, CorpusExample


ToolResult = dict[str, Any]
ToolFn = Callable[..., ToolResult]


_STOPWORDS = {
    "the", "and", "for", "with", "you", "your", "our", "are", "will", "this", "that",
    "from", "to", "in", "of", "on", "a", "an", "as", "by", "or", "at", "be", "is",
    "it", "we", "they", "their", "them", "us", "can", "may", "all", "any", "more",
    "most", "years", "year", "experience", "role", "team", "work", "strong", "skills",
    "job", "summary", "preferred", "qualifications", "what", "youll", "do", "using",
    "used", "build", "building", "develop", "development", "engineer", "engineering",
}

# Single-token requirement hints
_REQUIREMENT_HINTS = {
    "python", "java", "api", "apis", "backend", "cloud", "aws", "gcp",
    "services", "service", "applications", "application", "enterprise",
    "automation", "workflow", "workflows", "integration", "integrations",
    "data", "security", "distributed", "systems", "scalable", "scalability",
    "microservices", "platform", "platforms", "genai", "llm", "llms",
    "ai", "ml", "infrastructure", "reliability", "testing", "observability",
    "stakeholders", "partners", "cross-functional", "customer", "customers",
    "architecture", "architect", "pipelines", "internal", "tools", "tooling",
}

# Multi-word phrases that matter more than isolated words
_REQUIREMENT_PHRASES = [
    "internal applications",
    "enterprise applications",
    "workflow automation",
    "api integrations",
    "backend services",
    "distributed systems",
    "cross-functional partners",
    "software engineering",
    "internal tooling",
    "cloud platforms",
    "large language models",
    "genai applications",
    "machine learning",
    "python",
    "java",
    "aws",
    "gcp",
    "apis",
    "llms",
]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9\+\#\-]{2,}", text.lower())


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _resume_lines(state: AgentState) -> list[str]:
    return [ln.strip() for ln in state.resume_text.splitlines() if ln.strip()]


def _count_phrase_occurrences(text: str, phrase: str) -> int:
    pattern = re.escape(phrase.lower())
    return len(re.findall(pattern, text))


def _markdown_to_text(text: str) -> str:
    """
    Lightweight markdown-to-text normalization.
    Good enough for job specs stored as .md.
    """
    text = text.replace("\r\n", "\n")

    # links: [label](url) -> label
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # wiki links: [[PyTorch]] -> PyTorch
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)

    # emphasis / bold / code markers
    text = re.sub(r"[*_`>#]", "", text)

    # markdown headings / bullets
    text = re.sub(r"^\s*[-+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.MULTILINE)

    # collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()

# Generic reader for .txt and .md
def _read_textish_file(path: Path) -> str:
    raw = path.read_text(encoding="utf-8").strip()
    if path.suffix.lower() == ".md":
        return _markdown_to_text(raw)
    return raw


def _keyword_overlap_score(a: str, b: str) -> int:
    a_tokens = {t for t in _tokenize(a) if t not in _STOPWORDS}
    b_tokens = {t for t in _tokenize(b) if t not in _STOPWORDS}
    return len(a_tokens & b_tokens)


def tool_retrieve_similar_resume_examples(
    state: AgentState,
    top_k: int = 2,
) -> dict[str, Any]:
    """
    Rank corpus examples by target JD similarity.
    """
    if not state.corpus_examples:
        return {"matches": []}

    scored: list[tuple[int, CorpusExample]] = []
    for example in state.corpus_examples:
        score = _keyword_overlap_score(state.jd_text, example.jd_text)
        scored.append((score, example))

    scored.sort(key=lambda pair: (-pair[0], pair[1].slug))
    matches = [
        {
            "slug": example.slug,
            "score": score,
            "jd_preview": example.jd_text[:300],
            "resume_preview": example.resume_variant_text[:300],
        }
        for score, example in scored[:top_k]
        if score > 0
    ]

    state.artifacts["retrieved_examples_json"] = {"matches": matches}
    return {"matches": matches}


def tool_extract_jd_requirements(state: AgentState, top_k: int = 15) -> ToolResult:
    """
    Extract rough requirement keywords and phrases from the JD only.

    Strategy:
    1. detect important multi-word phrases
    2. detect important single tokens
    3. rank phrases first, then tokens by frequency
    """
    jd_text = _normalize_whitespace(state.jd_text)

    phrase_hits: list[tuple[str, int]] = []
    for phrase in _REQUIREMENT_PHRASES:
        count = _count_phrase_occurrences(jd_text, phrase)
        if count > 0:
            phrase_hits.append((phrase, count))

    words = _tokenize(jd_text)
    token_freq: dict[str, int] = {}

    for word in words:
        if word in _STOPWORDS:
            continue
        if word not in _REQUIREMENT_HINTS and len(word) < 4:
            continue
        if word in _REQUIREMENT_HINTS:
            token_freq[word] = token_freq.get(word, 0) + 1

    ranked_phrases = sorted(phrase_hits, key=lambda kv: (-kv[1], kv[0]))
    ranked_tokens = sorted(token_freq.items(), key=lambda kv: (-kv[1], kv[0]))

    combined: list[str] = []
    seen: set[str] = set()

    for phrase, _ in ranked_phrases:
        if phrase not in seen:
            combined.append(phrase)
            seen.add(phrase)

    for token, _ in ranked_tokens:
        if token not in seen:
            combined.append(token)
            seen.add(token)

    requirements = combined[:top_k]

    payload = {"requirements": requirements}
    state.artifacts["requirements_json"] = payload
    return payload


def tool_find_resume_evidence(
    state: AgentState,
    requirement: str,
    max_snippets: int = 5,
) -> ToolResult:
    """
    Find resume lines relevant to a given requirement.

    Matching is deterministic and transparent:
    - exact phrase match gets strong weight
    - token overlap adds additional score
    """
    req = requirement.lower().strip()
    if not req:
        return {"requirement": requirement, "snippets": []}

    req_tokens = [t for t in _tokenize(req) if t not in _STOPWORDS]
    lines = _resume_lines(state)

    scored_hits: list[tuple[int, str]] = []

    for line in lines:
        line_lower = line.lower()

        score = 0

        if req in line_lower:
            score += 5

        score += sum(1 for token in req_tokens if token in line_lower)

        # Helpful soft matches for SWE-ish roles
        if req in {"genai applications", "large language models", "llms"}:
            if any(term in line_lower for term in {"machine learning", "signal research", "scikit-learn", "ml"}):
                score += 2

        if req in {"backend services", "internal applications", "enterprise applications"}:
            if any(term in line_lower for term in {"backend", "platform", "api", "framework", "infrastructure"}):
                score += 2

        if req in {"cloud platforms", "aws", "gcp"}:
            if any(term in line_lower for term in {"aws", "cloud", "distributed", "infrastructure"}):
                score += 2

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

    score = round(100 * len(matched) / len(requirements)) if requirements else 0

    fit = {
        "score": score,
        "matched": matched,
        "gaps": gaps,
    }
    state.artifacts["fit_analysis_json"] = fit
    return fit


def tool_render_report(state: AgentState) -> ToolResult:
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
    lines.append("- Emphasize backend systems depth, platform engineering, and cross-functional execution.")
    lines.append("- Tie infrastructure and reliability work to scalable internal tools and enterprise applications.")
    lines.append("- Highlight ML/AI-adjacent experience where relevant, especially experimentation and applied modeling.")
    lines.append("")

    report_md = "\n".join(lines).strip() + "\n"
    state.artifacts["report_md"] = report_md

    return {
        "ok": True,
        "bytes": len(report_md.encode("utf-8")),
    }


def tool_dump_state_summary(state: AgentState) -> ToolResult:
    return {
        "num_notes": len(state.notes),
        "artifact_keys": sorted(state.artifacts.keys()),
        "num_tool_calls": len(state.tool_history),
    }


def tool_load_resume_corpus(state: AgentState, corpus_dir: str) -> dict[str, Any]:
    """
    Load corpus examples from:
      resume_corpus/<slug>/jd.txt or jd.md
      resume_corpus/<slug>/resume_variant.txt
    """
    base = Path(corpus_dir)
    if not base.exists():
        raise FileNotFoundError(f"Corpus directory does not exist: {corpus_dir}")

    examples: list[CorpusExample] = []

    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue

        jd_path_txt = child / "jd.txt"
        jd_path_md = child / "jd.md"
        resume_variant_path = child / "resume_variant.txt"

        if not resume_variant_path.exists():
            continue

        jd_path = None
        if jd_path_txt.exists():
            jd_path = jd_path_txt
        elif jd_path_md.exists():
            jd_path = jd_path_md

        if jd_path is None:
            continue

        examples.append(
            CorpusExample(
                slug=child.name,
                jd_text=_read_textish_file(jd_path),
                resume_variant_text=_read_textish_file(resume_variant_path),
            )
        )

    state.corpus_examples = examples
    payload = {
        "num_examples": len(examples),
        "slugs": [ex.slug for ex in examples],
    }
    state.artifacts["corpus_summary_json"] = payload
    return payload

TOOLS: dict[str, ToolFn] = {
    "extract_jd_requirements": tool_extract_jd_requirements,
    "find_resume_evidence": tool_find_resume_evidence,
    "score_resume_fit": tool_score_resume_fit,
    "render_report": tool_render_report,
    "dump_state_summary": tool_dump_state_summary,
}