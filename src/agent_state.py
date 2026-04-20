# ----------------------------
# 1) Agent state
# ----------------------------
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ArtifactKey = Literal[
    # Inputs / routing
    "corpus_dir",
    # Corpus loading
    "corpus_summary_json",
    # Requirement extraction
    "requirements_json",
    # Retrieval
    "retrieved_examples_json",
    # Fit scoring
    "fit_analysis_json",
    # Generation
    "target_resume_txt",
    "generation_metadata_json",
    # Evaluation of initial draft
    "resume_evaluation_json",
    # Revision pass (triggered when initial score < threshold)
    "revised_resume_txt",
    "revision_evaluation_json",
    "revision_metadata_json",
    # Report
    "report_md",
]

ToolHistoryEntry = dict[str, Any]

@dataclass
class CorpusExample:
    slug: str
    jd_text: str
    resume_variant_text: str


@dataclass
class AgentState:
    jd_text: str
    resume_text: str

    # Scratchpad the agent can append to across iterations
    notes: list[str] = field(default_factory=list)

    # Structured intermediate/final outputs
    artifacts: dict[str, Any] = field(default_factory=dict)

    # Executed tool calls + results for observability/debugging
    tool_history: list[ToolHistoryEntry] = field(default_factory=list)

    corpus_examples: list[CorpusExample] = field(default_factory=list)

    def add_note(self, message: str) -> None:
        self.notes.append(message)

    def add_tool_history(
        self,
        *,
        tool_name: str,
        args: dict[str, Any],
        result: dict[str, Any] | None = None,
        error: str | None = None,
        duration_ms: int | None = None,
        input_chars: int | None = None,
        output_chars: int | None = None,
    ) -> None:
        entry: ToolHistoryEntry = {
            "tool": tool_name,
            "args": args,
        }
        if result is not None:
            entry["result"] = result
        if error is not None:
            entry["error"] = error
        if duration_ms is not None:
            entry["duration_ms"] = duration_ms
        if input_chars is not None:
            entry["input_chars"] = input_chars
        if output_chars is not None:
            entry["output_chars"] = output_chars
        self.tool_history.append(entry)