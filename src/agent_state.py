# ----------------------------
# 1) Agent state
# ----------------------------
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ArtifactKey = Literal[
    "requirements_json",
    "fit_analysis_json",
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

    def add_note(self, message: str) -> None:
        self.notes.append(message)

    def add_tool_history(
        self,
        *,
        tool_name: str,
        args: dict[str, Any],
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        entry: ToolHistoryEntry = {
            "tool": tool_name,
            "args": args,
        }
        if result is not None:
            entry["result"] = result
        if error is not None:
            entry["error"] = error
        self.tool_history.append(entry)