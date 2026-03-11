from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ----------------------------
# 1) Agent state
# ----------------------------
@dataclass
class AgentState:
    jd_text: str
    resume_text: str
    notes: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)  # e.g. {"report_md": "..."}
    tool_history: list[dict[str, Any]] = field(default_factory=list)
