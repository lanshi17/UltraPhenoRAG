"""KAG index result."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class IndexResult:
    outputs: list[Any]
    errors: list[str] = field(default_factory=list)
    has_errors: bool = False
    telemetry: dict[str, Any] = field(default_factory=dict)

    @property
    def workflow_names(self) -> list[str]:
        return [
            str(item.get("id", item)) if isinstance(item, dict) else str(item)
            for item in self.outputs
        ]

    @property
    def error_count(self) -> int:
        return len(self.errors)

    def __repr__(self) -> str:
        return (
            f"IndexResult(documents={len(self.outputs)}, "
            f"errors={self.error_count}, has_errors={self.has_errors})"
        )
