"""KAG query result."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryResult:
    response: str
    context_data: dict[str, Any] = field(default_factory=dict)
    query: str = ""
    method: str = ""
    raw_data: dict[str, Any] = field(default_factory=dict)
    telemetry: dict[str, Any] = field(default_factory=dict)

    @property
    def references(self) -> list[dict[str, Any]]:
        data = self.raw_data.get("data", self.context_data)
        return list(data.get("references", [])) if isinstance(data, dict) else []

    def get_context_df(self, name: str) -> Any:
        """Return a context table as a DataFrame when pandas is available."""

        value = self.context_data.get(name)
        if value is None:
            return None
        try:
            import pandas as pd
        except ImportError:
            return None
        if isinstance(value, pd.DataFrame):
            return value
        if isinstance(value, list):
            return pd.DataFrame(value)
        return None

    def __repr__(self) -> str:
        return (
            f"QueryResult(method={self.method!r}, "
            f"response_len={len(self.response)}, "
            f"context_keys={list(self.context_data)})"
        )
