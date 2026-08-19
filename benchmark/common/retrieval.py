"""Retrieved-context extraction shared by all baselines."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from benchmark.common.corpus import source_id_from_text as _source_id_from_text


def as_string_list(value: Any) -> list[str]:
    """Coerce a pandas cell (scalar, list, or NaN) into ``list[str]``."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item is not None]
    if isinstance(value, str):
        return [value] if value else []
    try:
        import pandas as pd

        if pd.isna(value):
            return []
    except (TypeError, ValueError, ImportError):
        pass
    return [str(value)]


def context_frames(context_data: Any) -> list[tuple[str, list[Any]]]:
    """Normalise DataFrame / dict / list context payloads into named rows.

    DataFrame inputs are converted to row dicts so LightRAG (list-of-dicts)
    and Microsoft GraphRAG (dict-of-DataFrames) payloads share one consumer.
    """
    if hasattr(context_data, "iterrows"):
        return [("sources", [row.to_dict() for _, row in context_data.iterrows()])]
    if isinstance(context_data, Mapping):
        data = context_data.get("data", context_data)
        if not isinstance(data, Mapping):
            return []
        frames: list[tuple[str, list[Any]]] = []
        for name, value in data.items():
            if hasattr(value, "iterrows"):
                frames.append(
                    (str(name), [row.to_dict() for _, row in value.iterrows()])
                )
            elif isinstance(value, list):
                frames.append((str(name), value))
        return frames
    if isinstance(context_data, list):
        return [("sources", context_data)]
    return []


CONTEXT_TABLE_PRIORITY = {
    "chunks": 0,
    "sources": 1,
    "text_units": 2,
    "entities": 3,
    "relationships": 4,
}


def extract_retrieved_context(
    context_data: Any,
    resolver: Any,
    k: int,
) -> list[dict[str, str]]:
    """Flatten retrieved context into auditable source records (capped at ``k``).

    ``resolver`` must expose ``resolve(record_id, text) -> source_id``; both
    baseline ``SourceResolver`` classes satisfy that contract.
    """
    records: list[dict[str, str]] = []
    frames = context_frames(context_data)
    for table, values in sorted(
        frames, key=lambda item: CONTEXT_TABLE_PRIORITY.get(item[0].casefold(), 9)
    ):
        for index, item in enumerate(values):
            if not isinstance(item, Mapping):
                continue
            text = str(
                item.get("content", item.get("text", item.get("description", ""))) or ""
            ).strip()
            if not text:
                continue
            record_id = item.get(
                "chunk_id",
                item.get("reference_id", item.get("id", index)),
            )
            file_path = item.get("file_path", item.get("source_id", ""))
            records.append(
                {
                    "context_table": table,
                    "record_id": str(record_id),
                    "source_id": resolver.resolve(file_path, text),
                    "text": text,
                }
            )
            if len(records) >= k:
                return records
    return records


def source_id_from_text(text: str) -> str | None:
    """Backward-compatible alias for :func:`corpus.source_id_from_text`."""
    return _source_id_from_text(text)


__all__ = [
    "CONTEXT_TABLE_PRIORITY",
    "as_string_list",
    "context_frames",
    "extract_retrieved_context",
    "source_id_from_text",
]
