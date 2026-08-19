"""Tests for shared retrieved-context extraction."""

from __future__ import annotations

from typing import Any

import pandas as pd

from benchmark.common.retrieval import (
    as_string_list,
    context_frames,
    extract_retrieved_context,
)


class _StaticResolver:
    """Minimal resolver satisfying the shared extractor contract."""

    def resolve(self, record_id: Any, text: str) -> str:
        if "fetal heart" in text:
            return "ISUOG-fetal-cardiac-screening-2023"
        return f"unresolved:{record_id}"


class TestContextFrames:
    def test_dataframe_becomes_named_rows(self) -> None:
        frame = pd.DataFrame([{"text": "a"}])

        frames = context_frames(frame)

        assert frames[0][0] == "sources"
        assert frames[0][1] == [{"text": "a"}]

    def test_mapping_payload_keeps_table_names(self) -> None:
        frames = context_frames(
            {"data": {"chunks": [{"content": "chunk"}], "entities": []}}
        )

        names = [name for name, _ in frames]
        assert names == ["chunks", "entities"]

    def test_list_payload_is_named_sources(self) -> None:
        frames = context_frames([{"content": "row"}])

        assert frames == [("sources", [{"content": "row"}])]


class TestExtractRetrievedContext:
    def test_chunks_table_takes_priority_and_caps_at_k(self) -> None:
        context = {
            "data": {
                "entities": [{"content": "entity text"}],
                "chunks": [
                    {"content": "fetal heart chunk", "chunk_id": "c1"},
                    {"content": "fetal heart chunk 2", "chunk_id": "c2"},
                ],
            }
        }

        records = extract_retrieved_context(context, _StaticResolver(), k=1)

        assert len(records) == 1
        assert records[0]["context_table"] == "chunks"
        assert records[0]["record_id"] == "c1"
        assert records[0]["source_id"] == "ISUOG-fetal-cardiac-screening-2023"

    def test_unresolvable_text_is_flagged(self) -> None:
        context = {"data": {"chunks": [{"content": "unknown topic"}]}}

        records = extract_retrieved_context(context, _StaticResolver(), k=16)

        assert records[0]["source_id"].startswith("unresolved:")

    def test_empty_text_rows_are_skipped(self) -> None:
        context = {"data": {"chunks": [{"content": ""}, {"content": "fetal heart"}]}}

        records = extract_retrieved_context(context, _StaticResolver(), k=16)

        assert len(records) == 1


class TestAsStringList:
    def test_none_and_empty(self) -> None:
        assert as_string_list(None) == []
        assert as_string_list("") == []

    def test_scalar_becomes_singleton(self) -> None:
        assert as_string_list("doc-1") == ["doc-1"]
        assert as_string_list(7) == ["7"]

    def test_iterables_are_stringified(self) -> None:
        assert as_string_list(["a", None, 2]) == ["a", "2"]

    def test_nan_is_dropped(self) -> None:
        assert as_string_list(float("nan")) == []
