"""Tests for the LightRAG benchmark source resolver."""

from __future__ import annotations

import pandas as pd

from benchmark.baseline.light_rag_client.benchmark import SourceResolver


class TestManifestResolution:
    def test_chunk_file_path_resolves_to_source_id(self) -> None:
        manifest = {
            "documents": [
                {
                    "source_id": "ISUOG-cns-2020",
                    "input_file": "input/ISUOG-cns-2020--abc.txt",
                    "status": "ok",
                }
            ]
        }
        resolver = SourceResolver(manifest)

        assert resolver.resolve("input/ISUOG-cns-2020--abc.txt") == "ISUOG-cns-2020"
        assert resolver.resolve("ISUOG-cns-2020--abc.txt") == "ISUOG-cns-2020"

    def test_source_id_header_in_text_is_last_resort(self) -> None:
        resolver = SourceResolver({"documents": []})

        resolved = resolver.resolve("unknown.txt", "SOURCE_ID: guide-x\n\nbody")

        assert resolved == "guide-x"

    def test_unresolvable_paths_are_flagged(self) -> None:
        resolver = SourceResolver({"documents": []})

        resolved = resolver.resolve("mystery.txt", "no header")

        assert resolved == "unresolved:mystery.txt"


class TestDataFrameShapeCompatibility:
    def test_accepts_microsoft_resolver_shape(self) -> None:
        """LightRAG resolver must accept (documents, text_units, manifest)."""
        documents = pd.DataFrame(
            [
                {
                    "id": "doc-1",
                    "title": "ISUOG-cns-2020--abc.txt",
                    "text": "SOURCE_ID: ISUOG-cns-2020\n",
                }
            ]
        )
        text_units = pd.DataFrame(
            [
                {
                    "id": "unit-1",
                    "human_readable_id": 3,
                    "document_id": ["doc-1"],
                    "text": "cns content",
                }
            ]
        )
        manifest = {
            "documents": [
                {
                    "source_id": "ISUOG-cns-2020",
                    "input_file": "input/ISUOG-cns-2020--abc.txt",
                }
            ]
        }

        resolver = SourceResolver(documents, text_units, manifest)

        assert resolver.resolve("unit-1", "cns content") == "ISUOG-cns-2020"
        assert resolver.resolve(3, "cns content") == "ISUOG-cns-2020"
