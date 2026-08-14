from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from benchmark.baseline.microsoft_graphrag_client.benchmark import (
    SourceResolver,
    _extract_retrieved_context,
    _selected_method,
    canonical_source_id,
)
from benchmark.qa.models import Question


class CorpusMappingTests(unittest.TestCase):
    def test_known_pdf_uses_dataset_source_id(self) -> None:
        source_id = canonical_source_id(
            Path("ISUOG_2022_routine-mid-trimester-scan.pdf")
        )

        self.assertEqual(source_id, "ISUOG-midtrimester-2022")


class SourceResolverTests(unittest.TestCase):
    def test_context_text_unit_resolves_to_manifest_source(self) -> None:
        documents = pd.DataFrame(
            [
                {
                    "id": "document-1",
                    "title": "ISUOG-midtrimester-2022--abc123.txt",
                    "text": "SOURCE_ID: ISUOG-midtrimester-2022\n",
                }
            ]
        )
        text_units = pd.DataFrame(
            [
                {
                    "id": "text-unit-1",
                    "human_readable_id": 7,
                    "document_id": "document-1",
                    "text": "The scan should assess fetal anatomy.",
                }
            ]
        )
        manifest = {
            "documents": [
                {
                    "source_id": "ISUOG-midtrimester-2022",
                    "input_file": (
                        "input/ISUOG-midtrimester-2022--abc123.txt"
                    ),
                    "status": "ok",
                }
            ]
        }
        resolver = SourceResolver(documents, text_units, manifest)
        context = {
            "sources": pd.DataFrame(
                [
                    {
                        "id": "7",
                        "text": "The scan should assess fetal anatomy.",
                    }
                ]
            )
        }

        records = _extract_retrieved_context(context, resolver, k=16)

        self.assertEqual(len(records), 1)
        self.assertEqual(
            records[0]["source_id"],
            "ISUOG-midtrimester-2022",
        )


class AdaptiveSearchTests(unittest.TestCase):
    def test_graph_enhanced_question_uses_drift_search(self) -> None:
        question = Question(
            question_id="PU-L3-001",
            question="How are the findings related?",
            question_type="differential_diagnosis",
            difficulty="L3",
            rag_arch_type="graph-enhanced",
        )

        self.assertEqual(_selected_method(question, "adaptive"), "drift")


if __name__ == "__main__":
    unittest.main()
