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
from benchmark.qa.models import GoldSource
from benchmark.qa.scoring import score_question


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


class ScoringPolicyTests(unittest.TestCase):
    def _question(self) -> Question:
        return Question(
            question_id="PU-L2-001",
            question="唐氏筛查如何进行？",
            question_type="factual",
            difficulty="L2",
            rag_arch_type="basic",
            gold_answer="建议进行21三体综合征血清学筛查。",
            gold_sources=[GoldSource(guide="ISUOG-midtrimester-2022", section="screening")],
            must_have_statements=["唐氏筛查"],
        )

    def test_hybrid_source_match_accepts_equivalent_evidence(self) -> None:
        result = score_question(
            self._question(),
            answer="建议进行21三体综合征血清学筛查。",
            retrieved_sources=["ACOG-midtrimester-ultrasound"],
            retrieved_context="唐氏筛查相关证据。",
            retrieved_supports=[{"唐氏筛查"}],
            source_equivalence={
                "ISUOG-midtrimester-2022": ["ACOG-midtrimester-ultrasound"]
            },
        )

        self.assertFalse(result.source_match.exact_source_hit)
        self.assertTrue(result.source_match.equivalent_source_hit)
        self.assertTrue(result.source_match.evidence_supported)
        self.assertEqual(result.retrieval.miss_at_k, 0.0)

    def test_judge_error_falls_back_to_lexical(self) -> None:
        result = score_question(
            self._question(),
            answer="无法确定。",
            retrieved_sources=[],
            retrieved_context="",
            retrieved_supports=[],
            judge_result={
                "model": "gpt-5",
                "error": "timeout",
                "usage": {"request_count": 1},
            },
        )

        self.assertEqual(result.scoring_method, "lexical")
        self.assertIsNone(result.judge_generation)
        self.assertEqual(result.judge.error, "timeout")


if __name__ == "__main__":
    unittest.main()
