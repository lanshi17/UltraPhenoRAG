"""Tests for the Microsoft GraphRAG baseline benchmark entry point."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from benchmark.baseline.microsoft_graphrag_client.benchmark import (
    SourceResolver,
    _selected_method,
)
from benchmark.common import extract_retrieved_context
from benchmark.qa.models import GoldSource, Question
from benchmark.qa.scoring import score_question


class CorpusMappingTests:
    def test_known_pdf_uses_dataset_source_id(self) -> None:
        from benchmark.common import canonical_source_id

        source_id = canonical_source_id(
            Path("ISUOG_2022_routine-mid-trimester-scan.pdf")
        )

        assert source_id == "ISUOG-midtrimester-2022"


class SourceResolverTests:
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
                    "input_file": "input/ISUOG-midtrimester-2022--abc123.txt",
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

        records = extract_retrieved_context(context, resolver, k=16)

        assert len(records) == 1
        assert records[0]["source_id"] == "ISUOG-midtrimester-2022"


class AdaptiveSearchTests:
    def test_graph_enhanced_question_uses_drift_search(self) -> None:
        question = Question(
            question_id="PU-L3-001",
            question="How are the findings related?",
            question_type="differential_diagnosis",
            difficulty="L3",
            rag_arch_type="graph-enhanced",
        )

        assert _selected_method(question, "adaptive") == "drift"


class ScoringPolicyTests:
    def _question(self) -> Question:
        return Question(
            question_id="PU-L2-001",
            question="唐氏筛查如何进行？",
            question_type="factual",
            difficulty="L2",
            rag_arch_type="basic",
            gold_answer="建议进行21三体综合征血清学筛查。",
            gold_sources=[
                GoldSource(guide="ISUOG-midtrimester-2022", section="screening")
            ],
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

        assert not result.source_match.exact_source_hit
        assert result.source_match.equivalent_source_hit
        assert result.source_match.evidence_supported
        assert result.retrieval.miss_at_k == 0.0

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

        assert result.scoring_method == "lexical"
        assert result.judge_generation is None
        assert result.judge.error == "timeout"
