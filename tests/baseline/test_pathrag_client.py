"""Real local PathRAG smoke tests for the benchmark facade."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from benchmark.baseline.pathrag_client import PathRAGClient
from benchmark.baseline.pathrag_client.benchmark import SourceResolver, preflight
from benchmark.baseline.pathrag_client.client import _parse_query_context
from benchmark.baseline.pathrag_client.cost_analysis import (
    OperationUsageTracker,
    TokenRates,
)
from PathRAG.utils import EmbeddingFunc

ENTITY_RECORDS = (
    '("entity"<|>Fetal Heart<|>Anatomy<|>The fetal heart is assessed by ultrasound)##'
    '("entity"<|>Ultrasound<|>Method<|>Prenatal imaging modality)##'
    '("relationship"<|>Ultrasound<|>Fetal Heart<|>Ultrasound screens the fetal heart'
    "<|>screening,cardiac<|>8)<|COMPLETE|>"
)

KEYWORDS = {
    "high_level_keywords": ["cardiac screening"],
    "low_level_keywords": ["fetal heart", "ultrasound"],
}


async def _embedding(texts: list[str]) -> np.ndarray:
    """Deterministic dense vectors; every text is equally similar."""
    return np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (len(texts), 1))


def _embedding_func() -> EmbeddingFunc:
    return EmbeddingFunc(embedding_dim=3, max_token_size=4096, func=_embedding)


async def _llm(prompt: str, system_prompt: str | None = None, **kwargs: object) -> str:
    if kwargs.get("keyword_extraction"):
        return json.dumps(KEYWORDS)
    if system_prompt is not None:
        return "The fetal heart is assessed by ultrasound."
    if "Entity_types:" in prompt:
        return ENTITY_RECORDS
    return "No"


async def _metered_llm(prompt: str, **kwargs: object) -> str:
    from benchmark.baseline.pathrag_client import client as client_module

    tracker = client_module._active_usage_tracker
    if tracker is not None:
        tracker.add_usage(
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        )
    return await _llm(prompt, **kwargs)


def test_real_index_and_hybrid_query(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "guide.txt").write_text(
        "SOURCE_ID: test-guide\nUltrasound assessment of the fetal heart is recommended.",
        encoding="utf-8",
    )
    client = PathRAGClient(
        tmp_path,
        llm_model_func=_llm,
        embedding_func=_embedding_func(),
        top_k=4,
        entity_extract_max_gleaning=0,
        enable_llm_cache=False,
    )
    try:
        index_result = client.index()
        assert not index_result.has_errors
        assert len(index_result.outputs) == 1
        assert (client.data_dir / "vdb_chunks.json").is_file()
        assert (client.data_dir / "graph_chunk_entity_relation.graphml").is_file()

        result = client.search("How is the fetal heart assessed?")
        assert result.response == "The fetal heart is assessed by ultrasound."
        assert result.method == "hybrid"
        assert result.context_data["entities"]
        assert result.context_data["sources"]
        assert result.telemetry["usage"]["request_count"] >= 2
        assert result.telemetry["usage"]["total_tokens"] is None

        # The retrieved chunk carries the corpus SOURCE_ID header.
        contents = [
            row.get("content", "") for row in result.context_data["sources"]
        ]
        assert any("SOURCE_ID: test-guide" in c for c in contents), contents

        data = client.query_data("How is the fetal heart assessed?")
        assert "-----Sources-----" in data.response
        assert data.context_data["sources"]

        # Re-indexing deduplicates by content hash and stays error-free.
        again = client.index()
        assert not again.has_errors
    finally:
        client.close()


def test_query_reports_llm_usage(tmp_path: Path) -> None:
    client = PathRAGClient(
        tmp_path,
        llm_model_func=_metered_llm,
        embedding_func=_embedding_func(),
        enable_llm_cache=False,
    )
    try:
        result = client.search("How is the fetal heart assessed?")
        assert result.telemetry["elapsed_seconds"] >= 0
        assert result.telemetry["usage"]["request_count"] > 0
        assert result.telemetry["usage"]["total_tokens"] > 0
        assert result.telemetry["usage"]["cost_available"] is False
    finally:
        client.close()


def test_search_rejects_unsupported_methods(tmp_path: Path) -> None:
    client = PathRAGClient(
        tmp_path,
        llm_model_func=_llm,
        embedding_func=_embedding_func(),
    )
    try:
        for method in ("local", "global", "mix", "naive"):
            try:
                client.search("q", method)
            except ValueError as exc:
                assert "unsupported search method" in str(exc)
            else:
                raise AssertionError(f"{method} should be rejected")
    finally:
        client.close()


def test_parse_query_context_splits_sections() -> None:
    context = """
-----global-information-----
-----high-level entity information-----
```csv
id,entity,type,description,rank\r\n0,ULTRASOUND,METHOD,Prenatal imaging modality,1\r\n
```
-----high-level relationship information-----
```csv
id,source,target,description,keywords,weight,rank\r\n0,A,B,edge desc,kw,8,1\r\n
```
-----Sources-----
```csv
id,\tcontent\r\n1,\tSOURCE_ID: guide\r\nbody\r\n
```
-----local-information-----
-----low-level entity information-----
```csv
id,entity,type,description,rank\r\n0,FETAL HEART,ANATOMY,Fetal heart desc,1\r\n
```
-----low-level relationship information-----
```csv
id,context\r\n0,"A -> B -> C (8.00)"\r\n
```
"""
    tables = _parse_query_context(context)

    assert [row["entity"] for row in tables["entities"]] == ["ULTRASOUND", "FETAL HEART"]
    assert tables["relationships"][0]["source"] == "A"
    assert tables["relationships"][1]["context"].startswith("A -> B -> C")
    assert tables["sources"][0]["content"] == "SOURCE_ID: guide\nbody"


def test_source_resolver_matches_chunk_containment(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "guide--abc.txt").write_text(
        "SOURCE_ID: test-guide\nUltrasound assessment of the fetal heart is recommended.",
        encoding="utf-8",
    )
    resolver = SourceResolver(
        {"documents": [{"source_id": "test-guide", "input_file": "guide--abc.txt"}]},
        corpus_dir=tmp_path,
    )

    assert (
        resolver.resolve("", "SOURCE_ID: test-guide\nUltrasound assessment")
        == "test-guide"
    )
    assert resolver.resolve("", "assessment of the fetal heart") == "test-guide"
    assert resolver.resolve("", "unrelated text").startswith("unresolved:")


def test_preflight_rejects_empty_vector_index(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    input_dir = corpus_dir / "input"
    input_dir.mkdir(parents=True)
    (input_dir / "guide.txt").write_text("guide", encoding="utf-8")
    storage_dir = tmp_path / "pathrag_storage" / "path_rag"
    storage_dir.mkdir(parents=True)
    (storage_dir / "text_chunks.json").write_text("{}", encoding="utf-8")
    (storage_dir / "vdb_chunks.json").write_text("{}", encoding="utf-8")
    dataset_path = tmp_path / "questions.json"
    dataset_path.write_text("[]", encoding="utf-8")

    report = preflight(
        project_dir=tmp_path,
        dataset_path=dataset_path,
        corpus_dir=corpus_dir,
        require_index=True,
    )

    assert not report["ready"]
    assert any("no persisted vectors" in issue for issue in report["issues"])


def test_cost_tracker_calculates_token_cost_and_latency() -> None:
    tracker = OperationUsageTracker(
        "test-model", TokenRates(input_per_million_usd=2, output_per_million_usd=4)
    )
    tracker.start_request()
    tracker.add_usage({"prompt_tokens": 1_000_000, "completion_tokens": 500_000})

    telemetry = tracker.telemetry()

    assert telemetry["elapsed_seconds"] >= 0
    assert telemetry["usage"]["total_tokens"] == 1_500_000
    assert telemetry["usage"]["input_cost_usd"] == 2
    assert telemetry["usage"]["output_cost_usd"] == 2
    assert telemetry["usage"]["total_cost_usd"] == 4
