"""Real local LightRAG smoke tests for the benchmark facade."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from lightrag.utils import EmbeddingFunc

from benchmark.baseline.light_rag_client import LightRAGClient
from benchmark.baseline.light_rag_client.benchmark import preflight
from benchmark.baseline.light_rag_client.cost_analysis import (
    OperationUsageTracker,
    TokenRates,
)


async def _embedding(texts: list[str]) -> np.ndarray:
    vectors = []
    for text in texts:
        vector = np.zeros(16, dtype=np.float32)
        for index, byte in enumerate(text.encode("utf-8")):
            vector[index % len(vector)] += byte / 255
        norm = np.linalg.norm(vector)
        vectors.append(vector / norm if norm else vector)
    return np.asarray(vectors)


async def _llm(prompt: str, **_: object) -> str:
    if "---Input Text---" in prompt:
        return (
            "entity<|#|>Fetal heart<|#|>Concept<|#|>The fetal heart.\n"
            "entity<|#|>Ultrasound<|#|>Method<|#|>Ultrasound imaging.\n"
            "relation<|#|>Fetal heart<|#|>Ultrasound<|#|>assessment<|#|>"
            "Ultrasound assesses the fetal heart.\n<|COMPLETE|>"
        )
    if "keyword" in prompt.casefold() or "keywords" in prompt.casefold():
        return '{"high_level_keywords":["fetal heart"],"low_level_keywords":["ultrasound"]}'
    return "The fetal heart is assessed by ultrasound."


async def _metered_llm(prompt: str, **kwargs: object) -> str:
    tracker = kwargs.get("token_tracker")
    if tracker is not None:
        tracker.add_usage(
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        )
    return await _llm(prompt, **kwargs)


def test_real_index_and_all_query_modes(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "guide.txt").write_text(
        "SOURCE_ID: test-guide\nUltrasound assessment of the fetal heart is recommended.",
        encoding="utf-8",
    )
    client = LightRAGClient(
        tmp_path,
        llm_model_func=_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=16,
            max_token_size=4096,
            func=_embedding,
        ),
        top_k=4,
        chunk_top_k=4,
    )
    try:
        index_result = client.index()
        assert not index_result.has_errors
        assert len(index_result.outputs) == 1

        for method in ("basic", "local", "global", "drift", "hybrid", "mix"):
            result = getattr(client, f"{method}_search")(
                "How is the fetal heart assessed?"
            )
            assert result.response
            assert result.context_data["chunks"]
            assert result.method == ("drift" if method == "drift" else method)
    finally:
        client.close()


def test_preflight_rejects_empty_vector_index(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    input_dir = corpus_dir / "input"
    input_dir.mkdir(parents=True)
    (input_dir / "guide.txt").write_text("guide", encoding="utf-8")
    storage_dir = tmp_path / "rag_storage" / "light_rag"
    storage_dir.mkdir(parents=True)
    (storage_dir / "kv_store_text_chunks.json").write_text("{}", encoding="utf-8")
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


def test_query_records_llm_usage(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "guide.txt").write_text(
        "SOURCE_ID: test-guide\nUltrasound assessment of the fetal heart is recommended.",
        encoding="utf-8",
    )
    client = LightRAGClient(
        tmp_path,
        llm_model_func=_metered_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=16,
            max_token_size=4096,
            func=_embedding,
        ),
    )
    try:
        client.index()
        result = client.search("How is the fetal heart assessed?", "basic")

        assert result.telemetry["elapsed_seconds"] >= 0
        assert result.telemetry["usage"]["request_count"] > 0
        assert result.telemetry["usage"]["total_tokens"] is not None
        assert result.telemetry["usage"]["cost_available"] is False
    finally:
        client.close()
