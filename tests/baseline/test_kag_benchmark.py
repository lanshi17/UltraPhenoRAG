"""Focused offline tests for the KAG benchmark entry point."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

from benchmark.baseline.kag_client import benchmark as kag_benchmark


SOURCE_ID = "ISUOG-11-14w-2023"
INPUT_NAME = "ISUOG-11-14w-2023--1824b542dc28.txt"


def _question(question_id: str, rag_arch_type: str) -> dict[str, object]:
    """Return the smallest QA record that passes the shared validator."""
    return {
        "question_id": question_id,
        "question": f"What does {question_id} ask?",
        "question_type": "measurement_standard",
        "difficulty": "L1",
        "rag_arch_type": rag_arch_type,
        "gold_answer": "Evidence supports the answer.",
        "gold_sources": [{"guide": SOURCE_ID, "section": "section-1"}],
        "must_have_statements": ["Evidence supports the answer."],
        "annotators": [{"id": "reviewer-1"}, {"id": "reviewer-2"}],
    }


def _write_benchmark_inputs(
    tmp_path: Path,
    *,
    questions: list[dict[str, object]] | None = None,
) -> tuple[Path, Path, Path]:
    """Create a valid local corpus, dataset, and empty KAG project directory."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    corpus_dir = tmp_path / "corpus"
    input_dir = corpus_dir / "input"
    input_dir.mkdir(parents=True)
    (input_dir / INPUT_NAME).write_text(
        f"SOURCE_ID: {SOURCE_ID}\n\nEvidence supports the answer.\n",
        encoding="utf-8",
    )
    (corpus_dir / "corpus_manifest.json").write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "source_id": SOURCE_ID,
                        "input_file": f"input/{INPUT_NAME}",
                        "status": "ok",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    dataset_path = tmp_path / "questions.json"
    dataset_path.write_text(
        json.dumps(questions or [_question("PU-L1-001", "basic")]),
        encoding="utf-8",
    )
    return project_dir, corpus_dir, dataset_path


def _make_checkpoint(project_dir: Path, *, records: int) -> Path:
    """Build just enough of diskcache's on-disk shape for read-only preflight."""
    graph_path = project_dir / "kag_storage" / "ckpt" / "MemoryGraphWriter"
    graph_path.mkdir(parents=True)
    database = graph_path / "cache.db"
    connection = sqlite3.connect(database)
    try:
        connection.execute("CREATE TABLE Cache (key TEXT, filename TEXT, value BLOB)")
        connection.executemany(
            "INSERT INTO Cache (key, filename, value) VALUES (?, NULL, ?)",
            [
                (
                    f"record-{index}",
                    b"KAG.Chunk _content_vector [0.1, 0.2]",
                )
                for index in range(records)
            ],
        )
        connection.commit()
    finally:
        connection.close()
    return database


class TestSourceResolver:
    def test_resolves_kag_split_title_variants(self) -> None:
        resolver = kag_benchmark.SourceResolver(
            {
                "documents": [
                    {
                        "source_id": SOURCE_ID,
                        "input_file": f"input/{INPUT_NAME}",
                    }
                ]
            }
        )

        titles = (
            "ISUOG-11-14w-2023--1824b542dc28_split_1",
            "ISUOG-11-14w-2023--1824b542dc28_split_12.txt",
            "input/ISUOG-11-14w-2023--1824b542dc28_split_3.txt",
            r"input\ISUOG-11-14w-2023--1824b542dc28_split_4",
        )

        assert [resolver.resolve(title) for title in titles] == [SOURCE_ID] * len(
            titles
        )


class TestPreflight:
    def test_diskcache_checkpoint_requires_at_least_one_record(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        project_dir, corpus_dir, dataset_path = _write_benchmark_inputs(tmp_path)
        _make_checkpoint(project_dir, records=0)

        class UnexpectedKAGClient:
            def __init__(self, *args: object, **kwargs: object) -> None:
                del args, kwargs
                raise AssertionError("preflight must not construct a KAG client")

        # Preflight must inspect only the local SQLite checkpoint; no model or
        # remote service is needed to distinguish an empty index from a real one.
        monkeypatch.setattr(kag_benchmark, "KAGClient", UnexpectedKAGClient)

        empty = kag_benchmark.preflight(
            project_dir=project_dir,
            dataset_path=dataset_path,
            corpus_dir=corpus_dir,
            require_index=True,
        )

        assert empty["checkpoint_record_count"] == 0
        assert empty["has_checkpoint_records"] is False
        assert empty["missing_index"] is True
        assert empty["ready"] is False
        assert empty["has_chunk_vectors"] is False
        assert any(
            "no persisted vectorized Chunk" in issue for issue in empty["issues"]
        )

        # A nonempty graph snapshot alone is not evidence of a retrievable
        # vector index: it may be stale, entity-only, or otherwise empty.  The
        # read-only preflight deliberately requires the verified diskcache
        # chunk-vector marker instead of deserializing arbitrary graph bytes.
        graph_path = project_dir / "kag_storage" / "ckpt" / "MemoryGraphWriter"
        (graph_path / "graph").write_bytes(b"nonempty but unverified snapshot")
        snapshot_only = kag_benchmark.preflight(
            project_dir=project_dir,
            dataset_path=dataset_path,
            corpus_dir=corpus_dir,
            require_index=True,
        )
        assert snapshot_only["has_graph_snapshot"] is True
        assert snapshot_only["has_chunk_vectors"] is False
        assert snapshot_only["ready"] is False

        with sqlite3.connect(graph_path / "cache.db") as connection:
            connection.execute(
                "INSERT INTO Cache (key, filename, value) VALUES (?, NULL, ?)",
                ("persisted-record", b"KAG.Chunk _content_vector [0.1, 0.2]"),
            )

        populated = kag_benchmark.preflight(
            project_dir=project_dir,
            dataset_path=dataset_path,
            corpus_dir=corpus_dir,
            require_index=True,
        )

        assert populated["checkpoint_record_count"] == 1
        assert populated["has_checkpoint_records"] is True
        assert populated["chunk_vector_record_count"] == 1
        assert populated["has_chunk_vectors"] is True
        assert populated["missing_index"] is False
        assert populated["ready"] is True


class TestEvaluationAudit:
    def test_mocked_evaluation_writes_audit_and_adapts_methods(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        questions = [
            _question("PU-L1-001", "basic"),
            _question("PU-L1-002", "multi-vector"),
            _question("PU-L1-003", "graph-enhanced"),
        ]
        project_dir, corpus_dir, dataset_path = _write_benchmark_inputs(
            tmp_path, questions=questions
        )
        _make_checkpoint(project_dir, records=1)

        class FakeKAGClient:
            instances: list["FakeKAGClient"] = []

            def __init__(self, project_path: Path, **kwargs: object) -> None:
                del kwargs
                self.project_path = project_path
                self.calls: list[tuple[str, str, int]] = []
                self.closed = False
                self.instances.append(self)

            def search(self, query: str, method: str, *, top_k: int) -> SimpleNamespace:
                self.calls.append((query, method, top_k))
                return SimpleNamespace(
                    response="Evidence supports the answer.",
                    context_data={
                        "chunks": [
                            {
                                "chunk_id": "chunk-1",
                                "title": ("ISUOG-11-14w-2023--1824b542dc28_split_1"),
                                "file_path": (
                                    "ISUOG-11-14w-2023--1824b542dc28_split_1"
                                ),
                                "content": "Evidence supports the answer.",
                            }
                        ]
                    },
                    telemetry={
                        "usage": {
                            "request_count": 1,
                            "failed_request_count": 0,
                            "prompt_tokens": 2,
                            "completion_tokens": 1,
                            "total_tokens": 3,
                            "cost_available": False,
                            "models": ["fake-kag"],
                        }
                    },
                )

            def close(self) -> None:
                self.closed = True

        monkeypatch.setattr(kag_benchmark, "KAGClient", FakeKAGClient)
        output_path = tmp_path / "results" / "evaluation.json"

        report = kag_benchmark.evaluate(
            project_dir=project_dir,
            dataset_path=dataset_path,
            corpus_dir=corpus_dir,
            requested_method="adaptive",
            k=2,
            output_path=output_path,
        )

        assert [call[1] for call in FakeKAGClient.instances[0].calls] == [
            "basic",
            "naive",
            "solver",
        ]
        assert all(call[2] == 2 for call in FakeKAGClient.instances[0].calls)
        assert FakeKAGClient.instances[0].closed is True
        assert report["metadata"]["actual_search_methods"] == {
            "basic": 1,
            "naive": 1,
            "solver": 1,
        }
        assert report["summary"]["usage"]["query"]["request_count"] == 3
        assert all(
            result["retrieved_sources"] == [SOURCE_ID] for result in report["results"]
        )

        raw_path = output_path.with_suffix(".jsonl")
        assert output_path.is_file()
        assert raw_path.is_file()
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        raw_rows = [
            json.loads(line)
            for line in raw_path.read_text(encoding="utf-8").splitlines()
        ]
        assert saved["metadata"]["requested_search_method"] == "adaptive"
        assert [row["search_method"] for row in raw_rows] == [
            "basic",
            "naive",
            "solver",
        ]


def test_build_index_passes_selected_corpus_directory_to_client(
    tmp_path: Path, monkeypatch
) -> None:
    """The CLI-facing build path must not infer corpus from project parents."""

    project_dir, corpus_dir, dataset_path = _write_benchmark_inputs(tmp_path)
    observed: dict[str, object] = {}

    class FakeKAGClient:
        def __init__(self, project_path: Path, **kwargs: object) -> None:
            observed["project_path"] = project_path
            observed["input_dir"] = kwargs.get("input_dir")
            self.graph_path = (
                project_path / "kag_storage" / "ckpt" / "MemoryGraphWriter"
            )

        def index(self, *, cache: bool) -> SimpleNamespace:
            observed["cache"] = cache
            return SimpleNamespace(
                outputs=[{"id": "selected-corpus-document"}],
                errors=[],
                has_errors=False,
                telemetry={},
            )

        def close(self) -> None:
            observed["closed"] = True

    monkeypatch.setattr(kag_benchmark, "KAGClient", FakeKAGClient)

    result = kag_benchmark.build_index(
        project_dir=project_dir,
        dataset_path=dataset_path,
        corpus_dir=corpus_dir,
        cache=False,
    )

    expected_input = corpus_dir / "input"
    assert observed["project_path"] == project_dir
    assert observed["input_dir"] == expected_input
    assert observed["cache"] is False
    assert observed["closed"] is True
    assert result["corpus_input_dir"] == str(expected_input)


def test_shared_model_provider_cli_override_applies_to_embedding() -> None:
    args = kag_benchmark.build_parser().parse_args(
        [
            "index",
            "--model-provider",
            "azure",
            "--api-base",
            "https://example.openai.azure.com",
            "--api-version",
            "2025-01-01-preview",
        ]
    )

    overrides = kag_benchmark._llm_overrides_from_args(args)

    assert overrides is not None
    assert overrides.completion.model_provider == "azure"
    assert overrides.embedding.model_provider == "azure"
    assert overrides.embedding.api_base == "https://example.openai.azure.com"
    assert overrides.embedding.api_version == "2025-01-01-preview"
