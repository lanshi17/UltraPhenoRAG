"""Tests for the unified corpus layout."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.common.unified_corpus import (
    corpus_input_dir,
    corpus_manifest_path,
    load_corpus_manifest,
    prepare_corpus,
    read_corpus_documents,
    source_coverage,
)


class TestCorpusPaths:
    def test_input_dir_is_created(self, tmp_path: Path) -> None:
        path = corpus_input_dir(tmp_path)

        assert path == tmp_path / "input"
        assert path.is_dir()

    def test_manifest_path(self, tmp_path: Path) -> None:
        assert corpus_manifest_path(tmp_path) == tmp_path / "corpus_manifest.json"

    def test_load_missing_manifest_is_empty(self, tmp_path: Path) -> None:
        assert load_corpus_manifest(tmp_path) == {"documents": []}


class _FakeSource:
    def __init__(self, guide: str) -> None:
        self.guide = guide


class _FakeQuestion:
    def __init__(self, question_id: str, guides: list[str]) -> None:
        self.question_id = question_id
        self.gold_sources = [_FakeSource(guide) for guide in guides]


class TestSourceCoverage:
    def test_coverage_counts(self) -> None:
        questions = [
            _FakeQuestion("q1", ["A", "B"]),
            _FakeQuestion("q2", ["C"]),
        ]
        coverage = source_coverage(questions, {"A", "B", "D"})

        assert coverage["fully_covered_questions"] == 1
        assert coverage["missing_source_ids"] == ["C"]
        assert coverage["uncovered_question_ids"] == ["q2"]


@pytest.fixture()
def fake_pdf(tmp_path: Path) -> Path:
    """Minimal valid PDF recognized by pypdf."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pdf = raw_dir / "guide.pdf"
    # 最小合法 PDF：一页、无文本
    pdf.write_bytes(
        b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF"
    )
    return raw_dir


class TestPrepareCorpus:
    def test_prepare_writes_manifest_and_records_error(
        self, tmp_path: Path, fake_pdf: Path
    ) -> None:
        corpus_dir = tmp_path / "corpus"

        manifest = prepare_corpus(
            raw_dir=fake_pdf,
            corpus_dir=corpus_dir,
        )

        # 空文本页 -> status=error 但流程不中断
        assert manifest["pdf_count"] == 1
        assert manifest["error_count"] == 1
        assert manifest["documents"][0]["status"] == "error"
        assert (corpus_dir / "corpus_manifest.json").is_file()


class TestReadCorpusDocuments:
    def test_reads_unified_input_with_source_id(self, tmp_path: Path) -> None:
        input_dir = corpus_input_dir(tmp_path)
        (input_dir / "guide--abc.txt").write_text(
            "SOURCE_ID: ISUOG-cns-2020\n\nbody text", encoding="utf-8"
        )

        documents = read_corpus_documents(tmp_path)

        assert len(documents) == 1
        source_id, path, text = documents[0]
        assert source_id == "ISUOG-cns-2020"
        assert path.name == "guide--abc.txt"
        assert "body text" in text

    def test_empty_input_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_corpus_documents(tmp_path)
