"""Tests for shared corpus helpers."""

from __future__ import annotations

from pathlib import Path

from benchmark.common.corpus import (
    canonical_source_id,
    extract_pdf_text,
    safe_slug,
    source_id_from_text,
)


class TestCanonicalSourceId:
    def test_known_pdf_maps_to_dataset_id(self) -> None:
        assert (
            canonical_source_id(Path("ISUOG_2022_routine-mid-trimester-scan.pdf"))
            == "ISUOG-midtrimester-2022"
        )

    def test_alias_pdf_maps_to_same_id(self) -> None:
        alias = Path(
            "ISUOG-Practice-Guidelines-Updated-performance-of-11-14-week-"
            "ultrasound-scan.pdf"
        )
        assert canonical_source_id(alias) == "ISUOG-11-14w-2023"

    def test_unknown_pdf_falls_back_to_slug(self) -> None:
        assert canonical_source_id(Path("Some_Other Guide.pdf")) == "Some_Other-Guide"


class TestSafeSlug:
    def test_chinese_only_falls_back_to_document(self) -> None:
        # CJK 字符被 NFKD/ASCII 过滤后无剩余内容
        assert safe_slug("产前 超声指南") == "document"

    def test_mixed_content_keeps_ascii(self) -> None:
        assert safe_slug("ISUOG 2022 指南") == "ISUOG-2022"

    def test_empty_falls_back_to_document(self) -> None:
        assert safe_slug("???") == "document"

    def test_length_is_capped(self) -> None:
        assert len(safe_slug("a" * 500)) == 120


class TestSourceIdFromText:
    def test_header_is_extracted(self) -> None:
        text = "SOURCE_ID: ISUOG-cns-2020\n\nbody"
        assert source_id_from_text(text) == "ISUOG-cns-2020"

    def test_missing_header_returns_none(self) -> None:
        assert source_id_from_text("no header here") is None


class TestExtractPdfText:
    def test_non_pdf_returns_empty(self, tmp_path: Path) -> None:
        fake = tmp_path / "fake.pdf"
        fake.write_text("not a pdf", encoding="utf-8")

        text, pages, warnings = extract_pdf_text(fake)

        assert text == ""
        assert pages == 0
        assert warnings  # pypdf 无法解析时记录告警而非崩溃
