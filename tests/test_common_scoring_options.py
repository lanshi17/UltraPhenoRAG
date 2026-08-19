"""Tests for shared scoring-option loading."""

from __future__ import annotations

from pathlib import Path

from benchmark.common.scoring_options import load_scoring_options


class TestLoadScoringOptions:
    def test_missing_file_falls_back_to_hybrid(self, tmp_path: Path) -> None:
        options = load_scoring_options(tmp_path / "missing.yaml")

        assert options["source_match_mode"] == "hybrid"
        assert options["source_equivalence"] == {}

    def test_retrieval_section_is_parsed(self, tmp_path: Path) -> None:
        config = tmp_path / "scoring.yaml"
        config.write_text(
            "retrieval:\n"
            "  source_match_mode: exact\n"
            "  source_equivalence:\n"
            "    ISUOG-midtrimester-2022:\n"
            "      - ACOG-midtrimester-ultrasound\n",
            encoding="utf-8",
        )

        options = load_scoring_options(config)

        assert options["source_match_mode"] == "exact"
        assert options["source_equivalence"] == {
            "ISUOG-midtrimester-2022": ["ACOG-midtrimester-ultrasound"]
        }

    def test_invalid_equivalence_is_ignored(self, tmp_path: Path) -> None:
        config = tmp_path / "scoring.yaml"
        config.write_text(
            "retrieval:\n  source_equivalence: not-a-mapping\n",
            encoding="utf-8",
        )

        options = load_scoring_options(config)

        assert options["source_equivalence"] == {}
