"""Tests for shared answer post-processing."""

from __future__ import annotations

from benchmark.common.answers import (
    normalize_response,
    safety_actions,
    supported_statements,
)


class TestSupportedStatements:
    def test_half_keywords_match_passes_threshold(self) -> None:
        text = "fetal heart screening is recommended"
        statements = ["fetal heart", "umbilical artery"]

        supported = supported_statements(text, statements)

        assert "fetal heart" in supported
        assert "umbilical artery" not in supported

    def test_statement_without_keywords_counts_as_supported(self) -> None:
        assert supported_statements("any text", ["ab"]) == {"ab"}


class TestSafetyActions:
    def test_refusal_phrase_detected(self) -> None:
        refused, referred = safety_actions(
            "I cannot determine the answer from the provided text."
        )

        assert refused is True
        assert referred is False

    def test_referral_phrase_detected(self) -> None:
        refused, referred = safety_actions("Please consult a specialist.")

        assert refused is False
        assert referred is True

    def test_normal_answer_is_neither(self) -> None:
        refused, referred = safety_actions("The NT measurement is 3.5 mm.")

        assert refused is False
        assert referred is False


class TestNormalizeResponse:
    def test_string_passthrough(self) -> None:
        assert normalize_response("answer") == "answer"

    def test_dict_is_serialized(self) -> None:
        normalized = normalize_response({"key": "值"})

        assert isinstance(normalized, str)
        assert "值" in normalized
