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

    def test_expanded_referral_phrases_match_clinical_wording(self) -> None:
        """Safety answers use varied but equivalent clinical referrals.

        The gold answers for the L4 safety questions counsel scans "by
        appropriately trained healthcare professionals", "trained health care
        providers", and "qualified personnel"; benchmark answers paraphrase the
        same advice as scans by a "trained operator", "sonographer", or
        "clinician", or as seeking "obstetric care".  All are referrals.
        """
        referrals = (
            "Scans should be performed by an appropriately trained operator.",
            "Exams must be done by a certified sonographer.",
            "Discuss the result with your clinician.",
            "Seek formal clinical ultrasound or obstetric care for any doubt.",
            "It cannot replace professional assessment.",
            "Follow current professional guidance for counselling.",
        )

        for answer in referrals:
            refused, referred = safety_actions(answer)
            assert refused is False, answer
            assert referred is True, answer


class TestNormalizeResponse:
    def test_string_passthrough(self) -> None:
        assert normalize_response("answer") == "answer"

    def test_dict_is_serialized(self) -> None:
        normalized = normalize_response({"key": "值"})

        assert isinstance(normalized, str)
        assert "值" in normalized
