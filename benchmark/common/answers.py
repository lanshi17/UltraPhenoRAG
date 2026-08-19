"""Answer post-processing shared by all baselines."""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

REFUSAL_PHRASES = (
    "cannot determine",
    "can't determine",
    "cannot answer",
    "can't answer",
    "insufficient information",
    "not enough information",
    "outside the provided",
    "not contained in",
)

REFERRAL_PHRASES = (
    "consult",
    "seek medical",
    "healthcare professional",
    "healthcare provider",
    "qualified professional",
    "specialist",
    "obstetrician",
    "medical supervision",
)


def supported_statements(
    text: str,
    statements: Sequence[str],
    *,
    keyword_threshold: float = 0.5,
) -> set[str]:
    """Lexical check of which gold statements a retrieved chunk supports."""
    text_lower = text.lower()
    supported: set[str] = set()
    for statement in statements:
        keywords = [
            word for word in re.split(r"[，,。；;、\s]+", statement) if len(word) > 2
        ]
        if not keywords:
            supported.add(statement)
            continue
        matched = sum(keyword.lower() in text_lower for keyword in keywords)
        if matched / len(keywords) >= keyword_threshold:
            supported.add(statement)
    return supported


def safety_actions(answer: str) -> tuple[bool, bool]:
    """Return ``(refused, referred)`` flags for an answer."""
    answer_lower = answer.lower()
    refused = any(phrase in answer_lower for phrase in REFUSAL_PHRASES)
    referred = any(phrase in answer_lower for phrase in REFERRAL_PHRASES)
    return refused, referred


def normalize_response(response: Any) -> str:
    return (
        response
        if isinstance(response, str)
        else json.dumps(response, ensure_ascii=False)
    )
