"""Enums shared by the LightRAG benchmark client."""

from __future__ import annotations

from enum import Enum


class SearchMethod(str, Enum):
    GLOBAL = "global"
    LOCAL = "local"
    HYBRID = "hybrid"
    MIX = "mix"
    NAIVE = "naive"
    BASIC = "basic"
    DRIFT = "drift"

    def __str__(self) -> str:
        return self.value


class IndexMethod(str, Enum):
    STANDARD = "standard"
    FAST = "fast"
    STANDARD_UPDATE = "standard-update"
    FAST_UPDATE = "fast-update"

    def __str__(self) -> str:
        return self.value


DEFAULT_RESPONSE_TYPE = "multiple paragraphs"
DEFAULT_TOP_K = 40
DEFAULT_CHUNK_TOP_K = 20


__all__ = [
    "DEFAULT_CHUNK_TOP_K",
    "DEFAULT_RESPONSE_TYPE",
    "DEFAULT_TOP_K",
    "IndexMethod",
    "SearchMethod",
]
