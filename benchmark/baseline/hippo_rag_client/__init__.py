"""HippoRAG 2 baseline client for the prenatal GraphRAG benchmark."""

from .client import (
    DEFAULT_SAVE_DIR,
    DEFAULT_TOP_K,
    HippoRAGClient,
    IndexResult,
    QueryResult,
)
from .documents import CorpusChunk, load_corpus_chunks

__all__ = [
    "DEFAULT_SAVE_DIR",
    "DEFAULT_TOP_K",
    "CorpusChunk",
    "HippoRAGClient",
    "IndexResult",
    "QueryResult",
    "load_corpus_chunks",
]
