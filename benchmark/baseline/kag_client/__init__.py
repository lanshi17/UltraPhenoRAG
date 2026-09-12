"""KAG baseline client package (vendored KAG 0.8)."""

from .client import KAGClient
from .config.llm_config import (
    DEFAULT_COMPLETION_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    LLMConfigOverrides,
    ModelConfigOverride,
)
from .index_methods import IndexMethod
from .models import IndexResult, QueryResult
from .search_methods import SearchMethod
from .states import ClientState

__all__ = [
    "DEFAULT_COMPLETION_MODEL",
    "DEFAULT_EMBEDDING_MODEL",
    "ClientState",
    "IndexMethod",
    "IndexResult",
    "KAGClient",
    "LLMConfigOverrides",
    "ModelConfigOverride",
    "QueryResult",
    "SearchMethod",
]
