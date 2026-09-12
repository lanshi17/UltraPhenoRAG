"""Configuration compatibility exports for the PathRAG baseline."""

from benchmark.baseline.pathrag_client.config.llm_config import (
    DEFAULT_COMPLETION_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    LLMConfigOverrides,
    ModelConfigOverride,
)

__all__ = [
    "DEFAULT_COMPLETION_MODEL",
    "DEFAULT_EMBEDDING_MODEL",
    "LLMConfigOverrides",
    "ModelConfigOverride",
]
