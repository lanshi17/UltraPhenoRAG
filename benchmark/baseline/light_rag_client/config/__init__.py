"""Configuration compatibility exports for the LightRAG baseline."""

from benchmark.baseline.light_rag_client.config.llm_config import (
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
