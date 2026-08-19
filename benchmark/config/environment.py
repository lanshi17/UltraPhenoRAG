"""Framework-neutral environment configuration for both RAG baselines."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

RAG_ENVIRONMENT = {
    "provider": "RAG_MODEL_PROVIDER",
    "api_key": "RAG_API_KEY",
    "api_base": "RAG_API_BASE",
    "completion_model": "RAG_COMPLETION_MODEL",
    "embedding_model": "RAG_EMBEDDING_MODEL",
    "embedding_dimension": "RAG_EMBEDDING_DIMENSION",
    "embedding_max_token_size": "RAG_EMBEDDING_MAX_TOKEN_SIZE",
    "api_version": "RAG_API_VERSION",
}

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ENV_FILE = REPOSITORY_ROOT / "benchmark" / ".env"

DEFAULT_COMPLETION_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSION = 1536
DEFAULT_EMBEDDING_MAX_TOKEN_SIZE = 8192


@dataclass(frozen=True)
class RagEnvironment:
    """Resolved provider settings shared by Microsoft GraphRAG and LightRAG."""

    provider: str
    api_key: str
    api_base: str
    completion_model: str
    embedding_model: str
    embedding_dimension: int
    embedding_max_token_size: int
    api_version: str

    @classmethod
    def from_process(cls) -> "RagEnvironment":
        def value(name: str, default: str = "") -> str:
            return os.getenv(name, default).strip()

        return cls(
            provider=value(RAG_ENVIRONMENT["provider"], "openai").casefold(),
            api_key=value(RAG_ENVIRONMENT["api_key"]),
            api_base=value(RAG_ENVIRONMENT["api_base"]),
            completion_model=value(
                RAG_ENVIRONMENT["completion_model"], DEFAULT_COMPLETION_MODEL
            ),
            embedding_model=value(
                RAG_ENVIRONMENT["embedding_model"], DEFAULT_EMBEDDING_MODEL
            ),
            embedding_dimension=int(
                value(
                    RAG_ENVIRONMENT["embedding_dimension"],
                    str(DEFAULT_EMBEDDING_DIMENSION),
                )
            ),
            embedding_max_token_size=int(
                value(
                    RAG_ENVIRONMENT["embedding_max_token_size"],
                    str(DEFAULT_EMBEDDING_MAX_TOKEN_SIZE),
                )
            ),
            api_version=value(RAG_ENVIRONMENT["api_version"]),
        )


def load_environment() -> RagEnvironment:
    """Load the single shared environment file at ``benchmark/.env``."""
    load_dotenv(BENCHMARK_ENV_FILE, override=False)
    return RagEnvironment.from_process()
