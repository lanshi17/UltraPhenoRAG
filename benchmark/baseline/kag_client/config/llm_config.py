"""Provider override data classes for the KAG baseline.

The field names intentionally mirror the Microsoft and LightRAG baselines,
but this module has no GraphRAG/LightRAG dependency so the KAG client remains
independently importable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from benchmark.config import (
    DEFAULT_COMPLETION_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    load_environment,
)

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class ModelConfigOverride:
    model: str | None = None
    model_env: str | None = None
    model_provider: str | None = None
    type: str | None = None
    api_base: str | None = None
    api_base_env: str | None = None
    api_version: str | None = None
    api_key_env: str | None = None
    azure_deployment_name: str | None = None
    call_args: Mapping[str, Any] | None = None

    def is_empty(self) -> bool:
        return not any(
            value is not None
            for value in (
                self.model,
                self.model_env,
                self.model_provider,
                self.type,
                self.api_base,
                self.api_base_env,
                self.api_version,
                self.api_key_env,
                self.azure_deployment_name,
                self.call_args,
            )
        )

    def validate(self) -> None:
        for name in (self.model_env, self.api_base_env, self.api_key_env):
            if name and not _ENV_NAME.fullmatch(name):
                raise ValueError(f"invalid environment variable name: {name}")


@dataclass(frozen=True)
class LLMConfigOverrides:
    completion: ModelConfigOverride = field(default_factory=ModelConfigOverride)
    embedding: ModelConfigOverride = field(default_factory=ModelConfigOverride)

    def is_empty(self) -> bool:
        return self.completion.is_empty() and self.embedding.is_empty()

    @property
    def requires_environment(self) -> bool:
        return bool(self.completion.api_key_env or self.embedding.api_key_env)


def load_project_env() -> None:
    """Load the shared ``benchmark/.env`` file."""
    load_environment()


__all__ = [
    "DEFAULT_COMPLETION_MODEL",
    "DEFAULT_EMBEDDING_MODEL",
    "LLMConfigOverrides",
    "ModelConfigOverride",
    "load_project_env",
]
