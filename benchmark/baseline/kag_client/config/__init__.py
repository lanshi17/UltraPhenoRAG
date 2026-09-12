"""KAG baseline client configuration package."""

from .kag_project import (
    DEFAULT_API_KEY_ENV,
    build_kag_config,
    initialize_kag_config,
    resolve_model_settings,
    substitute_instances,
    write_config_file,
)
from .llm_config import (
    DEFAULT_COMPLETION_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    LLMConfigOverrides,
    ModelConfigOverride,
    load_project_env,
)

__all__ = [
    "DEFAULT_API_KEY_ENV",
    "DEFAULT_COMPLETION_MODEL",
    "DEFAULT_EMBEDDING_MODEL",
    "LLMConfigOverrides",
    "ModelConfigOverride",
    "build_kag_config",
    "initialize_kag_config",
    "load_project_env",
    "resolve_model_settings",
    "substitute_instances",
    "write_config_file",
]
