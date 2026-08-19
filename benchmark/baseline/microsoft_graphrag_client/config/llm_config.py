"""GraphRAG LLM 供应商配置覆盖。

GraphRAG 使用 ``graphrag-llm``/LiteLLM 创建模型。该模块只负责把命令行或
Python 调用中的供应商参数转换成 GraphRAG 能识别的嵌套配置，不把 API key
写入 settings.yaml。
"""

from __future__ import annotations

import os
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping

from benchmark.config import load_environment

DEFAULT_COMPLETION_MODEL = "gpt-4.1"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"

_ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SENSITIVE_KEYS = {"api_key", "password", "secret", "token"}


@dataclass(frozen=True)
class ModelConfigOverride:
    """单个 completion 或 embedding 模型的可选覆盖项。

    ``model_provider`` 是 LiteLLM provider 名称，例如 ``openai``、``azure``
    或 LiteLLM 支持的其他 provider。OpenAI-compatible 网关通常仍使用
    ``openai``，并通过 ``api_base`` 指定网关地址。
    """

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
        """是否没有设置任何覆盖项。"""
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

    def _validate_env_names(self) -> None:
        for name in (self.api_key_env, self.api_base_env, self.model_env):
            if name and not _ENV_NAME_PATTERN.fullmatch(name):
                raise ValueError("环境变量名必须是合法标识符，例如 GRAPHRAG_API_KEY")

    def to_settings_dict(self) -> dict[str, Any]:
        """转换为可写入 settings.yaml 的配置。

        API key 只写入环境变量引用，不写入实际密钥。
        """
        self._validate_env_names()
        result: dict[str, Any] = {}
        field_values = {
            "model": self.model,
            "model_provider": self.model_provider,
            "type": self.type,
            "api_base": self.api_base,
            "api_version": self.api_version,
            "azure_deployment_name": self.azure_deployment_name,
        }
        result.update(
            {key: value for key, value in field_values.items() if value is not None}
        )
        if self.model_env:
            result["model"] = f"${{{self.model_env}}}"
        if self.api_key_env:
            result["api_key"] = f"${{{self.api_key_env}}}"
        if self.api_base_env:
            result["api_base"] = f"${{{self.api_base_env}}}"
        if self.call_args is not None:
            result["call_args"] = deepcopy(dict(self.call_args))
        return result

    def to_runtime_dict(self) -> dict[str, Any]:
        """转换为传给 ``load_config`` 的运行时配置。"""
        result = self.to_settings_dict()
        if self.model_env:
            result["model"] = os.getenv(self.model_env, "")
        if self.api_key_env:
            # 缺失时保留空字符串，让 preflight 给出可读的凭据错误，而不是
            # 在配置覆盖阶段直接泄露或抛出不明确的异常。
            result["api_key"] = os.getenv(self.api_key_env, "")
        if self.api_base_env:
            result["api_base"] = os.getenv(self.api_base_env, "")
        return result


def load_project_env() -> None:
    """加载共享的 ``benchmark/.env``。"""
    load_environment()


@dataclass(frozen=True)
class LLMConfigOverrides:
    """completion 和 embedding 的统一配置覆盖。"""

    completion: ModelConfigOverride = field(default_factory=ModelConfigOverride)
    embedding: ModelConfigOverride = field(default_factory=ModelConfigOverride)

    @property
    def requires_environment(self) -> bool:
        """是否需要先加载项目 ``.env`` 才能解析 API key。"""
        return bool(self.completion.api_key_env or self.embedding.api_key_env)

    def is_empty(self) -> bool:
        return self.completion.is_empty() and self.embedding.is_empty()

    def to_settings_overrides(self) -> dict[str, Any]:
        """生成可持久化到 GraphRAG settings 的嵌套覆盖。"""
        result: dict[str, Any] = {}
        if not self.completion.is_empty():
            result["completion_models"] = {
                "default_completion_model": self.completion.to_settings_dict()
            }
        if not self.embedding.is_empty():
            result["embedding_models"] = {
                "default_embedding_model": self.embedding.to_settings_dict()
            }
        return result

    def to_runtime_overrides(self) -> dict[str, Any]:
        """生成传给 GraphRAG ``load_config`` 的运行时覆盖。"""
        result: dict[str, Any] = {}
        if not self.completion.is_empty():
            result["completion_models"] = {
                "default_completion_model": self.completion.to_runtime_dict()
            }
        if not self.embedding.is_empty():
            result["embedding_models"] = {
                "default_embedding_model": self.embedding.to_runtime_dict()
            }
        return result


def merge_overrides(
    *overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """递归合并多个 GraphRAG 配置覆盖，后面的值优先。"""
    result: dict[str, Any] = {}

    def merge(destination: dict[str, Any], source: Mapping[str, Any]) -> None:
        for key, value in source.items():
            if isinstance(value, Mapping) and isinstance(destination.get(key), dict):
                merge(destination[key], value)
            elif isinstance(value, Mapping):
                destination[key] = deepcopy(dict(value))
            else:
                destination[key] = deepcopy(value)

    for override in overrides:
        if override:
            merge(result, override)
    return result


def redact_overrides(value: Any, key: str | None = None) -> Any:
    """递归脱敏配置日志中的密钥。"""
    if key and key.casefold() in _SENSITIVE_KEYS:
        return "<redacted>"
    if isinstance(value, Mapping):
        return {
            str(item_key): redact_overrides(item_value, str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [redact_overrides(item) for item in value]
    return value


__all__ = [
    "DEFAULT_COMPLETION_MODEL",
    "DEFAULT_EMBEDDING_MODEL",
    "LLMConfigOverrides",
    "ModelConfigOverride",
    "load_project_env",
    "merge_overrides",
    "redact_overrides",
]
