"""Judge 环境变量解析（JUDGE_* / OPENAI_* 回退）测试。"""

from __future__ import annotations

import pytest

from benchmark.qa import (
    DEFAULT_JUDGE_MODEL,
    JUDGE_ENVIRONMENT,
    JudgeConfig,
    judge_model_from_env,
)

ENV_KEYS = (
    "JUDGE_API_KEY",
    "JUDGE_API_BASE",
    "JUDGE_COMPLETION_MODEL",
    "OPENAI_API_KEY",
    "OPENAI_API_BASE",
)


@pytest.fixture()
def clean_judge_env(monkeypatch: pytest.MonkeyPatch):
    """隔离真实 benchmark/.env 与进程环境中的 Judge 相关变量。"""
    # 阻止 judge 模块加载开发机上的真实 benchmark/.env。
    monkeypatch.setattr(
        "benchmark.qa.judge.load_environment", lambda: None, raising=True
    )
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


class TestResolveModel:
    def test_explicit_model_wins(self, clean_judge_env) -> None:
        clean_judge_env.setenv("JUDGE_COMPLETION_MODEL", "env-model")
        config = JudgeConfig(model="cli-model")
        assert config.resolve_model() == "cli-model"

    def test_env_model_used_when_no_explicit(self, clean_judge_env) -> None:
        clean_judge_env.setenv("JUDGE_COMPLETION_MODEL", "env-model")
        assert JudgeConfig().resolve_model() == "env-model"

    def test_default_when_nothing_configured(self, clean_judge_env) -> None:
        assert JudgeConfig().resolve_model() == DEFAULT_JUDGE_MODEL

    def test_judge_model_from_env_missing(self, clean_judge_env) -> None:
        assert judge_model_from_env() is None

    def test_judge_model_from_env_present(self, clean_judge_env) -> None:
        clean_judge_env.setenv("JUDGE_COMPLETION_MODEL", "env-model")
        assert judge_model_from_env() == "env-model"


class TestResolveApiKeyEnv:
    def test_explicit_env_name_wins(self, clean_judge_env) -> None:
        clean_judge_env.setenv("JUDGE_API_KEY", "judge-key")
        config = JudgeConfig(api_key_env="CUSTOM_KEY")
        assert config.resolve_api_key_env() == "CUSTOM_KEY"

    def test_judge_key_preferred_over_openai(self, clean_judge_env) -> None:
        clean_judge_env.setenv("JUDGE_API_KEY", "judge-key")
        clean_judge_env.setenv("OPENAI_API_KEY", "openai-key")
        assert JudgeConfig().resolve_api_key_env() == "JUDGE_API_KEY"

    def test_fallback_to_openai_when_judge_missing(self, clean_judge_env) -> None:
        clean_judge_env.setenv("OPENAI_API_KEY", "openai-key")
        assert (
            JudgeConfig().resolve_api_key_env() == JUDGE_ENVIRONMENT["api_key_fallback"]
        )

    def test_fallback_when_judge_empty(self, clean_judge_env) -> None:
        clean_judge_env.setenv("JUDGE_API_KEY", "  ")
        assert (
            JudgeConfig().resolve_api_key_env() == JUDGE_ENVIRONMENT["api_key_fallback"]
        )


class TestResolveApiBase:
    def test_explicit_base_wins(self, clean_judge_env) -> None:
        clean_judge_env.setenv("JUDGE_API_BASE", "https://judge.example.com/v1")
        config = JudgeConfig(api_base="https://cli.example.com/v1")
        assert config.resolve_api_base() == "https://cli.example.com/v1"

    def test_judge_base_preferred_over_openai(self, clean_judge_env) -> None:
        clean_judge_env.setenv("JUDGE_API_BASE", "https://judge.example.com/v1")
        clean_judge_env.setenv("OPENAI_API_BASE", "https://openai.example.com/v1")
        assert JudgeConfig().resolve_api_base() == "https://judge.example.com/v1"

    def test_fallback_to_openai_base(self, clean_judge_env) -> None:
        clean_judge_env.setenv("OPENAI_API_BASE", "https://openai.example.com/v1")
        assert JudgeConfig().resolve_api_base() == "https://openai.example.com/v1"

    def test_none_when_unconfigured(self, clean_judge_env) -> None:
        assert JudgeConfig().resolve_api_base() is None

    def test_empty_string_treated_as_none(self, clean_judge_env) -> None:
        clean_judge_env.setenv("JUDGE_API_BASE", "")
        assert JudgeConfig().resolve_api_base() is None
