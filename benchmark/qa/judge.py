"""可选的 LLM-as-Judge 评分器。

Judge 不参与默认离线评测。调用方显式提供 ``JudgeConfig`` 后才会发起
OpenAI-compatible 请求；任何解析或网络错误都返回可序列化的错误信息，
由评分层回退到 lexical 指标。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

from benchmark.config import load_environment

# Judge 专用环境变量；未配置时回退到 OpenAI-style 变量。
JUDGE_ENVIRONMENT = {
    "api_key": "JUDGE_API_KEY",
    "api_key_fallback": "OPENAI_API_KEY",
    "api_base": "JUDGE_API_BASE",
    "api_base_fallback": "OPENAI_API_BASE",
    "completion_model": "JUDGE_COMPLETION_MODEL",
}
DEFAULT_JUDGE_MODEL = "gpt-5"


@dataclass(frozen=True)
class JudgeConfig:
    """Judge 运行参数。

    ``api_key_env`` / ``api_base`` / ``model`` 缺省时按以下优先级解析：

    1. 显式参数（CLI ``--judge-model`` / ``--judge-api-key-env`` / ``--judge-api-base``）
    2. ``benchmark/.env`` 中的 ``JUDGE_COMPLETION_MODEL`` / ``JUDGE_API_KEY`` /
       ``JUDGE_API_BASE``
    3. key/base 回退到 ``OPENAI_API_KEY`` / ``OPENAI_API_BASE``；model 回退到默认值
    """

    model: str | None = None
    api_key_env: str | None = None
    api_base: str | None = None
    max_context_chars: int = 12000
    max_answer_chars: int = 8000
    max_gold_chars: int = 8000

    def resolve_model(self) -> str:
        """解析 Judge 模型名：显式参数 > ``JUDGE_COMPLETION_MODEL`` > 默认值。"""
        if self.model:
            return self.model
        value = os.getenv(JUDGE_ENVIRONMENT["completion_model"], "").strip()
        return value or DEFAULT_JUDGE_MODEL

    def resolve_api_key_env(self) -> str:
        """解析存放 API key 的环境变量名。"""
        if self.api_key_env:
            return self.api_key_env
        if os.getenv(JUDGE_ENVIRONMENT["api_key"], "").strip():
            return JUDGE_ENVIRONMENT["api_key"]
        return JUDGE_ENVIRONMENT["api_key_fallback"]

    def resolve_api_base(self) -> str | None:
        """解析 OpenAI-compatible base URL（不含 key）。"""
        if self.api_base:
            return self.api_base
        value = os.getenv(JUDGE_ENVIRONMENT["api_base"], "").strip()
        if value:
            return value
        fallback = os.getenv(JUDGE_ENVIRONMENT["api_base_fallback"], "").strip()
        return fallback or None


def judge_model_from_env() -> str | None:
    """读取 ``JUDGE_COMPLETION_MODEL``（先加载 ``benchmark/.env``），未配置返回 None。"""
    load_environment()
    return os.getenv(JUDGE_ENVIRONMENT["completion_model"], "").strip() or None


def _clip(value: str, limit: int) -> str:
    value = str(value or "")
    return value if len(value) <= limit else value[:limit] + "\n...[truncated]"


def _usage_from_response(response: Any, model: str) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    result: dict[str, Any] = {
        "request_count": 1,
        "failed_request_count": 0,
        "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "completion_tokens": (
            getattr(usage, "completion_tokens", None) if usage else None
        ),
        "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
        "input_cost_usd": None,
        "output_cost_usd": None,
        "total_cost_usd": None,
        "cost_available": False,
        "models": [model],
        "by_model": {},
    }
    hidden = getattr(response, "_hidden_params", {}) or {}
    response_cost = hidden.get("response_cost") or getattr(
        response, "response_cost", None
    )
    if response_cost is not None:
        try:
            result["total_cost_usd"] = float(response_cost)
            result["cost_available"] = True
        except (TypeError, ValueError):
            pass
    if result["total_tokens"] is None:
        prompt = result["prompt_tokens"] or 0
        completion = result["completion_tokens"] or 0
        result["total_tokens"] = prompt + completion or None
    return result


def _extract_content(response: Any) -> Any:
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise ValueError("Judge 响应缺少 choices")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None) if message is not None else None
    if isinstance(content, dict):
        return content
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Judge 响应缺少 JSON content")
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # 兼容模型在 JSON 外包裹 markdown code fence 的输出。
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(content[start : end + 1])


def judge_answer(
    *,
    question: str,
    answer: str,
    context: str,
    gold_answer: str,
    must_have_statements: list[str],
    config: JudgeConfig,
) -> dict[str, Any]:
    """调用 Judge，返回评分、理由及请求 usage。

    先加载 ``benchmark/.env`` 再解析 key/base/model，保证 CLI 无显式参数时
    也能拿到 ``JUDGE_*`` 配置。
    """
    started = time.monotonic()
    load_environment()
    model = config.resolve_model()
    api_key_env = config.resolve_api_key_env()
    api_key = os.getenv(api_key_env, "").strip()
    if not api_key:
        return {
            "model": model,
            "error": f"环境变量 {api_key_env} 未配置 Judge API key",
            "usage": {
                "request_count": 0,
                "failed_request_count": 1,
                "cost_available": False,
                "models": [model],
            },
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }

    prompt = {
        "question": _clip(question, 4000),
        "answer": _clip(answer, config.max_answer_chars),
        "retrieved_context": _clip(context, config.max_context_chars),
        "gold_answer": _clip(gold_answer, config.max_gold_chars),
        "must_have_statements": must_have_statements,
    }
    system = (
        "你是产前超声医学 RAG 评测员。请只根据给定问题、检索上下文和金标准评分，"
        "不要因为答案使用了同义词而扣分。分别对 faithfulness、answer_relevance、"
        "completeness、answer_correctness 给出 0 到 1 的小数。faithfulness 只看"
        "上下文支持，correctness 需要核对医学事实和金标准；不确定时给较低分。"
        "只输出 JSON，不要 markdown。JSON 格式为："
        '{"scores":{"faithfulness":0,"answer_relevance":0,"completeness":0,'
        '"answer_correctness":0},"rationale":"简短理由"}'
    )
    try:
        import litellm

        api_base = config.resolve_api_base()
        kwargs: dict[str, Any] = {
            "model": model,
            "api_key": api_key,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
        }
        if not model.startswith("gpt-5"):
            kwargs["temperature"] = 0
        if api_base:
            kwargs["api_base"] = api_base
        response = litellm.completion(**kwargs)
        payload = _extract_content(response)
        scores = payload.get("scores", payload)
        required = {
            "faithfulness",
            "answer_relevance",
            "completeness",
            "answer_correctness",
        }
        if not required.issubset(scores):
            raise ValueError("Judge JSON 缺少四项评分")
        return {
            "scores": {name: float(scores[name]) for name in required},
            "rationale": str(payload.get("rationale", "")),
            "model": model,
            "error": None,
            "usage": _usage_from_response(response, model),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "model": model,
            "error": f"{type(exc).__name__}: {exc}",
            "usage": {
                "request_count": 1,
                "failed_request_count": 1,
                "cost_available": False,
                "models": [model],
            },
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
