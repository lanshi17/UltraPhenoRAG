"""Per-operation LLM token, latency, and configurable cost accounting."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

INPUT_COST_ENV = "RAG_INPUT_COST_PER_MILLION_TOKENS_USD"
OUTPUT_COST_ENV = "RAG_OUTPUT_COST_PER_MILLION_TOKENS_USD"


@dataclass(frozen=True)
class TokenRates:
    input_per_million_usd: float | None = None
    output_per_million_usd: float | None = None

    @classmethod
    def from_environment(cls) -> "TokenRates":
        return cls(
            input_per_million_usd=_read_rate(INPUT_COST_ENV),
            output_per_million_usd=_read_rate(OUTPUT_COST_ENV),
        )


def _read_rate(name: str) -> float | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    try:
        rate = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative number") from exc
    if rate < 0:
        raise ValueError(f"{name} must be a non-negative number")
    return rate


@dataclass
class OperationUsageTracker:
    """Collect API-reported token usage for one LightRAG operation."""

    model: str
    rates: TokenRates = field(default_factory=TokenRates.from_environment)
    _started: float = field(default_factory=time.monotonic, init=False)
    _request_count: int = field(default=0, init=False)
    _failed_request_count: int = field(default=0, init=False)
    _prompt_tokens: int = field(default=0, init=False)
    _completion_tokens: int = field(default=0, init=False)
    _total_tokens: int = field(default=0, init=False)
    _has_tokens: bool = field(default=False, init=False)

    def start_request(self) -> None:
        self._request_count += 1

    def fail_request(self) -> None:
        self._failed_request_count += 1

    def add_usage(self, token_counts: dict[str, Any]) -> None:
        self._has_tokens = True
        prompt = token_counts.get("prompt_tokens", 0)
        completion = token_counts.get("completion_tokens", 0)
        total = token_counts.get("total_tokens", prompt + completion)
        self._prompt_tokens += int(prompt or 0)
        self._completion_tokens += int(completion or 0)
        self._total_tokens += int(total or 0)

    def telemetry(self) -> dict[str, Any]:
        prompt = self._prompt_tokens if self._has_tokens else None
        completion = self._completion_tokens if self._has_tokens else None
        total = self._total_tokens if self._has_tokens else None
        cost_available = (
            self._has_tokens
            and self.rates.input_per_million_usd is not None
            and self.rates.output_per_million_usd is not None
        )
        input_cost = (
            prompt * self.rates.input_per_million_usd / 1_000_000
            if cost_available and prompt is not None
            else None
        )
        output_cost = (
            completion * self.rates.output_per_million_usd / 1_000_000
            if cost_available and completion is not None
            else None
        )
        usage = {
            "request_count": self._request_count,
            "failed_request_count": self._failed_request_count,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
            "input_cost_usd": input_cost,
            "output_cost_usd": output_cost,
            "total_cost_usd": (
                input_cost + output_cost
                if input_cost is not None and output_cost is not None
                else None
            ),
            "cost_available": cost_available,
            "models": [self.model] if self.model else [],
            "by_model": {
                self.model: {
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                    "total_tokens": total,
                    "input_cost_usd": input_cost,
                    "output_cost_usd": output_cost,
                    "total_cost_usd": (
                        input_cost + output_cost
                        if input_cost is not None and output_cost is not None
                        else None
                    ),
                    "cost_available": cost_available,
                }
            }
            if self.model
            else {},
        }
        return {
            "elapsed_seconds": round(time.monotonic() - self._started, 3),
            "usage": usage,
        }
