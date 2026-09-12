"""Per-operation LLM token, latency, and configurable cost accounting."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from threading import Lock
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


def _coerce_token_count(value: Any) -> int | None:
    """Return a non-negative integral token count, or ``None`` if unknown.

    Provider and test doubles do not always agree on the concrete token-count
    type.  In particular, strings and floating-point JSON values are common,
    while ``None``, booleans, non-finite values, and malformed values should
    not turn telemetry collection itself into an operation failure.
    """

    if value is None or isinstance(value, bool):
        return None
    try:
        count = int(value)
    except (TypeError, ValueError, OverflowError):
        try:
            numeric_value = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(numeric_value):
            return None
        count = int(numeric_value)
    return max(0, count)


@dataclass
class OperationUsageTracker:
    """Collect token usage for one KAG operation.

    KAG accounts tokens per process through its in-memory ``TokenMeter``; the
    tracker snapshots those counters around an operation and converts the
    delta into the telemetry shape shared by all baseline clients.
    """

    model: str
    rates: TokenRates = field(default_factory=TokenRates.from_environment)
    _started: float = field(default_factory=time.monotonic, init=False)
    _request_count: int = field(default=0, init=False)
    _failed_request_count: int = field(default=0, init=False)
    _prompt_tokens: int = field(default=0, init=False)
    _completion_tokens: int = field(default=0, init=False)
    _total_tokens: int = field(default=0, init=False)
    _has_tokens: bool = field(default=False, init=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def start_request(self) -> None:
        with self._lock:
            self._request_count += 1

    def fail_request(self) -> None:
        """Record one attempted provider request that raised an exception."""

        with self._lock:
            self._failed_request_count += 1

    def add_usage(self, token_counts: Mapping[str, Any] | None) -> None:
        """Add provider token usage without letting malformed telemetry fail work."""

        if not isinstance(token_counts, Mapping):
            return

        prompt_value = _coerce_token_count(token_counts.get("prompt_tokens"))
        completion_value = _coerce_token_count(token_counts.get("completion_tokens"))
        total_value = _coerce_token_count(token_counts.get("total_tokens"))
        if prompt_value is None and completion_value is None and total_value is None:
            return

        with self._lock:
            self._has_tokens = True
            prompt = prompt_value or 0
            completion = completion_value or 0
            total = total_value if total_value is not None else prompt + completion
            self._prompt_tokens += prompt
            self._completion_tokens += completion
            self._total_tokens += total

    def telemetry(self) -> dict[str, Any]:
        with self._lock:
            prompt = self._prompt_tokens if self._has_tokens else None
            completion = self._completion_tokens if self._has_tokens else None
            total = self._total_tokens if self._has_tokens else None
            request_count = self._request_count
            failed_request_count = self._failed_request_count
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
        total_cost = (
            input_cost + output_cost
            if input_cost is not None and output_cost is not None
            else None
        )
        model_usage = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
            "input_cost_usd": input_cost,
            "output_cost_usd": output_cost,
            "total_cost_usd": total_cost,
            "cost_available": cost_available,
        }
        usage: dict[str, Any] = {
            "request_count": request_count,
            "failed_request_count": failed_request_count,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
            "input_cost_usd": input_cost,
            "output_cost_usd": output_cost,
            "total_cost_usd": total_cost,
            "cost_available": cost_available,
            "models": [self.model] if self.model else [],
            "by_model": {self.model: model_usage} if self.model else {},
        }
        if self.model:
            # Older KAG callers read model usage directly from the top-level
            # usage mapping.  Keep that compatibility alongside ``by_model``.
            usage[self.model] = dict(model_usage)
        return {
            "elapsed_seconds": round(time.monotonic() - self._started, 3),
            "usage": usage,
        }


def snapshot_token_totals() -> dict[str, int]:
    """Return KAG's process-wide token usage across all task meters."""

    from kag.interface.common.llm_client import TokenMeterFactory

    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for meter in TokenMeterFactory().get_all_meters().values():
        data = meter.to_dict()
        if not isinstance(data, Mapping):
            continue
        for token_field in totals:
            totals[token_field] += _coerce_token_count(data.get(token_field)) or 0
    return totals


__all__ = [
    "INPUT_COST_ENV",
    "OUTPUT_COST_ENV",
    "OperationUsageTracker",
    "TokenRates",
    "snapshot_token_totals",
]
