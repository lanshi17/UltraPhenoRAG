"""Usage accounting shared by the query and judge pipelines."""

from __future__ import annotations

from typing import Any, Sequence


def empty_usage() -> dict[str, Any]:
    """Shape of a usage record before any request has been observed."""
    return {
        "request_count": 0,
        "failed_request_count": 0,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "input_cost_usd": None,
        "output_cost_usd": None,
        "total_cost_usd": None,
        "cost_available": False,
        "models": [],
        "by_model": {},
    }


def merge_usage(*usages: dict[str, Any]) -> dict[str, Any]:
    """Merge query/Judge usage; unknown tokens/cost stay None, never 0."""
    numeric_fields = (
        "request_count",
        "failed_request_count",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_cost_usd",
        "output_cost_usd",
        "total_cost_usd",
    )
    result = empty_usage()
    for field in numeric_fields:
        values = [item.get(field) for item in usages if item.get(field) is not None]
        if values:
            result[field] = sum(float(value) for value in values)
            if field.endswith("tokens"):
                result[field] = int(result[field])
    result["cost_available"] = bool(usages) and all(
        item.get("cost_available") is True
        for item in usages
        if item.get("request_count", 0) or item.get("total_tokens") is not None
    )
    result["models"] = sorted(
        {str(model) for item in usages for model in item.get("models", [])}
    )
    return result


def aggregate_usage(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-question usage, preserving visibility of missing costs."""
    result: dict[str, Any] = {
        "recorded_query_count": 0,
        "recorded_judge_count": 0,
        "cost_missing_count": 0,
        "query": empty_usage(),
        "judge": empty_usage(),
        "total": empty_usage(),
    }
    for scope in ("query", "judge", "total"):
        values = [
            row.get("usage", {}).get(scope)
            for row in rows
            if isinstance(row.get("usage", {}).get(scope), dict)
        ]
        result[scope] = merge_usage(*values) if values else empty_usage()
    result["recorded_query_count"] = sum(
        isinstance(row.get("usage", {}).get("query"), dict) for row in rows
    )
    result["recorded_judge_count"] = sum(
        isinstance(row.get("usage", {}).get("judge"), dict) for row in rows
    )
    result["cost_missing_count"] = sum(
        isinstance(row.get("usage", {}).get("total"), dict)
        and not row["usage"]["total"].get("cost_available", False)
        for row in rows
    )
    return result
