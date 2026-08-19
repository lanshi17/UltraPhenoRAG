"""Tests for shared usage accounting."""

from __future__ import annotations

from benchmark.common.usage import aggregate_usage, empty_usage, merge_usage


class TestEmptyUsage:
    def test_defaults_keep_unknown_values_as_none(self) -> None:
        usage = empty_usage()

        assert usage["prompt_tokens"] is None
        assert usage["total_cost_usd"] is None
        assert usage["request_count"] == 0
        assert usage["cost_available"] is False


class TestMergeUsage:
    def test_sums_numeric_fields(self) -> None:
        merged = merge_usage(
            {"request_count": 2, "prompt_tokens": 10, "total_tokens": 20},
            {"request_count": 3, "prompt_tokens": 15, "total_tokens": 30},
        )

        assert merged["request_count"] == 5
        assert merged["prompt_tokens"] == 25
        assert merged["total_tokens"] == 50

    def test_missing_values_stay_none(self) -> None:
        merged = merge_usage({"request_count": 1}, {"request_count": 2})

        assert merged["prompt_tokens"] is None
        assert merged["total_cost_usd"] is None

    def test_cost_available_requires_all_sides(self) -> None:
        with_cost = {
            "request_count": 1,
            "total_cost_usd": 0.1,
            "cost_available": True,
        }
        without_cost = {"request_count": 1}

        assert merge_usage(with_cost, with_cost)["cost_available"] is True
        assert merge_usage(with_cost, without_cost)["cost_available"] is False

    def test_models_are_deduplicated(self) -> None:
        merged = merge_usage(
            {"models": ["gpt-5", "gpt-5-mini"]},
            {"models": ["gpt-5"]},
        )

        assert merged["models"] == ["gpt-5", "gpt-5-mini"]


class TestAggregateUsage:
    def test_counts_scopes_per_row(self) -> None:
        rows = [
            {
                "usage": {
                    "query": {"request_count": 1},
                    "judge": {"request_count": 2},
                    "total": {"request_count": 3},
                }
            },
            {"usage": {"query": {"request_count": 1}}},
        ]

        result = aggregate_usage(rows)

        assert result["recorded_query_count"] == 2
        assert result["recorded_judge_count"] == 1
        assert result["query"]["request_count"] == 2
        assert result["judge"]["request_count"] == 2

    def test_cost_missing_count_flags_unavailable_costs(self) -> None:
        rows = [
            {"usage": {"total": {"request_count": 1, "cost_available": False}}},
            {"usage": {"total": {"request_count": 1, "cost_available": True}}},
        ]

        result = aggregate_usage(rows)

        assert result["cost_missing_count"] == 1
