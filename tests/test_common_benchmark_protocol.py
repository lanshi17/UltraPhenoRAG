"""Tests for controlled cross-framework benchmark comparisons."""

from __future__ import annotations

import pytest

from benchmark.common.benchmark_protocol import assert_comparable_conditions


def test_matching_conditions_are_comparable() -> None:
    conditions = {"dataset_sha256": "dataset", "top_k": 16}

    assert_comparable_conditions(conditions, dict(conditions))


def test_mismatched_conditions_are_rejected() -> None:
    with pytest.raises(ValueError, match="top_k"):
        assert_comparable_conditions(
            {"dataset_sha256": "dataset", "top_k": 16},
            {"dataset_sha256": "dataset", "top_k": 8},
        )
