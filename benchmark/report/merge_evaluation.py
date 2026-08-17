"""Merge partial benchmark reruns into an evaluation result.

The merge is deliberately question-id based.  A rerun may replace only a
failed row from the original run; rows from another model must not be used as
fallbacks because that would make the aggregate incomparable.

Example
-------
.. code-block:: bash

    uv run python -m benchmark.report.merge_evaluation \
        --base benchmark/results/microsoft_graphrag/evaluation-20260815T133456Z.json \
        --retry benchmark/data/microsoft_graphrag/benchmark/results/microsoft_graphrag/evaluation-gpt5-retry.jsonl \
        --output benchmark/results/microsoft_graphrag/evaluation-gpt5-merged-20260817.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from benchmark.qa.scoring import (
    DatasetScoringReport,
    GenerationMetrics,
    JudgeMetrics,
    RetrievalMetrics,
    SafetyMetrics,
    ScoringResult,
    SourceMatchMetrics,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSONL 第 {line_number} 行无法解析: {path}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL 第 {line_number} 行不是对象: {path}")
        rows.append(value)
    return rows


def _raw_path(evaluation: dict[str, Any], evaluation_path: Path) -> Path:
    value = evaluation.get("metadata", {}).get("raw_results_path")
    if not value:
        raise ValueError(f"评测结果缺少 metadata.raw_results_path: {evaluation_path}")
    path = Path(value)
    if path.is_file():
        return path
    # Result metadata can contain the path from another workspace mount.
    candidate = evaluation_path.parent / path.name
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"找不到原始 JSONL: {value}")


def _scoring_result(row: dict[str, Any]) -> ScoringResult:
    scoring = row.get("scoring")
    if not isinstance(scoring, dict):
        raise ValueError(f"题目缺少 scoring: {row.get('question_id')}")
    try:
        return ScoringResult(
            question_id=str(scoring.get("question_id", row["question_id"])),
            retrieval=RetrievalMetrics(**scoring["retrieval"]),
            generation=GenerationMetrics(**scoring["generation"]),
            safety=SafetyMetrics(**scoring["safety"]),
            final_score=float(scoring["final_score"]),
            safety_violation=bool(scoring.get("safety_violation", False)),
            notes=str(scoring.get("notes", "")),
            source_match=(
                SourceMatchMetrics(**scoring["source_match"])
                if isinstance(scoring.get("source_match"), dict)
                else None
            ),
            lexical_generation=(
                GenerationMetrics(**scoring["lexical_generation"])
                if scoring.get("lexical_generation")
                else None
            ),
            judge_generation=(
                GenerationMetrics(**scoring["judge_generation"])
                if scoring.get("judge_generation")
                else None
            ),
            judge=(JudgeMetrics(**scoring["judge"]) if scoring.get("judge") else None),
            scoring_method=str(scoring.get("scoring_method", "lexical")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"题目 scoring 字段不完整: {row.get('question_id')}") from exc


def _validate_rows(rows: Iterable[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        question_id = row.get("question_id")
        if not question_id:
            raise ValueError(f"{label} 中存在缺少 question_id 的记录")
        if question_id in index:
            raise ValueError(f"{label} 中 question_id 重复: {question_id}")
        index[str(question_id)] = row
    return index


def _aggregate_usage(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """合并结果时同步聚合可选的 query/Judge usage。"""
    output: dict[str, Any] = {
        "recorded_query_count": 0,
        "recorded_judge_count": 0,
        "cost_missing_count": 0,
    }
    for scope in ("query", "judge", "total"):
        values = [
            row.get("usage", {}).get(scope)
            for row in rows
            if isinstance(row.get("usage", {}).get(scope), dict)
        ]
        aggregate: dict[str, Any] = {}
        for field in (
            "request_count",
            "failed_request_count",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "input_cost_usd",
            "output_cost_usd",
            "total_cost_usd",
        ):
            present = [item.get(field) for item in values if item.get(field) is not None]
            if present:
                aggregate[field] = sum(present)
        aggregate["cost_available"] = bool(values) and all(
            item.get("cost_available") is True
            for item in values
            if item.get("request_count", 0) or item.get("total_tokens") is not None
        )
        aggregate["models"] = sorted(
            {model for item in values for model in item.get("models", [])}
        )
        output[scope] = aggregate
    output["recorded_query_count"] = sum(
        isinstance(row.get("usage", {}).get("query"), dict) for row in rows
    )
    output["recorded_judge_count"] = sum(
        isinstance(row.get("usage", {}).get("judge"), dict) for row in rows
    )
    output["cost_missing_count"] = sum(
        isinstance(row.get("usage", {}).get("total"), dict)
        and not row["usage"]["total"].get("cost_available", False)
        for row in rows
    )
    return output


def merge_evaluation(
    base_path: Path,
    retry_path: Path,
    output_path: Path,
    *,
    model: str | None = None,
) -> tuple[Path, Path]:
    """Merge successful retry rows into the base evaluation.

    The output JSON keeps the original question order and the output JSONL is
    written next to it.  Existing error rows remain visible when no successful
    same-model retry is available.
    """
    base_path = base_path.resolve()
    retry_path = retry_path.resolve()
    output_path = output_path.resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if not isinstance(base, dict) or not isinstance(base.get("metadata"), dict):
        raise ValueError(f"基础评测 JSON 结构无效: {base_path}")

    base_rows = _read_jsonl(_raw_path(base, base_path))
    retry_rows = _read_jsonl(retry_path)
    base_index = _validate_rows(base_rows, "基础评测")
    retry_index = _validate_rows(retry_rows, "重跑结果")
    unknown = sorted(set(retry_index) - set(base_index))
    if unknown:
        raise ValueError("重跑结果包含基础评测不存在的题目: " + ", ".join(unknown))

    invalid_retry = sorted(
        qid for qid, row in retry_index.items() if row.get("error")
    )
    if invalid_retry:
        raise ValueError("重跑结果仍失败，不能覆盖基础记录: " + ", ".join(invalid_retry))

    merged_rows: list[dict[str, Any]] = []
    replaced: list[str] = []
    for original in base_rows:
        question_id = str(original["question_id"])
        replacement = retry_index.get(question_id)
        if replacement is not None:
            merged_rows.append(replacement)
            replaced.append(question_id)
        else:
            merged_rows.append(original)

    scoring_report = DatasetScoringReport()
    for row in merged_rows:
        scoring_report.add(_scoring_result(row))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output_path.with_suffix(".jsonl")
    raw_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged_rows),
        encoding="utf-8",
    )

    metadata = dict(base["metadata"])
    if model:
        metadata["completion_model"] = model
    metadata.update(
        {
            "created_at": datetime.now(UTC).isoformat(),
            "question_count": len(merged_rows),
            "failed_query_count": sum(bool(row.get("error")) for row in merged_rows),
            "elapsed_seconds": round(
                sum(float(row.get("elapsed_seconds", 0.0)) for row in merged_rows),
                3,
            ),
            "actual_search_methods": dict(
                Counter(str(row["search_method"]) for row in merged_rows)
            ),
            "raw_results_path": str(raw_path),
            "source_match_mode": next(
                (
                    row.get("scoring", {})
                    .get("source_match", {})
                    .get("source_match_mode")
                    for row in merged_rows
                    if isinstance(row.get("scoring", {}).get("source_match"), dict)
                ),
                metadata.get("source_match_mode", "hybrid"),
            ),
            "merge": {
                "base_evaluation": str(base_path),
                "retry_results": str(retry_path),
                "base_row_count": len(base_rows),
                "retry_row_count": len(retry_rows),
                "replaced_question_ids": replaced,
                "replaced_count": len(replaced),
                "successful_count": sum(not bool(row.get("error")) for row in merged_rows),
                "remaining_failed_question_ids": [
                    str(row["question_id"])
                    for row in merged_rows
                    if row.get("error")
                ],
            },
        }
    )
    report = dict(base)
    report["metadata"] = metadata
    summary = scoring_report.summary()
    summary["usage"] = _aggregate_usage(merged_rows)
    report["summary"] = summary
    report["results"] = merged_rows
    report["metadata"]["output_path"] = str(output_path)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path, raw_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="合并同模型评测重跑结果")
    parser.add_argument("--base", type=Path, required=True, help="基础评测 JSON")
    parser.add_argument("--retry", type=Path, required=True, help="重跑 JSONL")
    parser.add_argument("--output", type=Path, required=True, help="合并评测 JSON")
    parser.add_argument("--model", default=None, help="completion 模型标签（如 gpt-5）")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output_path, raw_path = merge_evaluation(
            args.base,
            args.retry,
            args.output,
            model=args.model,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"合并失败: {type(exc).__name__}: {exc}")
        return 1
    print(f"合并评测已生成: {output_path}")
    print(f"原始 JSONL 已生成: {raw_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
