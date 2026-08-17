"""评测数据分析报告生成器。

读取最新的完整评测结果（``evaluation-*.json`` / ``.jsonl``）以及语料
清单与向量化清单，计算细分指标并生成 Markdown 报告到 ``benchmark/report``。

用法
----
.. code-block:: bash

    uv run python -m benchmark.report.build_report
    # 或指定评测结果：
    uv run python -m benchmark.report.build_report \\
        --evaluation benchmark/results/microsoft_graphrag/evaluation-xxx.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPOSITORY_ROOT / "benchmark" / "results" / "microsoft_graphrag"
DEFAULT_REPORT_DIR = REPOSITORY_ROOT / "benchmark" / "report"
VECTORIZE_MANIFEST = (
    REPOSITORY_ROOT / "benchmark" / "data" / "proceed" / "vectorize_manifest.json"
)
CORPUS_MANIFEST = (
    REPOSITORY_ROOT
    / "benchmark"
    / "data"
    / "microsoft_graphrag"
    / "corpus_manifest.json"
)


def _pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def _f(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(len(ordered) * 0.95) - 1))]


def _mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def _usage(row: dict[str, Any], scope: str = "total") -> dict[str, Any] | None:
    value = row.get("usage")
    if not isinstance(value, dict):
        return None
    if scope in value and isinstance(value[scope], dict):
        return value[scope]
    # 兼容早期试验结果把 usage 直接写成扁平结构的情况。
    return value if any(key in value for key in ("total_tokens", "prompt_tokens")) else None


def _query_elapsed(row: dict[str, Any]) -> float | None:
    telemetry = row.get("telemetry")
    if isinstance(telemetry, dict) and telemetry.get("elapsed_seconds") is not None:
        return float(telemetry["elapsed_seconds"])
    if row.get("elapsed_seconds") is not None:
        return float(row["elapsed_seconds"])
    return None


def _pick_latest_evaluation() -> Path:
    candidates = sorted(RESULTS_DIR.glob("evaluation-*.json"))
    if not candidates:
        raise FileNotFoundError(f"未找到评测结果: {RESULTS_DIR}")
    return candidates[-1]


def _load_manifest(path: Path) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _build_header(report: dict[str, Any], label: str | None = None) -> str:
    meta = report["metadata"]
    scoring_method = meta.get("scoring_method", "lexical")
    judge_mode = meta.get("judge_mode", "off")
    if judge_mode == "off":
        scoring_note = f"词法指标作为 baseline；本轮 selected={scoring_method}，未启用 LLM-as-Judge。"
    else:
        scoring_note = (
            f"本轮 Judge 模式为 `{judge_mode}`；每题保留 lexical，Judge 成功时 selected=judge，"
            "失败记录明确回退。"
        )
    lines = [
        "# 产前超声 GraphRAG 基准评测数据分析报告",
        "",
        f"- **生成时间**: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- **评测时间**: {meta['created_at']}",
        f"- **GraphRAG 版本**: {meta['graphrag_version']}",
        f"- **评测方式**: {meta['requested_search_method']}",
        f"- **题目数**: {meta['question_count']}（失败 {meta['failed_query_count']}）",
        f"- **检索 top-k**: {meta['k']}",
        f"- **来源校验口径**: {meta.get('source_match_mode', '历史结果未记录')}",
        f"- **评测耗时**: {meta['elapsed_seconds']:.0f} 秒",
        "",
        f"> 评分说明：{scoring_note} L3/L4 与 pending 标注仍需医学专家复核。",
        "",
    ]
    model_label = label or meta.get("completion_model")
    if model_label:
        lines.insert(3, f"- **completion 模型**: {model_label}")
    return "\n".join(lines)


def _build_corpus_section() -> str:
    corpus = _load_manifest(CORPUS_MANIFEST)
    vector = _load_manifest(VECTORIZE_MANIFEST)
    lines = [
        "## 1. 数据资产概况",
        "",
        "### 1.1 语料",
        "",
        f"- 输入文档数：**{corpus.get('indexed_document_count', 'N/A')}** 份"
        f"（PDF {corpus.get('pdf_count', 'N/A')} 份，去重后）",
        f"- 提取失败：{corpus.get('error_count', 0)} 份；部分提取："
        f"{corpus.get('partial_count', 0)} 份",
        "",
    ]
    coverage = corpus.get("dataset_source_coverage", {})
    if coverage:
        lines += [
            "- 数据集金标准来源覆盖：",
            f"  - 完整覆盖题目：{coverage.get('fully_covered_questions')}/"
            f"{coverage.get('total_questions')}",
            f"  - 缺失来源：{', '.join(coverage.get('missing_source_ids', [])) or '无'}",
            "",
        ]
    lines += ["### 1.2 向量化", ""]
    if vector:
        lines += [
            f"- 向量模型：`{vector.get('embedding_model')}`（维度 "
            f"{vector.get('embedding_dimension')}）",
            f"- API base：`{vector.get('embedding_api_base')}`",
            f"- 文本块：**{vector.get('chunk_count')}** 块 / "
            f"{vector.get('token_count'):,} token",
            f"- 分块：{vector.get('chunk_size_tokens')} token，"
            f"重叠 {vector.get('chunk_overlap_tokens')}",
            f"- 文档数：{vector.get('vectorized_document_count')}；"
            f"错误：{len(vector.get('errors', []))}",
            "",
        ]
    else:
        lines += ["（未找到向量化清单）", ""]
    return "\n".join(lines)


def _build_merge_section(report: dict[str, Any]) -> str:
    """Describe how a partial same-model rerun was incorporated."""
    merge = report.get("metadata", {}).get("merge")
    if not merge:
        return ""
    replaced = merge.get("replaced_question_ids", [])
    remaining = merge.get("remaining_failed_question_ids", [])
    lines = [
        "## 合并说明",
        "",
        f"- 基础结果：{merge.get('base_row_count', 'N/A')} 题；重跑结果："
        f"{merge.get('retry_row_count', 'N/A')} 题。",
        f"- 按题号替换成功记录：**{merge.get('replaced_count', len(replaced))}** 题。",
        f"- 当前成功记录：**{merge.get('successful_count', 'N/A')}** / "
        f"{report['metadata'].get('question_count', 'N/A')}。",
        "- 替换题号：" + (", ".join(replaced) if replaced else "无"),
        "- 仍失败题号：" + (", ".join(remaining) if remaining else "无"),
        "",
        "> 合并仅使用同一 completion 模型的成功重跑记录；未跨模型回填，"
        "因此剩余失败题不应解读为该模型的成功结果。",
        "",
    ]
    return "\n".join(lines)


def _build_overall_section(report: dict[str, Any]) -> str:
    summary = report["summary"]
    meta = report["metadata"]
    lines = [
        "## 2. 总体评测指标",
        "",
        "| 维度 | 指标 | 值 |",
        "|---|---|---|",
        f"| 检索 | context_precision@{meta['k']} | {_f(summary['retrieval']['context_precision_at_k'])} |",
        f"| 检索 | context_recall@{meta['k']} | {_f(summary['retrieval']['context_recall_at_k'])} |",
        f"| 检索 | coverage@{meta['k']} | {_f(summary['retrieval']['coverage_at_k'])} |",
        f"| 检索 | miss@{meta['k']} | {_f(summary['retrieval']['miss_at_k'])} |",
        f"| 生成 | faithfulness | {_f(summary['generation']['faithfulness'])} |",
        f"| 生成 | answer_relevance | {_f(summary['generation']['answer_relevance'])} |",
        f"| 生成 | completeness | {_f(summary['generation']['completeness'])} |",
        f"| 生成 | answer_correctness | {_f(summary['generation']['answer_correctness'])} |",
        f"| 安全 | safety_score | {_f(summary['safety']['safety_score'])} |",
        f"| 安全 | hallucination_rate | {_f(summary['safety']['hallucination_rate'])} |",
        f"| 综合 | **final_score** | **{_f(summary['final_score'])}** |",
        f"| 综合 | safety_gate | {'✅ 通过' if summary['safety_gate_passed'] else '❌ 未通过'} |",
        "",
    ]
    return "\n".join(lines)


def _build_scoring_section(rows: list[dict[str, Any]], report: dict[str, Any]) -> str:
    """说明 lexical/Judge 双轨评分覆盖，避免把历史结果伪装成 Judge。"""
    scoring = report.get("summary", {}).get("scoring", {})
    judge_rows = [r for r in rows if r.get("scoring", {}).get("judge_generation")]
    failed_rows = [
        r
        for r in rows
        if isinstance(r.get("scoring", {}).get("judge"), dict)
        and r["scoring"]["judge"].get("error")
    ]
    lines = [
        "## 2.1 评分方法与覆盖",
        "",
        f"- selected 口径：`{scoring.get('selected_method', report.get('metadata', {}).get('scoring_method', 'lexical'))}`；"
        f"Judge 成功 {scoring.get('judge_success_count', len(judge_rows))}/{len(rows)} 题，"
        f"失败/回退 {scoring.get('judge_failure_count', len(failed_rows))} 题。",
        "- lexical 指标仍完整保留，用于回归对比；Judge 评分包含医学同义词和事实一致性判断。",
    ]
    if judge_rows:
        specs = (
            ("faithfulness", "faithfulness"),
            ("answer_relevance", "answer_relevance"),
            ("completeness", "completeness"),
            ("answer_correctness", "answer_correctness"),
        )
        lines += [
            "",
            "| 指标 | lexical 均值 | Judge 均值 | Δ |",
            "|---|---:|---:|---:|",
        ]
        for key, label in specs:
            lexical = [
                r["scoring"].get("lexical_generation", {}).get(key)
                for r in rows
                if r["scoring"].get("lexical_generation")
            ]
            judge = [
                r["scoring"].get("judge_generation", {}).get(key)
                for r in judge_rows
            ]
            lexical_mean = mean(lexical) if lexical else None
            judge_mean = mean(judge) if judge else None
            if lexical_mean is None or judge_mean is None:
                continue
            lines.append(
                f"| {label} | {_f(lexical_mean)} | {_f(judge_mean)} | {judge_mean - lexical_mean:+.3f} |"
            )
    else:
        lines.append("- 当前结果没有 Judge 字段（历史结果或 Judge 关闭）。")
    lines.append("")
    return "\n".join(lines)


def _build_source_section(rows: list[dict[str, Any]]) -> str:
    matches = [r.get("scoring", {}).get("source_match") for r in rows]
    matches = [item for item in matches if isinstance(item, dict)]
    lines = [
        "## 6.1 来源等价与证据校验",
        "",
    ]
    if not matches:
        lines.append("历史结果没有 `source_match` 字段，无法区分 exact 与 evidence 口径。")
        return "\n".join(lines + [""])
    total = len(rows)
    recorded = len(matches)
    exact = sum(bool(item.get("exact_source_hit")) for item in matches)
    equivalent = sum(bool(item.get("equivalent_source_hit")) for item in matches)
    evidence = sum(bool(item.get("evidence_supported")) for item in matches)
    false_negative_reduction = sum(
        bool(item.get("evidence_supported")) and not bool(item.get("exact_source_hit"))
        for item in matches
    )
    lines += [
        f"- `source_match` 字段覆盖：**{recorded}/{total}**；以下命中率仅在已记录题目中计算。",
        f"- exact 指定来源命中：**{exact}/{recorded}**（{_pct(exact / recorded if recorded else 0)}）",
        f"- equivalent 权威来源命中：**{equivalent}/{recorded}**（{_pct(equivalent / recorded if recorded else 0)}）",
        f"- evidence 覆盖全部 must-have：**{evidence}/{recorded}**（{_pct(evidence / recorded if recorded else 0)}）",
        f"- 仅因来源 ID 未精确命中、但证据完整支持的潜在 False Negative：**{false_negative_reduction}** 题",
        "- `exact_source_hit` 保留旧口径；综合检索分析使用配置的 `source_match_mode`，两者不可混读。",
        "",
    ]
    return "\n".join(lines)


def _build_method_section(rows: list[dict[str, Any]]) -> str:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_method[r["search_method"]].append(r)
    lines = [
        "## 3. 分检索方法分析",
        "",
        "| 方法 | 题数 | coverage | miss | precision | faithfulness | completeness | final_score |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for method in ("basic", "local", "drift", "global"):
        items = by_method.get(method)
        if not items:
            continue
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                method,
                len(items),
                _f(mean(r["scoring"]["retrieval"]["coverage_at_k"] for r in items)),
                _f(mean(r["scoring"]["retrieval"]["miss_at_k"] for r in items)),
                _f(
                    mean(
                        r["scoring"]["retrieval"]["context_precision_at_k"]
                        for r in items
                    )
                ),
                _f(mean(r["scoring"]["generation"]["faithfulness"] for r in items)),
                _f(mean(r["scoring"]["generation"]["completeness"] for r in items)),
                _f(mean(r["scoring"]["final_score"] for r in items)),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _build_difficulty_section(rows: list[dict[str, Any]]) -> str:
    by_diff: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_diff[r["difficulty"]].append(r)
    lines = [
        "## 4. 分难度分析",
        "",
        "| 难度 | 题数 | coverage | faithfulness | completeness | final_score |",
        "|---|---|---|---|---|---|",
    ]
    for diff in ("L1", "L2", "L3", "L4"):
        items = by_diff.get(diff)
        if not items:
            continue
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                diff,
                len(items),
                _f(mean(r["scoring"]["retrieval"]["coverage_at_k"] for r in items)),
                _f(mean(r["scoring"]["generation"]["faithfulness"] for r in items)),
                _f(mean(r["scoring"]["generation"]["completeness"] for r in items)),
                _f(mean(r["scoring"]["final_score"] for r in items)),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _build_question_section(rows: list[dict[str, Any]]) -> str:
    ranked = sorted(rows, key=lambda r: r["scoring"]["final_score"], reverse=True)
    lines = [
        "## 5. 逐题表现",
        "",
        "### 5.1 表现最好（Top 10）",
        "",
        "| 题目 | 难度 | 方法 | coverage | faithfulness | final_score |",
        "|---|---|---|---|---|---|",
    ]
    for r in ranked[:10]:
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                r["question_id"],
                r["difficulty"],
                r["search_method"],
                _f(r["scoring"]["retrieval"]["coverage_at_k"]),
                _f(r["scoring"]["generation"]["faithfulness"]),
                _f(r["scoring"]["final_score"]),
            )
        )
    lines += [
        "",
        "### 5.2 表现最差（Bottom 10）",
        "",
        "| 题目 | 难度 | 方法 | coverage | faithfulness | final_score |",
        "|---|---|---|---|---|---|",
    ]
    for r in ranked[-10:][::-1]:
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                r["question_id"],
                r["difficulty"],
                r["search_method"],
                _f(r["scoring"]["retrieval"]["coverage_at_k"]),
                _f(r["scoring"]["generation"]["faithfulness"]),
                _f(r["scoring"]["final_score"]),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _build_retrieval_section(rows: list[dict[str, Any]], k: int) -> str:
    missed = [r for r in rows if r["scoring"]["retrieval"]["miss_at_k"] > 0]
    unresolved = [
        r
        for r in rows
        if any(s.startswith("unresolved:") for s in r.get("retrieved_sources", []))
    ]
    lines = [
        "## 6. 检索质量分析",
        "",
        f"- 命中关联证据（miss@k = 0，按 source_match_mode）：**{len(rows) - len(missed)}/{len(rows)}**"
        f"（{_pct(1 - len(missed) / len(rows))}）",
        f"- 未命中关联证据（miss@k > 0）：**{len(missed)}/{len(rows)}**"
        f"（{_pct(len(missed) / len(rows))}）",
        f"- 检索来源含 unresolved 的题目：{len(unresolved)}",
        "",
        "未命中关联证据的题目分布：",
        "",
        "| 题目 | 难度 | 方法 | coverage | miss |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(
        missed, key=lambda x: x["scoring"]["retrieval"]["miss_at_k"], reverse=True
    ):
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                r["question_id"],
                r["difficulty"],
                r["search_method"],
                _f(r["scoring"]["retrieval"]["coverage_at_k"]),
                _f(r["scoring"]["retrieval"]["miss_at_k"]),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _build_safety_section(rows: list[dict[str, Any]]) -> str:
    l4 = [r for r in rows if r["difficulty"] == "L4"]
    referred = [r for r in rows if r.get("referred")]
    refused = [r for r in rows if r.get("refused")]
    violations = [r for r in rows if r["scoring"].get("safety_violation")]
    lines = [
        "## 7. 安全与合规分析",
        "",
        f"- 转诊行为（referred）：{len(referred)} 题",
        f"- 拒答行为（refused）：{len(refused)} 题",
        f"- safety_violation 标记：{len(violations)} 题",
        f"- L4 安全题：{len(l4)} 题",
        "",
        "L4 安全题明细：",
        "",
        "| 题目 | 方法 | refused | referred | safety_score | 说明 |",
        "|---|---|---|---|---|---|",
    ]
    for r in l4:
        notes = r["scoring"].get("notes", "")
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                r["question_id"],
                r["search_method"],
                r.get("refused"),
                r.get("referred"),
                _f(r["scoring"]["safety"]["safety_score"]),
                (notes or "").replace("|", "\\|")[:40],
            )
        )
    lines.append("")
    return "\n".join(lines)


def _build_runtime_section(rows: list[dict[str, Any]], meta: dict[str, Any]) -> str:
    times = [elapsed for r in rows if (elapsed := _query_elapsed(r)) is not None]
    if not times:
        return "## 8. 运行特征\n\n（结果没有耗时字段）\n"
    by_method: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        elapsed = _query_elapsed(r)
        if elapsed is not None:
            by_method[r["search_method"]].append(elapsed)
    p95 = _p95(times)
    lines = [
        "## 8. 运行特征",
        "",
        f"- 总耗时：{meta['elapsed_seconds']:.0f} 秒"
        f"（约 {meta['elapsed_seconds'] / 60:.1f} 分钟）",
        f"- 单题耗时：中位数 {median(times):.1f} 秒，p95 {p95:.1f} 秒（范围 {min(times):.1f}–{max(times):.1f} 秒）",
        "- 分方法平均耗时：",
    ]
    for method, ts in sorted(by_method.items()):
        lines.append(f"  - `{method}`: {mean(ts):.1f} 秒/题")
    lines.append("")
    return "\n".join(lines)


def _build_telemetry_section(rows: list[dict[str, Any]]) -> str:
    telemetry_rows = [r for r in rows if isinstance(r.get("telemetry"), dict)]
    usage_rows = [r for r in rows if _usage(r, "total") is not None]
    lines = [
        "## 8.1 GraphRAG 监控与成本",
        "",
    ]
    if not telemetry_rows and not usage_rows:
        lines.append("历史结果没有 telemetry/usage 字段，无法回溯图探索事件或 token 成本。")
        return "\n".join(lines + [""])

    def total_metric(name: str) -> int:
        return sum(int((r.get("telemetry") or {}).get(name, 0) or 0) for r in telemetry_rows)

    lines += [
        f"- 记录 telemetry：**{len(telemetry_rows)}/{len(rows)}** 题；usage：**{len(usage_rows)}/{len(rows)}** 题。",
        f"- context events：{total_metric('context_event_count')}；context records：{total_metric('context_record_count')}；"
        f"context chars：{total_metric('context_chars'):,}。",
        f"- graph nodes：{total_metric('graph_node_count')}；community records：{total_metric('community_record_count')}；"
        f"graph exploration events：{total_metric('graph_exploration_event_count')}。",
        f"- map responses：{total_metric('map_response_count')}；reduce responses：{total_metric('reduce_response_count')}；"
        f"stream chunks：{total_metric('stream_output_chunks')}。",
        "",
        "| 方法 | 题数 | 平均耗时(s) | p95(s) | 图节点 | 社区记录 | 配置深度 | prompt tokens | completion tokens | total tokens | cost(USD) | 成本可用 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[str(row.get("search_method", "unknown"))].append(row)
    for method, items in sorted(by_method.items()):
        durations = [elapsed for r in items if (elapsed := _query_elapsed(r)) is not None]
        token_values = [
            float((_usage(r, "total") or {}).get("total_tokens"))
            for r in items
            if (_usage(r, "total") or {}).get("total_tokens") is not None
        ]
        prompt_values = [
            float((_usage(r, "total") or {}).get("prompt_tokens"))
            for r in items
            if (_usage(r, "total") or {}).get("prompt_tokens") is not None
        ]
        completion_values = [
            float((_usage(r, "total") or {}).get("completion_tokens"))
            for r in items
            if (_usage(r, "total") or {}).get("completion_tokens") is not None
        ]
        cost_values = [
            float((_usage(r, "total") or {}).get("total_cost_usd"))
            for r in items
            if (_usage(r, "total") or {}).get("total_cost_usd") is not None
        ]
        cost_available = sum(bool((_usage(r, "total") or {}).get("cost_available")) for r in items)
        graph_nodes = [
            float((r.get("telemetry") or {}).get("graph_node_count"))
            for r in items
            if (r.get("telemetry") or {}).get("graph_node_count") is not None
        ]
        community_records = [
            float((r.get("telemetry") or {}).get("community_record_count"))
            for r in items
            if (r.get("telemetry") or {}).get("community_record_count") is not None
        ]
        configured_depth = [
            float((r.get("telemetry") or {}).get("configured_traversal_depth"))
            for r in items
            if (r.get("telemetry") or {}).get("configured_traversal_depth") is not None
        ]
        lines.append(
            f"| {method} | {len(items)} | "
            f"{_f(mean(durations), 1) if durations else 'N/A'} | "
            f"{_f(_p95(durations), 1) if durations else 'N/A'} | "
            f"{_f(mean(graph_nodes), 0) if graph_nodes else 'N/A'} | "
            f"{_f(mean(community_records), 0) if community_records else 'N/A'} | "
            f"{_f(mean(configured_depth), 1) if configured_depth else 'N/A'} | "
            f"{_f(mean(prompt_values), 0) if prompt_values else 'N/A'} | "
            f"{_f(mean(completion_values), 0) if completion_values else 'N/A'} | "
            f"{_f(mean(token_values), 0) if token_values else 'N/A'} | "
            f"{_f(mean(cost_values), 6) if cost_values else 'N/A'} | "
            f"{cost_available}/{len(items)} |"
        )
    lines += [
        "",
        "成本说明：`null/N/A` 表示供应商未返回价格或历史结果未记录，不能按 0 美元解释。",
    ]
    drift = sorted(
        (r for r in rows if r.get("search_method") == "drift"),
        key=lambda r: _query_elapsed(r) or 0.0,
        reverse=True,
    )[:5]
    if drift:
        lines += [
            "",
            "DRIFT 慢查询 Top 5：",
            "",
            "| 题目 | 耗时(s) | exploration events | graph nodes | community records | total tokens |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in drift:
            telemetry = row.get("telemetry") or {}
            usage = _usage(row, "total") or {}
            lines.append(
                f"| {row.get('question_id')} | {(_query_elapsed(row) or 0.0):.1f} | "
                f"{telemetry.get('graph_exploration_event_count', 'N/A')} | {telemetry.get('graph_node_count', 'N/A')} | "
                f"{telemetry.get('community_record_count', 'N/A')} | "
                f"{usage.get('total_tokens', 'N/A')} |"
            )
    lines.append("")
    return "\n".join(lines)


def _build_conclusion_section(
    rows: list[dict[str, Any]],
    report: dict[str, Any],
    has_comparison: bool = False,
) -> str:
    summary = report["summary"]
    failed = [r for r in rows if r.get("error")]
    lines = [
        "## 9. 结论与建议",
        "",
        "### 9.1 主要结论",
        "",
    ]
    lines.append(
        f"- **检索召回尚可、精度偏低**：整体 coverage@{report['metadata']['k']} "
        f"{_pct(summary['retrieval']['coverage_at_k'])}，但 context_precision "
        f"{_f(summary['retrieval']['context_precision_at_k'])} 较低，"
        f"miss@k {_pct(summary['retrieval']['miss_at_k'])} 的题目需优化。"
    )
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[row["search_method"]].append(row)
    method_coverages = {
        method: mean(r["scoring"]["retrieval"]["coverage_at_k"] for r in items)
        for method, items in by_method.items()
        if items
    }
    if method_coverages:
        weakest_method, weakest_coverage = min(
            method_coverages.items(), key=lambda item: item[1]
        )
        lines.append(
            f"- **{weakest_method} 检索覆盖最低**：{len(by_method[weakest_method])} 题 "
            f"coverage {_pct(weakest_coverage)}，是当前主要优化对象。"
        )
    lines.append(
        f"- **安全门{'已通过' if summary['safety_gate_passed'] else '未通过'}**："
        f"safety_score {_f(summary['safety']['safety_score'])}"
        f"{'，需补充拒答/转诊行为' if not summary['safety_gate_passed'] else ''}。"
    )
    scoring_meta = report.get("metadata", {})
    if scoring_meta.get("judge_mode", "off") == "off":
        lines.append(
            f"- **正确性数值需谨慎解读**：当前 answer_correctness "
            f"{_f(summary['generation']['answer_correctness'])} 是 lexical baseline，"
            "不能作为 GPT-5 医学语义能力的最终结论；下一轮应启用 Judge 并保留专家抽检。"
        )
    if not any(isinstance(row.get("scoring", {}).get("source_match"), dict) for row in rows):
        lines.append(
            "- **来源命中存在口径盲区**：历史记录没有 exact/equivalent/evidence 字段，"
            "本报告无法量化来源强绑定造成的 False Negative。"
        )
    if failed:
        lines.append(f"- **查询失败 {len(failed)} 题**，详见逐题 JSONL。")
    if report.get("metadata", {}).get("merge"):
        lines += [
            "",
            "> 运行时间口径：合并报告的耗时为保留题目的单题耗时之和，"
            "包含基础运行与重跑，不代表一次连续运行的墙钟时间。",
        ]
    lines += [
        "",
        "### 9.2 建议",
        "",
        "1. **检索优化**：针对 coverage 最低的方法调优检索参数（如 DRIFT 的 "
        "`drift_k_followups`、`n_depth`、社区层级）或增大 top-k。",
        "2. **安全合规**：在系统提示词中强化「未确证问题应拒答或转诊」行为，"
        "重点覆盖 L4 安全题。",
        "3. **评分口径**：L3/L4 指标需医学专家复核；正式比较时启用 "
        "`--judge-mode optional --judge-model ...`，同时保留 lexical baseline。",
    ]
    if has_comparison:
        lines.append(
            "4. **模型对比**：第 10 节给出了共同成功题的对比；继续更换模型时，"
            "建议在同一数据集复跑以保持可比性。"
        )
    else:
        lines.append(
            "4. **模型对比**：如切换模型，建议在同一数据集上复跑并生成对比章节"
            "（`--compare`）。"
        )
    lines.append("")
    return "\n".join(lines)


def _build_model_comparison_section(
    current_rows: list[dict[str, Any]],
    compare_path: Path,
    current_label: str,
    compare_label: str,
) -> str:
    """在双方共同成功的题目上对比当前评测与上一轮评测（通常为不同模型）。"""
    compare_data = json.loads(compare_path.read_text(encoding="utf-8"))
    compare_raw = Path(compare_data["metadata"]["raw_results_path"])
    compare_rows = [
        json.loads(line)
        for line in compare_raw.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    current_idx = {r["question_id"]: r for r in current_rows}
    compare_idx = {r["question_id"]: r for r in compare_rows}
    common = [
        qid
        for qid, r in current_idx.items()
        if not r.get("error")
        and qid in compare_idx
        and not compare_idx[qid].get("error")
    ]
    if not common:
        return ""

    def agg(source: dict[str, dict[str, Any]], key: str, sub: str | None) -> float:
        vals = [
            (source[qid]["scoring"][sub][key] if sub else source[qid]["scoring"][key])
            for qid in common
        ]
        return mean(vals)

    metric_specs = [
        ("retrieval", "coverage_at_k"),
        ("retrieval", "miss_at_k"),
        ("retrieval", "context_precision_at_k"),
        ("generation", "faithfulness"),
        ("generation", "completeness"),
        ("generation", "answer_relevance"),
        ("generation", "answer_correctness"),
        (None, "final_score"),
    ]
    lines = [
        "## 10. 模型对比",
        "",
        f"在双方均成功的 **{len(common)} 题** 上对比 `{compare_label}` 与 "
        f"`{current_label}`：",
        "",
        f"| 指标 | {compare_label} | {current_label} | Δ |",
        "|---|---|---|---|",
    ]
    for sub, key in metric_specs:
        a = agg(compare_idx, key, sub)
        b = agg(current_idx, key, sub)
        lines.append(f"| {key} | {_f(a)} | {_f(b)} | {b - a:+.4f} |")

    by_method: dict[str, list[str]] = defaultdict(list)
    for qid in common:
        by_method[current_idx[qid]["search_method"]].append(qid)
    lines += [
        "",
        "分方法 final_score：",
        "",
        f"| 方法 | 题数 | {compare_label} | {current_label} | Δ |",
        "|---|---|---|---|---|",
    ]
    for method in ("basic", "local", "drift", "global"):
        qids = by_method.get(method)
        if not qids:
            continue
        a = mean(compare_idx[q]["scoring"]["final_score"] for q in qids)
        b = mean(current_idx[q]["scoring"]["final_score"] for q in qids)
        lines.append(f"| {method} | {len(qids)} | {_f(a)} | {_f(b)} | {b - a:+.4f} |")
    lines.append("")
    return "\n".join(lines)


def build_report(
    evaluation_path: Path,
    output_dir: Path,
    *,
    label: str | None = None,
    compare_path: Path | None = None,
    compare_label: str | None = None,
) -> Path:
    """根据评测结果生成 Markdown 分析报告。"""
    evaluation_path = evaluation_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report_data = json.loads(evaluation_path.read_text(encoding="utf-8"))
    raw_path = Path(report_data["metadata"]["raw_results_path"])
    rows = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    sections = [
        _build_header(report_data, label=label),
        _build_merge_section(report_data),
        _build_corpus_section(),
        _build_overall_section(report_data),
        _build_scoring_section(rows, report_data),
        _build_method_section(rows),
        _build_difficulty_section(rows),
        _build_question_section(rows),
        _build_retrieval_section(rows, report_data["metadata"]["k"]),
        _build_source_section(rows),
        _build_safety_section(rows),
        _build_runtime_section(rows, report_data["metadata"]),
        _build_telemetry_section(rows),
        _build_conclusion_section(rows, report_data, has_comparison=bool(compare_path)),
    ]
    if compare_path:
        sections.append(
            _build_model_comparison_section(
                rows,
                compare_path,
                current_label=label or "当前",
                compare_label=compare_label or "上一轮",
            )
        )
    report_md = "\n".join(sections)

    stamp = datetime.now(UTC).strftime("%Y%m%d")
    out_path = output_dir / f"evaluation-analysis-{stamp}.md"
    out_path.write_text(report_md, encoding="utf-8")
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="生成评测数据分析报告",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=None,
        help="评测结果 JSON（缺省自动选择最新完整评测）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="报告输出目录",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="当前评测模型标签（如 gpt-5），显示在报告头部",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="上一轮评测 JSON，用于生成模型对比章节",
    )
    parser.add_argument(
        "--compare-label",
        default=None,
        help="对比评测的模型标签（如 gpt-4.1）",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evaluation_path = args.evaluation or _pick_latest_evaluation()
    try:
        out_path = build_report(
            evaluation_path,
            args.output_dir,
            label=args.label,
            compare_path=args.compare,
            compare_label=args.compare_label,
        )
        print(f"报告已生成: {out_path}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"生成失败: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
