"""Generate a comparison report for Microsoft GraphRAG and LightRAG evaluations."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from benchmark.common.benchmark_protocol import assert_comparable_conditions

ROOT = Path(__file__).resolve().parents[2]
RESULTS = {
    "Microsoft GraphRAG": ROOT
    / "benchmark/results/microsoft_graphrag/evaluation-20260821-judge-rerun.json",
    "LightRAG": ROOT / "benchmark/results/light_rag/evaluation-20260821T142218Z.json",
}
OUTPUT = ROOT / "benchmark/report/graphrag-lightrag-comparison-20260822.md"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _retrieval_mean(row: dict[str, Any]) -> float:
    metrics = row["scoring"]["retrieval"]
    return (
        metrics["context_precision_at_k"]
        + metrics["context_recall_at_k"]
        + metrics["coverage_at_k"]
        + 1.0
        - metrics["miss_at_k"]
    ) / 4.0


def _generation_mean(row: dict[str, Any]) -> float:
    metrics = row["scoring"]["generation"]
    return (
        0.25 * metrics["faithfulness"]
        + 0.15 * metrics["answer_relevance"]
        + 0.25 * metrics["completeness"]
        + 0.20 * metrics["answer_correctness"]
    ) / 0.85


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "retrieval": mean(_retrieval_mean(row) for row in rows),
        "generation": mean(_generation_mean(row) for row in rows),
        "final": mean(row["scoring"]["final_score"] for row in rows),
        "source": mean(
            row["scoring"]["source_match"]["exact_source_hit"] for row in rows
        ),
    }


def _rows_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return grouped


def _format_section(
    title: str,
    rows_by_result: dict[str, list[dict[str, Any]]],
    key: str,
) -> list[str]:
    groups = sorted({row[key] for rows in rows_by_result.values() for row in rows})
    lines = [
        f"## {title}",
        "",
        "| 分组 | 系统 | 题数 | 检索均值 | 生成加权均值 | Final | 严格来源命中 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in groups:
        for name, rows in rows_by_result.items():
            selected = [row for row in rows if row[key] == group]
            if not selected:
                continue
            metrics = _aggregate(selected)
            lines.append(
                f"| {group} | {name} | {len(selected)} | {metrics['retrieval']:.4f} | "
                f"{metrics['generation']:.4f} | {metrics['final']:.4f} | {metrics['source']:.4f} |"
            )
    return lines + [""]


def main() -> None:
    reports = {name: _load(path) for name, path in RESULTS.items()}
    rows = {name: report["results"] for name, report in reports.items()}
    microsoft = reports["Microsoft GraphRAG"]
    light = reports["LightRAG"]
    assert_comparable_conditions(
        microsoft["metadata"].get("benchmark_conditions", {}),
        light["metadata"].get("benchmark_conditions", {}),
    )
    ms_summary = microsoft["summary"]
    lr_summary = light["summary"]
    ms_rows = {row["question_id"]: row for row in rows["Microsoft GraphRAG"]}
    lr_rows = {row["question_id"]: row for row in rows["LightRAG"]}
    deltas = []
    for question_id, ms_row in ms_rows.items():
        lr_row = lr_rows[question_id]
        ms_score = ms_row["scoring"]["generation"]["answer_correctness"]
        lr_score = lr_row["scoring"]["generation"]["answer_correctness"]
        deltas.append(
            (
                lr_score - ms_score,
                question_id,
                ms_row["difficulty"],
                ms_row["search_method"],
                ms_score,
                lr_score,
            )
        )
    lower = sorted(deltas)[:5]
    higher = sorted(deltas, reverse=True)[:5]

    lines = [
        "# Microsoft GraphRAG 与 LightRAG 对比分析",
        "",
        "## 结论",
        "",
        "本轮有效结果中，LightRAG 的检索和 Judge 生成指标均高于 Microsoft GraphRAG。两者的 Final Score 都是 0.3333，原因不是性能相同，而是安全门控分数均为 0.6667，低于 0.90 阈值后 Final Score 被重置为安全分数的一半。",
        "",
        "本报告只使用完整索引重建后的 LightRAG 结果。此前 `evaluation-20260821T021910Z.json` 的索引没有持久化向量，50 题均为空上下文，不能用于比较。",
        "",
        "## 评测设置",
        "",
        f"- 数据集：50 题；两套结果的 dataset fingerprint 均为 `{microsoft['metadata']['dataset_fingerprint']}`。",
        f"- 检索：adaptive，top-k={microsoft['metadata']['k']}；策略分布均为 basic 23、local 17、drift 10。",
        "- 评分：GPT-5 Judge，50/50 题成功，无 lexical 回退；来源校验口径为 hybrid。",
        "- 版本：Microsoft GraphRAG 3.1.1；LightRAG 1.5.7。",
        "",
        "## 总体指标",
        "",
        "| 指标 | Microsoft GraphRAG | LightRAG | 差值（LightRAG - Microsoft） |",
        "|---|---:|---:|---:|",
    ]
    overall = [
        ("Context Precision@16", "retrieval", "context_precision_at_k"),
        ("Context Recall@16", "retrieval", "context_recall_at_k"),
        ("Coverage@16", "retrieval", "coverage_at_k"),
        ("Miss@16", "retrieval", "miss_at_k"),
        ("检索均值", "retrieval", "mean"),
        ("Faithfulness", "generation", "faithfulness"),
        ("Answer Relevance", "generation", "answer_relevance"),
        ("Completeness", "generation", "completeness"),
        ("Answer Correctness", "generation", "answer_correctness"),
        ("生成加权均值", "generation", "weighted_mean"),
        ("严格来源命中率", "source_match", "exact_source_hit_rate"),
        ("证据支持率", "source_match", "evidence_supported_rate"),
        ("Final Score", None, "final_score"),
    ]
    for label, section, key in overall:
        ms_value = ms_summary[key] if section is None else ms_summary[section][key]
        lr_value = lr_summary[key] if section is None else lr_summary[section][key]
        lines.append(
            f"| {label} | {ms_value:.4f} | {lr_value:.4f} | {lr_value - ms_value:+.4f} |"
        )
    lines += [
        "",
        "## 运行与成本记录",
        "",
        "| 项目 | Microsoft GraphRAG | LightRAG |",
        "|---|---:|---:|",
        f"| 评测耗时 | {microsoft['metadata']['elapsed_seconds'] / 60:.1f} 分钟 | {light['metadata']['elapsed_seconds'] / 60:.1f} 分钟 |",
        f"| Judge token | {ms_summary['usage']['judge']['total_tokens']:,} | {lr_summary['usage']['judge']['total_tokens']:,} |",
        f"| Judge cost | ${ms_summary['usage']['judge']['total_cost_usd']:.4f} | ${lr_summary['usage']['judge']['total_cost_usd']:.4f} |",
        "| 查询 token/cost | 已记录 token，但未提供 cost | 未记录查询 usage |",
        "",
        "LightRAG 的评测耗时约为 Microsoft GraphRAG 的三分之一，但两套结果的 usage 埋点不对等：LightRAG 未记录查询 token/cost，不能据此比较端到端成本。",
        "",
    ]
    lines += _format_section("按难度分层", rows, "difficulty")
    lines += _format_section("按检索策略分层", rows, "search_method")
    lines += [
        "## 逐题差异",
        "",
        "### LightRAG 优势最大的题目（Answer Correctness）",
        "",
        "| 题号 | 难度 | 策略 | Microsoft | LightRAG | 差值 |",
        "|---|---|---|---:|---:|---:|",
    ]
    for delta, question_id, difficulty, method, ms_score, lr_score in higher:
        lines.append(
            f"| {question_id} | {difficulty} | {method} | {ms_score:.3f} | {lr_score:.3f} | {delta:+.3f} |"
        )
    lines += [
        "",
        "### Microsoft GraphRAG 优势最大的题目（Answer Correctness）",
        "",
        "| 题号 | 难度 | 策略 | Microsoft | LightRAG | 差值（LightRAG - Microsoft） |",
        "|---|---|---|---:|---:|---:|",
    ]
    for delta, question_id, difficulty, method, ms_score, lr_score in lower:
        lines.append(
            f"| {question_id} | {difficulty} | {method} | {ms_score:.3f} | {lr_score:.3f} | {delta:+.3f} |"
        )
    lines += [
        "",
        "## 解读与后续工作",
        "",
        "- LightRAG 在严格来源命中、检索覆盖和 Judge 完整性上均更高。本轮差异首先出现在检索层：检索均值高 0.0780，严格来源命中率高 0.3400。",
        "- 两套系统的安全门控都未通过，因此 Final Score 不能用于区分系统优劣。需要单独审阅 L4 安全题的拒答、转诊与安全标注规则。",
        "- Microsoft GraphRAG 的 Judge 评分来自已保存查询结果的重评分；LightRAG 评分来自重建索引后的完整查询。两者数据集、检索策略映射、top-k 和 Judge 模型一致，但运行时间不同，不能把差值解释为严格的同一时点在线 A/B 实验。",
        "- LightRAG 索引曾因 embedding 包装和中断残留状态导致空检索。本报告使用重建后且失败文档已补跑的索引；索引状态为 32/32 processed。",
        "",
        "## 输入结果",
        "",
        "- Microsoft GraphRAG：[evaluation-20260821-judge-rerun.json](../results/microsoft_graphrag/evaluation-20260821-judge-rerun.json)",
        "- LightRAG：[evaluation-20260821T142218Z.json](../results/light_rag/evaluation-20260821T142218Z.json)",
        "",
    ]
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
