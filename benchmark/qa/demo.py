#!/usr/bin/env python
"""产前超声诊断 GraphRAG 评测 -- 快速入门示例。

演示如何使用 qa 模块完成一次完整的评测流程：
1. 加载标准问答集
2. 校验数据集
3. 检查难度分布
4. 执行评分（模拟）
5. 生成评分报告
6. 计算标注一致性
7. 分层划分数据集

用法::

    python -m benchmark.qa.demo
"""

from __future__ import annotations

import json

from benchmark.qa import (
    DatasetScoringReport,
    build_dataset_report,
    check_difficulty_distribution,
    cohens_kappa,
    compute_dataset_fingerprint,
    interpret_kappa,
    load_questions,
    score_question,
    stratified_split,
    validate_dataset,
)


def main() -> None:
    print("=" * 70)
    print("产前超声诊断 GraphRAG 评测数据集 -- 快速入门")
    print("=" * 70)

    # 1. 加载标准问答集
    print("\n[1] 加载标准问答集")
    questions = load_questions("benchmark/qa/dataset/sample_questions.json")
    print(f"    已加载 {len(questions)} 道题目")
    for q in questions:
        print(f"      {q.question_id} [{q.difficulty}] {q.question_type}: {q.question[:50]}...")

    # 2. 字段校验
    print("\n[2] 字段校验")
    errors = validate_dataset(questions)
    if errors:
        for qid, errs in errors.items():
            print(f"    ✗ {qid}: {errs}")
    else:
        print(f"    ✓ 全部 {len(questions)} 道题目校验通过")

    # 3. 难度分布检查
    print("\n[3] 难度分布")
    dist = check_difficulty_distribution(questions)
    print(f"    实际分布: {dist['actual']}")
    print(f"    目标分布: {dist['target']}")
    print(f"    差距:     {dist['gap']}")

    # 4. 数据集指纹
    print("\n[4] 数据集指纹")
    fp = compute_dataset_fingerprint(questions)
    print(f"    SHA-256: {fp}")

    # 5. 分层划分
    print("\n[5] 分层划分 (dev:val:test = 7:2:1)")
    split = stratified_split(questions, seed=42)
    print(f"    dev={len(split.dev)}, val={len(split.val)}, test={len(split.test)}")
    for name, qs in [("dev", split.dev), ("val", split.val), ("test", split.test)]:
        if qs:
            difficulties = [q.difficulty for q in qs]
            print(f"      {name}: {dict((d, difficulties.count(d)) for d in set(difficulties))}")

    # 6. 构建报告
    print("\n[6] 数据集构建报告")
    report = build_dataset_report(questions, split)
    print(f"    版本: {report['version']}")
    print(f"    总题数: {report['total_questions']}")
    print(f"    指纹: {report['fingerprint'][:32]}...")

    # 7. 标注一致性示例
    print("\n[7] 标注一致性计算示例")
    rater1 = ["correct", "correct", "correct", "incorrect", "correct",
              "partial", "incorrect", "correct", "correct", "partial"]
    rater2 = ["correct", "correct", "partial", "incorrect", "correct",
              "partial", "incorrect", "correct", "correct", "correct"]
    k = cohens_kappa(rater1, rater2)
    print(f"    Cohen's κ = {k:.4f} ({interpret_kappa(k)})")
    print(f"    阈值: ≥ 0.80 (几乎完全一致)")
    print(f"    达标: {'✓' if k >= 0.80 else '✗ (需更多训练或仲裁)'}")

    # 8. 评分框架演示
    print("\n[8] 评分框架演示 (使用金标准答案模拟系统输出)")
    scoring_report = DatasetScoringReport()
    for q in questions:
        # 模拟：系统检索到金标准来源，用金标准答案作为系统输出
        retrieved_sources = [s.guide for s in q.gold_sources]
        retrieved_context = q.gold_answer
        retrieved_supports = [set(q.must_have_statements)] * len(q.gold_sources)

        result = score_question(
            question=q,
            answer=q.gold_answer,
            retrieved_sources=retrieved_sources,
            retrieved_context=retrieved_context,
            retrieved_supports=retrieved_supports,
        )
        scoring_report.add(result)

    summary = scoring_report.summary()
    print(json.dumps(summary, ensure_ascii=False, indent=4))

    print("\n" + "=" * 70)
    print("快速入门完成。")
    print(f"文档: docs/benchmark-sop.md")
    print(f"样例: benchmark/qa/dataset/sample_questions.json")
    print(f"配置: benchmark/qa/dataset/scoring_config.yaml")
    print("=" * 70)


if __name__ == "__main__":
    main()
