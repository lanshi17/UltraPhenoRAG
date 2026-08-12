"""产前超声诊断 GraphRAG 评测 —— 数据集构建工具。

提供数据集加载、字段校验、去污染、分层划分等功能。
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Sequence

from benchmark.qa.models import DatasetSplit, Question


# ── 难度分布与划分比例 ────────────────────────────────────────────────────────

DIFFICULTY_DISTRIBUTION: dict[str, float] = {
    "L1": 0.40,
    "L2": 0.35,
    "L3": 0.20,
    "L4": 0.05,
}
"""目标难度分布比例。"""

SPLIT_RATIOS: dict[str, float] = {
    "dev": 0.7,
    "val": 0.2,
    "test": 0.1,
}
"""开发集/验证集/测试集划分比例。"""

NGRAM_THRESHOLD: int = 18
"""18-gram 文本重叠过滤阈值。"""


# ── 加载与导出 ────────────────────────────────────────────────────────────────


def load_questions(json_path: str | Path) -> list[Question]:
    """从 JSON 文件加载问题列表。

    Parameters
    ----------
    json_path : str | Path
        JSON 文件路径。文件格式为 ``[question_dict, ...]``。

    Returns
    -------
    list[Question]
    """
    path = Path(json_path)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return [Question.from_dict(item) for item in data]


def save_questions(questions: Sequence[Question], json_path: str | Path) -> None:
    """导出问题列表到 JSON 文件。

    Parameters
    ----------
    questions : Sequence[Question]
        问题列表。
    json_path : str | Path
        输出路径。
    """
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            [q.to_dict() for q in questions],
            f,
            ensure_ascii=False,
            indent=2,
        )


# ── 校验 ──────────────────────────────────────────────────────────────────


def validate_dataset(questions: Sequence[Question]) -> dict[str, list[str]]:
    """校验整个数据集的字段完整性。

    Parameters
    ----------
    questions : Sequence[Question]
        问题列表。

    Returns
    -------
    dict[str, list[str]]
        ``{question_id: [error_msg, ...]}``，空列表表示通过。
    """
    errors: dict[str, list[str]] = {}
    seen_ids: set[str] = set()

    for q in questions:
        errs = q.validate()

        # 唯一性检查
        if q.question_id in seen_ids:
            errs.append(f"question_id 重复: {q.question_id}")
        seen_ids.add(q.question_id)

        if errs:
            errors[q.question_id] = errs

    return errors


def check_difficulty_distribution(questions: Sequence[Question]) -> dict[str, dict[str, float]]:
    """检查难度分布是否达标。

    Returns
    -------
    dict[str, dict[str, float]]
        ``{"actual": {"L1": 0.42, ...}, "target": {...}, "gap": {"L1": +0.02, ...}}``
    """
    total = len(questions)
    if total == 0:
        return {"actual": {}, "target": DIFFICULTY_DISTRIBUTION, "gap": {}}

    actual: dict[str, float] = {}
    for q in questions:
        actual[q.difficulty] = actual.get(q.difficulty, 0) + 1
    actual = {k: v / total for k, v in actual.items()}

    gap = {
        k: actual.get(k, 0) - DIFFICULTY_DISTRIBUTION.get(k, 0)
        for k in set(actual) | set(DIFFICULTY_DISTRIBUTION)
    }

    return {"actual": actual, "target": DIFFICULTY_DISTRIBUTION, "gap": gap}


# ── 去污染 ──────────────────────────────────────────────────────────────────


def ngram_set(text: str, n: int = NGRAM_THRESHOLD) -> set[str]:
    """提取文本的 n-gram 集合（字符级）。

    Parameters
    ----------
    text : str
        输入文本。
    n : int
        n-gram 长度。

    Returns
    -------
    set[str]
    """
    # 去除空白和标点，保留中文和英文
    cleaned = "".join(c for c in text if c.isalnum())
    if len(cleaned) < n:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + n] for i in range(len(cleaned) - n + 1)}


def deduplicate_by_ngram(
    questions: Sequence[Question],
    reference_texts: Sequence[str] | None = None,
    overlap_threshold: float = 0.8,
) -> tuple[list[Question], list[tuple[str, float]]]:
    """基于 n-gram 重叠度去除与参考语料高度重叠的题目。

    Parameters
    ----------
    questions : Sequence[Question]
        待过滤的问题列表。
    reference_texts : Sequence[str] | None
        参考语料（如公开训练集文本）。为 ``None`` 时仅做题目间去重。
    overlap_threshold : float
        Jaccard 重叠度阈值，超过此值视为污染。

    Returns
    -------
    tuple[list[Question], list[tuple[str, float]]]
        (过滤后的问题列表, 被移除的 (question_id, overlap_score) 列表)
    """
    # 预计算参考语料的 n-gram 集合
    ref_ngrams: set[str] = set()
    if reference_texts:
        for text in reference_texts:
            ref_ngrams |= ngram_set(text)

    # 预计算每个问题的 n-gram
    q_ngrams: dict[str, set[str]] = {}
    for q in questions:
        q_ngrams[q.question_id] = ngram_set(q.question + q.gold_answer)

    removed: list[tuple[str, float]] = []
    kept: list[Question] = []
    seen_ngrams: list[set[str]] = []

    for q in questions:
        ng = q_ngrams[q.question_id]

        # 与参考语料比较
        if ref_ngrams:
            overlap = len(ng & ref_ngrams) / max(1, len(ng))
            if overlap > overlap_threshold:
                removed.append((q.question_id, overlap))
                continue

        # 题目间去重
        is_dup = False
        for prev_ng in seen_ngrams:
            if not ng or not prev_ng:
                continue
            overlap = len(ng & prev_ng) / max(1, len(ng | prev_ng))
            if overlap > overlap_threshold:
                is_dup = True
                removed.append((q.question_id, overlap))
                break

        if not is_dup:
            kept.append(q)
            seen_ngrams.append(ng)

    return kept, removed


# ── 分层划分 ──────────────────────────────────────────────────────────────────


def stratified_split(
    questions: Sequence[Question],
    ratios: dict[str, float] | None = None,
    seed: int = 42,
    version: str = "1.0.0",
) -> DatasetSplit:
    """按难度分层划分数据集。

    确保每个划分内部的难度分布与整体一致。

    Parameters
    ----------
    questions : Sequence[Question]
        问题列表。
    ratios : dict[str, float] | None
        划分比例，默认 ``{"dev": 0.7, "val": 0.2, "test": 0.1}``。
    seed : int
        随机种子。
    version : str
        版本号。

    Returns
    -------
    DatasetSplit
    """
    if ratios is None:
        ratios = SPLIT_RATIOS

    rng = random.Random(seed)

    # 按难度分组
    by_difficulty: dict[str, list[Question]] = {}
    for q in questions:
        by_difficulty.setdefault(q.difficulty, []).append(q)

    dev: list[Question] = []
    val: list[Question] = []
    test: list[Question] = []

    for difficulty, qs in by_difficulty.items():
        shuffled = list(qs)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_dev = round(n * ratios["dev"])
        n_val = round(n * ratios["val"])
        # test 取剩余，确保无遗漏
        n_test = n - n_dev - n_val
        if n_test < 0:
            n_dev += n_test
            n_test = 0

        for q in shuffled[:n_dev]:
            q.split = "dev"
            dev.append(q)
        for q in shuffled[n_dev : n_dev + n_val]:
            q.split = "val"
            val.append(q)
        for q in shuffled[n_dev + n_val :]:
            q.split = "test"
            test.append(q)

    return DatasetSplit(dev=dev, val=val, test=test, version=version)


# ── 数据集指纹 ────────────────────────────────────────────────────────────────


def compute_dataset_fingerprint(questions: Sequence[Question]) -> str:
    """计算数据集内容指纹（SHA-256）。

    用于版本锁定和污染检测。

    Parameters
    ----------
    questions : Sequence[Question]
        问题列表。

    Returns
    -------
    str
        SHA-256 十六进制摘要。
    """
    content = json.dumps(
        [q.to_dict() for q in questions],
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def build_dataset_report(
    questions: Sequence[Question],
    split: DatasetSplit,
) -> dict:
    """生成数据集构建报告。

    Parameters
    ----------
    questions : Sequence[Question]
        全部问题列表。
    split : DatasetSplit
        划分结果。

    Returns
    -------
    dict
        报告字典。
    """
    dist = check_difficulty_distribution(questions)
    return {
        "total_questions": len(questions),
        "fingerprint": compute_dataset_fingerprint(questions),
        "difficulty_distribution": dist,
        "split": {
            "dev": len(split.dev),
            "val": len(split.val),
            "test": len(split.test),
            "distribution": split.difficulty_distribution(),
        },
        "version": split.version,
    }
