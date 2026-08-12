"""产前超声诊断 GraphRAG 评测 —— 标注一致性计算。

提供 Cohen's κ、Fleiss' κ、Krippendorff's α 三种
评估者间一致性（Inter-Annotator Agreement, IAA）指标的计算方法。

阈值参考
--------
- Cohen's κ ≥ 0.80（2 名医师对答案正确性判定）
- Fleiss' κ ≥ 0.75（3+ 名医师对检索片段相关性判定）
- Krippendorff's α ≥ 0.70（must_have_statements 完整性判定）
- Krippendorff's α ≥ 0.50（事实性逐句判定）
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence


def cohens_kappa(rater1: Sequence[str], rater2: Sequence[str]) -> float:
    """计算 Cohen's κ（两名标注者的一致性）。

    Parameters
    ----------
    rater1 : Sequence[str]
        标注者 1 的标注序列（每个元素为类别标签）。
    rater2 : Sequence[str]
        标注者 2 的标注序列（长度需与 rater1 相同）。

    Returns
    -------
    float
        Cohen's κ 值，范围 [-1, 1]。``< 0`` 表示低于随机一致。

    Raises
    ------
    ValueError
        两个序列长度不一致时。
    """
    if len(rater1) != len(rater2):
        raise ValueError(
            f"序列长度不一致: rater1={len(rater1)}, rater2={len(rater2)}"
        )

    n = len(rater1)
    if n == 0:
        return 0.0

    # 观察一致率
    observed = sum(1 for a, b in zip(rater1, rater2) if a == b) / n

    # 期望一致率
    labels = sorted(set(rater1) | set(rater2))
    expected = 0.0
    for label in labels:
        p1 = sum(1 for x in rater1 if x == label) / n
        p2 = sum(1 for x in rater2 if x == label) / n
        expected += p1 * p2

    if expected == 1.0:
        # 两人全部标同一类别，κ 无定义，视为完全一致
        return 1.0

    return (observed - expected) / (1.0 - expected)


def fleiss_kappa(ratings: Sequence[Sequence[str]]) -> float:
    """计算 Fleiss' κ（多名标注者的一致性）。

    Parameters
    ----------
    ratings : Sequence[Sequence[str]]
        每个元素是一个标注者的标注序列（所有序列等长）。
        例如 3 名标注者对 10 个项目的标注::

            [["correct", "partial", ...],
             ["correct", "correct", ...],
             ["partial", "partial", ...]]

    Returns
    -------
    float
        Fleiss' κ 值。

    Raises
    ------
    ValueError
        标注者数量 < 2 或序列长度不一致时。
    """
    n_raters = len(ratings)
    if n_raters < 2:
        raise ValueError("Fleiss' κ 至少需要 2 名标注者")

    n_items = len(ratings[0])
    for r in ratings:
        if len(r) != n_items:
            raise ValueError("所有标注者的序列长度必须一致")

    if n_items == 0:
        return 0.0

    # 收集所有类别
    categories = sorted({label for rater in ratings for label in rater})

    # 构建 n_items × n_categories 的计数矩阵
    matrix: list[list[int]] = []
    for i in range(n_items):
        counts = Counter(rater[i] for rater in ratings)
        matrix.append([counts.get(cat, 0) for cat in categories])

    # P_i: 每个项目的标注者一致率
    p_i_list = []
    for row in matrix:
        total = sum(row)  # = n_raters
        p_i = sum(c * c for c in row) - total
        p_i /= total * (total - 1) if total > 1 else 1
        p_i_list.append(p_i)

    p_bar = sum(p_i_list) / n_items  # 观察一致率

    # p_j: 每个类别的边际比例
    p_j_list = []
    for j, _cat in enumerate(categories):
        col_sum = sum(row[j] for row in matrix)
        p_j_list.append(col_sum / (n_items * n_raters))

    # 期望一致率
    p_e = sum(p * p for p in p_j_list)

    if p_e == 1.0:
        return 1.0

    return (p_bar - p_e) / (1.0 - p_e)


def krippendorffs_alpha(
    ratings: Sequence[Sequence[str | None]],
    level: str = "nominal",
) -> float:
    """计算 Krippendorff's α（支持缺失值与多标注者）。

    适用于 must_have_statements 完整性判定和事实性逐句判定，
    这些场景中标注者可能对部分条目标注 ``None``（不确定）。

    Parameters
    ----------
    ratings : Sequence[Sequence[str | None]]
        每个元素是一个标注者的标注序列，``None`` 表示缺失/不确定。
    level : str, default "nominal"
        测量层级：``"nominal"`` / ``"ordinal"`` / ``"interval"`` / ``"ratio"``。
        产前超声标注默认 nominal。

    Returns
    -------
    float
        Krippendorff's α 值。
    """
    n_raters = len(ratings)
    if n_raters < 2:
        raise ValueError("Krippendorff's α 至少需要 2 名标注者")

    n_items = len(ratings[0])
    if n_items == 0:
        return 0.0

    # 收集所有非 None 类别
    categories = sorted(
        {label for rater in ratings for label in rater if label is not None}
    )
    cat_index = {c: i for i, c in enumerate(categories)}
    n_cats = len(categories)

    # 构建 unit-by-coder 计数矩阵
    # 每个标注者对每个项目标注或缺失
    # 计算每个 (item, category) 的计数
    counts: list[list[int]] = [[0] * n_cats for _ in range(n_items)]
    for item_idx in range(n_items):
        for rater in ratings:
            val = rater[item_idx]
            if val is not None and val in cat_index:
                counts[item_idx][cat_index[val]] += 1

    # 每个 item 的标注者数（排除缺失）
    unit_totals = [sum(row) for row in counts]
    # 有效 unit 数
    n_valid = sum(1 for t in unit_totals if t > 0)

    if n_valid == 0:
        return 0.0

    # 每个类别的总计数
    cat_totals = [0] * n_cats
    for row in counts:
        for j in range(n_cats):
            cat_totals[j] += row[j]

    total_pairs = sum(t * (t - 1) for t in unit_totals)
    if total_pairs == 0:
        return 0.0

    # Observed disagreement (Do)
    if level == "nominal":
        do = 0.0
        for row in counts:
            item_total = sum(row)
            if item_total <= 1:
                continue
            # n_u * (n_u - 1) - sum(n_uk * (n_uk - 1))
            disagreement = item_total * (item_total - 1) - sum(
                c * (c - 1) for c in row
            )
            do += disagreement
        do /= total_pairs

        # Expected disagreement (De)
        de = 0.0
        grand_total = sum(cat_totals)
        for j in range(n_cats):
            p_j = cat_totals[j] / grand_total
            de += p_j * (1 - p_j)

        if de == 0:
            return 1.0
        return 1.0 - do / de

    # ordinal / interval / ratio 简化为 nominal（产前超声标注以 nominal 为主）
    # 如需扩展可在此处添加
    return krippendorffs_alpha(ratings, level="nominal")


def statement_iou(
    statements_a: Sequence[str],
    statements_b: Sequence[str],
) -> float:
    """计算两份 must_have_statements 的 IoU（关键陈述重叠度）。

    采用基于词级 Jaccard 相似度的软匹配：对 A 中每条陈述，
    在 B 中寻找最高相似度，超过阈值 0.5 视为匹配。

    Parameters
    ----------
    statements_a : Sequence[str]
        标注者 A 的关键陈述列表。
    statements_b : Sequence[str]
        标注者 B 的关键陈述列表。

    Returns
    -------
    float
        IoU 值，范围 [0, 1]。``< 0.7`` 时触发仲裁。
    """
    if not statements_a and not statements_b:
        return 1.0
    if not statements_a or not statements_b:
        return 0.0

    threshold = 0.5

    def word_jaccard(s1: str, s2: str) -> float:
        w1 = set(s1.lower().split())
        w2 = set(s2.lower().split())
        if not w1 or not w2:
            return 0.0
        return len(w1 & w2) / len(w1 | w2)

    # A 中每条是否在 B 中有匹配
    a_matched = sum(
        1 for sa in statements_a
        if any(word_jaccard(sa, sb) >= threshold for sb in statements_b)
    )
    # B 中每条是否在 A 中有匹配
    b_matched = sum(
        1 for sb in statements_b
        if any(word_jaccard(sb, sa) >= threshold for sa in statements_a)
    )

    # IoU = 交集 / 并集
    intersection = (a_matched + b_matched) / 2
    union = len(statements_a) + len(statements_b) - intersection
    if union == 0:
        return 0.0
    return intersection / union


def interpret_kappa(kappa: float) -> str:
    """Landis-Koch 标准解读 κ 值。

    Parameters
    ----------
    kappa : float
        κ 或 α 值。

    Returns
    -------
    str
        一致性等级描述。
    """
    if kappa < 0:
        return "低于随机一致"
    if kappa < 0.20:
        return "极低一致性 (slight)"
    if kappa < 0.40:
        return "一般一致性 (fair)"
    if kappa < 0.60:
        return "中等一致性 (moderate)"
    if kappa < 0.80:
        return "较高一致性 (substantial)"
    return "几乎完全一致 (almost perfect)"
