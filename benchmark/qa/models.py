"""产前超声诊断 GraphRAG 评测 —— 数据模型。

定义问题、金标准答案、标注记录、评分结果等核心数据结构，
所有字段与 SOP 文档中的标注模板保持一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from benchmark.qa.schema import (
    DifficultyLevel,
    EvidenceLevel,
    QuestionType,
    RAGArchType,
)


@dataclass
class GoldSource:
    """金标准证据来源。

    Attributes
    ----------
    guide : str
        指南名称或文档 ID（对应 ``GUIDELINE_REGISTRY`` 的 key）。
    section : str
        章节/条款定位。
    evidence_level : str
        证据层级（L1/L2/L3）。
    page : int | None
        页码（可选）。
    """

    guide: str
    section: str
    evidence_level: str = EvidenceLevel.L1
    page: int | None = None


@dataclass
class GoldRelation:
    """金标准关系三元组。"""

    src: str
    """源实体。"""
    rel: str
    """关系类型（对应 ``RelationType`` 的 value）。"""
    tgt: str
    """目标实体。"""


@dataclass
class AnnotationRecord:
    """单名标注者的标注记录。

    Attributes
    ----------
    annotator_id : str
        标注者唯一标识。
    rating : str
        标注者对问题/答案的评定（``"correct"`` / ``"partial"`` / ``"incorrect"``）。
    must_have_coverage : int
        标注者认为答案覆盖了 ``must_have_statements`` 中多少条。
    notes : str
        标注备注（分歧原因、修改建议等）。
    """

    annotator_id: str
    rating: str = "correct"
    must_have_coverage: int = 0
    notes: str = ""


@dataclass
class Question:
    """评测问题。

    与 SOP 标注模板一一对应，是数据集的基本单元。

    Attributes
    ----------
    question_id : str
        问题唯一标识，格式 ``PU-L{难度}-{序号:03d}``，如 ``PU-L2-017``。
    question : str
        问题文本。
    question_type : str
        问题类型（对应 ``QuestionType`` 的 value）。
    difficulty : str
        难度层级（L1/L2/L3/L4）。
    rag_arch_type : str
        最优 RAG 架构类型（basic/multi-vector/graph-enhanced）。
    kg_hops : int | None
        预期知识图谱跳数（L3+ 题目必填）。
    clinical_scenario : str
        临床场景标签。
    gold_answer : str
        金标准答案。
    gold_entities : list[str]
        金标准关键实体列表。
    gold_relations : list[GoldRelation]
        金标准关系三元组列表。
    gold_sources : list[GoldSource]
        金标准证据来源列表。
    must_have_statements : list[str]
        必须覆盖的关键陈述列表，用于完整性评分。
    safety_flags : list[str]
        安全标注（如 "不应直接建议终止妊娠"）。
    annotators : list[AnnotationRecord]
        标注者记录列表。
    agreement_kappa : float | None
        标注者间一致性 κ 值（2 人用 Cohen's κ，3+ 人用 Fleiss' κ）。
    adjudicated : bool
        是否经过仲裁。
    split : str | None
        数据集划分（``"dev"`` / ``"val"`` / ``"test"``），组装阶段填充。
    """

    question_id: str
    question: str
    question_type: str
    difficulty: str
    rag_arch_type: str
    kg_hops: int | None = None
    clinical_scenario: str = ""
    gold_answer: str = ""
    gold_entities: list[str] = field(default_factory=list)
    gold_relations: list[GoldRelation] = field(default_factory=list)
    gold_sources: list[GoldSource] = field(default_factory=list)
    must_have_statements: list[str] = field(default_factory=list)
    safety_flags: list[str] = field(default_factory=list)
    annotators: list[AnnotationRecord] = field(default_factory=list)
    agreement_kappa: float | None = None
    adjudicated: bool = False
    split: str | None = None

    # ── 校验 ──────────────────────────────────────────────────────────

    def validate(self) -> list[str]:
        """校验字段完整性，返回错误信息列表（空列表表示通过）。

        Returns
        -------
        list[str]
            错误信息列表。
        """
        errors: list[str] = []

        # ID 格式
        valid_difficulties = {d.value for d in DifficultyLevel}
        prefix = self.question_id.split("-")
        if len(prefix) < 3 or prefix[0] != "PU":
            errors.append(f"question_id 格式错误: {self.question_id}")
        elif prefix[1] not in valid_difficulties:
            errors.append(f"difficulty 不合法: {prefix[1]}")

        # 难度合法性
        if self.difficulty not in valid_difficulties:
            errors.append(f"difficulty 不合法: {self.difficulty}")

        # 问题类型合法性
        valid_types = {t.value for t in QuestionType}
        if self.question_type not in valid_types:
            errors.append(f"question_type 不合法: {self.question_type}")

        # RAG 架构类型合法性
        valid_archs = {a.value for a in RAGArchType}
        if self.rag_arch_type not in valid_archs:
            errors.append(f"rag_arch_type 不合法: {self.rag_arch_type}")

        # L3+ 必须标注 kg_hops
        if self.difficulty in ("L3", "L4") and self.kg_hops is None:
            errors.append("L3/L4 题目必须标注 kg_hops")

        # L4 安全题必须有 safety_flags
        if self.difficulty == "L4" and not self.safety_flags:
            errors.append("L4 安全题必须有 safety_flags")

        # 金标准答案不能为空
        if not self.gold_answer.strip():
            errors.append("gold_answer 不能为空")

        # must_have_statements 不能为空
        if not self.must_have_statements:
            errors.append("must_have_statements 不能为空")

        # gold_sources 不能为空（必须可溯源）
        if not self.gold_sources:
            errors.append("gold_sources 不能为空（必须可溯源）")

        # 标注者至少 2 人
        if len(self.annotators) < 2:
            errors.append("annotators 至少需要 2 名标注者")

        return errors

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典（用于 JSON 导出）。"""
        return {
            "question_id": self.question_id,
            "question": self.question,
            "question_type": self.question_type,
            "difficulty": self.difficulty,
            "rag_arch_type": self.rag_arch_type,
            "kg_hops": self.kg_hops,
            "clinical_scenario": self.clinical_scenario,
            "gold_answer": self.gold_answer,
            "gold_entities": list(self.gold_entities),
            "gold_relations": [
                {"src": r.src, "rel": r.rel, "tgt": r.tgt}
                for r in self.gold_relations
            ],
            "gold_sources": [
                {
                    "guide": s.guide,
                    "section": s.section,
                    "evidence_level": s.evidence_level,
                    "page": s.page,
                }
                for s in self.gold_sources
            ],
            "must_have_statements": list(self.must_have_statements),
            "safety_flags": list(self.safety_flags),
            "annotators": [
                {
                    "id": a.annotator_id,
                    "rating": a.rating,
                    "must_have_coverage": a.must_have_coverage,
                    "notes": a.notes,
                }
                for a in self.annotators
            ],
            "agreement_kappa": self.agreement_kappa,
            "adjudicated": self.adjudicated,
            "split": self.split,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Question:
        """从字典反序列化。"""
        relations = [
            GoldRelation(src=r["src"], rel=r["rel"], tgt=r["tgt"])
            for r in data.get("gold_relations", [])
        ]
        sources = [
            GoldSource(
                guide=s["guide"],
                section=s["section"],
                evidence_level=s.get("evidence_level", EvidenceLevel.L1),
                page=s.get("page"),
            )
            for s in data.get("gold_sources", [])
        ]
        annotators = [
            AnnotationRecord(
                annotator_id=a["id"],
                rating=a.get("rating", "correct"),
                must_have_coverage=a.get("must_have_coverage", 0),
                notes=a.get("notes", ""),
            )
            for a in data.get("annotators", [])
        ]
        return cls(
            question_id=data["question_id"],
            question=data["question"],
            question_type=data["question_type"],
            difficulty=data["difficulty"],
            rag_arch_type=data["rag_arch_type"],
            kg_hops=data.get("kg_hops"),
            clinical_scenario=data.get("clinical_scenario", ""),
            gold_answer=data.get("gold_answer", ""),
            gold_entities=data.get("gold_entities", []),
            gold_relations=relations,
            gold_sources=sources,
            must_have_statements=data.get("must_have_statements", []),
            safety_flags=data.get("safety_flags", []),
            annotators=annotators,
            agreement_kappa=data.get("agreement_kappa"),
            adjudicated=data.get("adjudicated", False),
            split=data.get("split"),
        )


@dataclass
class DatasetSplit:
    """数据集划分结果。

    Attributes
    ----------
    dev : list[Question]
        开发集（70%）。
    val : list[Question]
        验证集（20%）。
    test : list[Question]
        测试集（10%），长期冻结。
    version : str
        版本号（语义化版本，如 ``"1.0.0"``）。
    """

    dev: list[Question]
    val: list[Question]
    test: list[Question]
    version: str = "1.0.0"

    @property
    def total(self) -> int:
        return len(self.dev) + len(self.val) + len(self.test)

    def difficulty_distribution(self) -> dict[str, dict[str, int]]:
        """各划分的难度分布统计。"""
        result: dict[str, dict[str, int]] = {}
        for name, questions in [("dev", self.dev), ("val", self.val), ("test", self.test)]:
            dist: dict[str, int] = {}
            for q in questions:
                dist[q.difficulty] = dist.get(q.difficulty, 0) + 1
            result[name] = dist
        return result
