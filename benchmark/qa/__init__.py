"""产前超声诊断 GraphRAG 评测 -- 模块入口。

提供问题、金标准、评分框架、标注一致性计算等核心组件。

主要导出::

    # Schema
    EntityType, RelationType, EvidenceLevel,
    DifficultyLevel, RAGArchType, QuestionType,
    GUIDELINE_REGISTRY

    # Models
    Question, GoldSource, GoldRelation, AnnotationRecord, DatasetSplit

    # Scoring
    ScoringResult, DatasetScoringReport,
    RetrievalMetrics, GenerationMetrics, SafetyMetrics,
    score_question, compute_retrieval_metrics,
    compute_faithfulness, compute_completeness,
    compute_answer_correctness, compute_safety_score

    # Annotation Agreement
    cohens_kappa, fleiss_kappa, krippendorffs_alpha,
    statement_iou, interpret_kappa

    # Dataset Building
    load_questions, save_questions, validate_dataset,
    check_difficulty_distribution, deduplicate_by_ngram,
    stratified_split, compute_dataset_fingerprint, build_dataset_report
"""

from benchmark.qa.annotation_agreement import (
    cohens_kappa,
    fleiss_kappa,
    interpret_kappa,
    krippendorffs_alpha,
    statement_iou,
)
from benchmark.qa.build_dataset import (
    build_dataset_report,
    check_difficulty_distribution,
    compute_dataset_fingerprint,
    deduplicate_by_ngram,
    load_questions,
    save_questions,
    stratified_split,
    validate_dataset,
)
from benchmark.qa.judge import JudgeConfig, judge_answer
from benchmark.qa.models import (
    AnnotationRecord,
    DatasetSplit,
    GoldRelation,
    GoldSource,
    Question,
)
from benchmark.qa.schema import (
    DifficultyLevel,
    EntityType,
    EvidenceLevel,
    GUIDELINE_REGISTRY,
    QuestionType,
    RAGArchType,
    RelationType,
)
from benchmark.qa.scoring import (
    DatasetScoringReport,
    GenerationMetrics,
    JudgeMetrics,
    RetrievalMetrics,
    SafetyMetrics,
    SourceMatchMetrics,
    ScoringResult,
    compute_answer_correctness,
    compute_completeness,
    compute_faithfulness,
    compute_retrieval_metrics,
    compute_safety_score,
    compute_source_match,
    score_question,
)

__all__ = [
    # Schema
    "EntityType",
    "RelationType",
    "EvidenceLevel",
    "DifficultyLevel",
    "RAGArchType",
    "QuestionType",
    "GUIDELINE_REGISTRY",
    # Models
    "Question",
    "GoldSource",
    "GoldRelation",
    "AnnotationRecord",
    "DatasetSplit",
    # Scoring
    "ScoringResult",
    "DatasetScoringReport",
    "RetrievalMetrics",
    "GenerationMetrics",
    "JudgeMetrics",
    "SafetyMetrics",
    "SourceMatchMetrics",
    "score_question",
    "compute_retrieval_metrics",
    "compute_faithfulness",
    "compute_completeness",
    "compute_answer_correctness",
    "compute_safety_score",
    "compute_source_match",
    "JudgeConfig",
    "judge_answer",
    # Annotation Agreement
    "cohens_kappa",
    "fleiss_kappa",
    "krippendorffs_alpha",
    "statement_iou",
    "interpret_kappa",
    # Dataset Building
    "load_questions",
    "save_questions",
    "validate_dataset",
    "check_difficulty_distribution",
    "deduplicate_by_ngram",
    "stratified_split",
    "compute_dataset_fingerprint",
    "build_dataset_report",
]
