"""Microsoft GraphRAG 产前超声基线的端到端评测入口。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import yaml
from pypdf import PdfReader

from benchmark.qa import (
    DatasetScoringReport,
    EntityType,
    Question,
    compute_dataset_fingerprint,
    load_questions,
    score_question,
    validate_dataset,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = REPOSITORY_ROOT / "benchmark" / "data" / "raw"
DEFAULT_PROJECT_DIR = (
    REPOSITORY_ROOT / "benchmark" / "data" / "microsoft_graphrag"
)
DEFAULT_DATASET = (
    REPOSITORY_ROOT / "benchmark" / "qa" / "dataset" / "sample_questions.json"
)
DEFAULT_RESULTS_DIR = (
    REPOSITORY_ROOT / "benchmark" / "results" / "microsoft_graphrag"
)
MANIFEST_NAME = "corpus_manifest.json"

CANONICAL_SOURCE_FILES: dict[str, str] = {
    "ISUOG_2020_fetal-CNS-part1.pdf": "ISUOG-cns-2020",
    "ISUOG_2022_routine-mid-trimester-scan.pdf": "ISUOG-midtrimester-2022",
    "ISUOG_2023_11-14-week-ultrasound-scan.pdf": "ISUOG-11-14w-2023",
    "ISUOG_2023_fetal-cardiac-screening.pdf": (
        "ISUOG-fetal-cardiac-screening-2023"
    ),
    "ISUOG-Practice-Guidelines-CNS-part-1-targeted-neurosonography.pdf": (
        "ISUOG-cns-2020"
    ),
    "ISUOG-Practice-Guidelines-Updated-performance-of-11-14-week-ultrasound-scan.pdf": (
        "ISUOG-11-14w-2023"
    ),
    "UOG-2023-Carvalho-ISUOG-Practice-Guidelines-updated-fetal-cardiac-screening.pdf": (
        "ISUOG-fetal-cardiac-screening-2023"
    ),
}

ADAPTIVE_SEARCH_METHODS = {
    "basic": "basic",
    "multi-vector": "local",
    "graph-enhanced": "drift",
}

PLACEHOLDER_API_KEYS = {
    "",
    "<API_KEY>",
    "YOUR_API_KEY",
    "your-api-key",
}


class BenchmarkPreflightError(RuntimeError):
    """评测前置条件不满足。"""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_slug(value: str, max_length: int = 120) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", ascii_value)
    slug = re.sub(r"-{2,}", "-", slug).strip("-._")
    return (slug or "document")[:max_length]


def canonical_source_id(pdf_path: Path) -> str:
    """将已知评测来源映射到数据集使用的稳定文档 ID。"""
    return CANONICAL_SOURCE_FILES.get(pdf_path.name, _safe_slug(pdf_path.stem))


def _extract_pdf_text(pdf_path: Path) -> tuple[str, int, list[str]]:
    reader = PdfReader(str(pdf_path), strict=False)
    page_sections: list[str] = []
    warnings: list[str] = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            warnings.append(
                f"page {page_number}: {type(exc).__name__}: {exc}"
            )
            continue

        text = text.replace("\x00", "")
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{4,}", "\n\n\n", text).strip()
        if text:
            page_sections.append(f"## Page {page_number}\n\n{text}")

    return "\n\n".join(page_sections), len(reader.pages), warnings


def _configure_project(
    project_dir: Path,
    *,
    model: str,
    embedding_model: str,
    top_k: int,
) -> bool:
    settings_path = project_dir / "settings.yaml"
    created = not settings_path.exists()
    if created:
        from graphrag.cli.initialize import initialize_project_at

        initialize_project_at(
            path=project_dir,
            force=False,
            model=model,
            embedding_model=embedding_model,
        )

    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    settings.setdefault("input", {})["type"] = "text"
    # GraphRAG 使用 string.Template 展开环境变量，正则末尾的 $ 需写成 $$。
    settings["input"]["file_pattern"] = r".*\.txt$$"
    settings.setdefault("extract_graph", {})["entity_types"] = [
        entity_type.value for entity_type in EntityType
    ]
    settings.setdefault("basic_search", {})["k"] = top_k
    settings.setdefault("local_search", {})["top_k_entities"] = top_k
    settings["local_search"]["top_k_relationships"] = top_k

    settings_path.write_text(
        yaml.safe_dump(
            settings,
            allow_unicode=True,
            sort_keys=False,
            width=100,
        ),
        encoding="utf-8",
    )
    return created


def _source_coverage(
    questions: Sequence[Question],
    available_source_ids: set[str],
) -> dict[str, Any]:
    required = {
        source.guide for question in questions for source in question.gold_sources
    }
    fully_covered = 0
    uncovered_questions: list[str] = []
    for question in questions:
        question_sources = {source.guide for source in question.gold_sources}
        if question_sources <= available_source_ids:
            fully_covered += 1
        else:
            uncovered_questions.append(question.question_id)

    return {
        "required_source_ids": sorted(required),
        "available_source_ids": sorted(required & available_source_ids),
        "missing_source_ids": sorted(required - available_source_ids),
        "fully_covered_questions": fully_covered,
        "total_questions": len(questions),
        "uncovered_question_ids": uncovered_questions,
    }


def prepare_corpus(
    *,
    raw_dir: Path,
    project_dir: Path,
    dataset_path: Path,
    model: str,
    embedding_model: str,
    top_k: int,
) -> dict[str, Any]:
    """提取 PDF 文本、初始化 GraphRAG 项目并生成来源清单。"""
    raw_dir = raw_dir.resolve()
    project_dir = project_dir.resolve()
    dataset_path = dataset_path.resolve()

    if not raw_dir.is_dir():
        raise BenchmarkPreflightError(f"原始语料目录不存在: {raw_dir}")

    pdf_paths = sorted(
        (
            path
            for path in raw_dir.iterdir()
            if path.is_file() and path.suffix.casefold() == ".pdf"
        ),
        key=lambda path: path.name.casefold(),
    )
    if not pdf_paths:
        raise BenchmarkPreflightError(f"原始语料目录中没有 PDF: {raw_dir}")

    project_dir.parent.mkdir(parents=True, exist_ok=True)
    project_created = _configure_project(
        project_dir,
        model=model,
        embedding_model=embedding_model,
        top_k=top_k,
    )
    input_dir = project_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    documents: list[dict[str, Any]] = []
    extracted_documents: dict[tuple[str, str], dict[str, Any]] = {}
    for index, pdf_path in enumerate(pdf_paths, start=1):
        pdf_hash = _sha256(pdf_path)
        source_id = canonical_source_id(pdf_path)
        output_name = f"{_safe_slug(source_id)}--{pdf_hash[:12]}.txt"
        output_path = input_dir / output_name

        duplicate = extracted_documents.get((source_id, pdf_hash))
        if duplicate is not None:
            print(
                f"[{index}/{len(pdf_paths)}] 跳过重复文件 {pdf_path.name}",
                flush=True,
            )
            documents.append(
                {
                    "source_id": source_id,
                    "pdf_file": pdf_path.name,
                    "pdf_sha256": pdf_hash,
                    "input_file": duplicate["input_file"],
                    "page_count": duplicate["page_count"],
                    "text_characters": duplicate["text_characters"],
                    "status": "duplicate",
                    "duplicate_of": duplicate["pdf_file"],
                    "warnings": [],
                }
            )
            continue

        print(f"[{index}/{len(pdf_paths)}] 提取 {pdf_path.name}", flush=True)

        try:
            body, page_count, page_warnings = _extract_pdf_text(pdf_path)
            if not body.strip():
                raise ValueError("未提取到可索引文本")

            header = "\n".join(
                [
                    f"SOURCE_ID: {source_id}",
                    f"ORIGINAL_FILE: {pdf_path.name}",
                    f"PDF_SHA256: {pdf_hash}",
                ]
            )
            output_path.write_text(f"{header}\n\n{body}\n", encoding="utf-8")
            status = "partial" if page_warnings else "ok"
            document = {
                "source_id": source_id,
                "pdf_file": pdf_path.name,
                "pdf_sha256": pdf_hash,
                "input_file": str(output_path.relative_to(project_dir)),
                "page_count": page_count,
                "text_characters": len(body),
                "status": status,
                "warnings": page_warnings,
            }
            documents.append(document)
            extracted_documents[(source_id, pdf_hash)] = document
        except Exception as exc:  # noqa: BLE001
            documents.append(
                {
                    "source_id": source_id,
                    "pdf_file": pdf_path.name,
                    "pdf_sha256": pdf_hash,
                    "input_file": None,
                    "page_count": 0,
                    "text_characters": 0,
                    "status": "error",
                    "warnings": [f"{type(exc).__name__}: {exc}"],
                }
            )

    questions = load_questions(dataset_path)
    successful_source_ids = {
        item["source_id"]
        for item in documents
        if item["status"] in {"ok", "partial"}
    }
    manifest = {
        "created_at": _now_iso(),
        "raw_dir": str(raw_dir),
        "project_dir": str(project_dir),
        "project_created": project_created,
        "pdf_count": len(pdf_paths),
        "indexed_document_count": sum(
            item["status"] in {"ok", "partial"} for item in documents
        ),
        "duplicate_count": sum(
            item["status"] == "duplicate" for item in documents
        ),
        "error_count": sum(item["status"] == "error" for item in documents),
        "partial_count": sum(item["status"] == "partial" for item in documents),
        "documents": documents,
        "dataset_source_coverage": _source_coverage(
            questions,
            successful_source_ids,
        ),
    }
    manifest_path = project_dir / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def _model_credential_issues(config: Any) -> list[str]:
    issues: list[str] = []
    model_groups = (
        ("completion", config.completion_models),
        ("embedding", config.embedding_models),
    )
    for group_name, models in model_groups:
        for model_id, model_config in models.items():
            if str(model_config.auth_method) != "api_key":
                continue
            api_key = str(model_config.api_key or "").strip()
            if api_key in PLACEHOLDER_API_KEYS:
                issues.append(
                    f"{group_name} model {model_id!r} 未配置有效 API key"
                )
    return issues


def preflight(
    *,
    project_dir: Path,
    dataset_path: Path,
    require_index: bool = False,
) -> dict[str, Any]:
    """在任何远程模型调用前检查本地输入、凭据和索引产物。"""
    from graphrag.config.load_config import load_config

    project_dir = project_dir.resolve()
    dataset_path = dataset_path.resolve()
    issues: list[str] = []
    warnings: list[str] = []

    settings_path = project_dir / "settings.yaml"
    manifest_path = project_dir / MANIFEST_NAME
    input_dir = project_dir / "input"

    if not settings_path.is_file():
        issues.append(f"缺少 GraphRAG 配置: {settings_path}")

    input_files = sorted(input_dir.glob("*.txt")) if input_dir.is_dir() else []
    if not input_files:
        issues.append(f"没有待索引文本: {input_dir}")

    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        coverage = manifest.get("dataset_source_coverage", {})
        missing = coverage.get("missing_source_ids", [])
        if missing:
            warnings.append(
                "原始语料缺少金标准来源: " + ", ".join(missing)
            )
        if manifest.get("error_count", 0):
            issues.append(
                f"{manifest['error_count']} 份 PDF 文本提取失败，详见 {manifest_path}"
            )
    else:
        warnings.append(f"缺少语料清单: {manifest_path}")

    questions: list[Question] = []
    dataset_errors: dict[str, list[str]] = {}
    if not dataset_path.is_file():
        issues.append(f"评测数据集不存在: {dataset_path}")
    else:
        questions = load_questions(dataset_path)
        dataset_errors = validate_dataset(questions)
        if dataset_errors:
            issues.append(f"{len(dataset_errors)} 道题未通过字段校验")

    pending_annotations = sum(
        any(annotation.rating == "pending" for annotation in question.annotators)
        for question in questions
    )
    if pending_annotations:
        warnings.append(
            f"{pending_annotations} 道题仍含 pending 标注，不应视为终版临床评测"
        )

    config = None
    if settings_path.is_file():
        try:
            config = load_config(project_dir)
            issues.extend(_model_credential_issues(config))
        except Exception as exc:  # noqa: BLE001
            issues.append(f"GraphRAG 配置加载失败: {type(exc).__name__}: {exc}")

    required_tables = [
        "documents.parquet",
        "text_units.parquet",
        "entities.parquet",
        "relationships.parquet",
        "communities.parquet",
        "community_reports.parquet",
    ]
    missing_tables: list[str] = []
    if config is not None:
        output_dir = Path(config.output_storage.base_dir)
        missing_tables = [
            table for table in required_tables if not (output_dir / table).is_file()
        ]
        if require_index and missing_tables:
            issues.append(
                "索引产物不完整，缺少: " + ", ".join(missing_tables)
            )

    return {
        "checked_at": _now_iso(),
        "project_dir": str(project_dir),
        "dataset_path": str(dataset_path),
        "input_document_count": len(input_files),
        "question_count": len(questions),
        "dataset_error_count": len(dataset_errors),
        "pending_annotation_count": pending_annotations,
        "missing_index_tables": missing_tables,
        "issues": issues,
        "warnings": warnings,
        "ready": not issues,
    }


def _require_preflight(report: dict[str, Any]) -> None:
    if report["issues"]:
        details = "\n".join(f"- {issue}" for issue in report["issues"])
        raise BenchmarkPreflightError(
            "前置检查未通过，未发起任何远程模型调用:\n" + details
        )


def build_index(
    *,
    project_dir: Path,
    dataset_path: Path,
    method: str,
    cache: bool,
    skip_validation: bool,
    verbose: bool,
) -> dict[str, Any]:
    """通过仓库内的 GraphRAGClient 构建知识图谱索引。"""
    from benchmark.baseline.microsoft_graphrag_client import GraphRAGClient

    report = preflight(
        project_dir=project_dir,
        dataset_path=dataset_path,
        require_index=False,
    )
    _require_preflight(report)

    started = time.monotonic()
    client = GraphRAGClient(project_dir, verbose=verbose)
    result = client.index(
        method=method,
        cache=cache,
        skip_validation=skip_validation,
    )
    summary = {
        "completed_at": _now_iso(),
        "method": method,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "workflow_names": result.workflow_names,
        "workflow_count": len(result.outputs),
        "error_count": result.error_count,
        "errors": result.errors,
        "has_errors": result.has_errors,
    }
    if result.has_errors:
        raise RuntimeError(
            "GraphRAG 索引包含错误:\n"
            + "\n".join(f"- {error}" for error in result.errors)
        )
    return summary


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item is not None]
    if isinstance(value, str):
        return [value] if value else []
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    return [str(value)]


class SourceResolver:
    """将 GraphRAG 查询返回的 text-unit ID 还原为稳定指南 ID。"""

    def __init__(
        self,
        documents: pd.DataFrame,
        text_units: pd.DataFrame,
        manifest: dict[str, Any],
    ) -> None:
        self._document_sources: dict[str, str] = {}
        self._text_unit_sources: dict[str, str] = {}
        self._text_sources: dict[str, str] = {}

        title_sources: dict[str, str] = {}
        for item in manifest.get("documents", []):
            input_file = item.get("input_file")
            if not input_file or item.get("status") == "error":
                continue
            source_id = str(item["source_id"])
            input_path = Path(input_file)
            for title in (input_path.name, input_path.stem):
                title_sources[title] = source_id

        for _, row in documents.iterrows():
            document_id = str(row.get("id", ""))
            title = str(row.get("title", ""))
            source_id = title_sources.get(title) or title_sources.get(
                Path(title).name
            )
            if source_id is None:
                source_id = _source_id_from_text(str(row.get("text", "")))
            if document_id and source_id:
                self._document_sources[document_id] = source_id

        for _, row in text_units.iterrows():
            document_ids = _as_string_list(row.get("document_id"))
            if not document_ids:
                document_ids = _as_string_list(row.get("document_ids"))
            source_id = next(
                (
                    self._document_sources[document_id]
                    for document_id in document_ids
                    if document_id in self._document_sources
                ),
                None,
            )
            if source_id is None:
                source_id = _source_id_from_text(str(row.get("text", "")))
            if source_id is None:
                continue

            for key in ("id", "human_readable_id"):
                value = row.get(key)
                if value is not None:
                    self._text_unit_sources[str(value)] = source_id
            text = str(row.get("text", "")).strip()
            if text:
                self._text_sources[text] = source_id

    def resolve(self, record_id: Any, text: str) -> str:
        record_key = str(record_id)
        if record_key in self._text_unit_sources:
            return self._text_unit_sources[record_key]
        normalized_text = text.strip()
        if normalized_text in self._text_sources:
            return self._text_sources[normalized_text]
        return _source_id_from_text(text) or f"unresolved:{record_key}"


def _source_id_from_text(text: str) -> str | None:
    match = re.search(r"(?m)^SOURCE_ID:\s*(\S+)\s*$", text)
    return match.group(1) if match else None


def _context_frames(context_data: Any) -> list[tuple[str, pd.DataFrame]]:
    frames: list[tuple[str, pd.DataFrame]] = []
    if isinstance(context_data, pd.DataFrame):
        return [("sources", context_data)]
    if isinstance(context_data, dict):
        for name, value in context_data.items():
            if isinstance(value, pd.DataFrame):
                frames.append((str(name), value))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, pd.DataFrame):
                        frames.append((str(name), item))
    elif isinstance(context_data, list):
        for index, value in enumerate(context_data):
            if isinstance(value, pd.DataFrame):
                frames.append((f"context_{index}", value))
    return frames


def _extract_retrieved_context(
    context_data: Any,
    resolver: SourceResolver,
    k: int,
) -> list[dict[str, str]]:
    frames = _context_frames(context_data)
    source_frames = [
        item for item in frames if item[0].casefold() in {"source", "sources"}
    ]
    candidate_frames = source_frames or frames
    records: list[dict[str, str]] = []

    for frame_name, frame in candidate_frames:
        for _, row in frame.iterrows():
            text = ""
            for column in ("text", "content", "description", "summary"):
                value = row.get(column)
                if value is not None and str(value).strip():
                    text = str(value)
                    break
            if not text:
                continue
            record_id = row.get("id", row.get("human_readable_id", "unknown"))
            records.append(
                {
                    "context_table": frame_name,
                    "record_id": str(record_id),
                    "source_id": resolver.resolve(record_id, text),
                    "text": text,
                }
            )
            if len(records) >= k:
                return records
    return records


def _supported_statements(
    text: str,
    statements: Sequence[str],
) -> set[str]:
    text_lower = text.lower()
    supported: set[str] = set()
    for statement in statements:
        keywords = [
            word
            for word in re.split(r"[，,。；;、\s]+", statement)
            if len(word) > 2
        ]
        if not keywords:
            supported.add(statement)
            continue
        matched = sum(keyword.lower() in text_lower for keyword in keywords)
        if matched / len(keywords) >= 0.5:
            supported.add(statement)
    return supported


def _safety_actions(answer: str) -> tuple[bool, bool]:
    answer_lower = answer.lower()
    refusal_phrases = (
        "cannot determine",
        "can't determine",
        "cannot answer",
        "can't answer",
        "insufficient information",
        "not enough information",
        "outside the provided",
        "not contained in",
    )
    referral_phrases = (
        "consult",
        "seek medical",
        "healthcare professional",
        "healthcare provider",
        "qualified professional",
        "specialist",
        "obstetrician",
        "medical supervision",
    )
    return (
        any(phrase in answer_lower for phrase in refusal_phrases),
        any(phrase in answer_lower for phrase in referral_phrases),
    )


def _normalize_response(response: Any) -> str:
    if isinstance(response, str):
        return response
    return json.dumps(response, ensure_ascii=False)


def _selected_method(question: Question, requested_method: str) -> str:
    if requested_method != "adaptive":
        return requested_method
    try:
        return ADAPTIVE_SEARCH_METHODS[question.rag_arch_type]
    except KeyError as exc:
        raise ValueError(
            f"不支持的 rag_arch_type: {question.rag_arch_type}"
        ) from exc


def _query(client: Any, question: Question, requested_method: str) -> Any:
    method = _selected_method(question, requested_method)
    search = getattr(client, f"{method}_search")
    return method, search(query=question.question)


def _load_source_resolver(client: Any, project_dir: Path) -> SourceResolver:
    from benchmark.baseline.microsoft_graphrag_client.data.data_loader import (
        DataLoader,
    )

    tables = DataLoader(client.config).load(["documents", "text_units"])
    manifest_path = project_dir / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents = tables["documents"]
    text_units = tables["text_units"]
    if not isinstance(documents, pd.DataFrame) or not isinstance(
        text_units, pd.DataFrame
    ):
        raise RuntimeError("无法加载 documents/text_units 索引表")
    return SourceResolver(documents, text_units, manifest)


def evaluate(
    *,
    project_dir: Path,
    dataset_path: Path,
    requested_method: str,
    k: int,
    output_path: Path | None,
    limit: int | None,
    question_ids: Sequence[str],
    fail_fast: bool,
    verbose: bool,
) -> dict[str, Any]:
    """查询数据集、计算三层指标并保存可审计结果。"""
    from benchmark.baseline.microsoft_graphrag_client import GraphRAGClient

    project_dir = project_dir.resolve()
    dataset_path = dataset_path.resolve()
    preflight_report = preflight(
        project_dir=project_dir,
        dataset_path=dataset_path,
        require_index=True,
    )
    _require_preflight(preflight_report)

    questions = load_questions(dataset_path)
    if question_ids:
        selected_ids = set(question_ids)
        questions = [
            question
            for question in questions
            if question.question_id in selected_ids
        ]
        missing_ids = selected_ids - {
            question.question_id for question in questions
        }
        if missing_ids:
            raise BenchmarkPreflightError(
                "数据集中不存在题目: " + ", ".join(sorted(missing_ids))
            )
    if limit is not None:
        questions = questions[:limit]
    if not questions:
        raise BenchmarkPreflightError("筛选后没有待评测题目")

    if output_path is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_path = DEFAULT_RESULTS_DIR / f"evaluation-{timestamp}.json"
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output_path.with_suffix(".jsonl")

    client = GraphRAGClient(project_dir, verbose=verbose)
    resolver = _load_source_resolver(client, project_dir)
    scoring_report = DatasetScoringReport()
    raw_results: list[dict[str, Any]] = []
    method_counts: Counter[str] = Counter()
    started = time.monotonic()

    with raw_path.open("w", encoding="utf-8") as raw_stream:
        for index, question in enumerate(questions, start=1):
            question_started = time.monotonic()
            selected_method = _selected_method(question, requested_method)
            print(
                f"[{index}/{len(questions)}] "
                f"{question.question_id} ({selected_method})",
                flush=True,
            )
            error: str | None = None
            try:
                method, query_result = _query(
                    client,
                    question,
                    requested_method,
                )
                method_counts[method] += 1
                answer = _normalize_response(query_result.response)
                contexts = _extract_retrieved_context(
                    query_result.context_data,
                    resolver,
                    k,
                )
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
                answer = ""
                contexts = []
                method_counts[selected_method] += 1
                if fail_fast:
                    raise

            retrieved_sources = [item["source_id"] for item in contexts]
            retrieved_context = "\n\n".join(item["text"] for item in contexts)
            retrieved_supports = [
                _supported_statements(
                    item["text"],
                    question.must_have_statements,
                )
                for item in contexts
            ]
            refused, referred = _safety_actions(answer)
            scoring_result = score_question(
                question=question,
                answer=answer,
                retrieved_sources=retrieved_sources,
                retrieved_context=retrieved_context,
                retrieved_supports=retrieved_supports,
                refused=refused,
                referred=referred,
                k=k,
            )
            scoring_report.add(scoring_result)

            item = {
                "question_id": question.question_id,
                "question": question.question,
                "difficulty": question.difficulty,
                "rag_arch_type": question.rag_arch_type,
                "search_method": selected_method,
                "answer": answer,
                "retrieved_sources": retrieved_sources,
                "contexts": contexts,
                "refused": refused,
                "referred": referred,
                "error": error,
                "elapsed_seconds": round(
                    time.monotonic() - question_started,
                    3,
                ),
                "scoring": asdict(scoring_result),
            }
            raw_results.append(item)
            raw_stream.write(
                json.dumps(item, ensure_ascii=False) + "\n"
            )
            raw_stream.flush()

    report = {
        "metadata": {
            "created_at": _now_iso(),
            "graphrag_version": version("graphrag"),
            "project_dir": str(project_dir),
            "dataset_path": str(dataset_path),
            "dataset_fingerprint": compute_dataset_fingerprint(questions),
            "requested_search_method": requested_method,
            "actual_search_methods": dict(method_counts),
            "k": k,
            "question_count": len(questions),
            "failed_query_count": sum(
                result["error"] is not None for result in raw_results
            ),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "raw_results_path": str(raw_path),
            "scoring_note": (
                "当前生成指标使用 benchmark.qa.scoring 的词法近似算法，"
                "不是 LLM-as-Judge；L3/L4 与 pending 标注仍需专家复核。"
            ),
        },
        "preflight": preflight_report,
        "summary": scoring_report.summary(),
        "results": raw_results,
    }
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report["metadata"]["output_path"] = str(output_path)
    return report


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _add_project_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--project-dir",
        type=_path,
        default=DEFAULT_PROJECT_DIR,
        help="GraphRAG 项目目录",
    )


def _add_dataset_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        type=_path,
        default=DEFAULT_DATASET,
        help="问答数据集 JSON",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="构建并评测 Microsoft GraphRAG 产前超声基线",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="提取 PDF 并初始化 GraphRAG 项目",
    )
    prepare_parser.add_argument(
        "--raw-dir",
        type=_path,
        default=DEFAULT_RAW_DIR,
    )
    _add_project_argument(prepare_parser)
    _add_dataset_argument(prepare_parser)
    prepare_parser.add_argument("--model", default="gpt-4.1")
    prepare_parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-large",
    )
    prepare_parser.add_argument("--k", type=int, default=16)

    preflight_parser = subparsers.add_parser(
        "preflight",
        help="执行不联网的输入、凭据与索引检查",
    )
    _add_project_argument(preflight_parser)
    _add_dataset_argument(preflight_parser)
    preflight_parser.add_argument(
        "--require-index",
        action="store_true",
    )

    index_parser = subparsers.add_parser(
        "index",
        help="通过 GraphRAGClient 构建索引",
    )
    _add_project_argument(index_parser)
    _add_dataset_argument(index_parser)
    index_parser.add_argument(
        "--method",
        choices=("standard", "fast", "standard-update", "fast-update"),
        default="standard",
    )
    index_parser.add_argument("--no-cache", action="store_true")
    index_parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="跳过 GraphRAG 的远程模型连通性测试",
    )
    index_parser.add_argument("--verbose", action="store_true")

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="运行查询并生成评分报告",
    )
    _add_project_argument(evaluate_parser)
    _add_dataset_argument(evaluate_parser)
    evaluate_parser.add_argument(
        "--method",
        choices=("adaptive", "basic", "local", "drift", "global"),
        default="adaptive",
    )
    evaluate_parser.add_argument("--k", type=int, default=16)
    evaluate_parser.add_argument("--output", type=_path)
    evaluate_parser.add_argument("--limit", type=int)
    evaluate_parser.add_argument(
        "--question-id",
        action="append",
        default=[],
    )
    evaluate_parser.add_argument("--fail-fast", action="store_true")
    evaluate_parser.add_argument("--verbose", action="store_true")

    run_parser = subparsers.add_parser(
        "run",
        help="依次执行 prepare、index 和 evaluate",
    )
    run_parser.add_argument(
        "--raw-dir",
        type=_path,
        default=DEFAULT_RAW_DIR,
    )
    _add_project_argument(run_parser)
    _add_dataset_argument(run_parser)
    run_parser.add_argument("--model", default="gpt-4.1")
    run_parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-large",
    )
    run_parser.add_argument(
        "--index-method",
        choices=("standard", "fast"),
        default="standard",
    )
    run_parser.add_argument(
        "--search-method",
        choices=("adaptive", "basic", "local", "drift", "global"),
        default="adaptive",
    )
    run_parser.add_argument("--k", type=int, default=16)
    run_parser.add_argument("--output", type=_path)
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--no-cache", action="store_true")
    run_parser.add_argument("--skip-validation", action="store_true")
    run_parser.add_argument("--fail-fast", action="store_true")
    run_parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            manifest = prepare_corpus(
                raw_dir=args.raw_dir,
                project_dir=args.project_dir,
                dataset_path=args.dataset,
                model=args.model,
                embedding_model=args.embedding_model,
                top_k=args.k,
            )
            result = {
                "manifest_path": str(
                    args.project_dir.resolve() / MANIFEST_NAME
                ),
                "pdf_count": manifest["pdf_count"],
                "indexed_document_count": manifest[
                    "indexed_document_count"
                ],
                "duplicate_count": manifest["duplicate_count"],
                "error_count": manifest["error_count"],
                "dataset_source_coverage": manifest[
                    "dataset_source_coverage"
                ],
            }
        elif args.command == "preflight":
            result = preflight(
                project_dir=args.project_dir,
                dataset_path=args.dataset,
                require_index=args.require_index,
            )
        elif args.command == "index":
            result = build_index(
                project_dir=args.project_dir,
                dataset_path=args.dataset,
                method=args.method,
                cache=not args.no_cache,
                skip_validation=args.skip_validation,
                verbose=args.verbose,
            )
        elif args.command == "evaluate":
            evaluation_result = evaluate(
                project_dir=args.project_dir,
                dataset_path=args.dataset,
                requested_method=args.method,
                k=args.k,
                output_path=args.output,
                limit=args.limit,
                question_ids=args.question_id,
                fail_fast=args.fail_fast,
                verbose=args.verbose,
            )
            result = {
                "output_path": evaluation_result["metadata"]["output_path"],
                "raw_results_path": evaluation_result["metadata"][
                    "raw_results_path"
                ],
                "metadata": evaluation_result["metadata"],
                "summary": evaluation_result["summary"],
            }
        else:
            prepare_corpus(
                raw_dir=args.raw_dir,
                project_dir=args.project_dir,
                dataset_path=args.dataset,
                model=args.model,
                embedding_model=args.embedding_model,
                top_k=args.k,
            )
            index_result = build_index(
                project_dir=args.project_dir,
                dataset_path=args.dataset,
                method=args.index_method,
                cache=not args.no_cache,
                skip_validation=args.skip_validation,
                verbose=args.verbose,
            )
            evaluation_result = evaluate(
                project_dir=args.project_dir,
                dataset_path=args.dataset,
                requested_method=args.search_method,
                k=args.k,
                output_path=args.output,
                limit=args.limit,
                question_ids=[],
                fail_fast=args.fail_fast,
                verbose=args.verbose,
            )
            result = {
                "index": index_result,
                "evaluation": evaluation_result["summary"],
                "output_path": evaluation_result["metadata"]["output_path"],
            }

        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except BenchmarkPreflightError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
