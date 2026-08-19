"""Microsoft GraphRAG 产前超声基线的端到端评测入口。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import yaml

from benchmark.baseline.microsoft_graphrag_client.config.llm_config import (
    DEFAULT_COMPLETION_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    LLMConfigOverrides,
    ModelConfigOverride,
)
from benchmark.common import (
    MANIFEST_NAME,
    aggregate_usage,
    canonical_source_id,
    empty_usage,
    extract_pdf_text,
    extract_retrieved_context,
    load_scoring_options,
    merge_usage,
    normalize_response,
    now_iso,
    safe_slug,
    safety_actions,
    sha256_file,
    source_id_from_text,
    supported_statements,
    write_manifest,
)
from benchmark.common.scoring_options import DEFAULT_SCORING_CONFIG
from benchmark.qa import (
    DatasetScoringReport,
    EntityType,
    JudgeConfig,
    Question,
    compute_dataset_fingerprint,
    judge_answer,
    load_questions,
    score_question,
    validate_dataset,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = REPOSITORY_ROOT / "benchmark" / "data" / "raw"
DEFAULT_PROJECT_DIR = REPOSITORY_ROOT / "benchmark" / "data" / "microsoft_graphrag"
DEFAULT_DATASET = (
    REPOSITORY_ROOT / "benchmark" / "qa" / "dataset" / "sample_questions.json"
)
DEFAULT_RESULTS_DIR = REPOSITORY_ROOT / "benchmark" / "results" / "microsoft_graphrag"

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


def _apply_llm_env_references(
    settings: dict[str, Any],
    *,
    model_env: str | None,
    api_key_env: str | None,
    api_base_env: str | None,
    embedding_model_env: str | None,
    embedding_api_key_env: str | None,
    embedding_api_base_env: str | None,
) -> None:
    """将自定义环境变量名写入 settings.yaml 的 LLM 模型配置（``${VAR}`` 引用）。

    仅当对应参数非空时写入，未指定则保留现有引用。embedding 缺省沿用
    completion 的变量名。
    """
    completion = settings.setdefault("completion_models", {}).setdefault(
        "default_completion_model", {}
    )
    embedding = settings.setdefault("embedding_models", {}).setdefault(
        "default_embedding_model", {}
    )
    if model_env:
        completion["model"] = "${" + model_env + "}"
    if api_key_env:
        completion["api_key"] = "${" + api_key_env + "}"
    if api_base_env:
        completion["api_base"] = "${" + api_base_env + "}"
    if embedding_model_env:
        embedding["model"] = "${" + embedding_model_env + "}"
    if embedding_api_key_env:
        embedding["api_key"] = "${" + embedding_api_key_env + "}"
    if embedding_api_base_env:
        embedding["api_base"] = "${" + embedding_api_base_env + "}"


def _configure_project(
    project_dir: Path,
    *,
    model: str,
    embedding_model: str,
    top_k: int,
    model_env: str | None = None,
    api_key_env: str | None = None,
    api_base_env: str | None = None,
    embedding_model_env: str | None = None,
    embedding_api_key_env: str | None = None,
    embedding_api_base_env: str | None = None,
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
    _apply_llm_env_references(
        settings,
        model_env=model_env,
        api_key_env=api_key_env,
        api_base_env=api_base_env,
        embedding_model_env=(embedding_model_env or model_env),
        embedding_api_key_env=(embedding_api_key_env or api_key_env),
        embedding_api_base_env=(embedding_api_base_env or api_base_env),
    )

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
    model_env: str | None = None,
    api_key_env: str | None = None,
    api_base_env: str | None = None,
    embedding_model_env: str | None = None,
    embedding_api_key_env: str | None = None,
    embedding_api_base_env: str | None = None,
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
        model_env=model_env,
        api_key_env=api_key_env,
        api_base_env=api_base_env,
        embedding_model_env=embedding_model_env,
        embedding_api_key_env=embedding_api_key_env,
        embedding_api_base_env=embedding_api_base_env,
    )
    input_dir = project_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    documents: list[dict[str, Any]] = []
    extracted_documents: dict[tuple[str, str], dict[str, Any]] = {}
    for index, pdf_path in enumerate(pdf_paths, start=1):
        pdf_hash = sha256_file(pdf_path)
        source_id = canonical_source_id(pdf_path)
        output_name = f"{safe_slug(source_id)}--{pdf_hash[:12]}.txt"
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
            body, page_count, page_warnings = extract_pdf_text(pdf_path)
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
        item["source_id"] for item in documents if item["status"] in {"ok", "partial"}
    }
    manifest = {
        "created_at": now_iso(),
        "raw_dir": str(raw_dir),
        "project_dir": str(project_dir),
        "project_created": project_created,
        "pdf_count": len(pdf_paths),
        "indexed_document_count": sum(
            item["status"] in {"ok", "partial"} for item in documents
        ),
        "duplicate_count": sum(item["status"] == "duplicate" for item in documents),
        "error_count": sum(item["status"] == "error" for item in documents),
        "partial_count": sum(item["status"] == "partial" for item in documents),
        "documents": documents,
        "dataset_source_coverage": _source_coverage(
            questions,
            successful_source_ids,
        ),
    }
    write_manifest(project_dir, manifest)
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
                issues.append(f"{group_name} model {model_id!r} 未配置有效 API key")
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
            warnings.append("原始语料缺少金标准来源: " + ", ".join(missing))
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
            issues.append("索引产物不完整，缺少: " + ", ".join(missing_tables))

    return {
        "checked_at": now_iso(),
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
    llm_overrides: LLMConfigOverrides | None = None,
) -> dict[str, Any]:
    """通过仓库内的 GraphRAGClient 构建知识图谱索引。"""
    from benchmark.baseline.microsoft_graphrag_client import GraphRAGClient

    # 必须先解析为绝对路径：preflight 中的 load_config 会 os.chdir 到配置目录，
    # 若保持相对路径，后续 GraphRAGClient 会基于新 cwd 二次拼接导致路径错误。
    project_dir = project_dir.resolve()
    dataset_path = dataset_path.resolve()

    report = preflight(
        project_dir=project_dir,
        dataset_path=dataset_path,
        require_index=False,
    )
    _require_preflight(report)

    started = time.monotonic()
    client = GraphRAGClient(
        project_dir,
        llm_overrides=llm_overrides,
        verbose=verbose,
    )
    result = client.index(
        method=method,
        cache=cache,
        skip_validation=skip_validation,
    )
    summary = {
        "completed_at": now_iso(),
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
    from benchmark.common.retrieval import as_string_list

    return as_string_list(value)


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
            source_id = title_sources.get(title) or title_sources.get(Path(title).name)
            if source_id is None:
                source_id = source_id_from_text(str(row.get("text", "")))
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
                source_id = source_id_from_text(str(row.get("text", "")))
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
        return source_id_from_text(text) or f"unresolved:{record_key}"


def _selected_method(question: Question, requested_method: str) -> str:
    if requested_method != "adaptive":
        return requested_method
    try:
        return ADAPTIVE_SEARCH_METHODS[question.rag_arch_type]
    except KeyError as exc:
        raise ValueError(f"不支持的 rag_arch_type: {question.rag_arch_type}") from exc


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
    llm_overrides: LLMConfigOverrides | None = None,
    source_match_mode: str | None = None,
    scoring_config_path: Path | None = None,
    judge_mode: str = "off",
    judge_model: str | None = None,
    judge_api_key_env: str = "OPENAI_API_KEY",
    judge_api_base: str | None = None,
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
    scoring_options = load_scoring_options(scoring_config_path)
    selected_source_mode = source_match_mode or scoring_options["source_match_mode"]
    if selected_source_mode not in {"exact", "evidence", "hybrid"}:
        raise BenchmarkPreflightError(
            "source-match-mode 必须为 exact、evidence 或 hybrid"
        )
    if judge_mode not in {"off", "optional", "required"}:
        raise BenchmarkPreflightError("judge-mode 必须为 off、optional 或 required")
    judge_config = (
        JudgeConfig(
            model=judge_model,
            api_key_env=judge_api_key_env,
            api_base=judge_api_base,
        )
        if judge_mode != "off" and judge_model
        else None
    )
    if question_ids:
        selected_ids = set(question_ids)
        questions = [
            question for question in questions if question.question_id in selected_ids
        ]
        missing_ids = selected_ids - {question.question_id for question in questions}
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

    client = GraphRAGClient(
        project_dir,
        llm_overrides=llm_overrides,
        verbose=verbose,
    )
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
            query_telemetry: dict[str, Any] = {}
            try:
                method, query_result = _query(
                    client,
                    question,
                    requested_method,
                )
                method_counts[method] += 1
                answer = normalize_response(query_result.response)
                contexts = extract_retrieved_context(
                    query_result.context_data,
                    resolver,
                    k,
                )
                query_telemetry = dict(query_result.telemetry or {})
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
                supported_statements(
                    item["text"],
                    question.must_have_statements,
                )
                for item in contexts
            ]
            refused, referred = safety_actions(answer)
            judge_result: dict[str, Any] | None = None
            if judge_mode != "off" and error is None:
                if judge_config is None:
                    judge_result = {
                        "model": judge_model or "",
                        "error": "未提供 --judge-model，已回退 lexical",
                        "usage": {
                            **empty_usage(),
                            "failed_request_count": 1,
                        },
                    }
                else:
                    judge_result = judge_answer(
                        question=question.question,
                        answer=answer,
                        context=retrieved_context,
                        gold_answer=question.gold_answer,
                        must_have_statements=question.must_have_statements,
                        config=judge_config,
                    )
            scoring_result = score_question(
                question=question,
                answer=answer,
                retrieved_sources=retrieved_sources,
                retrieved_context=retrieved_context,
                retrieved_supports=retrieved_supports,
                refused=refused,
                referred=referred,
                k=k,
                source_match_mode=selected_source_mode,
                source_equivalence=scoring_options["source_equivalence"],
                judge_result=judge_result,
            )
            scoring_report.add(scoring_result)

            query_usage = dict(query_telemetry.get("usage") or empty_usage())
            judge_usage = dict((judge_result or {}).get("usage") or empty_usage())
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
                "telemetry": query_telemetry,
                "usage": {
                    "query": query_usage,
                    "judge": judge_usage,
                    "total": merge_usage(query_usage, judge_usage),
                },
                "elapsed_seconds": round(
                    time.monotonic() - question_started,
                    3,
                ),
                "scoring": asdict(scoring_result),
            }
            raw_results.append(item)
            raw_stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            raw_stream.flush()

    summary = scoring_report.summary()
    summary["usage"] = aggregate_usage(raw_results)
    report = {
        "metadata": {
            "created_at": now_iso(),
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
                "每题同时保留 lexical 指标；Judge 开启且成功时 selected=judge，"
                "失败时回退 lexical。来源校验同时保留 exact 与 evidence/equivalent 口径。"
            ),
            "scoring_method": (
                "lexical"
                if judge_mode == "off"
                else (
                    "judge"
                    if raw_results
                    and all(
                        item["scoring"].get("scoring_method") == "judge"
                        for item in raw_results
                    )
                    else (
                        "mixed"
                        if any(
                            item["scoring"].get("scoring_method") == "judge"
                            for item in raw_results
                        )
                        else "lexical"
                    )
                )
            ),
            "judge_mode": judge_mode,
            "judge_model": judge_model,
            "source_match_mode": selected_source_mode,
            "scoring_config_path": str(
                (scoring_config_path or DEFAULT_SCORING_CONFIG).resolve()
            ),
        },
        "preflight": preflight_report,
        "summary": summary,
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


def _add_llm_override_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_model_names: bool = True,
) -> None:
    """添加运行时 LLM 模型/供应商覆盖参数。

    所有参数默认 ``None``：不提供时沿用 ``settings.yaml``，提供时仅运行时
    覆盖，不写回配置文件。
    """
    group = parser.add_argument_group(
        "LLM 模型/供应商覆盖（运行时生效，不改写 settings.yaml）"
    )
    if include_model_names:
        group.add_argument(
            "--model",
            default=None,
            help="completion 模型名（如 gpt-4.1）",
        )
        group.add_argument(
            "--embedding-model",
            default=None,
            help="embedding 模型名（如 text-embedding-3-large）",
        )
        group.add_argument(
            "--model-env",
            default=None,
            help="completion 模型名环境变量（运行时解析，如 RAG_COMPLETION_MODEL）",
        )
        group.add_argument(
            "--embedding-model-env",
            default=None,
            help="embedding 模型名环境变量（运行时解析）",
        )
    group.add_argument(
        "--model-provider",
        default=None,
        help="completion 供应商（如 openai/azure）",
    )
    group.add_argument(
        "--embedding-model-provider",
        default=None,
        help="embedding 供应商（如 openai/azure）",
    )
    group.add_argument(
        "--api-base",
        default=None,
        help="completion 与 embedding 共享 API base URL（含 /v1）",
    )
    group.add_argument(
        "--embedding-api-base",
        default=None,
        help="embedding 专属 API base URL（覆盖 --api-base）",
    )
    group.add_argument(
        "--api-base-env",
        default=None,
        help="completion 与 embedding 共享 API base 环境变量名（运行时解析）",
    )
    group.add_argument(
        "--embedding-api-base-env",
        default=None,
        help="embedding 专属 API base 环境变量名（覆盖 --api-base-env）",
    )
    group.add_argument(
        "--api-version",
        default=None,
        help="API version（Azure 等供应商需要时）",
    )
    group.add_argument(
        "--api-key-env",
        default=None,
        help="API key 环境变量名（如 RAG_API_KEY）",
    )
    group.add_argument(
        "--embedding-api-key-env",
        default=None,
        help="embedding 专属 API key 环境变量名（覆盖 --api-key-env）",
    )


def _llm_overrides_from_args(
    args: argparse.Namespace,
) -> LLMConfigOverrides | None:
    """从 CLI 参数构建 LLM 模型/供应商运行时覆盖。"""
    completion_kwargs: dict[str, str] = {}
    for attr, key in (
        ("model", "model"),
        ("model_env", "model_env"),
        ("model_provider", "model_provider"),
        ("api_base", "api_base"),
        ("api_base_env", "api_base_env"),
        ("api_version", "api_version"),
        ("api_key_env", "api_key_env"),
    ):
        value = getattr(args, attr, None)
        if value:
            completion_kwargs[key] = value

    embedding_kwargs: dict[str, str] = {}
    for attr, key in (
        ("embedding_model", "model"),
        ("embedding_model_env", "model_env"),
        ("embedding_model_provider", "model_provider"),
    ):
        value = getattr(args, attr, None)
        if value:
            embedding_kwargs[key] = value
    # 共享项也作用于 embedding；embedding 专属项优先。
    for attr, key in (
        ("model_env", "model_env"),
        ("api_base", "api_base"),
        ("api_base_env", "api_base_env"),
        ("api_version", "api_version"),
        ("api_key_env", "api_key_env"),
    ):
        value = getattr(args, attr, None)
        if value:
            embedding_kwargs.setdefault(key, value)
    for attr, key in (
        ("embedding_model_env", "model_env"),
        ("embedding_api_base", "api_base"),
        ("embedding_api_base_env", "api_base_env"),
        ("embedding_api_key_env", "api_key_env"),
    ):
        value = getattr(args, attr, None)
        if value:
            embedding_kwargs[key] = value

    overrides = LLMConfigOverrides(
        completion=ModelConfigOverride(**completion_kwargs),
        embedding=ModelConfigOverride(**embedding_kwargs),
    )
    return overrides if not overrides.is_empty() else None


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
    prepare_parser.add_argument(
        "--api-key-env",
        default=None,
        help="completion API key 环境变量名（写入 settings.yaml，如 MY_API_KEY）",
    )
    prepare_parser.add_argument(
        "--api-base-env",
        default=None,
        help="completion API base 环境变量名（写入 settings.yaml，如 MY_API_BASE）",
    )
    prepare_parser.add_argument(
        "--embedding-api-key-env",
        default=None,
        help="embedding 专属 API key 环境变量名（缺省沿用 --api-key-env）",
    )
    prepare_parser.add_argument(
        "--embedding-api-base-env",
        default=None,
        help="embedding 专属 API base 环境变量名（缺省沿用 --api-base-env）",
    )
    prepare_parser.add_argument(
        "--model-env",
        default=None,
        help="completion 模型名环境变量（写入 settings.yaml，如 RAG_COMPLETION_MODEL）",
    )
    prepare_parser.add_argument(
        "--embedding-model-env",
        default=None,
        help="embedding 模型名环境变量（写入 settings.yaml，缺省沿用 --model-env）",
    )

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
    _add_llm_override_arguments(index_parser)
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
    _add_llm_override_arguments(evaluate_parser)
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
    evaluate_parser.add_argument(
        "--source-match-mode",
        choices=("exact", "evidence", "hybrid"),
        default=None,
        help="金标准来源校验口径；默认读取 scoring_config.yaml",
    )
    evaluate_parser.add_argument("--scoring-config", type=_path, default=None)
    evaluate_parser.add_argument(
        "--judge-mode",
        choices=("off", "optional", "required"),
        default="off",
        help="LLM-as-Judge；默认关闭，失败时回退 lexical",
    )
    evaluate_parser.add_argument("--judge-model", default=None)
    evaluate_parser.add_argument("--judge-api-key-env", default="OPENAI_API_KEY")
    evaluate_parser.add_argument("--judge-api-base", default=None)
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
    run_parser.add_argument(
        "--model",
        default=None,
        help="completion 模型名（初始化与运行时覆盖）",
    )
    run_parser.add_argument(
        "--embedding-model",
        default=None,
        help="embedding 模型名（初始化与运行时覆盖）",
    )
    _add_llm_override_arguments(run_parser, include_model_names=False)
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
    run_parser.add_argument(
        "--source-match-mode",
        choices=("exact", "evidence", "hybrid"),
        default=None,
    )
    run_parser.add_argument("--scoring-config", type=_path, default=None)
    run_parser.add_argument(
        "--judge-mode",
        choices=("off", "optional", "required"),
        default="off",
    )
    run_parser.add_argument("--judge-model", default=None)
    run_parser.add_argument("--judge-api-key-env", default="OPENAI_API_KEY")
    run_parser.add_argument("--judge-api-base", default=None)
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
                model_env=args.model_env,
                api_key_env=args.api_key_env,
                api_base_env=args.api_base_env,
                embedding_model_env=args.embedding_model_env,
                embedding_api_key_env=args.embedding_api_key_env,
                embedding_api_base_env=args.embedding_api_base_env,
            )
            result = {
                "manifest_path": str(args.project_dir.resolve() / MANIFEST_NAME),
                "pdf_count": manifest["pdf_count"],
                "indexed_document_count": manifest["indexed_document_count"],
                "duplicate_count": manifest["duplicate_count"],
                "error_count": manifest["error_count"],
                "dataset_source_coverage": manifest["dataset_source_coverage"],
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
                llm_overrides=_llm_overrides_from_args(args),
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
                llm_overrides=_llm_overrides_from_args(args),
                source_match_mode=args.source_match_mode,
                scoring_config_path=args.scoring_config,
                judge_mode=args.judge_mode,
                judge_model=args.judge_model,
                judge_api_key_env=args.judge_api_key_env,
                judge_api_base=args.judge_api_base,
            )
            result = {
                "output_path": evaluation_result["metadata"]["output_path"],
                "raw_results_path": evaluation_result["metadata"]["raw_results_path"],
                "metadata": evaluation_result["metadata"],
                "summary": evaluation_result["summary"],
            }
        else:
            llm_overrides = _llm_overrides_from_args(args)
            prepare_corpus(
                raw_dir=args.raw_dir,
                project_dir=args.project_dir,
                dataset_path=args.dataset,
                model=args.model or DEFAULT_COMPLETION_MODEL,
                embedding_model=(args.embedding_model or DEFAULT_EMBEDDING_MODEL),
                top_k=args.k,
                model_env=args.model_env,
                api_key_env=args.api_key_env,
                api_base_env=args.api_base_env,
                embedding_model_env=args.embedding_model_env,
                embedding_api_key_env=args.embedding_api_key_env,
                embedding_api_base_env=args.embedding_api_base_env,
            )
            index_result = build_index(
                project_dir=args.project_dir,
                dataset_path=args.dataset,
                method=args.index_method,
                cache=not args.no_cache,
                skip_validation=args.skip_validation,
                verbose=args.verbose,
                llm_overrides=llm_overrides,
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
                llm_overrides=llm_overrides,
                source_match_mode=args.source_match_mode,
                scoring_config_path=args.scoring_config,
                judge_mode=args.judge_mode,
                judge_model=args.judge_model,
                judge_api_key_env=args.judge_api_key_env,
                judge_api_base=args.judge_api_base,
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
