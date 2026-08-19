"""End-to-end benchmark entry point for the LightRAG baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence

from benchmark.baseline.light_rag_client import LightRAGClient
from benchmark.baseline.light_rag_client.config.llm_config import (
    LLMConfigOverrides,
    ModelConfigOverride,
)
from benchmark.common import (
    MANIFEST_NAME,
    aggregate_usage,
    canonical_source_id,
    empty_usage,
    extract_retrieved_context,
    load_scoring_options,
    merge_usage,
    normalize_response,
    now_iso,
    safe_slug,
    safety_actions,
    source_id_from_text,
    supported_statements,
    write_manifest,
)
from benchmark.common.scoring_options import DEFAULT_SCORING_CONFIG
from benchmark.qa import (
    DatasetScoringReport,
    JudgeConfig,
    compute_dataset_fingerprint,
    judge_answer,
    load_questions,
    score_question,
    validate_dataset,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = REPOSITORY_ROOT / "benchmark" / "data" / "raw"
DEFAULT_PROJECT_DIR = REPOSITORY_ROOT / "benchmark" / "data" / "light_rag"
DEFAULT_DATASET = (
    REPOSITORY_ROOT / "benchmark" / "qa" / "dataset" / "sample_questions.json"
)
DEFAULT_RESULTS_DIR = REPOSITORY_ROOT / "benchmark" / "results" / "light_rag"

ADAPTIVE_SEARCH_METHODS = {
    "basic": "basic",
    "multi-vector": "local",
    "graph-enhanced": "drift",
}


class BenchmarkPreflightError(RuntimeError):
    """Local benchmark inputs are not ready."""


def prepare_corpus(
    *,
    raw_dir: Path | None = None,
    project_dir: Path | None = None,
    corpus_dir: Path | None = None,
) -> dict[str, Any]:
    """Extract PDFs into the unified corpus directory.

    The unified layout (``benchmark/data/corpus/input``) is shared with the
    Microsoft GraphRAG baseline; LightRAG keeps only its own index storage
    under ``project_dir``.
    """
    del project_dir  # input lives in the shared corpus dir now
    from benchmark.common.unified_corpus import prepare_corpus as unified_prepare

    return unified_prepare(
        raw_dir=raw_dir or UNIFIED_RAW_DIR,
        corpus_dir=corpus_dir or DEFAULT_CORPUS_DIR,
    )


def preflight(
    *, project_dir: Path, dataset_path: Path, require_index: bool = False
) -> dict[str, Any]:
    project_dir = project_dir.resolve()
    dataset_path = dataset_path.resolve()
    issues: list[str] = []
    warnings: list[str] = []
    if not project_dir.is_dir():
        issues.append(f"project directory does not exist: {project_dir}")
    if not dataset_path.is_file():
        issues.append(f"dataset does not exist: {dataset_path}")
        questions = []
    else:
        questions = load_questions(dataset_path)
        dataset_errors = validate_dataset(questions)
        issues.extend(
            f"{question_id}: {error}"
            for question_id, errors in dataset_errors.items()
            for error in errors
        )
    input_files = (
        sorted((project_dir / "input").glob("*.txt"))
        if (project_dir / "input").is_dir()
        else []
    )
    if not input_files:
        issues.append(
            f"no LightRAG input text files found under {project_dir / 'input'}"
        )
    storage_dir = project_dir / "rag_storage"
    storage_markers = (
        "vdb_chunks.json",
        "kv_store_text_chunks.json",
        "graph_chunk_entity_relation.graphml",
    )
    storage_files = (
        {path.name for path in storage_dir.rglob("*") if path.is_file()}
        if storage_dir.is_dir()
        else set()
    )
    missing_index = require_index and not any(
        marker in storage_files for marker in storage_markers
    )
    if missing_index:
        issues.append(f"LightRAG index is missing under {storage_dir}")
    manifest_path = project_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        warnings.append(f"manifest is missing: {manifest_path}")
    return {
        "checked_at": now_iso(),
        "project_dir": str(project_dir),
        "dataset_path": str(dataset_path),
        "input_document_count": len(input_files),
        "question_count": len(questions),
        "issues": issues,
        "warnings": warnings,
        "missing_index": missing_index,
        "ready": not issues,
    }


def _require_preflight(report: dict[str, Any]) -> None:
    if report["issues"]:
        raise BenchmarkPreflightError(
            "preflight failed:\n" + "\n".join(f"- {item}" for item in report["issues"])
        )


def build_index(
    *,
    project_dir: Path,
    dataset_path: Path,
    cache: bool = True,
    verbose: bool = False,
    llm_overrides: LLMConfigOverrides | None = None,
) -> dict[str, Any]:
    report = preflight(project_dir=project_dir, dataset_path=dataset_path)
    _require_preflight(report)
    started = time.monotonic()
    client = LightRAGClient(project_dir, llm_overrides=llm_overrides, verbose=verbose)
    try:
        result = client.index(cache=cache)
        if result.has_errors:
            raise RuntimeError("LightRAG indexing failed:\n" + "\n".join(result.errors))
        return {
            "completed_at": now_iso(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "document_count": len(result.outputs),
            "errors": result.errors,
        }
    finally:
        client.close()


class SourceResolver:
    """Resolve LightRAG chunk ``file_path`` values to benchmark source IDs."""

    def __init__(
        self,
        manifest_or_documents: dict[str, Any] | Any,
        text_units: Any | None = None,
        manifest: dict[str, Any] | None = None,
    ) -> None:
        # Accept the Microsoft resolver's (documents, text_units, manifest)
        # shape too; it makes migration scripts and comparison tests reusable.
        manifest = manifest or (
            manifest_or_documents
            if isinstance(manifest_or_documents, Mapping)
            else {"documents": []}
        )
        self._sources: dict[str, str] = {}
        self._record_sources: dict[str, str] = {}
        self._text_sources: dict[str, str] = {}
        for item in manifest.get("documents", []):
            source_id = item.get("source_id")
            input_file = item.get("input_file")
            if not source_id or not input_file:
                continue
            path = Path(input_file)
            self._sources[path.name] = source_id
            self._sources[path.stem] = source_id
        if hasattr(manifest_or_documents, "iterrows") and hasattr(
            text_units, "iterrows"
        ):
            document_sources: dict[str, str] = {}
            for _, row in manifest_or_documents.iterrows():
                title = str(row.get("title", ""))
                source = self._sources.get(title) or self._sources.get(Path(title).name)
                if source:
                    document_sources[str(row.get("id", ""))] = source
            for _, row in text_units.iterrows():
                document_ids = row.get("document_id", row.get("document_ids", []))
                if isinstance(document_ids, str):
                    document_ids = [document_ids]
                source = next(
                    (
                        document_sources.get(str(document_id))
                        for document_id in document_ids
                        if document_sources.get(str(document_id))
                    ),
                    None,
                )
                text = str(row.get("text", "") or "").strip()
                if source:
                    for key in (row.get("id"), row.get("human_readable_id")):
                        if key is not None:
                            self._record_sources[str(key)] = source
                    if text:
                        self._text_sources[text] = source

    def resolve(self, file_path: Any, text: str = "") -> str:
        record_key = str(file_path or "")
        if record_key in self._record_sources:
            return self._record_sources[record_key]
        if text.strip() in self._text_sources:
            return self._text_sources[text.strip()]
        path = Path(record_key)
        for key in (path.name, path.stem, str(file_path or "")):
            if key in self._sources:
                return self._sources[key]
        source = source_id_from_text(text)
        return source or f"unresolved:{path.name or 'unknown'}"


def _selected_method(question: Any, requested_method: str) -> str:
    if requested_method != "adaptive":
        return requested_method
    try:
        return ADAPTIVE_SEARCH_METHODS[question.rag_arch_type]
    except KeyError as exc:
        raise ValueError(
            f"unsupported rag_arch_type: {question.rag_arch_type}"
        ) from exc


def evaluate(
    *,
    project_dir: Path,
    dataset_path: Path,
    requested_method: str = "adaptive",
    k: int = 16,
    output_path: Path | None = None,
    limit: int | None = None,
    question_ids: Sequence[str] = (),
    fail_fast: bool = False,
    verbose: bool = False,
    llm_overrides: LLMConfigOverrides | None = None,
    source_match_mode: str | None = None,
    scoring_config_path: Path | None = None,
    judge_mode: str = "off",
    judge_model: str | None = None,
    judge_api_key_env: str = "OPENAI_API_KEY",
    judge_api_base: str | None = None,
) -> dict[str, Any]:
    """Query the dataset, compute three-layer metrics, and save auditable results."""
    report = preflight(
        project_dir=project_dir, dataset_path=dataset_path, require_index=True
    )
    _require_preflight(report)
    questions = load_questions(dataset_path)
    scoring_options = load_scoring_options(scoring_config_path)
    selected_source_mode = source_match_mode or scoring_options["source_match_mode"]
    if selected_source_mode not in {"exact", "evidence", "hybrid"}:
        raise BenchmarkPreflightError(
            "source-match-mode must be exact, evidence, or hybrid"
        )
    if judge_mode not in {"off", "optional", "required"}:
        raise BenchmarkPreflightError("judge-mode must be off, optional, or required")
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
        selected = set(question_ids)
        questions = [
            question for question in questions if question.question_id in selected
        ]
        missing = selected - {question.question_id for question in questions}
        if missing:
            raise BenchmarkPreflightError(
                "unknown question IDs: " + ", ".join(sorted(missing))
            )
    if limit is not None:
        questions = questions[:limit]
    if not questions:
        raise BenchmarkPreflightError("no questions selected")
    project_dir = project_dir.resolve()
    output_path = (
        output_path
        or DEFAULT_RESULTS_DIR
        / f"evaluation-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    ).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output_path.with_suffix(".jsonl")
    manifest_path = project_dir / MANIFEST_NAME
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "documents": [
                {
                    "source_id": safe_slug(path.stem.split("--", 1)[0]),
                    "input_file": str(path.relative_to(project_dir)),
                }
                for path in sorted((project_dir / "input").glob("*.txt"))
            ]
        }
    resolver = SourceResolver(manifest)
    client = LightRAGClient(project_dir, llm_overrides=llm_overrides, verbose=verbose)
    scoring = DatasetScoringReport()
    method_counts: Counter[str] = Counter()
    raw_results: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        with raw_path.open("w", encoding="utf-8") as raw_stream:
            for index, question in enumerate(questions, 1):
                question_started = time.monotonic()
                selected_method = _selected_method(question, requested_method)
                error: str | None = None
                query_result: Any = None
                try:
                    query_result = client.search(question.question, selected_method)
                    method_counts[selected_method] += 1
                    answer = normalize_response(query_result.response)
                    contexts = extract_retrieved_context(
                        query_result.context_data, resolver, k
                    )
                except Exception as exc:  # noqa: BLE001
                    if fail_fast:
                        raise
                    error = f"{type(exc).__name__}: {exc}"
                    answer, contexts = "", []
                    method_counts[selected_method] += 1
                retrieved_sources = [item["source_id"] for item in contexts]
                retrieved_context = "\n\n".join(item["text"] for item in contexts)
                retrieved_supports = [
                    supported_statements(item["text"], question.must_have_statements)
                    for item in contexts
                ]
                refused, referred = safety_actions(answer)
                judge_result: dict[str, Any] | None = None
                if judge_mode != "off" and error is None:
                    if judge_config is None:
                        judge_result = {
                            "model": judge_model or "",
                            "error": "no --judge-model provided, fell back to lexical",
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
                result = score_question(
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
                scoring.add(result)
                query_usage = dict(
                    (getattr(query_result, "telemetry", None) or {}).get("usage")
                    or empty_usage()
                )
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
                    "telemetry": getattr(query_result, "telemetry", {}) or {},
                    "usage": {
                        "query": query_usage,
                        "judge": judge_usage,
                        "total": merge_usage(query_usage, judge_usage),
                    },
                    "elapsed_seconds": round(time.monotonic() - question_started, 3),
                    "scoring": asdict(result),
                }
                raw_results.append(item)
                raw_stream.write(json.dumps(item, ensure_ascii=False) + "\n")
                raw_stream.flush()
    finally:
        client.close()
    summary = scoring.summary()
    summary["usage"] = aggregate_usage(raw_results)
    result = {
        "metadata": {
            "created_at": now_iso(),
            "lightrag_version": _lightrag_version(),
            "project_dir": str(project_dir),
            "dataset_path": str(dataset_path.resolve()),
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
            "output_path": str(output_path),
            "scoring_note": (
                "Each question preserves lexical metrics alongside; when Judge is "
                "enabled and successful selected=judge, otherwise falls back to lexical. "
                "Source validation preserves exact and evidence/equivalent modes."
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
        "preflight": report,
        "summary": summary,
        "results": raw_results,
    }
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def _lightrag_version() -> str:
    try:
        return version("lightrag-hku")
    except PackageNotFoundError:
        return "vendored"


def _add_llm_override_arguments(parser: argparse.ArgumentParser) -> None:
    """Add runtime LLM model/provider override arguments."""
    group = parser.add_argument_group(
        "LLM model/provider overrides (runtime only, not written to config)"
    )
    group.add_argument(
        "--model",
        default=None,
        help="completion model name (e.g. gpt-4.1)",
    )
    group.add_argument(
        "--embedding-model",
        default=None,
        help="embedding model name (e.g. text-embedding-3-large)",
    )
    group.add_argument(
        "--model-provider",
        default=None,
        help="completion provider (e.g. openai/azure)",
    )
    group.add_argument(
        "--embedding-model-provider",
        default=None,
        help="embedding provider (e.g. openai/azure)",
    )
    group.add_argument(
        "--api-base",
        default=None,
        help="completion and embedding shared API base URL (including /v1)",
    )
    group.add_argument(
        "--embedding-api-base",
        default=None,
        help="embedding-specific API base URL (overrides --api-base)",
    )
    group.add_argument(
        "--api-version",
        default=None,
        help="API version (required for some providers like Azure)",
    )
    group.add_argument(
        "--api-key-env",
        default=None,
        help="API key environment variable name (e.g. RAG_API_KEY)",
    )
    group.add_argument(
        "--embedding-api-key-env",
        default=None,
        help="embedding-specific API key environment variable name",
    )


def _llm_overrides_from_args(args: argparse.Namespace) -> LLMConfigOverrides | None:
    """Build LLM model/provider runtime overrides from CLI arguments."""
    completion_kwargs: dict[str, str] = {}
    for attr, key in (
        ("model", "model"),
        ("model_provider", "model_provider"),
        ("api_base", "api_base"),
        ("api_version", "api_version"),
        ("api_key_env", "api_key_env"),
    ):
        value = getattr(args, attr, None)
        if value:
            completion_kwargs[key] = value

    embedding_kwargs: dict[str, str] = {}
    for attr, key in (
        ("embedding_model", "model"),
        ("embedding_model_provider", "model_provider"),
    ):
        value = getattr(args, attr, None)
        if value:
            embedding_kwargs[key] = value
    # Shared items also apply to embedding; embedding-specific takes priority.
    for attr, key in (
        ("api_base", "api_base"),
        ("api_version", "api_version"),
        ("api_key_env", "api_key_env"),
    ):
        value = getattr(args, attr, None)
        if value:
            embedding_kwargs.setdefault(key, value)
    for attr, key in (
        ("embedding_api_base", "api_base"),
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
        description="Build and evaluate the LightRAG prenatal benchmark"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    prepare.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT_DIR)
    preflight_parser = sub.add_parser("preflight")
    preflight_parser.add_argument(
        "--project-dir", type=Path, default=DEFAULT_PROJECT_DIR
    )
    preflight_parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    preflight_parser.add_argument("--require-index", action="store_true")
    index = sub.add_parser("index")
    index.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT_DIR)
    index.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    index.add_argument("--no-cache", action="store_true")
    index.add_argument("--verbose", action="store_true")
    _add_llm_override_arguments(index)
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument(
        "--project-dir", type=Path, default=DEFAULT_PROJECT_DIR
    )
    evaluate_parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    evaluate_parser.add_argument(
        "--method",
        choices=(
            "adaptive",
            "basic",
            "naive",
            "local",
            "global",
            "hybrid",
            "mix",
            "drift",
        ),
        default="adaptive",
    )
    evaluate_parser.add_argument("--k", type=int, default=16)
    evaluate_parser.add_argument("--output", type=Path)
    evaluate_parser.add_argument("--limit", type=int)
    evaluate_parser.add_argument("--question-id", action="append", default=[])
    evaluate_parser.add_argument(
        "--source-match-mode",
        choices=("exact", "evidence", "hybrid"),
        default=None,
        help="gold source validation mode; defaults to scoring_config.yaml",
    )
    evaluate_parser.add_argument("--scoring-config", type=Path, default=None)
    evaluate_parser.add_argument(
        "--judge-mode",
        choices=("off", "optional", "required"),
        default="off",
        help="LLM-as-Judge; default off, falls back to lexical on failure",
    )
    evaluate_parser.add_argument("--judge-model", default=None)
    evaluate_parser.add_argument("--judge-api-key-env", default="OPENAI_API_KEY")
    evaluate_parser.add_argument("--judge-api-base", default=None)
    evaluate_parser.add_argument("--fail-fast", action="store_true")
    evaluate_parser.add_argument("--verbose", action="store_true")
    _add_llm_override_arguments(evaluate_parser)
    run = sub.add_parser("run")
    run.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    run.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT_DIR)
    run.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    run.add_argument(
        "--method",
        choices=(
            "adaptive",
            "basic",
            "naive",
            "local",
            "global",
            "hybrid",
            "mix",
            "drift",
        ),
        default="adaptive",
    )
    run.add_argument("--k", type=int, default=16)
    run.add_argument("--output", type=Path)
    run.add_argument("--limit", type=int)
    run.add_argument("--no-cache", action="store_true")
    run.add_argument("--verbose", action="store_true")
    _add_llm_override_arguments(run)
    run.add_argument(
        "--source-match-mode",
        choices=("exact", "evidence", "hybrid"),
        default=None,
    )
    run.add_argument("--scoring-config", type=Path, default=None)
    run.add_argument(
        "--judge-mode",
        choices=("off", "optional", "required"),
        default="off",
    )
    run.add_argument("--judge-model", default=None)
    run.add_argument("--judge-api-key-env", default="OPENAI_API_KEY")
    run.add_argument("--judge-api-base", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        llm_overrides = _llm_overrides_from_args(args)
        if args.command == "prepare":
            result = prepare_corpus(raw_dir=args.raw_dir, project_dir=args.project_dir)
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
                cache=not args.no_cache,
                verbose=args.verbose,
                llm_overrides=llm_overrides,
            )
        elif args.command == "run":
            prepare_corpus(raw_dir=args.raw_dir, project_dir=args.project_dir)
            index_result = build_index(
                project_dir=args.project_dir,
                dataset_path=args.dataset,
                cache=not args.no_cache,
                verbose=args.verbose,
                llm_overrides=llm_overrides,
            )
            evaluation = evaluate(
                project_dir=args.project_dir,
                dataset_path=args.dataset,
                requested_method=args.method,
                k=args.k,
                output_path=args.output,
                limit=args.limit,
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
                "output_path": evaluation["metadata"]["output_path"],
                "summary": evaluation["summary"],
            }
        else:
            evaluation = evaluate(
                project_dir=args.project_dir,
                dataset_path=args.dataset,
                requested_method=args.method,
                k=args.k,
                output_path=args.output,
                limit=args.limit,
                question_ids=args.question_id,
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
                "output_path": evaluation["metadata"]["output_path"],
                "summary": evaluation["summary"],
            }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except BenchmarkPreflightError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
