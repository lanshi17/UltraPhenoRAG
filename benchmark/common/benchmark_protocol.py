"""Shared, auditable controls for cross-framework benchmark runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from benchmark.common.unified_corpus import load_corpus_manifest
from benchmark.config import load_environment


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _corpus_fingerprint(corpus_dir: Path | None) -> str:
    manifest = load_corpus_manifest(corpus_dir)
    documents = [
        {
            "source_id": item.get("source_id"),
            "pdf_sha256": item.get("pdf_sha256"),
            "input_file": item.get("input_file"),
            "status": item.get("status"),
        }
        for item in manifest.get("documents", [])
    ]
    payload = json.dumps(documents, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _override_model(overrides: Any, role: str, fallback: str) -> str:
    override = getattr(overrides, role, None) if overrides is not None else None
    model = getattr(override, "model", None)
    return str(model or fallback)


def build_benchmark_conditions(
    *,
    dataset_path: Path,
    corpus_dir: Path | None,
    question_ids: Sequence[str],
    requested_method: str,
    k: int,
    source_match_mode: str,
    scoring_config_path: Path,
    judge_mode: str,
    judge_model: str | None,
    llm_overrides: Any | None,
) -> dict[str, Any]:
    """Record the controls that must match before comparing two runs."""
    environment = load_environment()
    return {
        "protocol_version": 1,
        "dataset_sha256": _sha256_file(dataset_path.resolve()),
        "corpus_fingerprint": _corpus_fingerprint(corpus_dir),
        "question_ids": list(question_ids),
        "requested_search_method": requested_method,
        "top_k": k,
        "source_match_mode": source_match_mode,
        "scoring_config_sha256": _sha256_file(scoring_config_path.resolve()),
        "judge_mode": judge_mode,
        "judge_model": judge_model or "",
        "completion_model": _override_model(
            llm_overrides, "completion", environment.completion_model
        ),
        "embedding_model": _override_model(
            llm_overrides, "embedding", environment.embedding_model
        ),
    }


def assert_comparable_conditions(left: dict[str, Any], right: dict[str, Any]) -> None:
    """Reject comparisons whose controlled inputs or evaluator settings differ."""
    mismatches = [
        key for key in sorted(set(left) | set(right)) if left.get(key) != right.get(key)
    ]
    if mismatches:
        raise ValueError("benchmark conditions differ: " + ", ".join(mismatches))
