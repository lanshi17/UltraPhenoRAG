"""Shared helpers reused by every RAG baseline benchmark.

Both the Microsoft GraphRAG and LightRAG entry points used to carry their own
copies of these helpers.  They live here now so behaviour fixes (usage
aggregation, source canonicalisation, scoring options) apply to every
baseline at once.
"""

from benchmark.common.answers import (
    normalize_response,
    safety_actions,
    supported_statements,
)
from benchmark.common.corpus import (
    MANIFEST_NAME,
    canonical_source_id,
    extract_pdf_text,
    now_iso,
    safe_slug,
    sha256_file,
    source_id_from_text,
    write_manifest,
)
from benchmark.common.retrieval import (
    as_string_list,
    context_frames,
    extract_retrieved_context,
)
from benchmark.common.scoring_options import load_scoring_options
from benchmark.common.usage import aggregate_usage, empty_usage, merge_usage

__all__ = [
    "MANIFEST_NAME",
    "aggregate_usage",
    "as_string_list",
    "canonical_source_id",
    "context_frames",
    "empty_usage",
    "extract_pdf_text",
    "extract_retrieved_context",
    "load_scoring_options",
    "merge_usage",
    "normalize_response",
    "now_iso",
    "safe_slug",
    "safety_actions",
    "sha256_file",
    "source_id_from_text",
    "supported_statements",
    "write_manifest",
]
