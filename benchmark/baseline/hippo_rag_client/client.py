"""HippoRAG 2 client for the prenatal-ultrasound benchmark.

HippoRAG is a retrieval framework: fact retrieval → recognition memory →
dense passage scoring → personalised PageRank.  Its bundled QA prompt is
a Wikipedia-style CoT template, which is a poor fit for clinical
guideline answering, so ``search`` keeps HippoRAG's retrieval verbatim
and renders the final answer with a fixed clinical prompt (the same one
used by the generation-isolation experiment, so retrieval quality stays
comparable with the other baselines).
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmark.config import load_environment

_HIPPORAG_SRC = (
    Path(__file__).resolve().parents[1]
    / "libs"
    / "HippoRAG"
    / "src"
)
if _HIPPORAG_SRC.is_dir() and str(_HIPPORAG_SRC) not in sys.path:
    sys.path.insert(0, str(_HIPPORAG_SRC))

from hipporag import HippoRAG  # noqa: E402
from hipporag.information_extraction.openie_openai import (  # noqa: E402
    _extract_json_list_field,
)
from hipporag.llm.openai_gpt import CacheOpenAI  # noqa: E402
from hipporag.utils.config_utils import BaseConfig  # noqa: E402
from hipporag.utils.misc_utils import Chunk as HippoChunk  # noqa: E402

_EMPTY_TEXT = "did not contain non-empty text"
# Fields the OpenIE parsers demand inside a JSON object (see
# ``_extract_json_list_field``); gpt-5-mini occasionally answers the
# one-shot templates with the BARE array instead of the wrapper object.
# Order is precedence: the TRIPLE prompt embeds a ``{"named_entities":
# [...]}`` input echo, so "triples" must win when both appear; NER prompts
# never name triples.
_LIST_FIELDS = ("triples", "named_entities")

def _prompt_text(messages: Any) -> str:
    return " ".join(
        str(
            item.get("content", "")
            if isinstance(item, dict)
            else getattr(item, "content", "")
        )
        for item in (messages or [])
    )


def _openie_field(messages: Any) -> str | None:
    """Return the JSON list field the OpenIE parser demands, if this is an
    OpenIE prompt (NER/triple one-shot templates name their field)."""
    prompt = _prompt_text(messages)
    for list_field in _LIST_FIELDS:
        if f'"{list_field}"' in prompt:
            return list_field
    return None


def _normalize_bare_json_list(messages: Any, text: str, field: str) -> str:
    stripped = (text or "").strip()
    if not stripped.startswith("["):
        return text
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return text
    if isinstance(payload, list):
        return json.dumps({field: payload})
    return text


def _openie_parseable(text: str, metadata: Any, field: str) -> bool:
    # The engine's scan (every '{' + raw_decode of the tail) is quadratic
    # in pathological reasoning-loop output; reject oversized replies
    # outright instead of scanning them.  Legitimate NER/triple replies for
    # a 1200-token chunk stay well under this ceiling.
    if not isinstance(text, str) or not text.strip() or len(text) > 100_000:
        return False
    try:
        candidate = text
        if isinstance(metadata, dict) and metadata.get("finish_reason") == "length":
            from hipporag.utils.llm_utils import fix_broken_generated_json

            candidate = fix_broken_generated_json(text)
        _extract_json_list_field(candidate, field)
        return True
    except Exception:  # noqa: BLE001 - any parse failure counts as not parseable
        return False

class _ResilientOpenAI(CacheOpenAI):
    """``CacheOpenAI`` hardened for reasoning-model quirks.

    gpt-5-mini through an OpenAI-compatible proxy sporadically returns
    empty messages (reasoning budget exhausted) or answers the one-shot
    OpenIE prompts with a bare array / prose instead of the wrapper object
    ``{"named_entities": [...]}`` / ``{"triples": [...]}`` the engine's
    parser demands.  The engine aborts the entire batch on one such chunk,
    so ``infer`` guarantees OpenIE-bound replies parse: normalize, retry
    the live API (``store=True`` bypasses the response cache), and only as
    a last resort substitute an empty list for that one chunk.
    """

    def infer(self, messages, **kwargs):  # type: ignore[override]
        attempts = max(2, int(os.getenv("HIPPO_OPENIE_RETRIES", "4")))
        field = _openie_field(messages)
        last_error: Exception | None = None
        fallback: tuple[Any, ...] | None = None
        for attempt in range(attempts):
            call_kwargs = dict(kwargs)
            if attempt:
                call_kwargs["store"] = True  # uncached fresh sampling
            try:
                text, *rest = super().infer(messages, **call_kwargs)
            except ValueError as exc:
                if _EMPTY_TEXT not in str(exc):
                    raise
                last_error = exc
                time.sleep(1.5 * (attempt + 1))
                continue
            if isinstance(text, str):
                if field is not None:
                    text = _normalize_bare_json_list(messages, text, field)
                    if _openie_parseable(text, rest[0] if rest else None, field):
                        return (text, *rest)
                    fallback = (text, *rest)
                    time.sleep(1.0 * (attempt + 1))
                    continue
            return (text, *rest)
        if field is not None:
            print(
                f"[hippo_rag] OpenIE {field} unparseable after {attempts} "
                f"attempts; degrading one chunk to an empty list. "
                f"last response: {(fallback[0] if fallback else last_error)!r:.200}",
                file=sys.stderr,
            )
            rest = fallback[1:] if fallback else [{}]
            return (json.dumps({field: []}), *rest)
        assert last_error is not None
        raise last_error

DEFAULT_SAVE_DIR = Path("benchmark/data/hipporag")
DEFAULT_TOP_K = 16

QA_SYSTEM = (
    "You are a clinical reference assistant for prenatal ultrasound. "
    "Answer the question using ONLY the numbered evidence provided. "
    "Cite evidence inline as [1], [2] after each claim. If the evidence "
    "does not contain the answer, say so explicitly instead of guessing."
)


@dataclass
class QueryResult:
    response: str
    context_data: dict[str, Any] = field(default_factory=dict)
    query: str = ""
    method: str = ""
    raw_data: dict[str, Any] = field(default_factory=dict)
    telemetry: dict[str, Any] = field(default_factory=dict)


@dataclass
class IndexResult:
    documents: list[str] = field(default_factory=list)
    chunk_count: int = 0
    elapsed_seconds: float = 0.0
    save_dir: str = ""

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"IndexResult(documents={len(self.documents)}, "
            f"chunks={self.chunk_count}, elapsed={self.elapsed_seconds:.1f}s)"
        )


class HippoRAGClient:
    """Thin wrapper aligning HippoRAG 2 with the shared benchmark protocol."""

    def __init__(
        self,
        save_dir: Path | str = DEFAULT_SAVE_DIR,
        *,
        top_k: int = DEFAULT_TOP_K,
        force_rebuild: bool = False,
    ) -> None:
        load_environment()
        self._save_dir = Path(save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._top_k = top_k
        self._force_rebuild = force_rebuild
        self._llm_calls = 0
        self._llm_tokens = 0
        self._llm_prompt_tokens = 0
        self._llm_completion_tokens = 0
        # Defaults (NER 512 / triples 2048) are too small for reasoning
        # models on 1200-token clinical chunks: the budget is spent on
        # invisible reasoning tokens and the API returns empty content,
        # which HippoRAG treats as a hard NER failure.
        config = BaseConfig()
        # Reasoning models may consume HippoRAG's 2K default entirely on
        # difficult QA prompts and return an empty message.
        config.max_new_tokens = int(os.getenv("HIPPO_QA_MAX_TOKENS", "8192"))
        config.openie_ner_max_tokens = int(
            os.getenv("HIPPO_NER_MAX_TOKENS", "8192")
        )
        config.openie_triple_max_tokens = int(
            os.getenv("HIPPO_TRIPLE_MAX_TOKENS", "16384")
        )
        config.openie_max_workers = int(os.getenv("HIPPO_OPENIE_WORKERS", "6"))
        # Resuming a partially-built store (embeddings present, graph
        # missing) needs the engine's explicit rebuild authorization.
        config.force_index_from_scratch = force_rebuild
        llm_name = os.getenv("RAG_COMPLETION_MODEL", "gpt-5-mini")
        api_base = os.getenv("RAG_API_BASE") or None
        config.llm_name = llm_name
        config.llm_base_url = api_base or ""
        config.save_dir = str(self._save_dir)
        extraction_llm = _ResilientOpenAI.from_experiment_config(config)
        self._hipporag = HippoRAG(
            global_config=config,
            extraction_llm=extraction_llm,
            index_identity="hippo_rag_client/v1",
            save_dir=str(self._save_dir),
            llm_model_name=llm_name,
            llm_base_url=api_base,
            embedding_model_name=os.getenv(
                "RAG_EMBEDDING_MODEL", "text-embedding-3-large"
            ),
            embedding_base_url=os.getenv("RAG_API_BASE") or None,
        )

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        hr = getattr(self, "_hipporag", None)
        if hr is not None:
            hr.close()
            self._hipporag = None

    def __enter__(self) -> "HippoRAGClient":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # -- indexing ----------------------------------------------------------

    def _hr(self) -> Any:
        if self._hipporag is None:
            raise RuntimeError("HippoRAGClient is closed")
        return self._hipporag

    def index(
        self,
        chunks: list[Any],
        *,
        document_names: list[str] | None = None,
    ) -> IndexResult:
        """Index corpus chunks (incremental; already-seen chunks are reused)."""

        started = time.monotonic()
        docs = [
            HippoChunk(
                content=chunk.content,
                source_id=chunk.source_id,
                metadata={
                    "file_path": chunk.file_path,
                    "chunk_index": chunk.chunk_index,
                },
            )
            for chunk in chunks
        ]
        self._hr().index(docs)
        elapsed = time.monotonic() - started
        names = document_names or sorted({chunk.source_id for chunk in chunks})
        return IndexResult(
            documents=names,
            chunk_count=len(docs),
            elapsed_seconds=elapsed,
            save_dir=str(self._save_dir),
        )

    def is_indexed(self) -> bool:
        hr = self._hr()
        try:
            return bool(
                hr.chunk_embedding_store.get_all_ids()
                and hr.graph.vcount() > 0
            )
        except Exception:  # noqa: BLE001 - stores not initialised yet
            return False

    # -- retrieval + generation -------------------------------------------

    def search(
        self,
        query: str,
        method: str = "hipporag",
        *,
        top_k: int | None = None,
    ) -> QueryResult:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        del method  # one retrieval pipeline for every question type
        k = top_k if top_k is not None else self._top_k
        calls0 = self._llm_calls
        tokens0 = self._llm_tokens
        prompt0 = self._llm_prompt_tokens
        completion0 = self._llm_completion_tokens
        started = time.monotonic()

        solution = self._hr().retrieve(queries=[query], num_to_retrieve=k)[0]
        contexts = self._contexts_from_solution(solution)
        answer = self._generate(query, contexts)
        return QueryResult(
            response=answer,
            context_data={"chunks": contexts},
            query=query,
            method="hipporag",
            raw_data={
                "doc_scores": [
                    float(score)
                    for score in (solution.doc_scores.tolist()
                                  if solution.doc_scores is not None else [])
                ],
            },
            telemetry={
                "usage": {
                    "request_count": self._llm_calls - calls0,
                    "prompt_tokens": self._llm_prompt_tokens - prompt0,
                    "completion_tokens": self._llm_completion_tokens - completion0,
                    "total_tokens": self._llm_tokens - tokens0,
                },
                "elapsed_seconds": round(time.monotonic() - started, 3),
            },
        )

    def query(self, query: str, **kwargs: Any) -> QueryResult:
        return self.search(query, **kwargs)

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _contexts_from_solution(solution: Any) -> list[dict[str, Any]]:
        contexts: list[dict[str, Any]] = []
        metadata = solution.doc_metadata or []
        for index, doc in enumerate(solution.docs):
            meta = dict(metadata[index]) if index < len(metadata) else {}
            source_id = str(meta.get("source_id") or "")
            contexts.append(
                {
                    "text": str(doc),
                    "content": str(doc),
                    "source_id": source_id,
                    "title": source_id,
                    "file_path": meta.get("file_path") or source_id,
                    "chunk_id": meta.get("file_path")
                    or f"{source_id}#{index}",
                }
            )
        return contexts

    def _generate(self, query: str, contexts: list[dict[str, Any]]) -> str:
        evidence = []
        for index, chunk in enumerate(contexts, start=1):
            evidence.append(f"[{index}] {chunk['text']}")
        if not evidence:
            return "No evidence retrieved."
        messages = [
            {"role": "system", "content": QA_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Question: {query}\n\nEvidence:\n" + "\n\n".join(evidence)
                ),
            },
        ]
        text, metadata, _ok = self._hr().llm_model.infer(messages)
        self._llm_calls += 1
        if isinstance(metadata, dict):
            prompt = int(metadata.get("prompt_tokens") or 0)
            completion = int(metadata.get("completion_tokens") or 0)
            usage_total = metadata.get("total_tokens") or prompt + completion
            self._llm_prompt_tokens += prompt
            self._llm_completion_tokens += completion
            self._llm_tokens += int(usage_total)
        return str(text)
