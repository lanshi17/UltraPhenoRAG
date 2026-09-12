"""Synchronous facade for the vendored PathRAG implementation.

PathRAG intentionally exposes a primarily asynchronous engine, and its
synchronous ``insert``/``query`` helpers assume they own the ambient event
loop.  The benchmark clients use synchronous search methods, so this module
owns the small amount of event-loop plumbing and keeps the public contract
aligned with the Microsoft GraphRAG and LightRAG baselines.

Two vendored quirks shape the design:

* PathRAG's query path returns only the final answer string.  The retrieved
  context (entities, relationships, source chunks) exists solely inside the
  RAG system prompt it renders for the final LLM call, so the tracked LLM
  wrapper captures that prompt and the client parses its ``-----...-----``
  sections into structured tables.  A query therefore reports exactly the
  context PathRAG actually used, in a single pass.
* ``kg_query`` only implements the ``hybrid`` pipeline, so
  :class:`~benchmark.baseline.pathrag_client.search_methods.SearchMethod`
  exposes a single public method.
"""

from __future__ import annotations

import csv
import io
import logging
import os
import re
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any, Callable

from benchmark.config import RAG_ENVIRONMENT, load_environment

from .compat import (
    ensure_pathrag_context_build_compatibility,
    ensure_pathrag_import_compatibility,
    ensure_pathrag_llm_cache_compatibility,
)
from .constants import DEFAULT_RESPONSE_TYPE, DEFAULT_TOP_K
from .cost_analysis import OperationUsageTracker
from .dependency import ensure_pathrag_import_path
from .documents import load_documents
from .event_loop import EventLoopRunner
from .index_methods import IndexMethod
from .search_methods import SearchMethod, resolve_search_method
from .states import ClientState

ensure_pathrag_import_path()
ensure_pathrag_import_compatibility()

from PathRAG import PathRAG, QueryParam  # noqa: E402
from PathRAG.llm import openai_embedding  # noqa: E402
from PathRAG.utils import EmbeddingFunc, safe_unicode_decode  # noqa: E402
ensure_pathrag_llm_cache_compatibility()
ensure_pathrag_context_build_compatibility()

from .models import IndexResult, QueryResult  # noqa: E402

logger = logging.getLogger(__name__)

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_active_llm_func: Callable[..., Any] | None = None
_active_usage_tracker: OperationUsageTracker | None = None
_captured_rag_contexts: list[str] = []

# Marker that only appears in the system prompt ``kg_query`` renders from
# ``_build_query_context``; entity extraction and keyword extraction never
# carry it.
_RAG_CONTEXT_MARKER = "-----Sources-----"

_CONTEXT_SECTION = re.compile(
    r"-----[ \t]*(?P<header>[^\n]+?)[ \t]*-----\s*```csv[ \t]*\n(?P<body>.*?)```",
    re.DOTALL,
)
_SECTION_TABLES = {
    "high-level entity information": "entities",
    "low-level entity information": "entities",
    "high-level relationship information": "relationships",
    "low-level relationship information": "relationships",
    "Sources": "sources",
}

# Files the default storages persist under ``working_dir`` after each
# successful insert: JSON KV stores, NanoVectorDB files, and the graphml
# graph.  ``cache=False`` on :meth:`PathRAGClient.index` removes them.
_STORAGE_FILES = (
    "kv_store_full_docs.json",
    "kv_store_text_chunks.json",
    "kv_store_llm_response_cache.json",
    "vdb_entities.json",
    "vdb_relationships.json",
    "vdb_chunks.json",
    "graph_chunk_entity_relation.graphml",
)


def _override_value(override: Any, name: str) -> Any:
    if override is None:
        return None
    if isinstance(override, Mapping):
        return override.get(name)
    return getattr(override, name, None)


def _response_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _usage_counts(usage: Any) -> dict[str, int]:
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    }


def _parse_context_section(body: str) -> list[dict[str, Any]]:
    rows = [row for row in csv.reader(io.StringIO(body)) if row]
    if len(rows) < 2:
        return []
    header = [cell.strip() for cell in rows[0]]
    records: list[list[str]] = []
    for row in rows[1:]:
        # PathRAG writes unquoted chunk text, so a chunk containing a
        # newline spills onto extra physical lines with a single field.
        if len(row) == 1 and len(header) > 1 and records:
            records[-1][-1] = f"{records[-1][-1]}\n{row[0]}"
            continue
        records.append(list(row))
    return [
        {name: value.strip() for name, value in zip(header, row, strict=False)}
        for row in records
    ]


def _parse_query_context(context: str) -> dict[str, list[dict[str, Any]]]:
    """Split a rendered PathRAG RAG prompt into named context tables.

    ``sources`` holds the retrieved text chunks, ``entities`` the node
    tables, and ``relationships`` the edge/path tables.  The table names
    match ``benchmark.common.retrieval.CONTEXT_TABLE_PRIORITY`` so shared
    scoring consumes chunks first.
    """
    tables: dict[str, list[dict[str, Any]]] = {
        "entities": [],
        "relationships": [],
        "sources": [],
    }
    for match in _CONTEXT_SECTION.finditer(context):
        table = _SECTION_TABLES.get(match.group("header").strip())
        if table is not None:
            tables[table].extend(_parse_context_section(match.group("body")))
    return {name: rows for name, rows in tables.items() if rows}


async def _tracked_llm(*args: Any, **kwargs: Any) -> Any:
    llm_func = _active_llm_func
    if llm_func is None:
        raise RuntimeError("PathRAG LLM telemetry context is not active")
    tracker = _active_usage_tracker
    if tracker is not None:
        tracker.start_request()
    system_prompt = kwargs.get("system_prompt")
    if (
        isinstance(system_prompt, str)
        and _RAG_CONTEXT_MARKER in system_prompt
    ):
        _captured_rag_contexts.append(system_prompt)
    try:
        return await llm_func(*args, **kwargs)
    except Exception:
        if tracker is not None:
            tracker.fail_request()
        raise


class PathRAGClient:
    """Synchronous PathRAG client used by the prenatal benchmark.

    ``root_dir`` is the client project directory.  Text files are read from
    ``root_dir/input`` (or directly from ``root_dir`` when that directory is
    absent, with the shared unified corpus preferred); PathRAG's generated
    stores are placed in ``data_dir`` when given, otherwise in
    ``root_dir/pathrag_storage``.

    The two model functions are injectable.  This is useful for deterministic
    local tests and also avoids coupling the facade to one provider SDK.
    """

    def __init__(
        self,
        root_dir: str | Path,
        data_dir: str | Path | None = None,
        llm_overrides: Any | None = None,
        *,
        llm_model_func: Callable[..., Any] | None = None,
        embedding_func: Any | None = None,
        top_k: int = DEFAULT_TOP_K,
        verbose: bool = False,
        **rag_options: Any,
    ) -> None:
        self._root_dir = Path(root_dir).expanduser().resolve()
        self._root_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir = (
            Path(data_dir).expanduser().resolve()
            if data_dir is not None
            else self._root_dir / "pathrag_storage"
        )
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self._llm_overrides = llm_overrides
        self._llm_model_func = llm_model_func
        self._embedding_func = embedding_func
        self._top_k = top_k
        self._rag_options = dict(rag_options)
        self._rag: PathRAG | None = None
        self._runner: EventLoopRunner | None = None
        self._usage_tracker: OperationUsageTracker | None = None
        self._closed = False
        self._state = ClientState.CREATED

        load_environment()
        self._apply_overrides(llm_overrides)

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    @property
    def rag(self) -> PathRAG:
        """The initialized underlying PathRAG engine."""
        return self._ensure_rag()

    @property
    def config(self) -> dict[str, Any]:
        """Runtime settings exposed for benchmark diagnostics."""
        return {
            "working_dir": str(self._data_dir),
            "top_k": self._top_k,
            **self._rag_options,
        }

    def _apply_overrides(self, overrides: Any | None) -> None:
        if overrides is None:
            return
        completion = _override_value(overrides, "completion")
        embedding = _override_value(overrides, "embedding")
        self._apply_model_environment(completion, role="completion")
        self._apply_model_environment(embedding, role="embedding")

    @staticmethod
    def _apply_model_environment(override: Any, *, role: str) -> None:
        if override is None:
            return
        values: dict[str, str] = {}
        model = _override_value(override, "model")
        model_env = _override_value(override, "model_env")
        if model:
            values["model"] = str(model)
        elif model_env:
            if not _ENV_NAME.fullmatch(str(model_env)):
                raise ValueError(f"invalid environment variable name: {model_env}")
            values["model"] = os.getenv(str(model_env), "")
        api_base = _override_value(override, "api_base")
        api_base_env = _override_value(override, "api_base_env")
        if api_base:
            values["api_base"] = str(api_base)
        elif api_base_env:
            if not _ENV_NAME.fullmatch(str(api_base_env)):
                raise ValueError(f"invalid environment variable name: {api_base_env}")
            values["api_base"] = os.getenv(str(api_base_env), "")
        provider = _override_value(override, "model_provider")
        if provider:
            values["model_provider"] = str(provider)
        api_key_env = _override_value(override, "api_key_env")
        if api_key_env:
            if not _ENV_NAME.fullmatch(str(api_key_env)):
                raise ValueError(f"invalid environment variable name: {api_key_env}")
            values["api_key"] = os.getenv(str(api_key_env), "")

        env_names = {
            "model": RAG_ENVIRONMENT[
                "completion_model" if role == "completion" else "embedding_model"
            ],
            "api_base": RAG_ENVIRONMENT["api_base"],
            "model_provider": RAG_ENVIRONMENT["provider"],
            "api_key": RAG_ENVIRONMENT["api_key"],
        }

        for key, value in values.items():
            target = env_names.get(key)
            if target and value:
                os.environ[target] = value

    @staticmethod
    async def _default_llm(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, str]] | None = None,
        keyword_extraction: bool = False,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """OpenAI-compatible completion used when none is injected.

        Mirrors the vendored ``openai_complete`` contract (keyword requests
        switch to JSON output, prompts are plain text or paired with a
        system prompt) while reporting API usage back to the active tracker
        and skipping the vendor's fixed two-second sleep between calls.
        """
        from openai import AsyncOpenAI

        kwargs.pop("hashing_kv", None)  # injected by PathRAG's partial wrapper
        environment = load_environment()
        client_kwargs: dict[str, Any] = {}
        if environment.api_base:
            client_kwargs["base_url"] = environment.api_base
        if environment.api_key:
            client_kwargs["api_key"] = environment.api_key
        client = AsyncOpenAI(**client_kwargs)
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages or [])
        messages.append({"role": "user", "content": prompt})
        request: dict[str, Any] = {
            "model": environment.completion_model,
            "messages": messages,
        }
        if keyword_extraction:
            request["response_format"] = {"type": "json_object"}
        tracker = _active_usage_tracker

        if stream:
            request["stream"] = True
            request["stream_options"] = {"include_usage": True}
            stream_response = await client.chat.completions.create(**request)

            async def _stream() -> Any:
                async for chunk in stream_response:
                    usage = getattr(chunk, "usage", None)
                    if usage is not None and tracker is not None:
                        tracker.add_usage(_usage_counts(usage))
                    if not chunk.choices:
                        continue
                    content = getattr(chunk.choices[0].delta, "content", None)
                    if content:
                        yield safe_unicode_decode(content.encode("utf-8"))

            return _stream()

        response = await client.chat.completions.create(**request)
        usage = getattr(response, "usage", None)
        if usage is not None and tracker is not None:
            tracker.add_usage(_usage_counts(usage))
        content = response.choices[0].message.content or ""
        if r"\u" in content:
            content = safe_unicode_decode(content.encode("utf-8"))
        return content

    @staticmethod
    def _default_embedding() -> EmbeddingFunc:
        environment = load_environment()
        func = partial(
            openai_embedding.func,
            model=environment.embedding_model,
            base_url=environment.api_base or None,
            api_key=environment.api_key,
        )
        return EmbeddingFunc(
            embedding_dim=environment.embedding_dimension,
            max_token_size=environment.embedding_max_token_size,
            func=func,
        )

    def _make_rag(self) -> PathRAG:
        embedding = self._embedding_func or self._default_embedding()
        options: dict[str, Any] = {
            "working_dir": str(self._data_dir),
            "embedding_func": embedding,
            "llm_model_func": _tracked_llm,
            "llm_model_name": load_environment().completion_model,
            **self._rag_options,
        }
        if self.verbose:
            options.setdefault("log_level", logging.DEBUG)
        return PathRAG(**options)

    def _begin_usage_tracking(self) -> OperationUsageTracker:
        tracker = OperationUsageTracker(load_environment().completion_model)
        self._usage_tracker = tracker
        return tracker

    async def _ensure_rag_async(self) -> PathRAG:
        if self._closed:
            raise RuntimeError("PathRAGClient is closed")
        if self._rag is None:
            self._rag = self._make_rag()
            self._state = ClientState.READY
        return self._rag

    def _ensure_rag(self) -> PathRAG:
        return self._run(self._ensure_rag_async())

    def _run(self, coro: Any) -> Any:
        if self._runner is None:
            self._runner = EventLoopRunner()
        return self._runner.run(coro)

    def reload_config(self) -> dict[str, Any]:
        self.close()
        self._closed = False
        self._rag = None
        return self.config

    def _clear_storage(self) -> None:
        """Drop the in-memory engine and remove its persisted store files."""
        self._rag = None
        for name in _STORAGE_FILES:
            path = self._data_dir / name
            if path.is_file():
                path.unlink()

    async def _index_async(self, documents: list[Any]) -> IndexResult:
        rag = await self._ensure_rag_async()
        if not documents:
            return IndexResult(outputs=[])
        outputs: list[dict[str, Any]] = []
        errors: list[str] = []
        # PathRAG has no document-status protocol: one ``ainsert`` persists
        # everything only when the whole call finishes, so a mid-run API
        # failure would discard (and half-pollute) an hour of extraction
        # work.  Inserting per document keeps every finished document
        # durable — re-running after a failure skips it via content-hash
        # dedup instead of redoing the corpus.
        for item in documents:
            try:
                await rag.ainsert([item.text])
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    f"{item.document_id}: {type(exc).__name__}: {exc}"
                )
                continue
            outputs.append(
                {"id": item.document_id, "file_path": item.file_path}
            )
        return IndexResult(
            outputs=outputs, errors=errors, has_errors=bool(errors)
        )

    def index(
        self,
        method: str | IndexMethod = IndexMethod.STANDARD,
        is_update_run: bool = False,
        cache: bool = True,
        dry_run: bool = False,
        skip_validation: bool = False,
        callbacks: list[Any] | None = None,
        input_documents: Any | None = None,
    ) -> IndexResult:
        """Insert project input documents into PathRAG.

        PathRAG has one ingestion pipeline rather than Microsoft's standard
        vs fast workflow split; ``method`` and ``is_update_run`` are accepted
        for API compatibility.  Ingestion deduplicates documents by content
        hash, so ``cache=False`` clears the persisted stores under
        ``data_dir`` first to force a full rebuild.  Each document is
        inserted separately, so a failure mid-run keeps earlier documents
        durable and a re-run skips them via content-hash dedup.
        """

        del is_update_run, skip_validation, callbacks
        try:
            IndexMethod(method)
        except ValueError as exc:
            raise ValueError(f"unsupported index method: {method}") from exc
        documents = load_documents(self._root_dir, input_documents)
        if dry_run:
            return IndexResult(outputs=documents)
        if not cache:
            self._clear_storage()
        tracker = self._begin_usage_tracking()
        global _active_llm_func, _active_usage_tracker
        _active_llm_func = self._llm_model_func or self._default_llm
        _active_usage_tracker = tracker
        try:
            result = self._run(self._index_async(documents))
            result.telemetry = tracker.telemetry()
            return result
        finally:
            _active_usage_tracker = None
            _active_llm_func = None
            self._usage_tracker = None

    def _resolve_mode(self, method: str | SearchMethod) -> tuple[str, str]:
        return resolve_search_method(method)

    async def _query_async(
        self,
        query: str,
        method: str,
        *,
        response_type: str,
        streaming: bool,
        top_k: int | None,
        only_need_context: bool = False,
    ) -> QueryResult:
        rag = await self._ensure_rag_async()
        backend_mode, public_method = self._resolve_mode(method)
        param = QueryParam(
            mode=backend_mode,
            response_type=response_type,
            stream=streaming,
            top_k=top_k if top_k is not None else self._top_k,
            only_need_context=only_need_context,
        )
        tracker = self._begin_usage_tracking()
        global _active_llm_func, _active_usage_tracker, _captured_rag_contexts
        _active_llm_func = self._llm_model_func or self._default_llm
        _active_usage_tracker = tracker
        _captured_rag_contexts = []
        try:
            response = await rag.aquery(query, param=param)
            if hasattr(response, "__aiter__"):
                chunks: list[str] = []
                async for chunk in response:
                    chunks.append(_response_text(chunk))
                response = "".join(chunks)
        finally:
            _active_usage_tracker = None
            _active_llm_func = None
            captured = list(_captured_rag_contexts)
            _captured_rag_contexts = []
            self._usage_tracker = None
        response_text = _response_text(response)
        # In context-only mode the context itself is the payload; otherwise
        # the captured RAG system prompt is the authoritative record.
        context_prompt = (
            response_text
            if only_need_context and not captured
            else (captured[-1] if captured else "")
        )
        return QueryResult(
            response=response_text,
            context_data=_parse_query_context(context_prompt),
            query=query,
            method=public_method,
            raw_data={"mode": backend_mode, "context": context_prompt},
            telemetry=tracker.telemetry(),
        )

    def search(
        self,
        query: str,
        method: str | SearchMethod = SearchMethod.HYBRID,
        *,
        response_type: str = DEFAULT_RESPONSE_TYPE,
        streaming: bool = False,
        top_k: int | None = None,
    ) -> QueryResult:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        self._resolve_mode(method)
        return self._run(
            self._query_async(
                query,
                str(method.value if isinstance(method, SearchMethod) else method),
                response_type=response_type,
                streaming=streaming,
                top_k=top_k,
            )
        )

    def query(
        self,
        query: str,
        mode: str | SearchMethod = SearchMethod.HYBRID,
        **kwargs: Any,
    ) -> QueryResult:
        """Generic query alias matching PathRAG's terminology."""

        return self.search(query, mode, **kwargs)

    def hybrid_search(
        self,
        query: str,
        response_type: str = DEFAULT_RESPONSE_TYPE,
        streaming: bool = False,
        **kwargs: Any,
    ) -> QueryResult:
        return self.search(
            query,
            SearchMethod.HYBRID,
            response_type=response_type,
            streaming=streaming,
            **kwargs,
        )

    def query_data(
        self,
        query: str,
        method: str | SearchMethod = SearchMethod.HYBRID,
        *,
        top_k: int | None = None,
    ) -> QueryResult:
        """Return the retrieved context without generating a final answer."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        return self._run(
            self._query_async(
                query,
                str(method.value if isinstance(method, SearchMethod) else method),
                response_type=DEFAULT_RESPONSE_TYPE,
                streaming=False,
                top_k=top_k,
                only_need_context=True,
            )
        )

    def close(self) -> None:
        if self._closed:
            return
        # PathRAG persists every store in ``index_done_callback`` after each
        # insert, so there is no async storage teardown to run here.
        if self._runner is not None:
            self._runner.close()
            self._runner = None
        self._closed = True
        self._state = ClientState.CLOSED

    def __enter__(self) -> "PathRAGClient":
        self._ensure_rag()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


__all__ = ["PathRAGClient"]
