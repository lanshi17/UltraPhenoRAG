"""Synchronous facade for the vendored LightRAG implementation.

LightRAG intentionally exposes a primarily asynchronous SDK.  The benchmark
clients use synchronous search methods, so this module owns the small amount of
event-loop plumbing and keeps the public contract aligned with the Microsoft
GraphRAG baseline.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any, Callable

from benchmark.config import RAG_ENVIRONMENT, load_environment

from .constants import DEFAULT_CHUNK_TOP_K, DEFAULT_RESPONSE_TYPE, DEFAULT_TOP_K
from .dependency import ensure_light_rag_import_path
from .documents import load_documents
from .event_loop import EventLoopRunner
from .index_methods import IndexMethod
from .search_methods import SearchMethod, resolve_search_method
from .states import ClientState

ensure_light_rag_import_path()

from lightrag import LightRAG, QueryParam  # noqa: E402
from lightrag.utils import EmbeddingFunc  # noqa: E402

from .models import IndexResult, QueryResult  # noqa: E402

logger = logging.getLogger(__name__)

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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


class LightRAGClient:
    """Synchronous LightRAG client used by the prenatal benchmark.

    ``root_dir`` is the client project directory.  Text files are read from
    ``root_dir/input`` (or directly from ``root_dir`` when that directory is
    absent); LightRAG's generated stores are placed in ``data_dir`` when given,
    otherwise in ``root_dir/rag_storage``.

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
        workspace: str | None = None,
        top_k: int = DEFAULT_TOP_K,
        chunk_top_k: int = DEFAULT_CHUNK_TOP_K,
        enable_rerank: bool | None = None,
        verbose: bool = False,
        **rag_options: Any,
    ) -> None:
        self._root_dir = Path(root_dir).expanduser().resolve()
        self._root_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir = (
            Path(data_dir).expanduser().resolve()
            if data_dir is not None
            else self._root_dir / "rag_storage"
        )
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self._llm_overrides = llm_overrides
        self._llm_model_func = llm_model_func
        self._embedding_func = embedding_func
        self._workspace = workspace or self._workspace_name(self._root_dir)
        self._top_k = top_k
        self._chunk_top_k = chunk_top_k
        self._enable_rerank = enable_rerank
        self._rag_options = dict(rag_options)
        self._rag: LightRAG | None = None
        self._runner: EventLoopRunner | None = None
        self._closed = False
        self._state = ClientState.CREATED

        load_environment()
        self._apply_overrides(llm_overrides)

    @staticmethod
    def _workspace_name(path: Path) -> str:
        value = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.name).strip("_.")
        return value or "light-rag"

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    @property
    def workspace(self) -> str:
        return self._workspace

    @property
    def rag(self) -> LightRAG:
        """The initialized underlying LightRAG instance."""
        return self._ensure_rag()

    @property
    def config(self) -> dict[str, Any]:
        """Runtime settings exposed for benchmark diagnostics."""
        return {
            "working_dir": str(self._data_dir),
            "workspace": self._workspace,
            "top_k": self._top_k,
            "chunk_top_k": self._chunk_top_k,
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
        **kwargs: Any,
    ) -> Any:
        environment = load_environment()
        binding = environment.provider
        model = environment.completion_model
        if binding == "ollama":
            from lightrag.llm.ollama import ollama_model_complete

            return await ollama_model_complete(
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                model=model,
                host=environment.api_base or "http://localhost:11434",
                **kwargs,
            )
        from lightrag.llm.openai import openai_complete_if_cache

        return await openai_complete_if_cache(
            model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            base_url=environment.api_base or None,
            api_key=environment.api_key,
            **kwargs,
        )

    @staticmethod
    def _default_embedding() -> EmbeddingFunc:
        environment = load_environment()
        binding = environment.provider
        model = environment.embedding_model
        dimension = environment.embedding_dimension
        if binding == "ollama":
            from lightrag.llm.ollama import ollama_embed

            func = partial(
                ollama_embed.func,
                embed_model=model,
                host=environment.api_base or "http://localhost:11434",
            )
        else:
            from lightrag.llm.openai import openai_embed

            func = partial(
                openai_embed.func,
                model=model,
                base_url=environment.api_base or None,
                api_key=environment.api_key,
            )
        return EmbeddingFunc(
            embedding_dim=dimension,
            max_token_size=environment.embedding_max_token_size,
            func=func,
        )

    def _make_rag(self) -> LightRAG:
        llm_func = self._llm_model_func or self._default_llm
        embedding = self._embedding_func or self._default_embedding()
        options = {
            "working_dir": str(self._data_dir),
            "workspace": self._workspace,
            "llm_model_func": llm_func,
            "embedding_func": embedding,
            "top_k": self._top_k,
            "chunk_top_k": self._chunk_top_k,
            **self._rag_options,
        }
        if self.verbose:
            options.setdefault("log_level", logging.DEBUG)
        return LightRAG(**options)

    async def _ensure_rag_async(self) -> LightRAG:
        if self._closed:
            raise RuntimeError("LightRAGClient is closed")
        if self._rag is None:
            self._rag = self._make_rag()
            await self._rag.initialize_storages()
            self._state = ClientState.READY
        return self._rag

    def _ensure_rag(self) -> LightRAG:
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

    async def _index_async(
        self, documents: list[dict[str, str]], cache: bool
    ) -> IndexResult:
        rag = await self._ensure_rag_async()
        if not cache:
            await rag.aclear_cache()
        if not documents:
            return IndexResult(outputs=[])
        try:
            track_id = await rag.ainsert(
                [item.text for item in documents],
                ids=[item.document_id for item in documents],
                file_paths=[item.file_path for item in documents],
            )
        except Exception as exc:  # noqa: BLE001
            return IndexResult(
                outputs=[],
                errors=[f"{type(exc).__name__}: {exc}"],
                has_errors=True,
            )
        return IndexResult(
            outputs=[
                {
                    "id": item.document_id,
                    "file_path": item.file_path,
                    "track_id": track_id,
                }
                for item in documents
            ]
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
        """Insert project input documents into LightRAG.

        LightRAG has one ingestion pipeline rather than Microsoft's standard vs
        fast workflow split; ``method`` and ``is_update_run`` are accepted for
        API compatibility and are recorded in the result metadata.
        """

        del is_update_run, skip_validation, callbacks
        try:
            IndexMethod(method)
        except ValueError as exc:
            raise ValueError(f"unsupported index method: {method}") from exc
        documents = load_documents(self._root_dir, input_documents)
        if dry_run:
            return IndexResult(outputs=documents)
        result = self._run(self._index_async(documents, cache))
        return result

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
        chunk_top_k: int | None,
        query_data_only: bool = False,
    ) -> QueryResult:
        rag = await self._ensure_rag_async()
        actual_mode, public_method = self._resolve_mode(method)
        param = QueryParam(
            mode=actual_mode,
            response_type=response_type,
            stream=streaming,
            top_k=top_k if top_k is not None else self._top_k,
            chunk_top_k=chunk_top_k if chunk_top_k is not None else self._chunk_top_k,
            enable_rerank=(
                self._enable_rerank
                if self._enable_rerank is not None
                else self._rag_options.get("rerank_model_func") is not None
            ),
        )
        if query_data_only:
            payload = await rag.aquery_data(query, param=param)
        else:
            payload = await rag.aquery_llm(query, param=param)
        payload = payload if isinstance(payload, dict) else {}
        llm_response = payload.get("llm_response", {})
        response = (
            llm_response.get("content") if isinstance(llm_response, dict) else None
        )
        if isinstance(llm_response, dict) and llm_response.get("is_streaming"):
            iterator = llm_response.get("response_iterator")
            if iterator is not None:
                chunks: list[str] = []
                async for chunk in iterator:
                    chunks.append(_response_text(chunk))
                response = "".join(chunks)
        if response is None and not query_data_only:
            response = (
                payload.get("message", "") if payload.get("status") == "failure" else ""
            )
        data = payload.get("data", {})
        return QueryResult(
            response=_response_text(response),
            context_data=data if isinstance(data, dict) else {},
            query=query,
            method=public_method,
            raw_data=payload,
        )

    def search(
        self,
        query: str,
        method: str | SearchMethod = SearchMethod.MIX,
        *,
        response_type: str = DEFAULT_RESPONSE_TYPE,
        streaming: bool = False,
        top_k: int | None = None,
        chunk_top_k: int | None = None,
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
                chunk_top_k=chunk_top_k,
            )
        )

    def query(
        self,
        query: str,
        mode: str | SearchMethod = SearchMethod.MIX,
        **kwargs: Any,
    ) -> QueryResult:
        """Generic query alias matching LightRAG's terminology."""

        return self.search(query, mode, **kwargs)

    def query_data(
        self,
        query: str,
        method: str | SearchMethod = SearchMethod.MIX,
        *,
        top_k: int | None = None,
        chunk_top_k: int | None = None,
    ) -> QueryResult:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        return self._run(
            self._query_async(
                query,
                str(method.value if isinstance(method, SearchMethod) else method),
                response_type=DEFAULT_RESPONSE_TYPE,
                streaming=False,
                top_k=top_k,
                chunk_top_k=chunk_top_k,
                query_data_only=True,
            )
        )

    def global_search(
        self,
        query: str,
        community_level: int | None = None,
        dynamic_community_selection: bool = False,
        response_type: str = DEFAULT_RESPONSE_TYPE,
        streaming: bool = False,
        **kwargs: Any,
    ) -> QueryResult:
        del community_level, dynamic_community_selection
        return self.search(
            query,
            SearchMethod.GLOBAL,
            response_type=response_type,
            streaming=streaming,
            **kwargs,
        )

    def local_search(
        self,
        query: str,
        community_level: int = 2,
        response_type: str = DEFAULT_RESPONSE_TYPE,
        streaming: bool = False,
        **kwargs: Any,
    ) -> QueryResult:
        del community_level
        return self.search(
            query,
            SearchMethod.LOCAL,
            response_type=response_type,
            streaming=streaming,
            **kwargs,
        )

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

    def mix_search(
        self,
        query: str,
        response_type: str = DEFAULT_RESPONSE_TYPE,
        streaming: bool = False,
        **kwargs: Any,
    ) -> QueryResult:
        return self.search(
            query,
            SearchMethod.MIX,
            response_type=response_type,
            streaming=streaming,
            **kwargs,
        )

    def basic_search(
        self,
        query: str,
        response_type: str = DEFAULT_RESPONSE_TYPE,
        streaming: bool = False,
        **kwargs: Any,
    ) -> QueryResult:
        return self.search(
            query,
            SearchMethod.BASIC,
            response_type=response_type,
            streaming=streaming,
            **kwargs,
        )

    def naive_search(
        self,
        query: str,
        response_type: str = DEFAULT_RESPONSE_TYPE,
        streaming: bool = False,
        **kwargs: Any,
    ) -> QueryResult:
        return self.search(
            query,
            SearchMethod.NAIVE,
            response_type=response_type,
            streaming=streaming,
            **kwargs,
        )

    def drift_search(
        self,
        query: str,
        community_level: int = 2,
        response_type: str = DEFAULT_RESPONSE_TYPE,
        streaming: bool = False,
        **kwargs: Any,
    ) -> QueryResult:
        del community_level
        return self.search(
            query,
            SearchMethod.DRIFT,
            response_type=response_type,
            streaming=streaming,
            **kwargs,
        )

    async def _close_async(self) -> None:
        if self._rag is not None:
            await self._rag.finalize_storages()

    def close(self) -> None:
        if self._closed:
            return
        if self._rag is not None:
            self._run(self._close_async())
        if self._runner is not None:
            self._runner.close()
            self._runner = None
        self._closed = True
        self._state = ClientState.CLOSED

    def __enter__(self) -> "LightRAGClient":
        self._ensure_rag()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


__all__ = ["LightRAGClient"]
