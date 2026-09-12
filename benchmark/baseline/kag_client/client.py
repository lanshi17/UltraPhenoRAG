"""Synchronous facade for the vendored KAG implementation.

Wraps the local (server-less) mode of OpenSPG KAG 0.8:

* **index** runs the unstructured builder chain (TXT reader → length
  splitter → schema-free OpenIE extractor → batch vectorizer → post
  processor → memory graph writer) over the project's input documents.
* **search** runs one of two solver pipelines on top of the memory graph:
  the logic-form static solver (``solver``) or a vector-retrieval-only
  pipeline (``naive``).

Usage
-----
.. code-block:: python

    from benchmark.baseline.kag_client import KAGClient

    client = KAGClient(root_dir="/path/to/kag/project")

    result = client.index()
    result = client.search(query="What is the NT measurement window?")
    print(result.response)

    client.close()
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import json
import logging
import re
import shutil
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import RLock
from typing import Any

from benchmark.config import load_environment

from .config.kag_project import (
    build_kag_config,
    initialize_kag_config,
    resolve_model_settings,
    substitute_instances,
    write_config_file,
)
from .provider_guard import ensure_provider_guard, guarded_query
from .azure_compat import ensure_azure_openai_compatibility
from .azure_compat import ensure_chunk_content_compatibility
from .azure_compat import ensure_logic_form_parse_compatibility
from .azure_compat import ensure_openai_extra_body_compatibility
from .azure_compat import ensure_ppr_chunk_content_compatibility
from .constants import (
    DEFAULT_MAX_ITERATION,
    DEFAULT_NUM_CHAINS,
    DEFAULT_NUM_THREADS_PER_CHAIN,
    DEFAULT_RESPONSE_TYPE,
    DEFAULT_SPLIT_LENGTH,
    DEFAULT_TOP_K,
)
from .cost_analysis import OperationUsageTracker, snapshot_token_totals
from .dependency import ensure_kag_import_path
from .documents import DocumentRecord, load_documents
from .event_loop import EventLoopRunner
from .index_methods import IndexMethod
from .models import IndexResult, QueryResult
from .search_methods import SearchMethod, resolve_search_method
from .states import ClientState

ensure_kag_import_path()
ensure_azure_openai_compatibility()
ensure_openai_extra_body_compatibility()
ensure_chunk_content_compatibility()
ensure_ppr_chunk_content_compatibility()
ensure_logic_form_parse_compatibility()
ensure_provider_guard()

from kag.common.checkpointer import CheckpointerManager  # noqa: E402
from kag.common.conf import KAG_CONFIG  # noqa: E402
from kag.interface import KAGBuilderChain, LLMClient, SolverPipelineABC  # noqa: E402
from kag.solver.reporter.trace_log_reporter import TraceLogReporter  # noqa: E402

logger = logging.getLogger(__name__)

_PIPELINE_KEYS = {
    "solver": "kag_solver_pipeline",
    "naive": "naive_rag_solver_pipeline",
}
_TOP_K_TYPES = {"rc_open_spg", "kg_fr_open_spg", "vector_chunk_retriever"}

# KAG stores configuration, checkpointers, schema sessions, and several
# retriever caches at process scope.  Serializing operations is intentional:
# it preserves correctness when callers create sequential clients for
# different knowledge bases in one Python process.
_KAG_RUNTIME_LOCK = RLock()
_ACTIVE_RUNTIME_CLIENT_ID: int | None = None
_ACTIVE_RUNTIME_GRAPH_PATH: Path | None = None
_INDEX_STATE_VERSION = 1
_INDEX_STATE_FILE_NAME = "kag_index_state.json"
_SENSITIVE_SETTING_PARTS = {
    "api_key",
    "apikey",
    "access_key",
    "access_token",
    "auth",
    "authorization",
    "credential",
    "credentials",
    "password",
    "secret",
    "token",
}

# These KAG caches are module globals keyed only by a query string.  They do
# not include a project namespace, graph path, or model identity, so retaining
# them while switching client instances can return evidence from another
# knowledge base.  The current local pipelines mainly exercise the vector
# cache, but clear all known query caches so future pipeline changes stay
# isolated too.
_KAG_QUERY_CACHE_LOCATIONS = (
    (
        "kag.common.tools.algorithm_tool.chunk_retriever.vector_chunk_retriever",
        "chunk_cached_by_query_map",
    ),
    (
        "kag.common.tools.algorithm_tool.chunk_retriever.vector_chunk_retriever_legacy",
        "chunk_cached_by_query_map",
    ),
    (
        "kag.common.tools.algorithm_tool.chunk_retriever.atomic_query_chunk_retriever",
        "chunk_cached_by_query_map",
    ),
    (
        "kag.common.tools.algorithm_tool.chunk_retriever.outline_chunk_retriever",
        "chunk_cached_by_query_map",
    ),
    (
        "kag.common.tools.algorithm_tool.chunk_retriever.summary_chunk_retriever",
        "chunk_cached_by_query_map",
    ),
    (
        "kag.common.tools.algorithm_tool.chunk_retriever.table_retriever",
        "chunk_cached_by_query_map",
    ),
    ("kag.common.tools.algorithm_tool.ner", "ner_tool_cache"),
)


def _response_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _patch_top_k(node: Any, top_k: int) -> None:
    if isinstance(node, dict):
        node_type = node.get("type")
        if node_type in _TOP_K_TYPES:
            node["top_k"] = top_k
        elif node_type == "rerank_by_vector":
            # RerankByVector reads ``rerank_top_k`` and defaults to 10; keep
            # it in sync so the fused ranking is not truncated below top_k.
            node["rerank_top_k"] = top_k
        for value in node.values():
            _patch_top_k(value, top_k)
    elif isinstance(node, list):
        for item in node:
            _patch_top_k(item, top_k)


def _file_safe_name(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_.")
    return value or "document"


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


class _TrackedLLMClient(LLMClient):
    """A lightweight LLMClient proxy that records actual KAG model calls.

    KAG's built-in TokenMeter records token totals but not request counts.  A
    proxy preserves the registered ``LLMClient`` interface while tracking
    synchronous and asynchronous calls made by builder and solver components.
    It deliberately returns itself from ``deepcopy`` because KAG copies
    configuration trees before constructing parallel builder chains.
    """

    def __init__(self, delegate: LLMClient, tracker: OperationUsageTracker) -> None:
        self._delegate = delegate
        self._tracker = tracker
        self.model = getattr(delegate, "model", "")

    def __deepcopy__(self, memo: dict[int, Any]) -> "_TrackedLLMClient":
        del memo
        return self

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def _invoke_delegate(self, method: Any, *args: Any, **kwargs: Any) -> Any:
        self._tracker.start_request()
        try:
            return method(*args, **kwargs)
        except Exception:
            self._tracker.fail_request()
            raise

    def __call__(self, prompt: Any, **kwargs: Any) -> Any:
        return self._invoke_delegate(self._delegate, prompt, **kwargs)

    async def acall(self, prompt: Any, **kwargs: Any) -> Any:
        self._tracker.start_request()
        try:
            return await self._delegate.acall(prompt, **kwargs)
        except Exception:
            self._tracker.fail_request()
            raise

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        return self._invoke_delegate(self._delegate.invoke, *args, **kwargs)

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        self._tracker.start_request()
        try:
            return await self._delegate.ainvoke(*args, **kwargs)
        except Exception:
            self._tracker.fail_request()
            raise

    def to_config(self) -> Any:
        return self._delegate.to_config()


class KAGClient:
    """Synchronous KAG client used by the prenatal benchmark.

    ``root_dir`` is the client project directory.  Text files are read from
    ``input_dir`` when supplied, otherwise from ``root_dir/input`` (or the
    unified corpus directory when that directory is absent); KAG's generated
    stores are placed in ``data_dir`` when given, otherwise in
    ``root_dir/kag_storage``.

    The two model functions are injectable.  ``llm`` (a KAG ``LLMClient``)
    and ``vectorize_model`` (a KAG ``VectorizeModelABC``) replace the
    YAML-configured clients, which is useful for deterministic local tests.
    Injected objects are wrapped in a lightweight telemetry proxy before KAG
    receives them, so they do not need to implement ``__deepcopy__``.

    KAG keeps its configuration and the memory graph in process-level
    singletons, so one client per knowledge-base directory should be active
    at a time; the facade re-points the global configuration before every
    operation to keep sequential usage correct.
    """

    def __init__(
        self,
        root_dir: str | Path,
        data_dir: str | Path | None = None,
        llm_overrides: Any | None = None,
        *,
        input_dir: str | Path | None = None,
        llm: Any | None = None,
        vectorize_model: Any | None = None,
        workspace: str | None = None,
        language: str = "en",
        top_k: int = DEFAULT_TOP_K,
        split_length: int = DEFAULT_SPLIT_LENGTH,
        max_iteration: int = DEFAULT_MAX_ITERATION,
        num_chains: int = DEFAULT_NUM_CHAINS,
        num_threads_per_chain: int = DEFAULT_NUM_THREADS_PER_CHAIN,
        verbose: bool = False,
        **kag_options: Any,
    ) -> None:
        if not isinstance(language, str) or not language.strip():
            raise ValueError("language must be a non-empty string")
        self._root_dir = Path(root_dir).expanduser().resolve()
        self._root_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir = (
            Path(data_dir).expanduser().resolve()
            if data_dir is not None
            else self._root_dir / "kag_storage"
        )
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._input_dir = (
            Path(input_dir).expanduser().resolve() if input_dir is not None else None
        )
        self.verbose = verbose
        self._llm_overrides = llm_overrides
        self._llm_instance = llm
        self._configured_llm: LLMClient | None = None
        self._vectorize_instance = vectorize_model
        if workspace is None:
            self._workspace = self._workspace_name(self._root_dir)
        elif not isinstance(workspace, str) or not workspace.strip():
            raise ValueError("workspace must be a non-empty string")
        else:
            self._workspace = workspace.strip()
        self._language = language.strip()
        self._top_k = _positive_int("top_k", top_k)
        self._split_length = _positive_int("split_length", split_length)
        self._max_iteration = _positive_int("max_iteration", max_iteration)
        self._num_chains = _positive_int("num_chains", num_chains)
        self._num_threads_per_chain = _positive_int(
            "num_threads_per_chain", num_threads_per_chain
        )
        self._kag_options = dict(kag_options)
        self._runner: EventLoopRunner | None = None
        self._closed = False
        self._state = ClientState.CREATED

        load_environment()
        self._llm_settings = resolve_model_settings(llm_overrides, "completion")
        self._vectorize_settings = resolve_model_settings(llm_overrides, "embedding")
        self._checkpoint_dir = self._data_dir / "ckpt"
        self._graph_path = self._checkpoint_dir / "MemoryGraphWriter"
        self._config_file = self._data_dir / "kag_config.yaml"
        self._index_state_file = self._data_dir / _INDEX_STATE_FILE_NAME
        self._write_and_load_config()
        self._state = ClientState.READY

    # ------------------------------------------------------------------
    # configuration
    # ------------------------------------------------------------------
    @staticmethod
    def _workspace_name(path: Path) -> str:
        value = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.name).strip("_.")
        return value or "kag"

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
    def graph_path(self) -> Path:
        return self._graph_path

    @property
    def state(self) -> ClientState:
        """Current lifecycle state of this facade."""
        return self._state

    @property
    def config(self) -> dict[str, Any]:
        """Runtime settings exposed for benchmark diagnostics."""
        settings = {
            "config_file": str(self._config_file),
            "index_state_file": str(self._index_state_file),
            "input_dir": str(self._input_dir) if self._input_dir is not None else None,
            "checkpoint_path": str(self._checkpoint_dir),
            "graph_path": str(self._graph_path),
            "namespace": self._workspace,
            "language": self._language,
            "top_k": self._top_k,
            "split_length": self._split_length,
            "max_iteration": self._max_iteration,
            "num_chains": self._num_chains,
            "num_threads_per_chain": self._num_threads_per_chain,
            "llm": self._llm_settings,
            "vectorize_model": self._vectorize_settings,
            **self._kag_options,
        }
        return self._safe_diagnostic_value(settings)

    def _build_config_dict(self) -> dict[str, Any]:
        return build_kag_config(
            namespace=self._workspace,
            language=self._language,
            checkpoint_path=self._checkpoint_dir,
            graph_path=self._graph_path,
            llm=self._llm_settings,
            vectorize_model=self._vectorize_settings,
            top_k=self._top_k,
            split_length=self._split_length,
            max_iteration=self._max_iteration,
            num_chains=self._num_chains,
            num_threads_per_chain=self._num_threads_per_chain,
            log_level="DEBUG" if self.verbose else "INFO",
        )

    def _write_and_load_config(self) -> None:
        with _KAG_RUNTIME_LOCK:
            config = self._build_config_dict()
            write_config_file(
                config,
                self._config_file,
                api_key_env=self._llm_settings["api_key_env"],
            )
            initialize_kag_config(self._config_file)

    def _activate_runtime(self) -> None:
        """Activate this client's global KAG configuration and clear leakage.

        KAG 0.8 stores ``KAG_CONFIG`` process-wide.  Constructing client B
        after client A therefore replaces A's active configuration even if A
        has never indexed or queried yet.  This method is called under
        :data:`_KAG_RUNTIME_LOCK` at the beginning of every public operation,
        before any configured LLM/vectorizer is constructed.
        """

        global _ACTIVE_RUNTIME_CLIENT_ID, _ACTIVE_RUNTIME_GRAPH_PATH

        self._ensure_open()
        previous_client_id = _ACTIVE_RUNTIME_CLIENT_ID
        previous_graph_path = _ACTIVE_RUNTIME_GRAPH_PATH
        # Re-seed the offline schema cache as part of configuration activation
        # (``initialize_kag_config`` performs both actions).
        initialize_kag_config(self._config_file)
        self._clear_retriever_caches()
        # MemoryGraph is keyed only by checkpoint path.  If two client
        # instances intentionally share a data directory but use different
        # namespaces or vectorizers, the old singleton is invalid for the new
        # active client.  Evict only on a client switch; repeated calls on the
        # same client retain its loaded graph.
        if previous_client_id != id(self):
            if previous_graph_path is not None:
                self._close_checkpointers_at(previous_graph_path.parent)
                self._evict_memory_graph_path(previous_graph_path)
            self._evict_memory_graph()
        _ACTIVE_RUNTIME_CLIENT_ID = id(self)
        _ACTIVE_RUNTIME_GRAPH_PATH = self._graph_path.resolve()

    def _operation_llm(self, tracker: OperationUsageTracker) -> _TrackedLLMClient:
        """Return a call-counting proxy around an injected or configured LLM."""

        delegate = self._llm_instance
        if delegate is None:
            if self._configured_llm is None:
                config = {
                    key: value
                    for key, value in self._llm_settings.items()
                    if key != "api_key_env"
                }
                self._configured_llm = LLMClient.from_config(copy.deepcopy(config))
            delegate = self._configured_llm
        if not isinstance(delegate, LLMClient):
            raise TypeError("llm must be an instance of kag.interface.LLMClient")
        return _TrackedLLMClient(delegate, tracker)

    def _pipeline_config(
        self, key: str, *, llm: LLMClient | None = None
    ) -> dict[str, Any]:
        """Fetch a pipeline section with this client's configuration active."""
        section = KAG_CONFIG.all_config[key]
        return substitute_instances(
            section,
            llm=llm if llm is not None else self._llm_instance,
            vectorize_model=self._vectorize_instance,
        )

    def reload_config(self) -> dict[str, Any]:
        """Regenerate the KAG configuration file and return diagnostics."""
        self._ensure_open()
        with _KAG_RUNTIME_LOCK:
            self._write_and_load_config()
            self._activate_runtime()
            return self.config

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("KAGClient is closed")

    # ------------------------------------------------------------------
    # event loop
    # ------------------------------------------------------------------
    def _run(self, coro: Any) -> Any:
        self._ensure_async_runtime()
        assert self._runner is not None
        return self._runner.run(coro)

    def _ensure_async_runtime(self) -> None:
        """Create this client's query loop and detach stale KAG limiters.

        KAG's ``RateLimiterManger`` caches :class:`aiolimiter.AsyncLimiter`
        instances by model name across the whole process.  An AsyncLimiter is
        loop-affine once it has been awaited.  This facade intentionally owns
        one background event loop per client, so a limiter from a closed or a
        different client's loop must not be reused for the next query.
        """

        if self._runner is None:
            self._runner = EventLoopRunner()
        self._refresh_async_limiters(self._runner.loop)

    def _refresh_async_limiters(self, target_loop: Any) -> None:
        """Replace only KAG limiters bound to a different event loop.

        Rate-limit state remains intact for repeated queries on this same
        client loop.  Configured KAG components are ephemeral per query, so
        removing their stale registry entries is sufficient; the injected or
        cached completion/vector models need their held limiter references
        updated as well.
        """

        try:
            from aiolimiter import AsyncLimiter
            from kag.common.rate_limiter import RATE_LIMITER_MANGER

            limiter_map = RATE_LIMITER_MANGER.limiter_map
            components = (
                self._llm_instance,
                self._configured_llm,
                self._vectorize_instance,
            )

            def is_stale(limiter: Any) -> bool:
                return isinstance(limiter, AsyncLimiter) and getattr(
                    limiter, "_event_loop", None
                ) not in (None, target_loop)

            stale_by_id: dict[int, Any] = {
                id(limiter): limiter
                for limiter in limiter_map.values()
                if is_stale(limiter)
            }
            stale_by_id.update(
                {
                    id(limiter): limiter
                    for limiter in (
                        getattr(component, "limiter", None) for component in components
                    )
                    if is_stale(limiter)
                }
            )
            if not stale_by_id:
                return

            replacements: dict[int, AsyncLimiter] = {}

            def replacement_for(limiter: AsyncLimiter) -> AsyncLimiter:
                key = id(limiter)
                if key not in replacements:
                    replacements[key] = AsyncLimiter(
                        limiter.max_rate,
                        limiter.time_period,
                    )
                return replacements[key]

            # Keep injected/cached client objects compatible with the runner,
            # rather than merely deleting their name from the global map.
            for component in components:
                limiter = getattr(component, "limiter", None)
                if id(limiter) not in stale_by_id:
                    continue
                try:
                    component.limiter = replacement_for(limiter)
                except (AttributeError, TypeError):
                    # Custom injected implementations may expose a read-only
                    # limiter; leave them untouched rather than changing their
                    # contract.  KAG's own implementations are writable.
                    logger.debug("failed to replace injected KAG async limiter")

            for name, limiter in list(limiter_map.items()):
                if id(limiter) not in stale_by_id:
                    continue
                limiter_map[name] = replacement_for(limiter)
        except Exception:  # noqa: BLE001 - vendor limiter layout may vary
            logger.debug("failed to refresh KAG async rate limiters", exc_info=True)

    # ------------------------------------------------------------------
    # indexing
    # ------------------------------------------------------------------
    def _reset_storage(self) -> None:
        """Drop KAG checkpoints so the next index() re-extracts everything."""
        # Release diskcache handles before removing their directory.  KAG's
        # manager is process-global, so close only pointers rooted in this
        # client's storage rather than indiscriminately closing another
        # inactive client's handles.
        self._close_client_checkpointers()
        self._evict_memory_graph()
        if self._checkpoint_dir.exists():
            shutil.rmtree(self._checkpoint_dir)
        if self._index_state_file.is_file():
            self._index_state_file.unlink()
        self._clear_retriever_caches()
        self._invalidate_graph_cache()

    @staticmethod
    def _is_sensitive_setting_key(key: Any) -> bool:
        normalized = (
            re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(key).strip())
            .casefold()
            .replace("-", "_")
        )
        if normalized in _SENSITIVE_SETTING_PARTS:
            return True
        return any(
            normalized.startswith(f"{part}_") or normalized.endswith(f"_{part}")
            for part in _SENSITIVE_SETTING_PARTS
        )

    @classmethod
    def _safe_fingerprint_value(cls, value: Any) -> Any:
        """Return a deterministic, credential-free value for index state."""

        if isinstance(value, Mapping):
            return {
                str(key): cls._safe_fingerprint_value(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
                if not cls._is_sensitive_setting_key(key)
            }
        if isinstance(value, (list, tuple)):
            return [cls._safe_fingerprint_value(item) for item in value]
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        # Do not use repr(): provider objects often include credential-bearing
        # connection details in it.  Class identity is enough to invalidate an
        # incompatible injected component conservatively.
        return {
            "implementation": f"{type(value).__module__}.{type(value).__qualname__}"
        }

    @classmethod
    def _safe_diagnostic_value(cls, value: Any) -> Any:
        """Redact credentials recursively before exposing client diagnostics."""

        if isinstance(value, Mapping):
            safe: dict[str, Any] = {}
            for key, item in value.items():
                normalized = (
                    re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(key).strip())
                    .casefold()
                    .replace("-", "_")
                )
                # Environment-variable names are not credentials themselves,
                # and retaining them makes diagnostics actionable.
                if cls._is_sensitive_setting_key(key) and not normalized.endswith(
                    "_env"
                ):
                    safe[str(key)] = "<redacted>"
                else:
                    safe[str(key)] = cls._safe_diagnostic_value(item)
            return safe
        if isinstance(value, (list, tuple)):
            return [cls._safe_diagnostic_value(item) for item in value]
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        return {
            "implementation": f"{type(value).__module__}.{type(value).__qualname__}"
        }

    def _component_fingerprint(
        self, instance: Any | None, settings: Mapping[str, Any]
    ) -> dict[str, Any]:
        if instance is None:
            return {
                "source": "configuration",
                "settings": self._safe_fingerprint_value(settings),
            }
        descriptor: dict[str, Any] = {
            "source": "injected",
            "implementation": f"{type(instance).__module__}.{type(instance).__qualname__}",
        }
        for attribute in ("model", "_vector_dimensions", "vector_dimensions"):
            value = getattr(instance, attribute, None)
            if value is None or callable(value):
                continue
            descriptor[attribute.removeprefix("_")] = self._safe_fingerprint_value(
                value
            )
        to_config = getattr(instance, "to_config", None)
        if callable(to_config):
            try:
                descriptor["config"] = self._safe_fingerprint_value(to_config())
            except Exception:  # noqa: BLE001 - opaque injection stays identifiable
                descriptor["config"] = {"available": False}
        return descriptor

    def _index_state_payload(self, documents: list[DocumentRecord]) -> dict[str, Any]:
        """Describe every input that can change persisted KAG graph content."""

        return {
            "version": _INDEX_STATE_VERSION,
            "workspace": self._workspace,
            "language": self._language,
            "split_length": self._split_length,
            "completion": self._component_fingerprint(
                self._llm_instance, self._llm_settings
            ),
            "embedding": self._component_fingerprint(
                self._vectorize_instance, self._vectorize_settings
            ),
            "documents": [
                {
                    "document_id": document.document_id,
                    "content_sha256": hashlib.sha256(
                        document.text.encode("utf-8")
                    ).hexdigest(),
                }
                for document in documents
            ],
        }

    @staticmethod
    def _fingerprint(payload: Mapping[str, Any]) -> str:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _read_index_fingerprint(self) -> str | None:
        try:
            data = json.loads(self._index_state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        if not isinstance(data, Mapping) or data.get("version") != _INDEX_STATE_VERSION:
            return None
        fingerprint = data.get("fingerprint")
        return fingerprint if isinstance(fingerprint, str) else None

    def _has_index_artifacts(self) -> bool:
        try:
            return self._checkpoint_dir.is_dir() and any(
                path.is_file() for path in self._checkpoint_dir.rglob("*")
            )
        except OSError:
            return True

    def _prepare_index_storage(
        self, documents: list[DocumentRecord], cache: bool
    ) -> tuple[dict[str, Any], str]:
        """Invalidate component checkpoints when their producing inputs drift."""

        payload = self._index_state_payload(documents)
        fingerprint = self._fingerprint(payload)
        if not cache:
            self._reset_storage()
        elif (
            self._has_index_artifacts()
            and self._read_index_fingerprint() != fingerprint
        ):
            logger.info(
                "KAG index inputs/configuration changed; rebuilding checkpoint storage"
            )
            self._reset_storage()
        return payload, fingerprint

    def _write_index_state(self, payload: Mapping[str, Any], fingerprint: str) -> None:
        self._index_state_file.write_text(
            json.dumps(
                {
                    "version": _INDEX_STATE_VERSION,
                    "fingerprint": fingerprint,
                    "inputs": payload,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _clear_retriever_caches() -> None:
        """Clear KAG's global query/config caches before changing clients."""

        for module_name, attribute in _KAG_QUERY_CACHE_LOCATIONS:
            try:
                cache = getattr(importlib.import_module(module_name), attribute)
                clear = getattr(getattr(cache, "cache", None), "clear", None)
                if callable(clear):
                    clear()
            except Exception:  # noqa: BLE001 - version-specific optional modules
                logger.debug(
                    "failed to clear KAG cache %s.%s",
                    module_name,
                    attribute,
                    exc_info=True,
                )
        try:
            from kag.common.conf import KAG_QA_TASK_CONFIG

            KAG_QA_TASK_CONFIG.cache.clear()
        except Exception:  # noqa: BLE001 - cache availability is version-specific
            logger.debug("failed to clear KAG task configuration cache", exc_info=True)

    def _evict_memory_graph(self) -> None:
        """Evict this storage path from KAG's in-process MemoryGraph cache.

        This intentionally leaves the checkpoint directory alone.  It is safe
        to call at close/reload boundaries and lets a later client reopen the
        durable local graph with its own namespace and vectorizer.
        """

        self._evict_memory_graph_path(self._graph_path)

    @staticmethod
    def _evict_memory_graph_path(graph_path: Path) -> None:
        """Evict one durable graph path without changing its checkpoint files."""

        try:
            from kag.common.graphstore.memory_graph import MemoryGraph

            target = graph_path.resolve()
            for key in list(MemoryGraph._instances):
                try:
                    key_path = Path(str(key)).expanduser().resolve()
                except OSError:
                    continue
                if key_path == target:
                    MemoryGraph._instances.pop(key, None)
        except Exception:  # noqa: BLE001 - best-effort vendor cache cleanup
            logger.debug("failed to evict KAG MemoryGraph singleton", exc_info=True)

    def _close_client_checkpointers(self) -> None:
        """Close only process-global KAG checkpoints owned by this client."""

        self._close_checkpointers_at(self._checkpoint_dir)

    @staticmethod
    def _close_checkpointers_at(checkpoint_dir: Path) -> None:
        """Close KAG checkpointers below one known client checkpoint root."""

        try:
            target = checkpoint_dir.resolve()
            with CheckpointerManager._LOCK:
                for key, pointer in list(CheckpointerManager._CKPT_OBJS.items()):
                    directory = getattr(pointer, "_ckpt_dir", None)
                    if not directory:
                        continue
                    try:
                        pointer_path = Path(str(directory)).expanduser().resolve()
                    except OSError:
                        continue
                    if pointer_path != target and target not in pointer_path.parents:
                        continue
                    try:
                        pointer.close()
                    finally:
                        CheckpointerManager._CKPT_OBJS.pop(key, None)
        except Exception:  # noqa: BLE001 - cleanup must not mask client work
            logger.debug("failed to close KAG checkpointers", exc_info=True)

    def _close_configured_llm(self) -> None:
        """Close HTTP clients owned by a YAML-configured KAG LLM.

        Caller-injected LLMs remain caller-owned.  KAG's OpenAI/Azure clients
        allocate both sync and async httpx pools, neither of which the vendor
        facade closes itself when a benchmark client goes away.
        """

        delegate = self._configured_llm
        self._configured_llm = None
        if delegate is None:
            return
        awaitables: list[Any] = []
        for attribute in ("client", "aclient"):
            transport = getattr(delegate, attribute, None)
            close = getattr(transport, "close", None)
            if not callable(close):
                continue
            try:
                result = close()
                if inspect.isawaitable(result):
                    awaitables.append(result)
            except Exception:  # noqa: BLE001 - cleanup must not mask close()
                logger.debug(
                    "failed to close KAG %s transport", attribute, exc_info=True
                )
        if not awaitables:
            return

        async def _await_closers() -> None:
            for awaitable in awaitables:
                try:
                    await awaitable
                except Exception:  # noqa: BLE001 - cleanup must not mask close()
                    logger.debug("failed to close KAG async transport", exc_info=True)

        temporary_runner: EventLoopRunner | None = None
        try:
            if self._runner is None:
                temporary_runner = EventLoopRunner()
                temporary_runner.run(_await_closers())
            else:
                self._runner.run(_await_closers())
        except Exception:  # noqa: BLE001 - cleanup must not mask close()
            logger.debug("failed to await KAG async transport close", exc_info=True)
        finally:
            if temporary_runner is not None:
                temporary_runner.close()

    def _invalidate_graph_cache(self) -> None:
        """Remove the persisted graph snapshot and in-process singleton."""
        graph_pickle = self._graph_path / "graph"
        if graph_pickle.is_file():
            graph_pickle.unlink()
        self._evict_memory_graph()

    def _stage_documents(self, documents: list[DocumentRecord]) -> list[DocumentRecord]:
        staging = self._data_dir / "input"
        staging.mkdir(parents=True, exist_ok=True)
        for old in staging.iterdir():
            if old.is_file():
                old.unlink()
        staged: list[DocumentRecord] = []
        used_names: set[str] = set()
        for index, document in enumerate(documents):
            name = _file_safe_name(document.document_id)
            while name in used_names:
                name = f"{name}-{index}"
            used_names.add(name)
            path = staging / f"{name}.txt"
            path.write_text(document.text, encoding="utf-8")
            staged.append(
                DocumentRecord(document.document_id, document.text, str(path))
            )
        return staged

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
        """Run an index operation while this client's KAG globals are active."""

        self._ensure_open()
        with _KAG_RUNTIME_LOCK:
            self._ensure_open()
            self._activate_runtime()
            return self._index(
                method=method,
                is_update_run=is_update_run,
                cache=cache,
                dry_run=dry_run,
                skip_validation=skip_validation,
                callbacks=callbacks,
                input_documents=input_documents,
            )

    def _index(
        self,
        method: str | IndexMethod = IndexMethod.STANDARD,
        is_update_run: bool = False,
        cache: bool = True,
        dry_run: bool = False,
        skip_validation: bool = False,
        callbacks: list[Any] | None = None,
        input_documents: Any | None = None,
    ) -> IndexResult:
        """Run the KAG builder chain over the project's input documents.

        KAG has one unstructured extraction pipeline rather than Microsoft's
        standard vs fast workflow split; ``method`` and ``is_update_run`` are
        accepted for API compatibility and recorded in the result metadata.
        ``cache=True`` reuses component checkpoints only when the source
        documents and graph-producing configuration fingerprint match; a
        mismatch triggers a safe full rebuild.
        """
        del is_update_run, skip_validation, callbacks
        try:
            IndexMethod(method)
        except ValueError as exc:
            raise ValueError(f"unsupported index method: {method}") from exc
        documents = load_documents(
            self._root_dir,
            input_documents,
            input_dir=self._input_dir,
        )
        if dry_run:
            return IndexResult(outputs=documents)
        index_state, index_fingerprint = self._prepare_index_storage(documents, cache)
        if not documents:
            # A successful empty index must never leave an earlier graph
            # queryable.  In particular, this handles a corpus that was
            # emptied between runs: KAG has no builder invocation in that
            # case, so it cannot invalidate its own component checkpoints.
            if self._has_index_artifacts():
                self._reset_storage()
            self._write_index_state(index_state, index_fingerprint)
            return IndexResult(outputs=[])

        tracker = OperationUsageTracker(self._llm_settings["model"])
        operation_llm = self._operation_llm(tracker)
        tokens_before = snapshot_token_totals()
        records = (
            self._stage_documents(documents)
            if input_documents is not None
            else documents
        )
        # Fetch the pipeline config once in the calling thread: initializing
        # KAG's process-level configuration mutates global state, which must
        # not race inside the worker pool below.
        chain_config = self._pipeline_config("kag_builder_pipeline", llm=operation_llm)[
            "chain"
        ]
        outputs: list[dict[str, Any]] = []
        errors: list[str] = []

        def _build(record: DocumentRecord) -> str | dict[str, Any]:
            try:
                chain = KAGBuilderChain.from_config(copy.deepcopy(chain_config))
                chain.invoke(record.file_path, max_workers=self._num_threads_per_chain)
                return {
                    "id": record.document_id,
                    "file_path": record.file_path,
                    "method": str(method),
                }
            except Exception as exc:  # noqa: BLE001
                logger.exception("KAG builder failed for %s", record.document_id)
                return f"{record.document_id}: {type(exc).__name__}: {exc}"

        try:
            with ThreadPoolExecutor(max_workers=self._num_chains) as pool:
                for result in pool.map(_build, records):
                    if isinstance(result, str):
                        errors.append(result)
                    else:
                        outputs.append(result)
        finally:
            self._close_client_checkpointers()
            self._invalidate_graph_cache()

        tokens_after = snapshot_token_totals()
        tracker.add_usage(
            {
                key: max(0, tokens_after[key] - tokens_before[key])
                for key in tokens_before
            }
        )
        if not errors:
            self._write_index_state(index_state, index_fingerprint)
        return IndexResult(
            outputs=outputs,
            errors=errors,
            has_errors=bool(errors),
            telemetry=tracker.telemetry(),
        )

    # ------------------------------------------------------------------
    # querying
    # ------------------------------------------------------------------
    def _trace_data(self, info: Any) -> dict[str, Any]:
        try:
            return info.to_dict()
        except Exception:  # noqa: BLE001 - trace data is diagnostic only
            return {}

    def _references(self, info: Any) -> list[dict[str, Any]]:
        references: list[dict[str, Any]] = []
        for ref in getattr(info, "reference", None) or []:
            try:
                references.append(ref.to_dict())
            except Exception:  # noqa: BLE001
                references.append({"content": str(ref)})
        return references

    @staticmethod
    def _chunks_from_references(
        references: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Normalize KAG references into the shared retrieved-context shape.

        KAG reports ``ChunkData`` as ``content``/``title``/``chunk_id``.
        The benchmark's generic evidence extractor expects a ``chunks`` table
        and a file-like identifier, so retain the original title and expose it
        as ``file_path`` as well.  KAG's title is the source filename stem
        (with an optional ``_split_N`` suffix), which the KAG benchmark
        resolver maps back to the unified-corpus manifest.
        """

        chunks: list[dict[str, Any]] = []
        for reference in references:
            chunk = dict(reference)
            title = str(
                chunk.get(
                    "title",
                    chunk.get("document_name", chunk.get("file_path", "")),
                )
                or ""
            )
            if title:
                chunk.setdefault("file_path", title)
                chunk.setdefault("source_id", title)
            if "chunk_id" not in chunk and chunk.get("id") is not None:
                chunk["chunk_id"] = chunk["id"]
            chunks.append(chunk)
        return chunks

    async def _query_async(
        self,
        query: str,
        backend: str,
        top_k: int | None,
        llm: LLMClient,
    ) -> Any:
        pipeline_config = self._pipeline_config(_PIPELINE_KEYS[backend], llm=llm)
        if top_k is not None:
            _patch_top_k(pipeline_config, top_k)
        pipeline = SolverPipelineABC.from_config(pipeline_config)
        reporter = TraceLogReporter()
        answer = await guarded_query(pipeline.ainvoke(query, reporter=reporter))
        info, status = reporter.generate_report_data()
        return answer, info, status

    def search(
        self,
        query: str,
        method: str | SearchMethod = SearchMethod.SOLVER,
        *,
        response_type: str = DEFAULT_RESPONSE_TYPE,
        top_k: int | None = None,
    ) -> QueryResult:
        """Answer a query while this client's KAG globals are active."""

        self._ensure_open()
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if top_k is not None:
            _positive_int("top_k", top_k)
        # Validate aliases before acquiring the global runtime lock.
        resolve_search_method(method)
        with _KAG_RUNTIME_LOCK:
            self._ensure_open()
            self._activate_runtime()
            return self._search(
                query,
                method,
                response_type=response_type,
                top_k=top_k,
            )

    def _search(
        self,
        query: str,
        method: str | SearchMethod = SearchMethod.SOLVER,
        *,
        response_type: str = DEFAULT_RESPONSE_TYPE,
        top_k: int | None = None,
    ) -> QueryResult:
        """Answer ``query`` with KAG and return the response with its trace."""
        del response_type  # KAG prompts fix their own output style.
        backend, public = resolve_search_method(method)

        # Do this before constructing a YAML-configured LLM.  Otherwise it
        # could retain a KAG process-global AsyncLimiter from another client's
        # event loop before ``_run`` gets a chance to activate this one.
        self._ensure_async_runtime()
        tracker = OperationUsageTracker(self._llm_settings["model"])
        operation_llm = self._operation_llm(tracker)
        tokens_before = snapshot_token_totals()
        try:
            answer, info, status = self._run(
                self._query_async(query, backend, top_k, operation_llm)
            )
        except Exception:
            # The tracking proxy records failed provider calls itself.  Do not
            # count an outer pipeline exception a second time.
            raise
        tokens_after = snapshot_token_totals()
        tracker.add_usage(
            {
                key: max(0, tokens_after[key] - tokens_before[key])
                for key in tokens_before
            }
        )
        references = self._references(info)
        chunks = self._chunks_from_references(references)
        trace = self._trace_data(info)
        return QueryResult(
            response=_response_text(answer),
            context_data={
                "chunks": chunks,
                "trace": trace,
                "references": references,
            },
            query=query,
            method=public,
            raw_data={
                "data": {"chunks": chunks, "references": references},
                "trace": trace,
                "status": status,
            },
            telemetry=tracker.telemetry(),
        )

    def query(
        self,
        query: str,
        method: str | SearchMethod = SearchMethod.SOLVER,
        **kwargs: Any,
    ) -> QueryResult:
        """Generic query alias matching KAG's terminology."""
        return self.search(query, method, **kwargs)

    def solver_search(
        self,
        query: str,
        response_type: str = DEFAULT_RESPONSE_TYPE,
        **kwargs: Any,
    ) -> QueryResult:
        """Full KAG logic-form solver pipeline (default method)."""
        return self.search(
            query,
            SearchMethod.SOLVER,
            response_type=response_type,
            **kwargs,
        )

    def naive_search(
        self,
        query: str,
        response_type: str = DEFAULT_RESPONSE_TYPE,
        **kwargs: Any,
    ) -> QueryResult:
        """Vector-retrieval-only RAG pipeline."""
        return self.search(
            query,
            SearchMethod.NAIVE,
            response_type=response_type,
            **kwargs,
        )

    def basic_search(
        self,
        query: str,
        response_type: str = DEFAULT_RESPONSE_TYPE,
        **kwargs: Any,
    ) -> QueryResult:
        """Alias of :meth:`naive_search` kept for cross-baseline parity."""
        return self.search(
            query,
            SearchMethod.BASIC,
            response_type=response_type,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        global _ACTIVE_RUNTIME_CLIENT_ID, _ACTIVE_RUNTIME_GRAPH_PATH

        with _KAG_RUNTIME_LOCK:
            if self._closed:
                return
            try:
                self._close_configured_llm()
                if self._runner is not None:
                    self._runner.close()
                    self._runner = None
            finally:
                self._clear_retriever_caches()
                self._close_client_checkpointers()
                self._evict_memory_graph()
                if _ACTIVE_RUNTIME_CLIENT_ID == id(self):
                    _ACTIVE_RUNTIME_CLIENT_ID = None
                    _ACTIVE_RUNTIME_GRAPH_PATH = None
                self._closed = True
                self._state = ClientState.CLOSED

    def __enter__(self) -> "KAGClient":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


__all__ = ["KAGClient"]
