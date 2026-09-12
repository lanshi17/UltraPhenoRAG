"""Compatibility shims for the vendored PathRAG checkout.

``PathRAG/llm.py`` imports every optional backend unconditionally: Amazon
Bedrock (``aioboto3``), ModelScope (``modelscope``), and vLLM (``vllm``).
None of them is installed — or needed — for the OpenAI-compatible benchmark
providers, and the import fails before any function can run.  The shim
installs inert module stubs for exactly the backends that are absent from
the current environment, so ``import PathRAG`` behaves like the vendored
LightRAG checkout: the supported dependency combination stays explicit and
reproducible.

``PathRAG/PathRAG.py`` additionally crashes when ``enable_llm_cache`` is
False: the fallback ``hashing_kv`` storage is constructed without the
``namespace`` and ``embedding_func`` its dataclass requires.  The second
shim substitutes an inert throwaway cache there.  The engine's fixed
relative ``PathRAG.log`` file is left untouched, matching how the
repository already tolerates ``lightrag.log`` at its root.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
import types
from typing import Any


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _install_stub(name: str, **attributes: object) -> None:
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    sys.modules.setdefault(name, module)


def ensure_pathrag_llm_cache_compatibility() -> None:
    """Fix PathRAG's crash when ``enable_llm_cache`` is False.

    ``PathRAG.__post_init__`` and ``aquery`` build their ``hashing_kv``
    fallback with ``key_string_value_json_storage_cls(global_config=...)``
    only, omitting the ``namespace`` and ``embedding_func`` the JSON KV
    dataclass requires — the constructor raises ``TypeError`` the moment a
    cache-disabled client touches the LLM wrapper.  The patch substitutes a
    lenient KV class at the storage-registry lookup: instances constructed
    with an explicit namespace keep the full vendor behavior (``full_docs``,
    ``text_chunks``, and the enabled cache are untouched), while the
    namespace-less fallback becomes an inert throwaway so a disabled cache
    behaves as documented — every lookup misses and nothing persists.
    """

    from PathRAG import PathRAG
    from PathRAG.storage import JsonKVStorage

    if getattr(PathRAG, "_PRENATAL_PATHRAG_LLM_CACHE_COMPAT", False):
        return

    original = PathRAG._get_storage_class
    inert_namespace = "__pathrag_llm_cache_disabled__"

    class _LenientJsonKVStorage(JsonKVStorage):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._inert = "namespace" not in kwargs and not args
            if self._inert:
                kwargs["namespace"] = inert_namespace
                kwargs.setdefault("embedding_func", None)
            super().__init__(*args, **kwargs)

        def __post_init__(self) -> None:
            if self._inert:
                self._data = {}
                self._file_name = None
                return
            super().__post_init__()

        async def get_by_id(self, id: str) -> Any:  # noqa: A002
            if self._inert:
                return None
            return await super().get_by_id(id)

        async def upsert(self, data: dict) -> Any:
            if self._inert:
                return None
            return await super().upsert(data)

        async def index_done_callback(self) -> None:
            if self._inert:
                return
            await super().index_done_callback()

    def patched(self: Any) -> dict:
        classes = original(self)
        if not self.enable_llm_cache:
            classes["JsonKVStorage"] = _LenientJsonKVStorage
        return classes

    PathRAG._get_storage_class = patched
    PathRAG._PRENATAL_PATHRAG_LLM_CACHE_COMPAT = True


class _ModePinnedQueryParam:
    """Proxy over a ``QueryParam`` copy that keeps ``mode`` constant.

    Reads see the pinned value; writes to ``mode`` are dropped.  Every other
    attribute lives on the copied ``QueryParam``, so callers never observe
    mutation.
    """

    def __init__(self, query_param: Any, mode: str = "hybrid") -> None:
        object.__setattr__(self, "_target", copy.copy(query_param))
        object.__setattr__(self, "_mode", mode)

    def __getattr__(self, name: str) -> Any:
        if name == "mode":
            return object.__getattribute__(self, "_mode")
        return getattr(object.__getattribute__(self, "_target"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "mode":
            return
        setattr(object.__getattribute__(self, "_target"), name, value)


def ensure_pathrag_context_build_compatibility() -> None:
    """Keep PathRAG's context assembly on the hybrid branch.

    ``operate._build_query_context`` (vendored ``libs/path_rag``) assigns
    ``query_param.mode = "global" | "local"`` whenever one keyword side comes
    back empty (operate.py 610/633/652).  The hybrid ``combine_contexts``
    branch at line 653 is then skipped, yet the return template still reads
    ``text_units_context`` for its ``-----Sources-----`` section (line 673) —
    so any degraded retrieval crashes inside ``aquery`` with an
    ``UnboundLocalError``.  Pinning the parameter copy to ``hybrid`` is safe:
    ``process_combine_contexts`` tolerates empty sides (returns the non-empty
    one, or ``""`` when both are empty), and ``mode`` is consulted by no code
    downstream of ``_build_query_context``.  The client only ever exposes the
    hybrid search method, so no vendor file is modified.
    """

    from PathRAG import operate as operate_module

    if getattr(operate_module, "_PRENATAL_PATHRAG_CONTEXT_COMPAT", False):
        return

    original = operate_module._build_query_context

    async def patched(
        query: Any,
        knowledge_graph_inst: Any,
        entities_vdb: Any,
        relationships_vdb: Any,
        text_chunks_db: Any,
        query_param: Any,
    ) -> Any:
        return await original(
            query,
            knowledge_graph_inst,
            entities_vdb,
            relationships_vdb,
            text_chunks_db,
            _ModePinnedQueryParam(query_param),
        )

    operate_module._build_query_context = patched
    operate_module._PRENATAL_PATHRAG_CONTEXT_COMPAT = True


def ensure_pathrag_import_compatibility() -> None:
    """Stub the optional backend imports PathRAG performs at module load."""

    if not _module_available("aioboto3"):
        # Only ``aioboto3.Session()`` inside the Bedrock backends is used.
        _install_stub("aioboto3")
    if not _module_available("modelscope"):
        # Accessed as ``ms.*`` inside ModelScope-only completion helpers.
        _install_stub("modelscope")
    if not _module_available("vllm"):
        # ``from vllm import LLM`` at module scope of PathRAG/llm.py.
        _install_stub("vllm", LLM=object)


__all__ = [
    "ensure_pathrag_import_compatibility",
    "ensure_pathrag_llm_cache_compatibility",
    "ensure_pathrag_context_build_compatibility",
]
