"""KAG project configuration assembly.

Builds the in-memory configuration dict (and its ``kag_config.yaml`` file) that
drives the vendored KAG 0.8 runtime in fully local mode:

* the builder pipeline writes subgraphs through ``memory_graph_writer`` into
  ``<checkpoint_path>/MemoryGraphWriter`` (a diskcache checkpoint), and
* the solver pipelines read that graph back through ``memory_graph_api`` /
  ``memory_search_api`` without any OpenSPG server.

``project.host_addr``/``project.id`` stay in the config because KAG's
``LogicFormConfiguration`` requires them, but they are never contacted in
memory mode.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from threading import RLock
from typing import Any

import yaml
from cachetools import LRUCache

from benchmark.config import RAG_ENVIRONMENT, load_environment

from .llm_config import LLMConfigOverrides

DEFAULT_API_KEY_ENV = RAG_ENVIRONMENT["api_key"]
DEFAULT_HOST_ADDR = "http://127.0.0.1:8887"
DEFAULT_PROJECT_ID = "1"
DEFAULT_BIZ_SCENE = "default"

_LLM_KEYS = {"llm", "llm_client", "llm_module", "context_select_llm"}
_VECTORIZE_KEYS = {"vectorize_model"}
_SENSITIVE_KEY_PARTS = {
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
_OFFLINE_SCHEMA_CACHE_LOCK = RLock()


class _EnvRef:
    """Marker for YAML emission as a ``!ENV <name>`` tag."""

    def __init__(self, name: str) -> None:
        self.name = name


class _EnvRefDumper(yaml.SafeDumper):
    """Dumper which keeps KAG's ``!ENV VAR`` syntax human-readable.

    PyYAML normally single-quotes a scalar with an explicit tag because the
    tag disables implicit scalar resolution.  KAG's example configurations
    use an unquoted environment variable name (``!ENV RAG_API_KEY``), and
    emitting the same form makes generated files both familiar and easy to
    audit without changing their YAML meaning.
    """

    def choose_scalar_style(self) -> str:
        if self.event.tag == "!ENV":
            return ""
        return super().choose_scalar_style()


def _env_ref_representer(dumper: "_EnvRefDumper", data: _EnvRef) -> Any:
    return dumper.represent_scalar("!ENV", data.name)


_EnvRefDumper.add_representer(_EnvRef, _env_ref_representer)


def _override_value(overrides: Any, role: str, name: str) -> Any:
    """Read one role override from a dataclass or mapping.

    Other baseline clients accept both their public dataclasses and mapping
    equivalents.  Supporting both here keeps ``KAGClient`` drop-in
    compatible with the benchmark entry points.
    """

    if overrides is None:
        return None
    override = (
        overrides.get(role)
        if isinstance(overrides, Mapping)
        else getattr(overrides, role, None)
    )
    if isinstance(override, Mapping):
        return override.get(name)
    return getattr(override, name, None)


def _validate_role_override(overrides: Any, role: str) -> None:
    if overrides is None:
        return
    override = (
        overrides.get(role)
        if isinstance(overrides, Mapping)
        else getattr(overrides, role, None)
    )
    validate = getattr(override, "validate", None)
    if callable(validate):
        validate()
        return
    for name in ("model_env", "api_base_env", "api_key_env"):
        value = override.get(name) if isinstance(override, Mapping) else None
        if value and not str(value).isidentifier():
            raise ValueError(f"invalid environment variable name: {value}")


def _env_or(name: str | None) -> str:
    if not name:
        return ""
    return os.getenv(name, "").strip()


def resolve_model_settings(
    overrides: LLMConfigOverrides | None, role: str
) -> dict[str, Any]:
    """Resolve one model role (``completion``/``embedding``) into a KAG block.

    Values merge, in decreasing precedence: explicit overrides, the shared
    ``benchmark/.env`` environment, then KAG-safe defaults.  The returned block
    carries both the resolved ``api_key`` (for in-memory consumption) and the
    ``api_key_env`` name (for ``!ENV`` emission into the YAML file).
    """
    if role not in {"completion", "embedding"}:
        raise ValueError(f"unsupported model role: {role}")
    _validate_role_override(overrides, role)
    environment = load_environment()
    model = (
        _override_value(overrides, role, "model")
        or _env_or(_override_value(overrides, role, "model_env"))
        or (
            environment.completion_model
            if role == "completion"
            else environment.embedding_model
        )
    )
    api_base = (
        _override_value(overrides, role, "api_base")
        or _env_or(_override_value(overrides, role, "api_base_env"))
        or environment.api_base
    )
    provider = (
        _override_value(overrides, role, "model_provider")
        or _override_value(overrides, role, "type")
        or environment.provider
        or "openai"
    ).casefold()
    if provider in {"azure", "azure-openai", "azure_openai"}:
        provider = "azure_openai"
    if role == "embedding" and provider in {"maas", "vllm"}:
        # KAG registers its OpenAI-compatible chat client under these aliases,
        # but its vector-model registry only exposes ``openai``.  VLLM and
        # MaaS endpoints use the same embeddings protocol, so retain the
        # caller's endpoint/model while selecting the compatible vector class.
        provider = "openai"
    api_key_env = _override_value(overrides, role, "api_key_env") or DEFAULT_API_KEY_ENV
    call_args = _override_value(overrides, role, "call_args")
    if call_args is not None and not isinstance(call_args, Mapping):
        raise TypeError(f"{role}.call_args must be a mapping")
    block: dict[str, Any] = {
        "type": provider,
        "base_url": api_base,
        "model": model,
        "api_key": os.getenv(api_key_env, ""),
        "api_key_env": api_key_env,
        "enable_check": False,
    }
    if call_args:
        # Provider-specific options (timeout, temperature, Azure AD settings,
        # etc.) are forwarded unchanged.  Explicit top-level override fields
        # below still win for their corresponding canonical names.
        block.update(dict(call_args))
    block.update(
        {
            "type": provider,
            "base_url": api_base,
            "model": model,
            "api_key": os.getenv(api_key_env, ""),
            "api_key_env": api_key_env,
            "enable_check": False,
        }
    )
    # Azure clients have a provider-default version, but a shared benchmark
    # endpoint frequently requires a specific version.  Honour a role override
    # first, then the common RAG_API_VERSION setting.  Passing it only for
    # Azure avoids adding an unsupported option to generic OpenAI clients.
    api_version = _override_value(overrides, role, "api_version")
    if provider == "azure_openai":
        api_version = api_version or environment.api_version
    deployment = _override_value(overrides, role, "azure_deployment_name")
    if api_version:
        block["api_version"] = api_version
    if deployment:
        block["azure_deployment"] = deployment
    if role == "completion":
        # KAG registers the OpenAI-compatible chat client under ``maas``;
        # ``ollama`` is registered under its own name.
        if provider == "openai":
            block["type"] = "maas"
    else:
        # Vector models are registered under ``openai``/``ollama`` directly.
        block["vector_dimensions"] = environment.embedding_dimension
    return block


def build_kag_config(
    *,
    namespace: str,
    language: str,
    checkpoint_path: Path,
    graph_path: Path,
    llm: dict[str, Any],
    vectorize_model: dict[str, Any],
    top_k: int,
    split_length: int,
    max_iteration: int,
    num_chains: int,
    num_threads_per_chain: int,
    log_level: str = "INFO",
) -> dict[str, Any]:
    """Assemble the full KAG configuration for local builder + solver runs."""
    # Keep role-specific environment-variable metadata until serialization.
    # ``write_config_file`` consumes it while replacing the actual key with a
    # ``!ENV`` tag, then removes the metadata before KAG parses the YAML.
    llm = dict(llm)
    vectorize_model = dict(vectorize_model)

    search_api = {
        "type": "memory_search_api",
        "graph_path": str(graph_path),
        "vectorize_model": dict(vectorize_model),
    }
    graph_api = {
        "type": "memory_graph_api",
        "graph_path": str(graph_path),
        "vectorize_model": dict(vectorize_model),
    }

    def _llm() -> dict[str, Any]:
        return dict(llm)

    def _vec() -> dict[str, Any]:
        return dict(vectorize_model)

    def _api() -> dict[str, Any]:
        return dict(search_api)

    def _graph() -> dict[str, Any]:
        return dict(graph_api)

    def _exclude_types() -> list[str]:
        return [
            "Chunk",
            "AtomicQuery",
            "KnowledgeUnit",
            "Summary",
            "Outline",
            "Doc",
        ]

    # Several KAG graph retrievers instantiate ``DefaultStdSchema`` when no
    # explicit value is supplied.  Its default is an OpenSPG HTTP search
    # client, which quietly defeats this facade's local-memory mode.  Keep
    # every standardization path on the same in-process graph instead.
    def _std_schema() -> dict[str, Any]:
        return {
            "type": "default_std_schema",
            "search_api": _api(),
            "vectorize_model": _vec(),
        }

    kg_cs = {
        "type": "kg_cs_open_spg",
        "priority": 0,
        "llm": _llm(),
        "std_schema": _std_schema(),
        "path_select": {
            "type": "exact_one_hop_select",
            "vectorize_model": _vec(),
            "graph_api": _graph(),
            "search_api": _api(),
        },
        "entity_linking": {
            "type": "entity_linking",
            "vectorize_model": _vec(),
            "graph_api": _graph(),
            "search_api": _api(),
            "recognition_threshold": 0.9,
            "exclude_types": _exclude_types(),
        },
    }
    kg_fr = {
        "type": "kg_fr_open_spg",
        "top_k": top_k,
        "llm": _llm(),
        "std_schema": _std_schema(),
        "graph_api": _graph(),
        "search_api": _api(),
        "vectorize_model": _vec(),
        "path_select": {
            "type": "fuzzy_one_hop_select",
            "llm_client": _llm(),
            "vectorize_model": _vec(),
            "graph_api": _graph(),
            "search_api": _api(),
        },
        "ppr_chunk_retriever_tool": {
            "type": "ppr_chunk_retriever",
            "llm_client": _llm(),
            "vectorize_model": _vec(),
            "graph_api": _graph(),
            "search_api": _api(),
        },
        "entity_linking": {
            "type": "entity_linking",
            "vectorize_model": _vec(),
            "graph_api": _graph(),
            "search_api": _api(),
            "recognition_threshold": 0.8,
            "exclude_types": _exclude_types(),
        },
    }
    # See the nested vector_chunk_retriever for the local-mode score-scale
    # rationale behind this threshold.
    rc = {
        "type": "rc_open_spg",
        "vector_chunk_retriever": {
            "type": "vector_chunk_retriever",
            "vectorize_model": _vec(),
            # Memory-mode search returns raw dot-product scores, whose scale
            # depends on the embedding model: text-embedding-3-large ranks
            # relevant guideline chunks at ~0.48-0.67, so KAG's upstream
            # demo value 0.65 (tuned for BGE) filters out every chunk.
            "score_threshold": 0.35,
            "search_api": _api(),
            # RCRetriever delegates directly to this nested retriever, so
            # the value must be present here as well as on ``rc``.
            "top_k": top_k,
        },
        "score_threshold": 0.35,
        "graph_api": _graph(),
        "search_api": _api(),
        "vectorize_model": _vec(),
        "top_k": top_k,
    }

    kg_merger = {
        # ``kag_merger`` is registered as a RetrieverOutputMerger.  The
        # similarly named ``kg_merger`` is a FlowComponent and cannot be
        # constructed in KAGHybridRetrievalExecutor's ``merger`` slot.
        "type": "kag_merger",
        "top_k": top_k,
        "llm_module": _llm(),
        "summary_prompt": {"type": "default_thought_then_answer"},
        "vectorize_model": _vec(),
    }
    kag_hybrid_executor = {
        "type": "kag_hybrid_retrieval_executor",
        "retrievers": [dict(kg_cs), dict(kg_fr), dict(rc)],
        "merger": dict(kg_merger),
        "llm_module": _llm(),
        "context_select_llm": _llm(),
        "with_llm_select": False,
        "enable_summary": True,
    }
    kag_output_executor = {"type": "kag_output_executor", "llm_module": _llm()}
    kag_deduce_executor = {"type": "kag_deduce_executor", "llm_module": _llm()}
    py_code_based_math_executor = {
        "type": "py_code_based_math_executor",
        "llm": _llm(),
    }

    return {
        "log": {"level": log_level},
        "project": {
            "biz_scene": DEFAULT_BIZ_SCENE,
            "host_addr": DEFAULT_HOST_ADDR,
            "id": DEFAULT_PROJECT_ID,
            "language": language,
            "namespace": namespace,
            "checkpoint_path": str(checkpoint_path),
        },
        "llm": _llm(),
        "ner_llm": _llm(),
        "vectorize_model": _vec(),
        "vectorizer": _vec(),
        "search_api": search_api,
        "graph_api": graph_api,
        "kg_cs": kg_cs,
        "kg_fr": kg_fr,
        "rc": rc,
        "kag_hybrid_executor": kag_hybrid_executor,
        "kag_output_executor": kag_output_executor,
        "kag_deduce_executor": kag_deduce_executor,
        "py_code_based_math_executor": py_code_based_math_executor,
        "kag_builder_pipeline": {
            "chain": {
                "type": "unstructured_builder_chain",
                "extractor": {
                    "type": "schema_free_extractor",
                    "llm": _llm(),
                    "ner_prompt": {"type": "default_ner"},
                    "std_prompt": {"type": "default_std"},
                    "triple_prompt": {"type": "default_triple"},
                },
                "reader": {"type": "txt_reader"},
                "post_processor": {"type": "kag_post_processor"},
                "splitter": {
                    "type": "length_splitter",
                    "split_length": split_length,
                    "window_length": 0,
                },
                "vectorizer": {
                    "type": "batch_vectorizer",
                    "vectorize_model": _vec(),
                },
                "writer": {"type": "memory_graph_writer"},
            },
            "num_threads_per_chain": num_threads_per_chain,
            "num_chains": num_chains,
        },
        "kag_solver_pipeline": {
            "type": "kag_static_pipeline",
            "max_iteration": max_iteration,
            "planner": {
                "type": "lf_kag_static_planner",
                "llm": _llm(),
                "plan_prompt": {
                    "type": "default_lf_static_planning",
                    "std_schema": _std_schema(),
                },
                "rewrite_prompt": {"type": "default_rewrite_sub_task_query"},
            },
            "executors": [
                dict(kag_hybrid_executor),
                dict(py_code_based_math_executor),
                dict(kag_deduce_executor),
                dict(kag_output_executor),
            ],
            "generator": {
                "type": "llm_index_generator",
                "llm_client": _llm(),
                "generated_prompt": {"type": "default_refer_generator_prompt"},
                # Without a reranker the generator concatenates per-step
                # chunks in step order (duplicates, no relevance ranking);
                # the benchmark then truncates that unranked list at k.
                "chunk_reranker": {
                    "type": "rerank_by_vector",
                    "vectorize_model": _vec(),
                },
                "enable_ref": True,
            },
        },
        "naive_rag_solver_pipeline": {
            "type": "naive_rag_pipeline",
            "executors": [
                {
                    "type": "kag_hybrid_retrieval_executor",
                    "retrievers": [dict(rc)],
                    "merger": dict(kg_merger),
                    "llm_module": _llm(),
                    "context_select_llm": _llm(),
                    "with_llm_select": False,
                    "enable_summary": False,
                }
            ],
            "generator": {
                "type": "llm_index_generator",
                "llm_client": _llm(),
                "generated_prompt": {"type": "default_refer_generator_prompt"},
                "chunk_reranker": {
                    "type": "rerank_by_vector",
                    "vectorize_model": _vec(),
                },
                "enable_ref": True,
            },
        },
    }


def substitute_instances(
    node: Any, *, llm: Any = None, vectorize_model: Any = None
) -> Any:
    """Replace config blocks with already-instantiated KAG objects.

    ``Registrable.from_config`` passes through values that already implement
    the annotated ABC, so injected LLM/vectorizer instances (used for
    deterministic tests) can replace the YAML-derived config dicts in place.
    """

    if isinstance(node, dict):
        replaced: dict[str, Any] = {}
        for key, value in node.items():
            if llm is not None and key in _LLM_KEYS and isinstance(value, dict):
                replaced[key] = llm
            elif (
                vectorize_model is not None
                and key in _VECTORIZE_KEYS
                and isinstance(value, dict)
            ):
                replaced[key] = vectorize_model
            else:
                replaced[key] = substitute_instances(
                    value, llm=llm, vectorize_model=vectorize_model
                )
        return replaced
    if isinstance(node, list):
        return [
            substitute_instances(item, llm=llm, vectorize_model=vectorize_model)
            for item in node
        ]
    return node


def write_config_file(
    config: dict[str, Any], path: Path, *, api_key_env: str = DEFAULT_API_KEY_ENV
) -> Path:
    """Serialize the configuration, keeping the API key off disk.

    The key is written as a ``!ENV <name>`` tag that KAG resolves from the
    process environment at load time, mirroring how the Microsoft baseline's
    ``settings.yaml`` references ``${RAG_API_KEY}``.
    """

    def _sensitive_key(key: Any) -> bool:
        """Whether a mapping key is likely to carry a credential value."""

        normalized = (
            re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(key).strip())
            .casefold()
            .replace("-", "_")
        )
        if normalized in _SENSITIVE_KEY_PARTS:
            return True
        return any(
            normalized.startswith(f"{part}_") or normalized.endswith(f"_{part}")
            for part in _SENSITIVE_KEY_PARTS
        )

    def _redacted_value(node: Mapping[str, Any], key: str, value: Any) -> Any:
        """Use an explicit sibling ``*_env`` reference when available.

        KAG reloads this generated YAML from disk, so an arbitrary literal
        credential in ``call_args`` cannot safely be preserved there.  Users
        can supply e.g. ``azure_ad_token_env`` alongside
        ``azure_ad_token``; otherwise leave an unmistakable non-secret marker
        instead of leaking the literal to the project directory.
        """

        env_name = node.get(f"{key}_env")
        if isinstance(env_name, str) and env_name:
            return _EnvRef(env_name)
        if value is None or isinstance(value, (bool, int, float)):
            return value
        return "<redacted>"

    def _hide_secrets(node: Any, inherited_api_key_env: str) -> Any:
        if isinstance(node, dict):
            local_api_key_env = str(node.get("api_key_env") or inherited_api_key_env)
            hidden: dict[str, Any] = {}
            for key, value in node.items():
                if key == "api_key_env":
                    continue
                if key == "api_key":
                    hidden[key] = _EnvRef(local_api_key_env)
                elif _sensitive_key(key):
                    hidden[key] = _redacted_value(node, str(key), value)
                else:
                    hidden[key] = _hide_secrets(value, local_api_key_env)
            return hidden
        if isinstance(node, list):
            return [_hide_secrets(item, inherited_api_key_env) for item in node]
        return node

    text = yaml.dump(
        _hide_secrets(config, api_key_env),
        Dumper=_EnvRefDumper,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _seed_offline_schema_cache(project_id: Any, namespace: str) -> None:
    """Seed knext's caches with the minimal schema required in local mode.

    KAG normally fetches schema metadata from an OpenSPG service.  Apart from
    causing unwanted network access, an *empty* substitute session leaves
    ``BatchVectorizer`` with no vector fields, so chunks are stored without
    embeddings and cannot be retrieved from ``MemoryGraph``.  The local
    facade needs only a compact schema: chunk title/content vectors plus a
    few generic entity types used by schema-free extraction and logic-form
    standardization.

    The caches are process-wide and KAG reuses the default project id.  They
    must therefore be replaced on each configuration load, rather than only
    seeded once, so sequential clients with different namespaces continue to
    read their own memory graph.
    """

    from knext.schema.client import SchemaSession
    from knext.schema.client import cache as schema_cache
    from knext.reasoner.client import reason_cache
    from knext.schema.model.base import IndexTypeEnum
    from knext.schema.model.property import Property
    from knext.schema.model.spg_type import EntityType

    def _type(
        name: str,
        properties: list[Property] | None = None,
    ) -> EntityType:
        return EntityType(
            name=f"{namespace}.{name}",
            name_zh=name,
            properties=properties or [],
        )

    def _make_session(pid: Any) -> Any:
        session = SchemaSession.__new__(SchemaSession)
        session._alter_spg_types = []
        session._rest_client = None
        session._project_id = pid
        # ``SchemaClient.load`` removes the namespace from mapping keys while
        # ``SchemaUtils`` retains the full type name.  This is exactly the
        # pairing expected by the vectorizer and memory search implementations.
        chunk = _type(
            "Chunk",
            [
                Property("name", "Text", index_type=IndexTypeEnum.Vector),
                Property(
                    "content",
                    "Text",
                    index_type=IndexTypeEnum.TextAndVector,
                ),
            ],
        )
        entity = _type(
            "Entity",
            [Property("name", "Text", index_type=IndexTypeEnum.Vector)],
        )
        others = _type(
            "Others",
            [Property("name", "Text", index_type=IndexTypeEnum.Vector)],
        )
        semantic_concept = _type(
            "SemanticConcept",
            [Property("name", "Text", index_type=IndexTypeEnum.Vector)],
        )
        types = (chunk, entity, others, semantic_concept)
        session._spg_types = {item.name: item for item in types}
        setattr(session, "_SchemaSession__spg_types", {})
        return session

    if project_id is None:
        return
    with _OFFLINE_SCHEMA_CACHE_LOCK:
        # knext's SchemaCache uses a five-minute TTL by default.  That is
        # sensible for an OpenSPG service, but a long local build may create a
        # later chain after the entry expires; SchemaClient would then silently
        # fall back to an HTTP schema request.  This facade owns the KAG
        # process-global runtime, so replace just these two local caches with
        # bounded non-expiring LRU caches and reseed them on every activation.
        # The active project's id/namespace is deliberately overwritten when
        # switching clients under the runtime lock.
        for local_cache in (schema_cache, reason_cache):
            if isinstance(local_cache.cache, LRUCache):
                continue
            capacity = getattr(local_cache.cache, "maxsize", 10)
            local_cache._cache = LRUCache(maxsize=capacity)
        schema_cache.put(project_id, _make_session(project_id))
        reason_id = str(project_id)
        reason_cache.put(reason_id, _make_session(reason_id))


def initialize_kag_config(config_file: Path) -> None:
    """Point KAG's global configuration at the generated file.

    Also seeds the offline schema cache so subsequent component
    construction never contacts the configured (absent) schema server.
    """
    from kag.common.conf import KAG_CONFIG

    KAG_CONFIG.initialize(prod=False, config_file=str(config_file))
    _seed_offline_schema_cache(
        KAG_CONFIG.global_config.project_id,
        KAG_CONFIG.global_config.namespace,
    )


__all__ = [
    "DEFAULT_API_KEY_ENV",
    "build_kag_config",
    "initialize_kag_config",
    "resolve_model_settings",
    "substitute_instances",
    "write_config_file",
]
