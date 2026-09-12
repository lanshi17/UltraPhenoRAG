"""Offline KAG client smoke tests using deterministic stub models."""

from __future__ import annotations

import json
import random
import warnings
from pathlib import Path

import pytest

from kag.interface import LLMClient, VectorizeModelABC

from benchmark.baseline.kag_client import KAGClient, SearchMethod
from benchmark.baseline.kag_client.config.kag_project import (
    resolve_model_settings,
    write_config_file,
)
from benchmark.baseline.kag_client.config.llm_config import (
    LLMConfigOverrides,
    ModelConfigOverride,
)
from benchmark.baseline.kag_client.cost_analysis import (
    OperationUsageTracker,
    TokenRates,
)
from benchmark.baseline.kag_client.documents import load_documents
from benchmark.baseline.kag_client.search_methods import resolve_search_method

ANSWER = "The nuchal translucency is measured between 11 and 14 weeks of gestation."

GUIDE_TEXT = (
    "Nuchal translucency (NT) is the sonographic appearance of a collection "
    "of fluid under the skin behind the fetal neck in the first-trimester "
    "fetus. The NT measurement is performed between 11 and 14 weeks of "
    "gestation, when the crown-rump length ranges from 45 to 84 mm. An "
    "increased NT is associated with chromosomal abnormalities."
)

_PLAN = (
    "Step1:When is the nuchal translucency measured?\n"
    "Action1:Retrieval(s=s1:Measurement[`nuchal translucency`],"
    "p=p1:measured_at,o=o1:TimeWindow)\n"
    "\n"
    "Step2:Output the measurement window\n"
    "Action2:output(o1)"
)

_ENTITIES = [
    {
        "name": "nuchal translucency",
        "type": "Measurement",
        "category": "Medical",
        "description": "First-trimester ultrasound marker measured at 11-14 weeks.",
    }
]

_TRIPLES = [["nuchal translucency", "measured at", "11 to 14 weeks of gestation"]]


class StubLLMClient(LLMClient):
    """Deterministic LLM covering every prompt family the pipelines issue."""

    def __init__(self, **kwargs):
        super().__init__(name="kag-stub-llm")
        self.model = "stub-model"
        self.prompts: list[str] = []

    def __deepcopy__(self, memo):
        return self

    def to_config(self):
        # Required by SchemaFreeExtractor when building its table extractor;
        # the table extractor is never invoked on plain text chunks.
        return {"type": "mock"}

    def __call__(self, prompt, **kwargs):
        return self._respond(prompt)

    async def acall(self, prompt, **kwargs):
        return self._respond(prompt)

    def _respond(self, prompt) -> str:
        text = prompt if isinstance(prompt, str) else str(prompt)
        self.prompts.append(text)
        self._account(text)
        if "entity extraction system" in text:
            return json.dumps(_ENTITIES)
        if "official names of" in text:
            return json.dumps(_ENTITIES)
        if "open information extraction" in text:
            return json.dumps({"triples": _TRIPLES})
        if "planning expert" in text:
            return _PLAN
        if "information analysis expert" in text:
            return ANSWER
        return f"Thought: the retrieved context covers the NT window. Answer: {ANSWER}"

    @staticmethod
    def _account(prompt) -> None:
        prompt_tokens = max(1, len(str(prompt)) // 4)
        completion_tokens = 12
        LLMClient.get_token_meter().update(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )


class StubVectorizeModel(VectorizeModelABC):
    """Deterministic hash-seeded embedding model with a fixed dimension."""

    def __init__(self, vector_dimensions: int = 16, **kwargs):
        super().__init__(
            name="kag-stub-vectorizer", vector_dimensions=vector_dimensions
        )
        self.model = "stub-vectorizer"

    def __deepcopy__(self, memo):
        return self

    def to_config(self):
        return {"type": "mock"}

    def _one(self, text: str) -> list[float]:
        seed = sum(ord(char) for char in text[:512])
        generator = random.Random(seed)
        return [generator.random() for _ in range(16)]

    def vectorize(self, texts):
        if isinstance(texts, str):
            return self._one(texts)
        return [self._one(text) for text in texts]

    async def avectorize(self, texts):
        return self.vectorize(texts)


def _make_client(tmp_path: Path, **kwargs) -> KAGClient:
    defaults = {
        "llm": StubLLMClient(),
        "vectorize_model": StubVectorizeModel(),
        "split_length": 800,
        "max_iteration": 1,
        "num_chains": 2,
        "num_threads_per_chain": 1,
    }
    defaults.update(kwargs)
    input_dir = tmp_path / "input"
    input_dir.mkdir(exist_ok=True)
    (input_dir / "guide.txt").write_text(GUIDE_TEXT, encoding="utf-8")
    return KAGClient(tmp_path, **defaults)


def test_search_method_resolution() -> None:
    assert resolve_search_method("solver") == ("solver", "solver")
    assert resolve_search_method("naive") == ("naive", "naive")
    assert resolve_search_method("basic") == ("naive", "basic")
    assert resolve_search_method(SearchMethod.SOLVER) == ("solver", "solver")

    try:
        resolve_search_method("drift")
    except ValueError as error:
        assert "unsupported search method" in str(error)
    else:
        raise AssertionError("expected ValueError for unknown method")


def test_index_and_all_query_modes(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    try:
        index_result = client.index(cache=False)
        assert not index_result.has_errors, index_result.errors
        assert len(index_result.outputs) == 1
        assert index_result.outputs[0]["id"] == "guide"
        assert client.graph_path.is_dir()
        from benchmark.baseline.kag_client.benchmark import _local_index_status

        assert _local_index_status(client.root_dir)["has_chunk_vectors"] is True

        naive = client.naive_search("When is the nuchal translucency measured?")
        assert naive.method == "naive"
        assert ANSWER[:30] in naive.response
        assert naive.query == "When is the nuchal translucency measured?"

        solver = client.solver_search("When is the nuchal translucency measured?")
        assert solver.method == "solver"
        assert solver.response

        basic = client.basic_search("When is the nuchal translucency measured?")
        assert basic.method == "basic"

        generic = client.query("When is the nuchal translucency measured?", "naive")
        assert generic.response
    finally:
        client.close()


def test_index_dry_run_and_input_documents(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    try:
        dry = client.index(dry_run=True)
        assert not dry.has_errors
        assert len(dry.outputs) == 1
        assert dry.outputs[0].document_id == "guide"
        # dry run must not build any checkpoint storage
        assert not (client.data_dir / "ckpt").exists()

        staged = client.index(
            cache=False,
            input_documents=[{"id": "nt-note", "text": "NT scan at 12 weeks."}],
        )
        assert not staged.has_errors, staged.errors
        assert len(staged.outputs) == 1
        assert staged.outputs[0]["id"] == "nt-note"
        staged_file = client.data_dir / "input" / "nt-note.txt"
        assert staged_file.is_file()
        assert staged_file.read_text(encoding="utf-8") == "NT scan at 12 weeks."

        result = client.naive_search("What happens at 12 weeks?")
        assert result.response
    finally:
        client.close()


def test_query_records_llm_usage(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    try:
        client.index(cache=False)
        result = client.naive_search("When is the nuchal translucency measured?")

        usage = result.telemetry["usage"]
        assert result.telemetry["elapsed_seconds"] >= 0
        assert usage["total_tokens"] > 0
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0
        assert usage["cost_available"] is False
        assert usage["total_cost_usd"] is None
    finally:
        client.close()


def test_local_solver_retrieves_memory_chunks_without_openspg(
    tmp_path: Path, monkeypatch
) -> None:
    """The local facade must never silently fall back to OpenSPG HTTP APIs."""

    from kag.common.tools.search_api.impl.openspg_search_api import OpenSPGSearchAPI

    calls: list[str] = []

    def _unexpected_remote_search(self, *args, **kwargs):
        del self, args, kwargs
        calls.append("openspg")
        raise AssertionError("local KAG client attempted an OpenSPG search")

    monkeypatch.setattr(OpenSPGSearchAPI, "search_vector", _unexpected_remote_search)
    monkeypatch.setattr(OpenSPGSearchAPI, "search_text", _unexpected_remote_search)

    client = _make_client(tmp_path)
    try:
        index_result = client.index(cache=False)
        assert not index_result.has_errors, index_result.errors

        result = client.solver_search("When is the nuchal translucency measured?")
        assert result.response
        assert result.references
        assert "11 and 14 weeks" in result.references[0]["content"]
        assert calls == []
    finally:
        client.close()


def test_index_rejects_unknown_method(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    try:
        try:
            client.index(method="turbo")
        except ValueError as error:
            assert "unsupported index method" in str(error)
        else:
            raise AssertionError("expected ValueError for unknown index method")
    finally:
        client.close()


def test_search_rejects_empty_query(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    try:
        try:
            client.search("   ")
        except ValueError as error:
            assert "query must be a non-empty string" in str(error)
        else:
            raise AssertionError("expected ValueError for empty query")
    finally:
        client.close()


def test_cost_tracker_calculates_token_cost_and_latency() -> None:
    tracker = OperationUsageTracker(
        "test-model", TokenRates(input_per_million_usd=2, output_per_million_usd=4)
    )
    tracker.start_request()
    tracker.add_usage({"prompt_tokens": 1_000_000, "completion_tokens": 500_000})

    telemetry = tracker.telemetry()

    assert telemetry["elapsed_seconds"] >= 0
    assert telemetry["usage"]["request_count"] == 1
    assert telemetry["usage"]["total_tokens"] == 1_500_000
    assert telemetry["usage"]["input_cost_usd"] == 2
    assert telemetry["usage"]["output_cost_usd"] == 2
    assert telemetry["usage"]["total_cost_usd"] == 4
    assert telemetry["usage"]["cost_available"] is True
    assert telemetry["usage"]["test-model"]["total_tokens"] == 1_500_000


def test_client_context_manager_and_close(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RAG_API_KEY", "sk-context-secret")
    with _make_client(tmp_path) as client:
        assert client.workspace
        assert client.config["namespace"] == client.workspace
        config_text = (client.data_dir / "kag_config.yaml").read_text(encoding="utf-8")
        # The variable name is deliberately stored as ``!ENV``; only the
        # resolved credential must stay out of the generated project file.
        assert "sk-context-secret" not in config_text
        assert "api_key: !ENV RAG_API_KEY" in config_text
        assert (client.data_dir / "kag_config.yaml").is_file()
    assert client._runner is None


def test_generated_config_hides_api_key(tmp_path: Path) -> None:
    from benchmark.baseline.kag_client.config.kag_project import (
        build_kag_config,
        write_config_file,
    )

    config = build_kag_config(
        namespace="secret-check",
        language="en",
        checkpoint_path=tmp_path / "ckpt",
        graph_path=tmp_path / "ckpt" / "MemoryGraphWriter",
        llm={
            "type": "maas",
            "model": "m",
            "api_key": "sk-plain",
            "base_url": "http://x",
            "api_key_env": "RAG_API_KEY",
        },
        vectorize_model={
            "type": "openai",
            "model": "e",
            "api_key": "sk-plain",
            "vector_dimensions": 16,
        },
        top_k=20,
        split_length=4000,
        max_iteration=1,
        num_chains=4,
        num_threads_per_chain=1,
    )
    path = tmp_path / "kag_config.yaml"
    write_config_file(config, path, api_key_env="RAG_API_KEY")
    text = path.read_text(encoding="utf-8")
    assert "sk-plain" not in text
    assert "api_key: !ENV RAG_API_KEY" in text


def test_config_uses_azure_environment_api_version(monkeypatch) -> None:
    monkeypatch.setenv("RAG_MODEL_PROVIDER", "azure")
    monkeypatch.setenv("RAG_API_VERSION", "2025-01-01-preview")

    settings = resolve_model_settings(None, "completion")

    assert settings["type"] == "azure_openai"
    assert settings["api_version"] == "2025-01-01-preview"


def test_offline_schema_cache_is_bounded_but_non_expiring(
    tmp_path: Path, monkeypatch
) -> None:
    """Long local builds must not fall back to the absent OpenSPG schema API."""

    from cachetools import LRUCache, TTLCache
    from knext.reasoner.client import reason_cache
    from knext.schema.client import cache as schema_cache

    # Model the vendor's default five-minute TTL cache.  KAGClient
    # initialization must replace it with the local facade's bounded LRU cache
    # before components can invoke SchemaClient.load().
    monkeypatch.setattr(schema_cache, "_cache", TTLCache(maxsize=10, ttl=0))
    monkeypatch.setattr(reason_cache, "_cache", TTLCache(maxsize=10, ttl=0))
    client = _make_client(tmp_path)
    try:
        assert isinstance(schema_cache.cache, LRUCache)
        assert isinstance(reason_cache.cache, LRUCache)
        assert schema_cache.get("1") is not None
        assert reason_cache.get("1") is not None
    finally:
        client.close()


def test_config_hides_separate_role_keys_and_credential_like_call_args(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("COMPLETION_TEST_KEY", "completion-actual-secret")
    monkeypatch.setenv("EMBEDDING_TEST_KEY", "embedding-actual-secret")
    overrides = LLMConfigOverrides(
        completion=ModelConfigOverride(api_key_env="COMPLETION_TEST_KEY"),
        embedding=ModelConfigOverride(api_key_env="EMBEDDING_TEST_KEY"),
    )
    client = _make_client(tmp_path, llm_overrides=overrides)
    try:
        generated = (client.data_dir / "kag_config.yaml").read_text(encoding="utf-8")
        assert "completion-actual-secret" not in generated
        assert "embedding-actual-secret" not in generated
        assert "api_key: !ENV COMPLETION_TEST_KEY" in generated
        assert "api_key: !ENV EMBEDDING_TEST_KEY" in generated
        index_state = json.dumps(
            client._index_state_payload(load_documents(client.root_dir))
        )
        assert "completion-actual-secret" not in index_state
        assert "embedding-actual-secret" not in index_state
    finally:
        client.close()

    config_path = tmp_path / "credential-call-args.yaml"
    write_config_file(
        {
            "llm": {
                "api_key": "api-key-secret",
                "api_key_env": "COMPLETION_TEST_KEY",
                "headers": {"Authorization": "Bearer header-secret"},
                "azure_ad_token": "ad-token-secret",
                "azure_ad_token_env": "AZURE_AD_TOKEN",
                "accessToken": "camel-case-secret",
            }
        },
        config_path,
    )
    serialized = config_path.read_text(encoding="utf-8")
    assert "api-key-secret" not in serialized
    assert "header-secret" not in serialized
    assert "ad-token-secret" not in serialized
    assert "camel-case-secret" not in serialized
    assert "Authorization: <redacted>" in serialized
    assert "azure_ad_token: !ENV AZURE_AD_TOKEN" in serialized


def test_documents_accept_strings_bytes_and_root_markdown_fallback(
    tmp_path: Path,
) -> None:
    root = tmp_path / "documents"
    root.mkdir()
    # An empty explicit input directory must not hide loose root-level docs.
    (root / "input").mkdir()
    (root / "legacy.md").write_text("# Legacy\n\nNT scan guidance.", encoding="utf-8")

    file_records = load_documents(root)
    assert [(record.document_id, record.text) for record in file_records] == [
        ("legacy", "# Legacy\n\nNT scan guidance.")
    ]
    assert load_documents(root, "single text")[0].text == "single text"
    assert load_documents(root, b"single bytes")[0].text == "single bytes"
    assert (
        load_documents(root, {"id": "bytes", "content": b"decoded bytes"})[0].text
        == "decoded bytes"
    )


def test_explicit_input_dir_overrides_project_and_empty_corpus_clears_index(
    tmp_path: Path,
) -> None:
    """A caller-selected corpus must be indexed and an empty rerun is not stale."""

    root = tmp_path / "project"
    project_input = root / "input"
    external_input = tmp_path / "alternate-corpus" / "input"
    project_input.mkdir(parents=True)
    external_input.mkdir(parents=True)
    (project_input / "wrong.txt").write_text(
        "PROJECT-ONLY document that must not be indexed.", encoding="utf-8"
    )
    selected = external_input / "selected.txt"
    selected.write_text(
        "EXTERNAL-ONLY evidence: NT is measured between 11 and 14 weeks.",
        encoding="utf-8",
    )
    client = KAGClient(
        root,
        input_dir=external_input,
        llm=StubLLMClient(),
        vectorize_model=StubVectorizeModel(),
        split_length=800,
        max_iteration=1,
        num_chains=1,
        num_threads_per_chain=1,
    )
    try:
        indexed = client.index(cache=False)
        assert not indexed.has_errors, indexed.errors
        assert [item["id"] for item in indexed.outputs] == ["selected"]
        assert client.config["input_dir"] == str(external_input.resolve())

        # With no source documents left, cache=True must clear the old graph
        # rather than treating its durable checkpoint as the new empty index.
        selected.unlink()
        emptied = client.index(cache=True)
        assert not emptied.has_errors
        assert emptied.outputs == []
        assert not client.graph_path.exists()
        state = json.loads(
            (client.data_dir / "kag_index_state.json").read_text(encoding="utf-8")
        )
        assert state["inputs"]["documents"] == []
    finally:
        client.close()


def test_async_limiters_are_not_reused_across_client_event_loops(
    tmp_path: Path,
) -> None:
    """KAG's process-global limiter must follow the active client's loop."""

    first_root = tmp_path / "first"
    first_root.mkdir()
    first = _make_client(first_root)
    try:
        assert not first.index(cache=False).has_errors
        first.solver_search("When is the nuchal translucency measured?")
        first_limiter = first._llm_instance.limiter
        assert getattr(first_limiter, "_event_loop") is first._runner.loop
    finally:
        first.close()

    second_root = tmp_path / "second"
    second_root.mkdir()
    second = _make_client(second_root)
    try:
        assert not second.index(cache=False).has_errors
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            second.solver_search("When is the nuchal translucency measured?")
        second_limiter = second._llm_instance.limiter
        assert second_limiter is not first_limiter
        assert getattr(second_limiter, "_event_loop") is second._runner.loop
    finally:
        second.close()


def test_client_validates_workspace_top_k_and_closed_lifecycle(tmp_path: Path) -> None:
    for invalid_workspace in ("", "   ", 7):
        with pytest.raises(ValueError, match="workspace must be a non-empty string"):
            KAGClient(
                tmp_path / f"workspace-{type(invalid_workspace).__name__}",
                llm=StubLLMClient(),
                vectorize_model=StubVectorizeModel(),
                workspace=invalid_workspace,
            )
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        KAGClient(
            tmp_path / "bad-top-k",
            llm=StubLLMClient(),
            vectorize_model=StubVectorizeModel(),
            top_k=0,
        )

    closed_root = tmp_path / "closed-client"
    closed_root.mkdir()
    client = _make_client(closed_root)
    try:
        assert client.config["num_threads_per_chain"] == 1
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            client.search("valid query", top_k=0)
    finally:
        client.close()

    for operation in (
        lambda: client.index(),
        lambda: client.search("valid query"),
        client.reload_config,
    ):
        with pytest.raises(RuntimeError, match="KAGClient is closed"):
            operation()


def test_operation_reactivates_its_own_config_before_default_llm_construction(
    tmp_path: Path, monkeypatch
) -> None:
    """A later client construction must not bind A's default LLM to B's config."""

    import benchmark.baseline.kag_client.client as client_module

    captured_namespaces: list[str] = []

    class ConfigCapturingLLM(LLMClient):
        def __init__(self) -> None:
            super().__init__(name="config-capturing-llm")
            self.model = "config-capturing-model"
            captured_namespaces.append(self.kag_project_config.namespace)

        def __call__(self, prompt, **kwargs):
            del prompt, kwargs
            return "unused"

        async def acall(self, prompt, **kwargs):
            return self(prompt, **kwargs)

        def to_config(self):
            return {"type": "mock"}

    class FakeBuilderChain:
        def invoke(self, file_path, **kwargs):
            del file_path, kwargs
            return []

    def fake_llm_from_config(cls, config):
        del cls, config
        return ConfigCapturingLLM()

    monkeypatch.setattr(
        client_module.LLMClient,
        "from_config",
        classmethod(fake_llm_from_config),
    )
    monkeypatch.setattr(
        client_module.KAGBuilderChain,
        "from_config",
        classmethod(lambda cls, config: FakeBuilderChain()),
    )

    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    for root, text in ((first_root, "first"), (second_root, "second")):
        (root / "input").mkdir(parents=True)
        (root / "input" / "guide.txt").write_text(text, encoding="utf-8")

    first = KAGClient(
        first_root,
        workspace="first-namespace",
        vectorize_model=StubVectorizeModel(),
        num_chains=1,
        num_threads_per_chain=1,
    )
    second = KAGClient(
        second_root,
        workspace="second-namespace",
        vectorize_model=StubVectorizeModel(),
        num_chains=1,
        num_threads_per_chain=1,
    )
    try:
        # second is constructed last, so a stale global config would report
        # second-namespace here.  index() must reactivate first before it
        # creates the YAML-configured LLM.
        result = first.index(cache=False)
        assert not result.has_errors, result.errors
        assert captured_namespaces == ["first-namespace"]
    finally:
        first.close()
        second.close()


def test_sequential_clients_do_not_reuse_global_query_cache(tmp_path: Path) -> None:
    """A repeated query must retrieve from the newly active knowledge base."""

    first_root = tmp_path / "first-cache"
    second_root = tmp_path / "second-cache"
    first_root.mkdir()
    second_root.mkdir()
    first = _make_client(first_root)
    second = _make_client(second_root)
    (first_root / "input" / "guide.txt").write_text(
        "FIRST-ONLY evidence: nuchal translucency is measured at 11 to 14 weeks.",
        encoding="utf-8",
    )
    (second_root / "input" / "guide.txt").write_text(
        "SECOND-ONLY evidence: nuchal translucency is measured at 11 to 14 weeks.",
        encoding="utf-8",
    )
    query = "When is nuchal translucency measured?"
    try:
        assert not first.index(cache=False).has_errors
        first_result = first.naive_search(query)
        assert any("FIRST-ONLY" in ref["content"] for ref in first_result.references)

        # This repeats the exact query but changes the active client/graph.
        assert not second.index(cache=False).has_errors
        second_result = second.naive_search(query)
        assert any("SECOND-ONLY" in ref["content"] for ref in second_result.references)
        assert not any(
            "FIRST-ONLY" in ref["content"] for ref in second_result.references
        )
    finally:
        first.close()
        second.close()


def test_close_evicts_memory_graph_but_keeps_persisted_checkpoint(
    tmp_path: Path,
) -> None:
    from kag.common.graphstore.memory_graph import MemoryGraph

    client = _make_client(tmp_path)
    try:
        assert not client.index(cache=False).has_errors
        client.naive_search("When is the nuchal translucency measured?")
        assert any(
            Path(str(key)).resolve() == client.graph_path.resolve()
            for key in MemoryGraph._instances
        )
        checkpoint = client.graph_path / "cache.db"
        assert checkpoint.is_file()
    finally:
        client.close()

    assert checkpoint.is_file()
    assert not any(
        Path(str(key)).resolve() == client.graph_path.resolve()
        for key in MemoryGraph._instances
    )


def test_cache_true_rebuilds_when_documents_or_index_config_change(
    tmp_path: Path, monkeypatch
) -> None:
    """KAG component checkpoints are valid only for identical graph inputs."""

    root = tmp_path / "fingerprint"
    root.mkdir()
    client = _make_client(root)
    guide = root / "input" / "guide.txt"
    guide.write_text(
        "FIRST-VERSION: nuchal translucency is measured at 11 to 14 weeks.",
        encoding="utf-8",
    )
    try:
        assert not client.index(cache=False).has_errors
        first_state = json.loads(
            (client.data_dir / "kag_index_state.json").read_text(encoding="utf-8")
        )

        guide.write_text(
            "SECOND-VERSION: nuchal translucency is measured at 11 to 14 weeks.",
            encoding="utf-8",
        )
        reset_calls = 0
        original_reset = client._reset_storage

        def tracked_reset() -> None:
            nonlocal reset_calls
            reset_calls += 1
            original_reset()

        monkeypatch.setattr(client, "_reset_storage", tracked_reset)
        assert not client.index(cache=True).has_errors
        assert reset_calls == 1
        second_state = json.loads(
            (client.data_dir / "kag_index_state.json").read_text(encoding="utf-8")
        )
        assert second_state["fingerprint"] != first_state["fingerprint"]
        result = client.naive_search("When is nuchal translucency measured?")
        assert any("SECOND-VERSION" in ref["content"] for ref in result.references)
    finally:
        client.close()

    # Reopening the same durable storage with a different splitter must also
    # invalidate cached component outputs rather than mix old chunk vectors.
    reconfigured = KAGClient(
        root,
        llm=StubLLMClient(),
        vectorize_model=StubVectorizeModel(),
        split_length=400,
        max_iteration=1,
        num_chains=1,
        num_threads_per_chain=1,
    )
    try:
        reset_calls = 0
        original_reset = reconfigured._reset_storage

        def tracked_reconfigured_reset() -> None:
            nonlocal reset_calls
            reset_calls += 1
            original_reset()

        monkeypatch.setattr(reconfigured, "_reset_storage", tracked_reconfigured_reset)
        assert not reconfigured.index(cache=True).has_errors
        assert reset_calls == 1
    finally:
        reconfigured.close()


def test_openai_extra_body_drops_default_thinking_marker(monkeypatch) -> None:
    """Strict OpenAI-compatible gateways reject KAG's vLLM template marker.

    The vendored chat client injects ``chat_template_kwargs`` into every
    request's ``extra_body``; gateways that 400 on the unknown field would
    otherwise empty every extraction/solve call (the builder invokes the LLM
    with ``with_except=False``).  The shim keeps an explicit ``think=True``
    opt-in for vLLM backends.
    """

    from kag.common.llm.openai_client import OpenAIClient

    from benchmark.baseline.kag_client.azure_compat import (
        ensure_openai_extra_body_compatibility,
    )

    monkeypatch.setattr(OpenAIClient, "check", lambda self: None)
    ensure_openai_extra_body_compatibility()
    ensure_openai_extra_body_compatibility()

    plain = OpenAIClient(
        base_url="https://example.invalid/v1",
        model="probe-model",
        api_key="probe-key",
    )
    assert plain.extra_body == {}

    thinking = OpenAIClient(
        base_url="https://example.invalid/v1",
        model="probe-model",
        api_key="probe-key",
        think=True,
    )
    assert thinking.extra_body == {"chat_template_kwargs": {"enable_thinking": True}}


def test_ppr_chunk_retriever_skips_contentless_nodes() -> None:
    """Pagerank chunk nodes without text must not abort graph retrieval.

    The vendored ``PprChunkRetriever`` dereferences ``node["content"].replace``
    on every pagerank-selected chunk; title-only nodes raise ``AttributeError``
    and the surrounding executor then drops every graph-retrieved chunk for
    that query.  The shim rebuilds the result defensively.
    """

    from types import SimpleNamespace

    from kag.common.tools.algorithm_tool.chunk_retriever import (
        ppr_chunk_retriever as ppr_module,
    )

    from benchmark.baseline.kag_client.azure_compat import (
        ensure_ppr_chunk_content_compatibility,
    )

    ensure_ppr_chunk_content_compatibility()
    ensure_ppr_chunk_content_compatibility()

    class FakePagerankRetriever:
        name = "ppr_chunk_retriever"
        top_k = 10

        def linking_matched_entities(self, query, **kwargs):
            return ["entity-1"]

        def calculate_pagerank_scores(self, entities, top_k=10):
            return {
                "doc-with-text": {
                    "score": 0.9,
                    "node": {
                        "content": "guideline text _split_0",
                        "name": "guide_split_0",
                    },
                },
                "doc-title-only": {
                    "score": 0.8,
                    "node": {"content": None, "name": "title-only"},
                },
            }

    retriever = FakePagerankRetriever()
    task = SimpleNamespace(arguments={"query": "probe query"})

    output = ppr_module.PprChunkRetriever.invoke(retriever, task)

    assert [chunk.chunk_id for chunk in output.chunks] == ["doc-with-text"]
    assert output.chunks[0].content == "guideline text "
    assert output.chunks[0].title == "guide"


def test_logic_form_parse_survives_backticked_plan_lines() -> None:
    """Backticked planner output must still produce executable tasks.

    gpt-5-mini answers the planner prompt's "code style" instruction
    literally: whole plan lines wrapped in backticks make the vendor's
    ``parse_steps`` see no steps at all (zero tasks, retrieval silently
    skipped), and backtick-wrapped actions make ``parse_logic_form`` raise.
    The shim unwraps both shapes before parsing.
    """

    from kag.solver.prompt.lf_static_planning_prompt import (
        RetrieverLFStaticPlanningPrompt,
    )

    from benchmark.baseline.kag_client.azure_compat import (
        ensure_logic_form_parse_compatibility,
    )

    ensure_logic_form_parse_compatibility()
    ensure_logic_form_parse_compatibility()

    prompt = RetrieverLFStaticPlanningPrompt()

    # Whole lines wrapped as inline code and carrying the action itself.
    fenced = (
        "First, retrieve the guideline's minimum requirements.\n"
        "`Step1: Retrieval(s=s1:Guideline[ISUOG Guideline], "
        "p=p1:MinimumRequirements, o=o1:RequirementList)`\n"
        "`Step2: Output(ded1)`"
    )
    tasks = prompt.parse_response(fenced)
    assert len(tasks) >= 2
    assert tasks[0].executor.lower() in {"retrieval", "retriever"}

    # Actions kept inside backticks on otherwise well-formed steps.
    wrapped_action = (
        "Step1: `What are the minimum requirements?`\n"
        "Action1: `Retrieval(s=s1:Guideline['ISUOG first-trimester guideline'], "
        "p=p1:MinimumRequirements, o=o1:RequirementList)`"
    )
    tasks = prompt.parse_response(wrapped_action)
    assert len(tasks) == 1
    assert tasks[0].executor.lower() in {"retrieval", "retriever"}
