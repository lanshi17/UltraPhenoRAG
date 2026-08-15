# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""GraphRAG 客户端门面。

封装 Microsoft GraphRAG 的索引构建与查询 API，提供同步调用接口。
内部委托给 ConfigManager、IndexEngine、各 SearchEngine 和 PromptTuneEngine。

使用方式
--------
.. code-block:: python

    from benchmark.baseline.microsoft_graphrag_client import GraphRAGClient

    client = GraphRAGClient(root_dir="/path/to/graphrag/project")

    # 构建索引
    result = client.index(method="standard", verbose=True)

    # 查询
    result = client.global_search(query="主要议题是什么？")
    print(result.response)

    result = client.local_search(query="某实体的详细信息？")
    print(result.response)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.config.models.graph_rag_config import GraphRagConfig

from benchmark.baseline.microsoft_graphrag_client.config.config_manager import (
    ConfigManager,
)
from benchmark.baseline.microsoft_graphrag_client.config.llm_config import (
    LLMConfigOverrides,
)
from benchmark.baseline.microsoft_graphrag_client.engine.basic_search_engine import (
    BasicSearchEngine,
)
from benchmark.baseline.microsoft_graphrag_client.engine.drift_search_engine import (
    DriftSearchEngine,
)
from benchmark.baseline.microsoft_graphrag_client.engine.global_search_engine import (
    GlobalSearchEngine,
)
from benchmark.baseline.microsoft_graphrag_client.engine.index_engine import IndexEngine
from benchmark.baseline.microsoft_graphrag_client.engine.local_search_engine import (
    LocalSearchEngine,
)
from benchmark.baseline.microsoft_graphrag_client.engine.prompt_tune_engine import (
    PromptTuneEngine,
)
from benchmark.baseline.microsoft_graphrag_client.enums import (
    DEFAULT_COMMUNITY_LEVEL,
    DEFAULT_PROMPT_TUNE_K,
    DEFAULT_PROMPT_TUNE_LIMIT,
    DEFAULT_PROMPT_TUNE_MAX_TOKENS,
    DEFAULT_PROMPT_TUNE_N_SUBSET_MAX,
    DEFAULT_RESPONSE_TYPE,
    DocSelectionMethod,
    IndexMethod,
)
from benchmark.baseline.microsoft_graphrag_client.models.index_result import IndexResult
from benchmark.baseline.microsoft_graphrag_client.models.query_result import QueryResult

logger = logging.getLogger(__name__)


class GraphRAGClient:
    """Microsoft GraphRAG 客户端门面。

    整合配置管理、索引构建与四种查询模式 (global / local / drift / basic)，
    将异步 API 转为同步调用，简化使用。

    内部委托关系::

        GraphRAGClient
        ├── ConfigManager      -> 配置加载与缓存
        ├── IndexEngine        -> 索引构建
        ├── GlobalSearchEngine -> 全局搜索
        ├── LocalSearchEngine  -> 局部搜索
        ├── DriftSearchEngine  -> DRIFT 搜索
        ├── BasicSearchEngine  -> 基础搜索
        └── PromptTuneEngine   -> 提示词调优

    Parameters
    ----------
    root_dir : str | Path
        GraphRAG 项目根目录（包含 ``settings.yaml``）。
    data_dir : str | Path | None, optional
        索引输出目录。为 ``None`` 时使用 ``settings.yaml`` 中的 ``output.base_dir``。
    llm_overrides : LLMConfigOverrides | Mapping[str, Any] | None, optional
        completion/embedding 模型的运行时覆盖。
    verbose : bool, default False
        是否输出详细日志。
    """

    def __init__(
        self,
        root_dir: str | Path,
        data_dir: str | Path | None = None,
        llm_overrides: LLMConfigOverrides | dict[str, object] | None = None,
        verbose: bool = False,
    ) -> None:
        self.verbose = verbose

        # 配置管理
        self._config_manager = ConfigManager(
            root_dir=root_dir,
            data_dir=data_dir,
            llm_overrides=llm_overrides,
        )

        # 懒加载引擎
        self._index_engine: IndexEngine | None = None
        self._global_engine: GlobalSearchEngine | None = None
        self._local_engine: LocalSearchEngine | None = None
        self._drift_engine: DriftSearchEngine | None = None
        self._basic_engine: BasicSearchEngine | None = None
        self._prompt_tune_engine: PromptTuneEngine | None = None

    # ── 属性 ──────────────────────────────────────────────────────────────────

    @property
    def root_dir(self) -> Path:
        """项目根目录。"""
        return self._config_manager.root_dir

    @property
    def data_dir(self) -> Path | None:
        """索引输出目录覆盖。"""
        return self._config_manager.data_dir

    @property
    def config(self) -> GraphRagConfig:
        """延迟加载并缓存的 GraphRagConfig。"""
        return self._config_manager.config

    def reload_config(self) -> GraphRagConfig:
        """强制重新加载配置。"""
        # 重置所有引擎以使用新配置
        self._index_engine = None
        self._global_engine = None
        self._local_engine = None
        self._drift_engine = None
        self._basic_engine = None
        self._prompt_tune_engine = None
        return self._config_manager.reload()

    # ── 引擎懒加载 ────────────────────────────────────────────────────────────

    @property
    def index_engine(self) -> IndexEngine:
        if self._index_engine is None:
            self._index_engine = IndexEngine(config=self.config, verbose=self.verbose)
        return self._index_engine

    @property
    def global_engine(self) -> GlobalSearchEngine:
        if self._global_engine is None:
            self._global_engine = GlobalSearchEngine(
                config=self.config, verbose=self.verbose
            )
        return self._global_engine

    @property
    def local_engine(self) -> LocalSearchEngine:
        if self._local_engine is None:
            self._local_engine = LocalSearchEngine(
                config=self.config, verbose=self.verbose
            )
        return self._local_engine

    @property
    def drift_engine(self) -> DriftSearchEngine:
        if self._drift_engine is None:
            self._drift_engine = DriftSearchEngine(
                config=self.config, verbose=self.verbose
            )
        return self._drift_engine

    @property
    def basic_engine(self) -> BasicSearchEngine:
        if self._basic_engine is None:
            self._basic_engine = BasicSearchEngine(
                config=self.config, verbose=self.verbose
            )
        return self._basic_engine

    @property
    def prompt_tune_engine(self) -> PromptTuneEngine:
        if self._prompt_tune_engine is None:
            self._prompt_tune_engine = PromptTuneEngine(
                config=self.config, verbose=self.verbose
            )
        return self._prompt_tune_engine

    # ── 索引 ──────────────────────────────────────────────────────────────────

    def index(
        self,
        method: str | IndexMethod = IndexMethod.STANDARD,
        is_update_run: bool = False,
        cache: bool = True,
        dry_run: bool = False,
        skip_validation: bool = False,
        callbacks: list[WorkflowCallbacks] | None = None,
        input_documents: pd.DataFrame | None = None,
    ) -> IndexResult:
        """构建（或增量更新）知识图谱索引。

        Parameters
        ----------
        method : str | IndexMethod, default "standard"
            索引方式：``"standard"`` / ``"fast"`` / ``"standard-update"`` / ``"fast-update"``。
        is_update_run : bool, default False
            是否为增量更新运行。
        cache : bool, default True
            是否启用缓存。``False`` 时禁用缓存。
        dry_run : bool, default False
            仅验证配置，不实际执行。
        skip_validation : bool, default False
            跳过配置验证。
        callbacks : list[WorkflowCallbacks] | None, optional
            自定义回调列表。为 ``None`` 时使用 ``ConsoleWorkflowCallbacks``。
        input_documents : pd.DataFrame | None, optional
            自定义输入文档 DataFrame，覆盖配置中的文档加载。

        Returns
        -------
        IndexResult
            索引构建结果。
        """
        return self.index_engine.build(
            method=method,
            is_update_run=is_update_run,
            cache=cache,
            dry_run=dry_run,
            skip_validation=skip_validation,
            callbacks=callbacks,
            input_documents=input_documents,
        )

    # ── 查询 ──────────────────────────────────────────────────────────────────

    def global_search(
        self,
        query: str,
        community_level: int | None = DEFAULT_COMMUNITY_LEVEL,
        dynamic_community_selection: bool = False,
        response_type: str = DEFAULT_RESPONSE_TYPE,
        streaming: bool = False,
    ) -> QueryResult:
        """全局搜索 - 基于社区报告回答整个语料库层面的问题。

        Parameters
        ----------
        query : str
            用户查询。
        community_level : int | None, default 2
            社区层级。
        dynamic_community_selection : bool, default False
            是否动态选择社区。
        response_type : str, default "multiple paragraphs"
            响应格式描述。
        streaming : bool, default False
            是否流式返回。

        Returns
        -------
        QueryResult
            查询结果，包含回答和上下文数据。
        """
        return self.global_engine.search(
            query=query,
            community_level=community_level,
            dynamic_community_selection=dynamic_community_selection,
            response_type=response_type,
            streaming=streaming,
        )

    def local_search(
        self,
        query: str,
        community_level: int = DEFAULT_COMMUNITY_LEVEL,
        response_type: str = DEFAULT_RESPONSE_TYPE,
        streaming: bool = False,
    ) -> QueryResult:
        """局部搜索 - 基于实体和相关文本单元回答具体问题。

        Parameters
        ----------
        query : str
            用户查询。
        community_level : int, default 2
            社区层级。
        response_type : str, default "multiple paragraphs"
            响应格式描述。
        streaming : bool, default False
            是否流式返回。

        Returns
        -------
        QueryResult
            查询结果。
        """
        return self.local_engine.search(
            query=query,
            community_level=community_level,
            response_type=response_type,
            streaming=streaming,
        )

    def drift_search(
        self,
        query: str,
        community_level: int = DEFAULT_COMMUNITY_LEVEL,
        response_type: str = DEFAULT_RESPONSE_TYPE,
        streaming: bool = False,
    ) -> QueryResult:
        """DRIFT 搜索 - 结合全局社区与局部实体进行多步推理。

        Parameters
        ----------
        query : str
            用户查询。
        community_level : int, default 2
            社区层级。
        response_type : str, default "multiple paragraphs"
            响应格式描述。
        streaming : bool, default False
            是否流式返回。

        Returns
        -------
        QueryResult
            查询结果。
        """
        return self.drift_engine.search(
            query=query,
            community_level=community_level,
            response_type=response_type,
            streaming=streaming,
        )

    def basic_search(
        self,
        query: str,
        response_type: str = DEFAULT_RESPONSE_TYPE,
        streaming: bool = False,
    ) -> QueryResult:
        """基础搜索 - 直接在文本块上进行 RAG 检索。

        Parameters
        ----------
        query : str
            用户查询。
        response_type : str, default "multiple paragraphs"
            响应格式描述。
        streaming : bool, default False
            是否流式返回。

        Returns
        -------
        QueryResult
            查询结果。
        """
        return self.basic_engine.search(
            query=query,
            response_type=response_type,
            streaming=streaming,
        )

    # ── Prompt Tuning ────────────────────────────────────────────────────────

    def prompt_tune(
        self,
        limit: int = DEFAULT_PROMPT_TUNE_LIMIT,
        selection_method: DocSelectionMethod = DocSelectionMethod.RANDOM,
        domain: str | None = None,
        language: str | None = None,
        max_tokens: int = DEFAULT_PROMPT_TUNE_MAX_TOKENS,
        discover_entity_types: bool = True,
        min_examples_required: int = 2,
        n_subset_max: int = DEFAULT_PROMPT_TUNE_N_SUBSET_MAX,
        k: int = DEFAULT_PROMPT_TUNE_K,
    ) -> tuple[str, str, str]:
        """生成自定义索引提示词。

        Parameters
        ----------
        limit : int, default 15
            加载的文本块数量上限。
        selection_method : DocSelectionMethod, default RANDOM
            文本块选择方式：``ALL`` / ``RANDOM`` / ``TOP`` / ``AUTO``。
        domain : str | None, optional
            领域描述。
        language : str | None, optional
            提示词语言。
        max_tokens : int, default 12000
            实体提取提示词的最大 token 数。
        discover_entity_types : bool, default True
            是否自动发现实体类型。
        min_examples_required : int, default 2
            最少示例数。
        n_subset_max : int, default 300
            auto 方式下嵌入的文本块数量。
        k : int, default 15
            auto 方式下选取的文档数量。

        Returns
        -------
        tuple[str, str, str]
            (domain, entity_extraction_prompt, community_report_prompt)
        """
        return self.prompt_tune_engine.generate(
            limit=limit,
            selection_method=selection_method,
            domain=domain,
            language=language,
            max_tokens=max_tokens,
            discover_entity_types=discover_entity_types,
            min_examples_required=min_examples_required,
            n_subset_max=n_subset_max,
            k=k,
        )
