# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""GraphRAG 客户端包。

提供 Microsoft GraphRAG 的同步调用封装。

主要导出::

    GraphRAGClient   — 客户端门面
    IndexResult      — 索引构建结果
    QueryResult      — 查询结果
    SearchMethod     — 搜索方法枚举
    IndexMethod      — 索引方法枚举
    ConfigManager    — 配置管理器
    DataLoader       — 数据加载器
    IndexEngine      — 索引引擎
    GlobalSearchEngine  — 全局搜索引擎
    LocalSearchEngine   — 局部搜索引擎
    DriftSearchEngine   — DRIFT 搜索引擎
    BasicSearchEngine   — 基础搜索引擎
    PromptTuneEngine    — 提示词调优引擎
"""

from benchmark.baseline.microsoft_graphrag_client.client import GraphRAGClient
from benchmark.baseline.microsoft_graphrag_client.config.config_manager import (
    ConfigManager,
)
from benchmark.baseline.microsoft_graphrag_client.config.llm_config import (
    DEFAULT_COMPLETION_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    LLMConfigOverrides,
    ModelConfigOverride,
)
from benchmark.baseline.microsoft_graphrag_client.data.data_loader import DataLoader
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
    DocSelectionMethod,
    IndexMethod,
    SearchMethod,
)
from benchmark.baseline.microsoft_graphrag_client.models.index_result import IndexResult
from benchmark.baseline.microsoft_graphrag_client.models.query_result import QueryResult

__all__ = [
    # 客户端
    "GraphRAGClient",
    # 数据结构
    "IndexResult",
    "QueryResult",
    # 枚举
    "SearchMethod",
    "IndexMethod",
    "DocSelectionMethod",
    # 配置
    "ConfigManager",
    "DEFAULT_COMPLETION_MODEL",
    "DEFAULT_EMBEDDING_MODEL",
    "LLMConfigOverrides",
    "ModelConfigOverride",
    # 组件
    "DataLoader",
    "IndexEngine",
    "GlobalSearchEngine",
    "LocalSearchEngine",
    "DriftSearchEngine",
    "BasicSearchEngine",
    "PromptTuneEngine",
]
