# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""搜索引擎基类。

封装所有搜索方法共享的逻辑：回调创建、流式/非流式执行、结果构建。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from graphrag.callbacks.noop_query_callbacks import NoopQueryCallbacks
from graphrag.config.models.graph_rag_config import GraphRagConfig

from benchmark.baseline.microsoft_graphrag_client.data.data_loader import DataLoader
from benchmark.baseline.microsoft_graphrag_client.enums import (
    DEFAULT_COMMUNITY_LEVEL,
    DEFAULT_RESPONSE_TYPE,
)
from benchmark.baseline.microsoft_graphrag_client.models.query_result import QueryResult
from benchmark.baseline.microsoft_graphrag_client.utils.async_runner import AsyncRunner

logger = logging.getLogger(__name__)


class BaseSearchEngine(ABC):
    """搜索引擎抽象基类。

    提供回调创建、数据加载、结果构建等通用方法，子类只需实现
    ``_build_search_call`` 和 ``_build_streaming_call``。

    Parameters
    ----------
    config : GraphRagConfig
        GraphRAG 配置实例。
    verbose : bool, default False
        是否输出详细日志。
    """

    def __init__(self, config: GraphRagConfig, verbose: bool = False) -> None:
        self._config = config
        self._verbose = verbose
        self._data_loader = DataLoader(config)

    @property
    @abstractmethod
    def method_name(self) -> str:
        """搜索方法名称 (global / local / drift / basic)。"""

    @abstractmethod
    def _load_data(self) -> dict[str, pd.DataFrame | None]:
        """加载该搜索方法所需的索引输出数据。"""

    @abstractmethod
    def _build_search_call(
        self,
        query: str,
        data: dict[str, pd.DataFrame | None],
        community_level: int | None,
        response_type: str,
        callbacks: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """构建非流式搜索协程。"""

    @abstractmethod
    def _build_streaming_call(
        self,
        query: str,
        data: dict[str, pd.DataFrame | None],
        community_level: int | None,
        response_type: str,
        callbacks: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """构建流式搜索异步生成器。"""

    def search(
        self,
        query: str,
        community_level: int = DEFAULT_COMMUNITY_LEVEL,
        response_type: str = DEFAULT_RESPONSE_TYPE,
        streaming: bool = False,
        **kwargs: Any,
    ) -> QueryResult:
        """执行搜索。

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
        **kwargs : Any
            传递给子类的额外参数。

        Returns
        -------
        QueryResult
            查询结果。
        """
        logger.debug("[%s] 加载索引数据...", self.method_name)
        data = self._load_data()

        context_data: dict[str, Any] = {}

        def on_context(context: Any) -> None:
            context_data.clear()
            context_data.update(context)

        cb = NoopQueryCallbacks()
        cb.on_context = on_context

        if streaming:
            logger.debug("[%s] 执行流式搜索...", self.method_name)

            async def _run_streaming() -> str:
                full = ""
                agen = self._build_streaming_call(
                    query=query,
                    data=data,
                    community_level=community_level,
                    response_type=response_type,
                    callbacks=[cb],
                    **kwargs,
                )
                async for chunk in agen:
                    full += chunk
                return full

            response = AsyncRunner.run(_run_streaming())
        else:
            logger.debug("[%s] 执行非流式搜索...", self.method_name)
            response, context_data = AsyncRunner.run(
                self._build_search_call(
                    query=query,
                    data=data,
                    community_level=community_level,
                    response_type=response_type,
                    callbacks=[cb],
                    **kwargs,
                )
            )

        return QueryResult(
            response=response,
            context_data=context_data,
            query=query,
            method=self.method_name,
        )
