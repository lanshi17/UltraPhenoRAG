# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""全局搜索引擎。

基于社区报告回答整个语料库层面的问题。
"""

from __future__ import annotations

from typing import Any

import graphrag.api as api
import pandas as pd

from benchmark.baseline.microsoft_graphrag_client.engine.base_search_engine import (
    BaseSearchEngine,
)
from benchmark.baseline.microsoft_graphrag_client.enums import (
    DEFAULT_COMMUNITY_LEVEL,
    DEFAULT_RESPONSE_TYPE,
)


class GlobalSearchEngine(BaseSearchEngine):
    """全局搜索引擎。

    利用社区报告进行 Map-Reduce 式搜索，适合回答宏观/概括性问题。
    """

    @property
    def method_name(self) -> str:
        return "global"

    def _load_data(self) -> dict[str, pd.DataFrame | None]:
        return self._data_loader.load_for_method("global")

    def _build_search_call(
        self,
        query: str,
        data: dict[str, pd.DataFrame | None],
        community_level: int | None,
        response_type: str,
        callbacks: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        return api.global_search(
            config=self._config,
            entities=data["entities"],
            communities=data["communities"],
            community_reports=data["community_reports"],
            community_level=community_level,
            dynamic_community_selection=kwargs.get(
                "dynamic_community_selection", False
            ),
            response_type=response_type,
            query=query,
            callbacks=callbacks,
            verbose=self._verbose,
        )

    def _build_streaming_call(
        self,
        query: str,
        data: dict[str, pd.DataFrame | None],
        community_level: int | None,
        response_type: str,
        callbacks: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        return api.global_search_streaming(
            config=self._config,
            entities=data["entities"],
            communities=data["communities"],
            community_reports=data["community_reports"],
            community_level=community_level,
            dynamic_community_selection=kwargs.get(
                "dynamic_community_selection", False
            ),
            response_type=response_type,
            query=query,
            callbacks=callbacks,
            verbose=self._verbose,
        )

    def search(
        self,
        query: str,
        community_level: int | None = DEFAULT_COMMUNITY_LEVEL,
        dynamic_community_selection: bool = False,
        response_type: str = DEFAULT_RESPONSE_TYPE,
        streaming: bool = False,
    ) -> Any:
        """执行全局搜索。

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
            查询结果。
        """
        return super().search(
            query=query,
            community_level=community_level,
            response_type=response_type,
            streaming=streaming,
            dynamic_community_selection=dynamic_community_selection,
        )
