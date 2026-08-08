# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""基础搜索引擎。

直接在文本块上进行 RAG 检索。
"""

from __future__ import annotations

from typing import Any

import graphrag.api as api
import pandas as pd

from benchmark.baseline.microsoft_graphrag_client.engine.base_search_engine import (
    BaseSearchEngine,
)
from benchmark.baseline.microsoft_graphrag_client.enums import DEFAULT_RESPONSE_TYPE


class BasicSearchEngine(BaseSearchEngine):
    """基础搜索引擎。

    不使用知识图谱，直接在文本块上进行向量检索和 LLM 生成，
    适合简单的文本问答。
    """

    @property
    def method_name(self) -> str:
        return "basic"

    def _load_data(self) -> dict[str, pd.DataFrame | None]:
        return self._data_loader.load_for_method("basic")

    def _build_search_call(
        self,
        query: str,
        data: dict[str, pd.DataFrame | None],
        community_level: int | None,
        response_type: str,
        callbacks: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        return api.basic_search(
            config=self._config,
            text_units=data["text_units"],
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
        return api.basic_search_streaming(
            config=self._config,
            text_units=data["text_units"],
            response_type=response_type,
            query=query,
            callbacks=callbacks,
            verbose=self._verbose,
        )

    def search(
        self,
        query: str,
        response_type: str = DEFAULT_RESPONSE_TYPE,
        streaming: bool = False,
    ) -> Any:
        """执行基础搜索。

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
        return super().search(
            query=query,
            response_type=response_type,
            streaming=streaming,
        )
