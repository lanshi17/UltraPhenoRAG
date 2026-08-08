# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""局部搜索引擎。

基于实体和相关文本单元回答具体问题。
"""

from __future__ import annotations

from typing import Any

import graphrag.api as api
import pandas as pd

from benchmark.baseline.microsoft_graphrag_client.engine.base_search_engine import (
    BaseSearchEngine,
)


class LocalSearchEngine(BaseSearchEngine):
    """局部搜索引擎。

    利用实体嵌入与相关文本单元进行精确检索，适合回答具体实体相关问题。
    """

    @property
    def method_name(self) -> str:
        return "local"

    def _load_data(self) -> dict[str, pd.DataFrame | None]:
        return self._data_loader.load_for_method("local")

    def _build_search_call(
        self,
        query: str,
        data: dict[str, pd.DataFrame | None],
        community_level: int | None,
        response_type: str,
        callbacks: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        return api.local_search(
            config=self._config,
            entities=data["entities"],
            communities=data["communities"],
            community_reports=data["community_reports"],
            text_units=data["text_units"],
            relationships=data["relationships"],
            covariates=data["covariates"],
            community_level=community_level,
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
        return api.local_search_streaming(
            config=self._config,
            entities=data["entities"],
            communities=data["communities"],
            community_reports=data["community_reports"],
            text_units=data["text_units"],
            relationships=data["relationships"],
            covariates=data["covariates"],
            community_level=community_level,
            response_type=response_type,
            query=query,
            callbacks=callbacks,
            verbose=self._verbose,
        )
