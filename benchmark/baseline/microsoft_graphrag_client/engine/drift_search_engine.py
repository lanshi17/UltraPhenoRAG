# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""DRIFT 搜索引擎。

结合全局社区与局部实体进行多步推理。
"""

from __future__ import annotations

from typing import Any

import graphrag.api as api
import pandas as pd

from benchmark.baseline.microsoft_graphrag_client.engine.base_search_engine import (
    BaseSearchEngine,
)


class DriftSearchEngine(BaseSearchEngine):
    """DRIFT 搜索引擎。

    先利用社区报告进行全局探索，再针对具体实体进行局部深入，
    适合需要多步推理的复杂问题。
    """

    @property
    def method_name(self) -> str:
        return "drift"

    def _load_data(self) -> dict[str, pd.DataFrame | None]:
        return self._data_loader.load_for_method("drift")

    def _build_search_call(
        self,
        query: str,
        data: dict[str, pd.DataFrame | None],
        community_level: int | None,
        response_type: str,
        callbacks: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        return api.drift_search(
            config=self._config,
            entities=data["entities"],
            communities=data["communities"],
            community_reports=data["community_reports"],
            text_units=data["text_units"],
            relationships=data["relationships"],
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
        return api.drift_search_streaming(
            config=self._config,
            entities=data["entities"],
            communities=data["communities"],
            community_reports=data["community_reports"],
            text_units=data["text_units"],
            relationships=data["relationships"],
            community_level=community_level,
            response_type=response_type,
            query=query,
            callbacks=callbacks,
            verbose=self._verbose,
        )
