# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""查询结果数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from benchmark.baseline.microsoft_graphrag_client.enums import SearchMethod


@dataclass
class QueryResult:
    """查询结果。

    封装 GraphRAG 查询引擎返回的回答文本和上下文数据。

    Attributes
    ----------
    response : str
        LLM 生成的回答文本。
    context_data : dict[str, Any]
        查询返回的上下文数据，包含 entities、relationships 等 DataFrame。
    query : str
        原始查询字符串。
    method : str
        使用的搜索方法 (global / local / drift / basic)。
    """

    response: str
    """LLM 生成的回答文本。"""
    context_data: dict[str, Any] = field(default_factory=dict)
    """查询返回的上下文数据，包含 entities、relationships 等 DataFrame。"""
    query: str = ""
    """原始查询字符串。"""
    method: str = ""
    """使用的搜索方法 (global / local / drift / basic)。"""
    telemetry: dict[str, Any] = field(default_factory=dict)
    """查询级事件、图探索和 token/cost 统计。"""

    def get_context_df(self, name: str) -> pd.DataFrame | None:
        """按名称获取上下文 DataFrame。

        Parameters
        ----------
        name : str
            上下文数据键名，如 'entities'、'relationships'、'sources' 等。

        Returns
        -------
        pd.DataFrame | None
            对应的 DataFrame，不存在时返回 None。
        """
        value = self.context_data.get(name)
        if isinstance(value, pd.DataFrame):
            return value
        if isinstance(value, list) and value and isinstance(value[0], pd.DataFrame):
            return value[0]
        return None

    @property
    def method_enum(self) -> SearchMethod | None:
        """获取搜索方法枚举。"""
        try:
            return SearchMethod(self.method)
        except ValueError:
            return None

    def __repr__(self) -> str:
        return (
            f"QueryResult(method={self.method!r}, "
            f"response_len={len(self.response)}, "
            f"context_keys={list(self.context_data.keys())})"
        )
