# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""索引构建结果数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class IndexResult:
    """索引构建结果。

    封装 GraphRAG 管道执行后的输出信息。

    Attributes
    ----------
    outputs : list[Any]
        管道运行结果列表 (PipelineRunResult)。
    errors : list[str]
        遇到的错误消息列表。
    has_errors : bool
        是否遇到了错误。
    """

    outputs: list[Any]
    """管道运行结果列表 (PipelineRunResult)。"""
    errors: list[str] = field(default_factory=list)
    """遇到的错误消息列表。"""
    has_errors: bool = False
    """是否遇到了错误。"""

    @property
    def workflow_names(self) -> list[str]:
        """获取已执行的 workflow 名称列表。"""
        return [getattr(o, "workflow", str(o)) for o in self.outputs]

    @property
    def error_count(self) -> int:
        """错误数量。"""
        return len(self.errors)

    def __repr__(self) -> str:
        return (
            f"IndexResult(workflows={len(self.outputs)}, "
            f"errors={self.error_count}, has_errors={self.has_errors})"
        )
