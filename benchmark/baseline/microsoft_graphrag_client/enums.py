# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""状态枚举与默认常量。"""

from __future__ import annotations

from enum import Enum


class SearchMethod(str, Enum):
    """搜索方法枚举。"""

    GLOBAL = "global"
    """全局搜索 - 基于社区报告回答语料库层面的问题。"""
    LOCAL = "local"
    """局部搜索 - 基于实体和相关文本单元回答具体问题。"""
    DRIFT = "drift"
    """DRIFT 搜索 - 结合全局社区与局部实体进行多步推理。"""
    BASIC = "basic"
    """基础搜索 - 直接在文本块上进行 RAG 检索。"""

    def __str__(self) -> str:
        return self.value


class IndexMethod(str, Enum):
    """索引方法枚举。"""

    STANDARD = "standard"
    """标准索引 - 所有图构建和摘要由 LLM 完成。"""
    FAST = "fast"
    """快速索引 - 使用 NLP 进行图构建，LLM 进行摘要。"""
    STANDARD_UPDATE = "standard-update"
    """标准增量更新。"""
    FAST_UPDATE = "fast-update"
    """快速增量更新。"""

    def __str__(self) -> str:
        return self.value


class DocSelectionMethod(str, Enum):
    """文档选择方式枚举（用于 prompt tuning）。"""

    ALL = "all"
    """全部选择。"""
    RANDOM = "random"
    """随机选择。"""
    TOP = "top"
    """选择前 N 个。"""
    AUTO = "auto"
    """自动选择。"""

    def __str__(self) -> str:
        return self.value


# ── 默认值 ────────────────────────────────────────────────────────────────────

DEFAULT_COMMUNITY_LEVEL: int = 2
"""默认社区层级。"""

DEFAULT_RESPONSE_TYPE: str = "multiple paragraphs"
"""默认响应格式描述。"""

DEFAULT_PROMPT_TUNE_LIMIT: int = 15
"""prompt tuning 默认加载文本块数量。"""

DEFAULT_PROMPT_TUNE_MAX_TOKENS: int = 12000
"""prompt tuning 默认最大 token 数。"""

DEFAULT_PROMPT_TUNE_N_SUBSET_MAX: int = 300
"""auto 选择方式默认嵌入文本块数量。"""

DEFAULT_PROMPT_TUNE_K: int = 15
"""auto 选择方式默认选取文档数量。"""
