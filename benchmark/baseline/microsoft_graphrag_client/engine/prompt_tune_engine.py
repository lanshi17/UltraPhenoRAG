# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""Prompt Tuning 引擎。

生成自定义索引提示词。
"""

from __future__ import annotations

import logging

import graphrag.api as api
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.prompt_tune.types import DocSelectionType

from benchmark.baseline.microsoft_graphrag_client.enums import (
    DEFAULT_PROMPT_TUNE_K,
    DEFAULT_PROMPT_TUNE_LIMIT,
    DEFAULT_PROMPT_TUNE_MAX_TOKENS,
    DEFAULT_PROMPT_TUNE_N_SUBSET_MAX,
    DocSelectionMethod,
)
from benchmark.baseline.microsoft_graphrag_client.utils.async_runner import AsyncRunner

logger = logging.getLogger(__name__)


class PromptTuneEngine:
    """Prompt Tuning 引擎。

    利用 LLM 分析输入文档，自动生成适合特定领域的索引提示词。

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

    def generate(
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
            文本块选择方式。
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
        doc_selection = DocSelectionType(selection_method.value)

        logger.info("开始生成提示词 (limit=%d, method=%s)", limit, selection_method)
        return AsyncRunner.run(
            api.generate_indexing_prompts(
                config=self._config,
                limit=limit,
                selection_method=doc_selection,
                domain=domain,
                language=language,
                max_tokens=max_tokens,
                discover_entity_types=discover_entity_types,
                min_examples_required=min_examples_required,
                n_subset_max=n_subset_max,
                k=k,
                verbose=self._verbose,
            )
        )
