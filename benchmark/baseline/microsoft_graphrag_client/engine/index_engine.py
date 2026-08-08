# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""索引引擎。

封装 GraphRAG 索引构建逻辑。
"""

from __future__ import annotations

import logging

import graphrag.api as api
import pandas as pd
from graphrag.callbacks.console_workflow_callbacks import ConsoleWorkflowCallbacks
from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.index.validate_config import validate_config_names
from graphrag_cache.cache_type import CacheType

from benchmark.baseline.microsoft_graphrag_client.enums import IndexMethod
from benchmark.baseline.microsoft_graphrag_client.models.index_result import IndexResult
from benchmark.baseline.microsoft_graphrag_client.utils.async_runner import AsyncRunner

logger = logging.getLogger(__name__)


class IndexEngine:
    """索引构建引擎。

    封装 GraphRAG 管道执行逻辑，包括配置验证、缓存控制、回调管理。

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

    def build(
        self,
        method: str | IndexMethod = IndexMethod.STANDARD,
        is_update_run: bool = False,
        cache: bool = True,
        dry_run: bool = False,
        skip_validation: bool = False,
        callbacks: list[WorkflowCallbacks] | None = None,
        input_documents: pd.DataFrame | None = None,
    ) -> IndexResult:
        """构建（或增量更新）知识图谱索引。

        Parameters
        ----------
        method : str | IndexMethod, default "standard"
            索引方式。
        is_update_run : bool, default False
            是否为增量更新运行。
        cache : bool, default True
            是否启用缓存。
        dry_run : bool, default False
            仅验证配置，不实际执行。
        skip_validation : bool, default False
            跳过配置验证。
        callbacks : list[WorkflowCallbacks] | None, optional
            自定义回调列表。
        input_documents : pd.DataFrame | None, optional
            自定义输入文档。

        Returns
        -------
        IndexResult
            索引构建结果。
        """
        config = self._config

        if not cache:
            config.cache.type = CacheType.Noop

        if not skip_validation:
            validate_config_names(config)

        if dry_run:
            logger.info("Dry run - 配置验证通过，未执行索引。")
            return IndexResult(outputs=[], has_errors=False)

        if isinstance(method, IndexMethod):
            method = method.value

        if callbacks is None:
            callbacks = [ConsoleWorkflowCallbacks(verbose=self._verbose)]

        logger.info("开始构建索引 (method=%s, update=%s)", method, is_update_run)
        outputs = AsyncRunner.run(
            api.build_index(
                config=config,
                method=method,
                is_update_run=is_update_run,
                callbacks=callbacks,
                verbose=self._verbose,
                input_documents=input_documents,
            )
        )

        errors = [str(o.error) for o in outputs if o.error is not None]
        result = IndexResult(
            outputs=outputs,
            errors=errors,
            has_errors=len(errors) > 0,
        )

        if result.has_errors:
            logger.warning("索引构建完成，但有 %d 个错误。", result.error_count)
        else:
            logger.info("索引构建成功，共 %d 个 workflow。", len(outputs))

        return result
