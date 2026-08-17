# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""搜索引擎基类。

封装所有搜索方法共享的逻辑：回调创建、流式/非流式执行、结果构建。
"""

from __future__ import annotations

import logging
import math
import time
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


def _context_stats(context: Any) -> tuple[int, int, set[str], int, int]:
    """统计回调上下文中的记录数、字符数和表名。"""
    frames: list[tuple[str, pd.DataFrame]] = []
    if isinstance(context, pd.DataFrame):
        frames.append(("context", context))
    elif isinstance(context, dict):
        for name, value in context.items():
            if isinstance(value, pd.DataFrame):
                frames.append((str(name), value))
            elif isinstance(value, list):
                frames.extend(
                    (str(name), item) for item in value if isinstance(item, pd.DataFrame)
                )
    elif isinstance(context, list):
        frames.extend(
            (f"context_{index}", item)
            for index, item in enumerate(context)
            if isinstance(item, pd.DataFrame)
        )
    records = 0
    chars = 0
    tables: set[str] = set()
    community_records = 0
    graph_nodes = 0
    for name, frame in frames:
        tables.add(name)
        records += len(frame.index)
        normalized_name = name.casefold()
        if "communit" in normalized_name or "report" in normalized_name:
            community_records += len(frame.index)
        if any(token in normalized_name for token in ("entit", "relationship", "communit")):
            graph_nodes += len(frame.index)
        text_columns = [
            column
            for column in ("text", "content", "description", "summary", "context")
            if column in frame.columns
        ]
        if text_columns:
            chars += int(
                frame[text_columns]
                .fillna("")
                .astype(str)
                .apply(lambda column: column.str.len())
                .sum()
                .sum()
            )
    return records, chars, tables, community_records, graph_nodes


class QueryMonitorCallbacks(NoopQueryCallbacks):
    """收集 GraphRAG 查询回调，不改变原始查询行为。"""

    def __init__(self) -> None:
        self.context_event_count = 0
        self.context_record_count = 0
        self.context_chars = 0
        self.context_tables: set[str] = set()
        self.community_record_count = 0
        self.graph_node_count = 0
        self.map_response_start_count = 0
        self.map_response_end_count = 0
        self.reduce_response_start_count = 0
        self.reduce_response_end_count = 0
        self.stream_output_chunks = 0
        self.stream_output_chars = 0

    def on_context(self, context: Any) -> None:
        self.context_event_count += 1
        records, chars, tables, community_records, graph_nodes = _context_stats(context)
        self.context_record_count += records
        self.context_chars += chars
        self.context_tables.update(tables)
        self.community_record_count += community_records
        self.graph_node_count += graph_nodes

    def on_map_response_start(self, map_response_contexts: list[str]) -> None:
        self.map_response_start_count += len(map_response_contexts)

    def on_map_response_end(self, map_response_outputs: list[Any]) -> None:
        self.map_response_end_count += len(map_response_outputs)

    def on_reduce_response_start(self, reduce_response_context: str | dict[str, Any]) -> None:
        self.reduce_response_start_count += 1

    def on_reduce_response_end(self, reduce_response_output: str) -> None:
        self.reduce_response_end_count += 1

    def on_llm_new_token(self, token: Any) -> None:
        self.stream_output_chunks += 1
        self.stream_output_chars += len(str(token or ""))

    def as_dict(self) -> dict[str, Any]:
        # 回调收到的 token 可能是文本 chunk，不一定是单 token，因此明确标记为估算。
        estimated_tokens = math.ceil(self.stream_output_chars / 4) if self.stream_output_chars else 0
        return {
            "context_event_count": self.context_event_count,
            "context_record_count": self.context_record_count,
            "context_chars": self.context_chars,
            "context_tokens_estimate": math.ceil(self.context_chars / 4) if self.context_chars else 0,
            "context_tables": sorted(self.context_tables),
            "community_record_count": self.community_record_count,
            "graph_node_count": self.graph_node_count,
            "graph_exploration_event_count": self.context_event_count,
            "map_response_count": self.map_response_end_count,
            "map_response_start_count": self.map_response_start_count,
            "reduce_response_count": self.reduce_response_end_count,
            "reduce_response_start_count": self.reduce_response_start_count,
            "stream_output_chunks": self.stream_output_chunks,
            "stream_output_chars": self.stream_output_chars,
            "stream_output_tokens_estimate": estimated_tokens,
        }


def _metric_stores(config: GraphRagConfig) -> dict[str, Any]:
    """获取 GraphRAG completion/embedding 的共享 metrics store。"""
    stores: dict[str, Any] = {}
    try:
        from graphrag_llm.metrics import create_metrics_store
    except ImportError:
        return stores

    model_ids: set[str] = set()
    for section_name in ("basic_search", "local_search", "drift_search", "global_search"):
        section = getattr(config, section_name, None)
        if section is None:
            continue
        for field_name, getter in (
            ("completion_model_id", config.get_completion_model_config),
            ("embedding_model_id", config.get_embedding_model_config),
        ):
            model_id = getattr(section, field_name, None)
            if not model_id:
                continue
            try:
                model_config = getter(model_id)
                if model_config.metrics is None:
                    continue
                resolved = f"{model_config.model_provider}/{model_config.model}"
                stores[resolved] = create_metrics_store(
                    config=model_config.metrics,
                    id=resolved,
                )
                model_ids.add(resolved)
            except Exception:  # noqa: BLE001
                logger.debug("无法读取模型 metrics store", exc_info=True)
    return {model_id: stores[model_id] for model_id in model_ids if model_id in stores}


def _metrics_snapshot(stores: dict[str, Any]) -> dict[str, dict[str, float]]:
    snapshot: dict[str, dict[str, float]] = {}
    for model_id, store in stores.items():
        try:
            metrics = store.get_metrics()
            snapshot[model_id] = {
                str(key): float(value)
                for key, value in metrics.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
        except Exception:  # noqa: BLE001
            logger.debug("无法读取模型 metrics", exc_info=True)
    return snapshot


def _usage_delta(
    before: dict[str, dict[str, float]], after: dict[str, dict[str, float]]
) -> dict[str, Any]:
    values: dict[str, float] = {}
    models: list[str] = []
    by_model: dict[str, dict[str, Any]] = {}
    for model_id in sorted(set(before) | set(after)):
        models.append(model_id)
        model_values: dict[str, float] = {}
        for key in set(before.get(model_id, {})) | set(after.get(model_id, {})):
            delta = (
                after.get(model_id, {}).get(key, 0.0)
                - before.get(model_id, {}).get(key, 0.0)
            )
            values[key] = values.get(key, 0.0) + delta
            model_values[key] = delta
        model_cost_available = (
            model_values.get("responses_with_tokens", 0.0) > 0
            and model_values.get("responses_with_cost", 0.0)
            >= model_values.get("responses_with_tokens", 0.0)
        )
        by_model[model_id] = {
            "prompt_tokens": (
                int(model_values["prompt_tokens"])
                if "prompt_tokens" in model_values
                else None
            ),
            "completion_tokens": (
                int(model_values["completion_tokens"])
                if "completion_tokens" in model_values
                else None
            ),
            "total_tokens": (
                int(model_values["total_tokens"])
                if "total_tokens" in model_values
                else None
            ),
            "input_cost_usd": (
                model_values.get("input_cost") if model_cost_available else None
            ),
            "output_cost_usd": (
                model_values.get("output_cost") if model_cost_available else None
            ),
            "total_cost_usd": (
                model_values.get("total_cost") if model_cost_available else None
            ),
            "cost_available": model_cost_available,
        }
    prompt = values.get("prompt_tokens")
    completion = values.get("completion_tokens")
    total = values.get("total_tokens")
    token_responses = values.get("responses_with_tokens", 0.0)
    cost_responses = values.get("responses_with_cost", 0.0)
    cost_available = token_responses > 0 and cost_responses >= token_responses
    usage: dict[str, Any] = {
        "request_count": int(
            values.get("attempted_request_count", 0.0)
            or values.get("successful_response_count", 0.0)
            or token_responses
        ),
        "failed_request_count": int(values.get("failed_response_count", 0.0)),
        "prompt_tokens": int(prompt) if prompt is not None else None,
        "completion_tokens": int(completion) if completion is not None else None,
        "total_tokens": int(total) if total is not None else None,
        "input_cost_usd": values.get("input_cost") if cost_available else None,
        "output_cost_usd": values.get("output_cost") if cost_available else None,
        "total_cost_usd": values.get("total_cost") if cost_available else None,
        "cost_available": cost_available,
        "models": models,
        "by_model": by_model,
    }
    return usage


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

        cb = QueryMonitorCallbacks()

        def on_context(context: Any) -> None:
            context_data.clear()
            context_data.update(context)
            QueryMonitorCallbacks.on_context(cb, context)

        cb.on_context = on_context
        stores = _metric_stores(self._config)
        metrics_before = _metrics_snapshot(stores)
        query_started = time.monotonic()

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

        telemetry = cb.as_dict()
        telemetry["elapsed_seconds"] = round(time.monotonic() - query_started, 3)
        if self.method_name == "drift":
            drift_config = getattr(self._config, "drift_search", None)
            telemetry["configured_traversal_depth"] = getattr(
                drift_config, "n_depth", None
            )
            telemetry["configured_followups"] = getattr(
                drift_config, "drift_k_followups", None
            )
        telemetry["usage"] = _usage_delta(metrics_before, _metrics_snapshot(stores))
        return QueryResult(
            response=response,
            context_data=context_data,
            query=query,
            method=self.method_name,
            telemetry=telemetry,
        )
