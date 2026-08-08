# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""数据加载器。

负责从索引输出目录读取 parquet 文件并转为 DataFrame。
"""

from __future__ import annotations

import logging

import pandas as pd
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.data_reader import DataReader
from graphrag_storage import create_storage
from graphrag_storage.tables.table_provider_factory import create_table_provider

from benchmark.baseline.microsoft_graphrag_client.utils.async_runner import AsyncRunner

logger = logging.getLogger(__name__)

# 各搜索方法所需的输出文件
REQUIRED_FILES: dict[str, list[str]] = {
    "global": ["entities", "communities", "community_reports"],
    "local": [
        "communities",
        "community_reports",
        "text_units",
        "relationships",
        "entities",
    ],
    "drift": [
        "communities",
        "community_reports",
        "text_units",
        "relationships",
        "entities",
    ],
    "basic": ["text_units"],
}

OPTIONAL_FILES: dict[str, list[str]] = {
    "local": ["covariates"],
}


class DataLoader:
    """索引输出数据加载器。

    从 GraphRAG 索引输出目录读取 parquet 文件，返回带有正确类型的 DataFrame。

    Parameters
    ----------
    config : GraphRagConfig
        GraphRAG 配置实例。
    """

    def __init__(self, config: GraphRagConfig) -> None:
        self._config = config

    def load(
        self,
        output_list: list[str],
        optional_list: list[str] | None = None,
    ) -> dict[str, pd.DataFrame | None]:
        """读取索引输出 parquet 文件为 DataFrame 字典。

        Parameters
        ----------
        output_list : list[str]
            必须加载的表名列表 (如 ``["entities", "communities"]``)。
        optional_list : list[str] | None, optional
            可选加载的表名列表，不存在时对应值为 ``None``。

        Returns
        -------
        dict[str, pd.DataFrame | None]
            表名到 DataFrame 的映射。
        """
        storage_obj = create_storage(self._config.output_storage)
        table_provider = create_table_provider(
            self._config.table_provider, storage=storage_obj
        )
        reader = DataReader(table_provider)

        dataframe_dict: dict[str, pd.DataFrame | None] = {}

        for name in output_list:
            logger.debug("加载表: %s", name)
            df_value = AsyncRunner.run(getattr(reader, name)())
            dataframe_dict[name] = df_value

        if optional_list:
            for optional_file in optional_list:
                file_exists = AsyncRunner.run(table_provider.has(optional_file))
                if file_exists:
                    df_value = AsyncRunner.run(getattr(reader, optional_file)())
                    dataframe_dict[optional_file] = df_value
                else:
                    logger.debug("可选表不存在: %s", optional_file)
                    dataframe_dict[optional_file] = None

        return dataframe_dict

    def load_for_method(self, method: str) -> dict[str, pd.DataFrame | None]:
        """根据搜索方法自动加载所需的输出文件。

        Parameters
        ----------
        method : str
            搜索方法 (``"global"`` / ``"local"`` / ``"drift"`` / ``"basic"``)。

        Returns
        -------
        dict[str, pd.DataFrame | None]
            表名到 DataFrame 的映射。
        """
        output_list = REQUIRED_FILES.get(method, [])
        optional_list = OPTIONAL_FILES.get(method)
        return self.load(
            output_list=output_list,
            optional_list=optional_list,
        )

    @staticmethod
    def get_required_files(method: str) -> list[str]:
        """获取指定搜索方法所需的文件列表。"""
        return REQUIRED_FILES.get(method, [])

    @staticmethod
    def get_optional_files(method: str) -> list[str]:
        """获取指定搜索方法的可选文件列表。"""
        return OPTIONAL_FILES.get(method, [])
