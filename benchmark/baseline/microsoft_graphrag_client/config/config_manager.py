# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""配置管理器。

负责加载和缓存 GraphRagConfig。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from graphrag.config.load_config import load_config
from graphrag.config.models.graph_rag_config import GraphRagConfig

logger = logging.getLogger(__name__)


class ConfigManager:
    """GraphRAG 配置管理器。

    从项目根目录的 ``settings.yaml`` 加载配置，支持 data_dir 覆盖输出路径，
    并缓存配置实例避免重复加载。

    Parameters
    ----------
    root_dir : str | Path
        GraphRAG 项目根目录（包含 ``settings.yaml``）。
    data_dir : str | Path | None, optional
        索引输出目录覆盖。为 ``None`` 时使用配置中的默认值。
    """

    def __init__(
        self,
        root_dir: str | Path,
        data_dir: str | Path | None = None,
    ) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.data_dir = Path(data_dir).resolve() if data_dir else None
        self._config: GraphRagConfig | None = None

    @property
    def config(self) -> GraphRagConfig:
        """延迟加载并缓存的 GraphRagConfig。首次访问时从磁盘加载。"""
        if self._config is None:
            self._config = self._load()
        return self._config

    def reload(self) -> GraphRagConfig:
        """强制重新加载配置。"""
        self._config = self._load()
        return self._config

    def _load(self) -> GraphRagConfig:
        """从 root_dir 加载配置，若设置了 data_dir 则覆盖输出路径。"""
        cli_overrides: dict[str, Any] = {}
        if self.data_dir:
            cli_overrides["output_storage"] = {"base_dir": str(self.data_dir)}

        logger.debug(
            "加载配置: root_dir=%s, data_dir=%s, overrides=%s",
            self.root_dir,
            self.data_dir,
            cli_overrides,
        )
        return load_config(
            root_dir=str(self.root_dir),
            cli_overrides=cli_overrides or None,
        )
