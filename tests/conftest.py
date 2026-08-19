"""Shared pytest fixtures for the benchmark test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 仓库根目录加入 sys.path，使 `benchmark.*` 在未安装为包时可导入。
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

# vendored LightRAG 需要在导入 benchmark.baseline.light_rag_client 前就位。
_LIGHTRAG_ROOT = REPOSITORY_ROOT / "benchmark" / "baseline" / "libs" / "light_rag"
if (_LIGHTRAG_ROOT / "lightrag" / "__init__.py").is_file():
    if str(_LIGHTRAG_ROOT) not in sys.path:
        sys.path.insert(0, str(_LIGHTRAG_ROOT))


@pytest.fixture()
def light_rag_root() -> Path:
    """Return the vendored LightRAG checkout path."""
    return _LIGHTRAG_ROOT
