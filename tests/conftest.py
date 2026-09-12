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

# vendored KAG 需要在导入 benchmark.baseline.kag_client 前就位。
_KAG_ROOT = REPOSITORY_ROOT / "benchmark" / "baseline" / "libs" / "kag"
if (_KAG_ROOT / "kag" / "__init__.py").is_file():
    if str(_KAG_ROOT) not in sys.path:
        sys.path.insert(0, str(_KAG_ROOT))

# vendored PathRAG 需要在导入 benchmark.baseline.pathrag_client 前就位。
_PATHRAG_ROOT = REPOSITORY_ROOT / "benchmark" / "baseline" / "libs" / "path_rag"
if (_PATHRAG_ROOT / "PathRAG" / "__init__.py").is_file():
    if str(_PATHRAG_ROOT) not in sys.path:
        sys.path.insert(0, str(_PATHRAG_ROOT))


@pytest.fixture()
def kag_root() -> Path:
    """Return the vendored KAG checkout path."""
    return _KAG_ROOT


@pytest.fixture()
def light_rag_root() -> Path:
    """Return the vendored LightRAG checkout path."""
    return _LIGHTRAG_ROOT


@pytest.fixture()
def path_rag_root() -> Path:
    """Return the vendored PathRAG checkout path."""
    return _PATHRAG_ROOT
