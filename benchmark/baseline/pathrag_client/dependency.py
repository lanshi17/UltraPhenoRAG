"""Bootstrap the vendored PathRAG distribution used by the benchmark."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_pathrag_import_path() -> Path:
    """Put the repository's PathRAG checkout ahead of site-packages.

    The benchmark deliberately vendors PathRAG under ``baseline/libs`` so its
    results are reproducible even when a different PathRAG version is
    installed in the caller's environment.
    """

    package_root = Path(__file__).resolve().parents[1] / "libs" / "path_rag"
    if not (package_root / "PathRAG" / "__init__.py").is_file():
        raise ImportError(f"Vendored PathRAG package not found at {package_root}")
    package_root_text = str(package_root)
    if package_root_text in sys.path:
        sys.path.remove(package_root_text)
    sys.path.insert(0, package_root_text)
    return package_root


__all__ = ["ensure_pathrag_import_path"]
