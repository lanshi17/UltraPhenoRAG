"""Bootstrap the vendored LightRAG distribution used by the benchmark."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_light_rag_import_path() -> Path:
    """Put the repository's LightRAG checkout ahead of site-packages.

    The benchmark deliberately vendors LightRAG under ``baseline/libs`` so its
    results are reproducible even when a different LightRAG version is
    installed in the caller's environment.
    """

    package_root = Path(__file__).resolve().parents[1] / "libs" / "light_rag"
    if not (package_root / "lightrag" / "__init__.py").is_file():
        raise ImportError(f"Vendored LightRAG package not found at {package_root}")
    package_root_text = str(package_root)
    if package_root_text in sys.path:
        sys.path.remove(package_root_text)
    sys.path.insert(0, package_root_text)
    return package_root


__all__ = ["ensure_light_rag_import_path"]
