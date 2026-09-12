"""Bootstrap the vendored KAG distribution used by the benchmark."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_kag_import_path() -> Path:
    """Put the repository's KAG checkout ahead of site-packages.

    The benchmark deliberately vendors KAG (openspg-kag) under ``baseline/libs``
    so its results are reproducible even when a different KAG version is
    installed in the caller's environment.  The checkout ships both the ``kag``
    and ``knext`` packages, so its root must be importable.
    """

    package_root = Path(__file__).resolve().parents[1] / "libs" / "kag"
    if not (package_root / "kag" / "__init__.py").is_file():
        raise ImportError(f"Vendored KAG package not found at {package_root}")
    package_root_text = str(package_root)
    if package_root_text in sys.path:
        sys.path.remove(package_root_text)
    sys.path.insert(0, package_root_text)
    return package_root


__all__ = ["ensure_kag_import_path"]
