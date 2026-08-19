"""Scoring policy loading shared by all baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_SCORING_CONFIG = (
    Path(__file__).resolve().parents[2] / "qa" / "dataset" / "scoring_config.yaml"
)

DEFAULT_SOURCE_MATCH_MODE = "hybrid"


def load_scoring_options(path: Path | None = None) -> dict[str, Any]:
    """Read the source-match mode and equivalence groups from the scoring config.

    Missing files degrade to the built-in defaults so old result files stay
    parseable.
    """
    config_path = (path or DEFAULT_SCORING_CONFIG).resolve()
    if not config_path.is_file():
        return {
            "source_match_mode": DEFAULT_SOURCE_MATCH_MODE,
            "source_equivalence": {},
        }
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    retrieval = data.get("retrieval", {}) if isinstance(data, dict) else {}
    mode = str(retrieval.get("source_match_mode", DEFAULT_SOURCE_MATCH_MODE))
    equivalence = retrieval.get("source_equivalence", {})
    if not isinstance(equivalence, dict):
        equivalence = {}
    return {
        "source_match_mode": mode,
        "source_equivalence": {
            str(key): [str(item) for item in values]
            for key, values in equivalence.items()
            if isinstance(values, (list, tuple, set))
        },
    }
