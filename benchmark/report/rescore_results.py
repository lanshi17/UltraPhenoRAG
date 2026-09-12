"""Re-score saved benchmark evaluations with the current shared scorer.

Re-runs :func:`score_question` against the answers, retrieved contexts, and
judge outputs already persisted in a finished evaluation's JSON/JSONL
artifacts, then recomputes the dataset summary.  Nothing is re-queried and no
model is called: answers and judge verdicts are frozen, so several baselines
can be compared under identical scoring rules after the shared scorer (for
example the referral-phrase list in ``benchmark/common/answers.py``) changes.

Usage::

    uv run python -m benchmark.report.rescore_results \\
        benchmark/results/kag/evaluation-*.json

Each input produces ``<stem>-rescored.json`` plus a matching ``.jsonl`` next
to the original; the original files are left untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from benchmark.common import aggregate_usage, safety_actions, supported_statements
from benchmark.common.scoring_options import load_scoring_options
from benchmark.qa import (
    DatasetScoringReport,
    load_questions,
    score_question,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCORING_CONFIG = (
    REPOSITORY_ROOT / "benchmark" / "qa" / "dataset" / "scoring_config.yaml"
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _judge_result_for(row: dict[str, Any]) -> dict[str, Any] | None:
    """Return the persisted judge outcome, mirroring the original run.

    ``evaluate`` skips the judge entirely for questions whose query failed;
    that state is preserved by returning ``None`` exactly for those rows.
    """

    if row.get("error") is not None:
        return None
    return (row.get("scoring") or {}).get("judge")


def rescore_file(report_path: Path) -> dict[str, Any]:
    """Re-score one evaluation artifact in place beside the original."""

    report_path = report_path.resolve()
    raw_path = report_path.with_suffix(".jsonl")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = _load_jsonl(raw_path)
    metadata = report.get("metadata", {})

    dataset_path = Path(metadata["dataset_path"])
    questions = {q.question_id: q for q in load_questions(dataset_path)}
    if set(questions) != {row["question_id"] for row in rows}:
        raise ValueError(f"dataset does not match saved rows: {dataset_path}")

    scoring_config_path = Path(metadata.get("scoring_config_path") or "")
    if not scoring_config_path.is_file():
        scoring_config_path = DEFAULT_SCORING_CONFIG
    source_equivalence = load_scoring_options(scoring_config_path)[
        "source_equivalence"
    ]
    k = metadata.get("k")
    source_match_mode = metadata.get("source_match_mode", "hybrid")

    scoring = DatasetScoringReport()
    raw_results: list[dict[str, Any]] = []
    for row in rows:
        question = questions[row["question_id"]]
        answer = row.get("answer") or ""
        contexts = row.get("contexts") or []
        refused, referred = safety_actions(answer)
        result = score_question(
            question=question,
            answer=answer,
            retrieved_sources=list(row.get("retrieved_sources") or []),
            retrieved_context="\n\n".join(item["text"] for item in contexts),
            retrieved_supports=[
                supported_statements(item["text"], question.must_have_statements)
                for item in contexts
            ],
            refused=refused,
            referred=referred,
            k=k,
            source_match_mode=source_match_mode,
            source_equivalence=source_equivalence,
            judge_result=_judge_result_for(row),
            safety_verdict=_safety_verdict_for(row),
        )
        scoring.add(result)
        row["refused"] = refused
        row["referred"] = referred
        row["scoring"] = asdict(result)
        raw_results.append(row)

    summary = scoring.summary()
    summary["usage"] = aggregate_usage(raw_results)
    report["summary"] = summary
    metadata["scoring_method"] = summary["scoring"]["selected_method"]
    metadata["rescored"] = {
        "source_report": str(report_path),
        "note": (
            "re-scored from persisted answers/contexts/judge outputs with the "
            "current shared scorer; no model was re-queried"
        ),
    }
    report["results"] = raw_results

    rescored_json = report_path.with_name(
        report_path.stem + "-rescored" + report_path.suffix
    )
    rescored_jsonl = rescored_json.with_suffix(".jsonl")
    rescored_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with rescored_jsonl.open("w", encoding="utf-8") as stream:
        for row in raw_results:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "file": str(rescored_json),
        "n_questions": summary["n_questions"],
        "safety_score": summary["safety"]["safety_score"],
        "final_score": summary["final_score"],
        "scoring_method": summary["scoring"]["selected_method"],
    }


def _safety_verdict_for(row: dict[str, Any]) -> dict[str, Any] | None:
    """Return the persisted judge safety verdict, if this artifact has one.

    Older artifacts predate judge-based safety verdicts; returning ``None``
    keeps them scoreable via the lexicon fallback.
    """

    return row.get("safety_verdict")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-score saved evaluations with the current shared scorer"
    )
    parser.add_argument("reports", nargs="+", type=Path, help="evaluation JSON files")
    args = parser.parse_args(argv)
    try:
        for summary in (rescore_file(path) for path in args.reports):
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"rescore failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

