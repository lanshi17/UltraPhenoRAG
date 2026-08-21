"""Re-run Judge scoring from an existing raw evaluation JSONL."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from benchmark.common.answers import safety_actions, supported_statements
from benchmark.common.scoring_options import load_scoring_options
from benchmark.common.usage import aggregate_usage, merge_usage
from benchmark.qa.build_dataset import load_questions
from benchmark.qa.judge import JudgeConfig, judge_answer
from benchmark.qa.scoring import DatasetScoringReport, score_question


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--judge-model", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line]
    questions = {item.question_id: item for item in load_questions(args.dataset)}
    options = load_scoring_options()
    judge_config = JudgeConfig(model=args.judge_model)
    report = DatasetScoringReport()

    for row in rows:
        question = questions[row["question_id"]]
        contexts = row.get("contexts", [])
        retrieved_sources = [item["source_id"] for item in contexts]
        retrieved_context = "\n\n".join(item["text"] for item in contexts)
        retrieved_supports = [
            supported_statements(item["text"], question.must_have_statements)
            for item in contexts
        ]
        refused, referred = safety_actions(row["answer"])
        judge_result = judge_answer(
            question=question.question,
            answer=row["answer"],
            context=retrieved_context,
            gold_answer=question.gold_answer,
            must_have_statements=question.must_have_statements,
            config=judge_config,
        )
        scoring = score_question(
            question=question,
            answer=row["answer"],
            retrieved_sources=retrieved_sources,
            retrieved_context=retrieved_context,
            retrieved_supports=retrieved_supports,
            refused=refused,
            referred=referred,
            k=16,
            source_match_mode=options["source_match_mode"],
            source_equivalence=options["source_equivalence"],
            judge_result=judge_result,
        )
        report.add(scoring)
        query_usage = row.get("usage", {}).get("query", {})
        judge_usage = judge_result.get("usage", {})
        row["refused"] = refused
        row["referred"] = referred
        row["usage"] = {
            "query": query_usage,
            "judge": judge_usage,
            "total": merge_usage(query_usage, judge_usage),
        }
        row["scoring"] = asdict(scoring)
        print(f"[{len(report.results)}/{len(rows)}] {row['question_id']}", flush=True)

    original = json.loads(args.input.with_suffix(".json").read_text())
    original["results"] = rows
    original["summary"] = report.summary()
    original["summary"]["usage"] = aggregate_usage(rows)
    original["metadata"]["judge_mode"] = "optional"
    original["metadata"]["judge_model"] = judge_config.resolve_model()
    original["metadata"]["scoring_method"] = original["summary"]["scoring"][
        "selected_method"
    ]
    original["metadata"]["raw_results_path"] = str(args.output.with_suffix(".jsonl"))
    args.output.write_text(json.dumps(original, ensure_ascii=False, indent=2))
    args.output.with_suffix(".jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    print(json.dumps(original["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
