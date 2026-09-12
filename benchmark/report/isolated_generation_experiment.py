"""Generation-isolated retrieval quality experiment.

Feeds each framework's frozen retrieval contexts to one identical
generator (gpt-5-mini) with one fixed evidence-citation prompt, then
scores the answers with the shared gpt-5 judge. Differences in
faithfulness therefore isolate retrieval-column quality, removing each
pipeline's own generation stack from the comparison.

Usage:
    .venv/bin/python benchmark/report/isolated_generation_experiment.py
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from benchmark.config import load_environment
from benchmark.qa import JudgeConfig, judge_answer, load_questions

RUNS = {
    "graphrag": Path(
        "benchmark/results/microsoft_graphrag/evaluation-20260821-judge-rerun.jsonl"
    ),
    "light_rag": Path(
        "benchmark/results/light_rag/evaluation-20260821T142218Z.jsonl"
    ),
    # KAG production form: run2 for basic/naive + reranked solver rows
    # (deduped, relevance-ranked) for graph-enhanced questions.
    "kag": Path("benchmark/results/kag/evaluation-20260908T032034Z.jsonl"),
}
KAG_SOLVER_OVERRIDE = Path(
    "benchmark/results/kag/evaluation-20260908T033139Z.jsonl"
)
OUT = Path("benchmark/results/isolated_generation_20260908.json")

SYSTEM = (
    "You are a clinical reference assistant for prenatal ultrasound. "
    "Answer the question using ONLY the numbered evidence provided. "
    "Cite evidence inline as [1], [2] after each claim. If the evidence "
    "does not contain the answer, say so explicitly instead of guessing."
)


def _generate(question: str, contexts: list[dict]) -> tuple[str, str]:
    load_environment()
    import litellm
    import os

    model = os.getenv("RAG_COMPLETION_MODEL", "gpt-5-mini")
    api_key = os.getenv("RAG_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    api_base = os.getenv("RAG_API_BASE") or os.getenv("OPENAI_API_BASE", "")

    evidence = []
    for i, c in enumerate(contexts, start=1):
        text = str(c.get("text") or c.get("content") or "").strip()
        if text:
            evidence.append(f"[{i}] {text}")
    if not evidence:
        return "", "no evidence"

    kwargs = {
        "model": model,
        "api_key": api_key,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\nEvidence:\n"
                    + "\n\n".join(evidence)
                ),
            },
        ],
    }
    if not model.startswith("gpt-5"):
        kwargs["temperature"] = 0
    if api_base:
        kwargs["api_base"] = api_base
    response = litellm.completion(**kwargs)
    content = response.choices[0].message.content or ""
    return str(content).strip(), ""


def main() -> None:
    questions = {q.question_id: q for q in load_questions(
        "benchmark/qa/dataset/sample_questions.json")}
    judge_config = JudgeConfig()

    kag_rows = {json.loads(line)["question_id"]: json.loads(line)
                for line in KAG_SOLVER_OVERRIDE.open(encoding="utf-8") if line.strip()}
    results: dict[str, list[dict]] = {}

    for name, path in RUNS.items():
        rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        if name == "kag":
            merged = {r["question_id"]: r for r in rows}
            merged.update(kag_rows)
            rows = [merged[qid] for qid in questions]
        jobs = []
        for r in rows:
            q = questions[r["question_id"]]
            contexts = r.get("contexts") or []
            jobs.append((r["question_id"], q, contexts))
        print(f"[{name}] {len(jobs)} questions", flush=True)

        def work(job):
            qid, q, contexts = job
            answer, err = _generate(q.question, contexts)
            if err:
                return {"question_id": qid, "error": err,
                        "faithfulness": None, "answer_correctness": None}
            context_text = "\n\n".join(
                str(c.get("text") or c.get("content") or "") for c in contexts
            )
            verdict = judge_answer(
                question=q.question,
                answer=answer,
                context=context_text,
                gold_answer=q.gold_answer,
                must_have_statements=q.must_have_statements,
                config=judge_config,
            )
            scores = verdict.get("scores") or {}
            return {
                "question_id": qid,
                "error": verdict.get("error"),
                "faithfulness": scores.get("faithfulness"),
                "answer_correctness": scores.get("answer_correctness"),
            }

        with ThreadPoolExecutor(max_workers=6) as pool:
            outcomes = list(pool.map(work, jobs))
        results[name] = outcomes
        valid = [o for o in outcomes if o["faithfulness"] is not None]
        if valid:
            f = sum(o["faithfulness"] for o in valid) / len(valid)
            c = sum(o["answer_correctness"] for o in valid) / len(valid)
            print(f"[{name}] faithfulness={f:.3f} correctness={c:.3f} "
                  f"({len(valid)}/{len(outcomes)} judged)", flush=True)

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print("written:", OUT, flush=True)


if __name__ == "__main__":
    started = time.monotonic()
    main()
    print(f"elapsed: {time.monotonic() - started:.0f}s")
