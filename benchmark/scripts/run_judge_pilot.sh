#!/usr/bin/env bash
# run_judge_pilot.sh
# Small helper to run a 20%/limited Judge-mode pilot evaluation locally.
# Usage:
#   cp .env.example .env
#   # edit .env to add JUDGE_API_KEY and JUDGE_COMPLETION_MODEL
#   source .env
#   ./benchmark/scripts/run_judge_pilot.sh [--limit N]

set -euo pipefail
LIMIT=10
if [ "$#" -ge 1 ]; then
  LIMIT="$1"
fi

if [ -z "${JUDGE_API_KEY:-}" ]; then
  echo "ERROR: JUDGE_API_KEY not set. Populate .env or export JUDGE_API_KEY before running." >&2
  exit 2
fi
if [ -z "${JUDGE_COMPLETION_MODEL:-}" ]; then
  echo "ERROR: JUDGE_COMPLETION_MODEL not set. Populate .env or export JUDGE_COMPLETION_MODEL before running." >&2
  exit 2
fi

# Ensure no proxy to avoid socks/httpx issues
unset ALL_PROXY
unset all_proxy
unset HTTP_PROXY
unset http_proxy
unset HTTPS_PROXY
unset https_proxy

echo "Running Judge pilot: limit=${LIMIT}, judge_model=${JUDGE_COMPLETION_MODEL}"
# Run evaluate on a deterministic subset using --limit
python -m benchmark.baseline.microsoft_graphrag_client.benchmark evaluate \
  --judge-mode optional \
  --judge-model "${JUDGE_COMPLETION_MODEL}" \
  --k 16 \
  --limit "${LIMIT}" \
  --fail-fast false

RC=$?
if [ $RC -eq 0 ]; then
  echo "Judge pilot finished successfully. Check benchmark/report and benchmark/data/.../results for new outputs." 
else
  echo "Judge pilot exited with code $RC" >&2
fi
exit $RC
