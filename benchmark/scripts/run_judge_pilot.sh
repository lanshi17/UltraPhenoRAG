#!/usr/bin/env bash
# run_judge_pilot.sh
# Small helper to run a 20%/limited Judge-mode pilot evaluation locally.
# Usage:
#   cp benchmark/.env.example benchmark/.env
#   # edit benchmark/.env to add JUDGE_API_KEY / JUDGE_API_BASE / JUDGE_COMPLETION_MODEL
#   ./benchmark/scripts/run_judge_pilot.sh [--limit N]
#
# Credentials are read from benchmark/.env by the benchmark itself
# (load_environment); exporting them here is optional.

set -euo pipefail
LIMIT=10
if [ "$#" -ge 1 ]; then
  LIMIT="$1"
fi

# Load the shared env file when present (exported vars still take precedence).
ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.env"
if [ -f "${ENV_FILE}" ]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

if [ -z "${JUDGE_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: JUDGE_API_KEY (or OPENAI_API_KEY) not set. Populate benchmark/.env" >&2
  exit 2
fi
if [ -z "${JUDGE_COMPLETION_MODEL:-}" ]; then
  echo "ERROR: JUDGE_COMPLETION_MODEL not set. Populate benchmark/.env before running." >&2
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
# Run evaluate on a deterministic subset using --limit.
# Model/key/base all fall back to JUDGE_* variables loaded from benchmark/.env.
python -m benchmark.baseline.microsoft_graphrag_client.benchmark evaluate \
  --judge-mode optional \
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
