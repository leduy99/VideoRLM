#!/usr/bin/env bash
set -euo pipefail

SAMPLE_ID_FILE="${SAMPLE_ID_FILE:-data/selected_sample_ids.txt}"

if [[ ! -f "${SAMPLE_ID_FILE}" ]]; then
  echo "Sample id file not found: ${SAMPLE_ID_FILE}" >&2
  exit 2
fi

sample_ids="$(
  python - "${SAMPLE_ID_FILE}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
ids = []
for line in path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    ids.extend(part.strip() for part in line.replace(",", " ").split() if part.strip())
print(",".join(dict.fromkeys(ids)))
PY
)"

if [[ -z "${sample_ids}" ]]; then
  echo "No sample ids found in ${SAMPLE_ID_FILE}" >&2
  exit 2
fi

# export SAMPLE_ID="${SAMPLE_ID:-${sample_ids}}"
export DATASET_PATH="${DATASET_PATH:-MBZUAI/longshot-bench}"
export DATASET_NAME="${DATASET_NAME:-postvalid_v2}"
export LONGSHOT_CONTEXT_DATASET_NAME="${LONGSHOT_CONTEXT_DATASET_NAME:-postvalid_v1}"
export SPLIT="${SPLIT:-test}"
export VIDEO_DIR="${VIDEO_DIR:-data/videos}"
export SAMPLE_LIMIT="${SAMPLE_LIMIT:-}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-output/finer_test_DP_qwen_whisper}"
export LAZY_SPEECH_REFINEMENT="${LAZY_SPEECH_REFINEMENT:-0}"
export ENABLE_TARGETED_ASR_REFINEMENT="${ENABLE_TARGETED_ASR_REFINEMENT:-1}"
export ENABLE_CONTROLLER_EVIDENCE_CLASSIFIER="${ENABLE_CONTROLLER_EVIDENCE_CLASSIFIER:-0}"
export RUN_INFERENCE="${RUN_INFERENCE:-1}"
export RUN_EVAL="${RUN_EVAL:-1}"
export ANSWER_ONLY_EVAL="${ANSWER_ONLY_EVAL:-0}"

# export CONTROLLER_API_BASE_URL="${CONTROLLER_API_BASE_URL:-https://api.deepseek.com}"
# export CONTROLLER_API_MODEL="${CONTROLLER_API_MODEL:-deepseek-v4-pro}"
# export CONTROLLER_API_MAX_TOKENS="${CONTROLLER_API_MAX_TOKENS:-512}"
# export CONTROLLER_API_TIMEOUT="${CONTROLLER_API_TIMEOUT:-600}"

exec scripts/run_longshot_postvalid_tools_v1_eval_local.sh "$@"
