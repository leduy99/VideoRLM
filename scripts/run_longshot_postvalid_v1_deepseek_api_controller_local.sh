#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${CONTROLLER_API_KEY:-}" && -z "${DEEPSEEK_API_KEY:-}" ]]; then
  export DEEPSEEK_API_KEY="${CONTROLLER_API_KEY}"
fi
unset CONTROLLER_API_KEY

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  cat >&2 <<'EOF'
Set your DeepSeek API key before running:
  export DEEPSEEK_API_KEY=sk-...

Then run:
  scripts/run_longshot_postvalid_v1_deepseek_api_controller_local.sh
EOF
  exit 2
fi

export OUTPUT_ROOT="${OUTPUT_ROOT:-output/longshot_postvalid_v1_deepseek_api_full_asr_hierarchical}"
export DATASET_PATH="${DATASET_PATH:-MBZUAI/longshot-bench}"
export DATASET_NAME="${DATASET_NAME:-postvalid_v2}"
export LONGSHOT_CONTEXT_DATASET_NAME="${LONGSHOT_CONTEXT_DATASET_NAME:-postvalid_v1}"
export SPLIT="${SPLIT:-test}"
export VIDEO_DIR="${VIDEO_DIR:-data/videos}"
export LAZY_SPEECH_REFINEMENT="${LAZY_SPEECH_REFINEMENT:-0}"
export ENABLE_CONTROLLER_EVIDENCE_CLASSIFIER="${ENABLE_CONTROLLER_EVIDENCE_CLASSIFIER:-0}"

export CONTROLLER_API_BASE_URL="${CONTROLLER_API_BASE_URL:-https://api.deepseek.com}"
export CONTROLLER_API_MODEL="${CONTROLLER_API_MODEL:-deepseek-v4-pro}"
export CONTROLLER_API_MAX_TOKENS="${CONTROLLER_API_MAX_TOKENS:-512}"
export CONTROLLER_API_TIMEOUT="${CONTROLLER_API_TIMEOUT:-600}"

# Default to 100 postvalid_v1 samples. Use SAMPLE_LIMIT= for the full split.
export SAMPLE_LIMIT="${SAMPLE_LIMIT-100}"
export RUN_INFERENCE="${RUN_INFERENCE:-1}"
export RUN_EVAL="${RUN_EVAL:-1}"
export ANSWER_ONLY_EVAL="${ANSWER_ONLY_EVAL:-0}"

exec scripts/run_longshot_postvalid_v1_eval_local.sh "$@"
