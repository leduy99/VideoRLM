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
  scripts/run_longshot_postvalid_tools_v1_deepseek_api_controller_local.sh
EOF
  exit 2
fi

export OUTPUT_ROOT="${OUTPUT_ROOT:-output/longshot_postvalid_tools_v1_deepseek_api_hierarchical}"
export CONTROLLER_API_BASE_URL="${CONTROLLER_API_BASE_URL:-https://api.deepseek.com}"
export CONTROLLER_API_MODEL="${CONTROLLER_API_MODEL:-deepseek-v4-pro}"
export CONTROLLER_API_MAX_TOKENS="${CONTROLLER_API_MAX_TOKENS:-1024}"
export CONTROLLER_API_TIMEOUT="${CONTROLLER_API_TIMEOUT:-600}"
export SAMPLE_LIMIT="${SAMPLE_LIMIT:-20}"
export RUN_INFERENCE="${RUN_INFERENCE:-1}"
export RUN_EVAL="${RUN_EVAL:-1}"
export ANSWER_ONLY_EVAL="${ANSWER_ONLY_EVAL:-1}"

exec scripts/run_longshot_postvalid_tools_v1_eval_local.sh "$@"
