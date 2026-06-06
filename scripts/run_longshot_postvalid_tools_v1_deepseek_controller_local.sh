#!/usr/bin/env bash
set -euo pipefail

DEEPSEEK_CONTROLLER_REPO="${DEEPSEEK_CONTROLLER_REPO:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
DEEPSEEK_CONTROLLER_LOCAL_DIR="${DEEPSEEK_CONTROLLER_LOCAL_DIR:-output/models/deepseek-ai__DeepSeek-R1-Distill-Qwen-7B}"
ALLOW_HF_DOWNLOAD="${ALLOW_HF_DOWNLOAD:-0}"

is_true() {
  case "${1,,}" in
    1 | true | yes | on) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ -z "${CONTROLLER_MODEL_PATH:-}" ]]; then
  if [[ -d "${DEEPSEEK_CONTROLLER_LOCAL_DIR}" ]]; then
    export CONTROLLER_MODEL_PATH="${DEEPSEEK_CONTROLLER_LOCAL_DIR}"
  elif is_true "${ALLOW_HF_DOWNLOAD}"; then
    export CONTROLLER_MODEL_PATH="${DEEPSEEK_CONTROLLER_REPO}"
  else
    cat >&2 <<EOF
DeepSeek controller model was not found at:
  ${DEEPSEEK_CONTROLLER_LOCAL_DIR}

Download it there first, or rerun with:
  ALLOW_HF_DOWNLOAD=1 scripts/run_longshot_postvalid_tools_v1_deepseek_controller_local.sh

For a smaller controller, set for example:
  DEEPSEEK_CONTROLLER_REPO=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  DEEPSEEK_CONTROLLER_LOCAL_DIR=output/models/deepseek-ai__DeepSeek-R1-Distill-Qwen-1.5B
EOF
    exit 2
  fi
fi

export OUTPUT_ROOT="${OUTPUT_ROOT:-output/longshot_postvalid_tools_v1_deepseek_controller}"
export CONTROLLER_REPO="${CONTROLLER_REPO:-${DEEPSEEK_CONTROLLER_REPO}}"
export CONTROLLER_MAX_NEW_TOKENS="${CONTROLLER_MAX_NEW_TOKENS:-768}"
export CONTROLLER_TRUST_REMOTE_CODE="${CONTROLLER_TRUST_REMOTE_CODE:-0}"
export TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
export SAMPLE_LIMIT="${SAMPLE_LIMIT:-20}"
export RUN_INFERENCE="${RUN_INFERENCE:-1}"
export RUN_EVAL="${RUN_EVAL:-1}"
export ANSWER_ONLY_EVAL="${ANSWER_ONLY_EVAL:-1}"

exec scripts/run_longshot_postvalid_tools_v1_eval_local.sh "$@"
