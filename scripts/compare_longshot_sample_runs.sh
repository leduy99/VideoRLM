#!/usr/bin/env bash
set -euo pipefail

DEFAULT_RUN_A="output/longshot_small_pitome_hybrid_local"
if [[ ! -d "${DEFAULT_RUN_A}" ]]; then
  DEFAULT_RUN_A="output/longshot_small_local"
fi

CONDA_ENV="${CONDA_ENV:-videorlm}"
RUN_A="${RUN_A:-${DEFAULT_RUN_A}}"
RUN_B="${RUN_B:-output/longshot_small_original_local}"
LABEL_A="${LABEL_A:-pitome_hybrid}"
LABEL_B="${LABEL_B:-original}"
EVAL_ROOT="${EVAL_ROOT:-output/longshot_compare_pitome_hybrid_vs_original}"
JUDGE_REPO="${JUDGE_REPO:-Qwen/Qwen3-0.6B}"
JUDGE_DEVICE="${JUDGE_DEVICE:-mps}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"

PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 conda run --no-capture-output -n "${CONDA_ENV}" python -u scripts/compare_longshot_runs.py \
  --run-a "${RUN_A}" \
  --run-b "${RUN_B}" \
  --label-a "${LABEL_A}" \
  --label-b "${LABEL_B}" \
  --eval-root "${EVAL_ROOT}" \
  --judge-repo "${JUDGE_REPO}" \
  --judge-device "${JUDGE_DEVICE}" \
  --torch-dtype "${TORCH_DTYPE}"
