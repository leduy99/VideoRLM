#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-videorlm}"
CONDA_BIN="${CONDA_BIN:-${CONDA_EXE:-}}"
if [[ -z "${CONDA_BIN}" ]] && command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
fi
if [[ -z "${CONDA_BIN}" ]]; then
  for candidate in \
    "${HOME}/miniforge3/bin/conda" \
    "${HOME}/miniconda3/bin/conda" \
    "${HOME}/anaconda3/bin/conda" \
    /opt/miniforge3/bin/conda \
    /opt/conda/bin/conda; do
    if [[ -x "${candidate}" ]]; then
      CONDA_BIN="${candidate}"
      break
    fi
  done
fi
if [[ -z "${CONDA_BIN}" || ! -x "${CONDA_BIN}" ]]; then
  echo "Could not find conda. Set CONDA_BIN=/path/to/conda." >&2
  exit 127
fi

REPO="${REPO:-OpenGVLab/InternVideo2-Stage2_6B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/models}"

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python -u scripts/download_hf_model_to_output.py \
  --repo "${REPO}" \
  --output-root "${OUTPUT_ROOT}" \
  "$@"
