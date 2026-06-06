#!/usr/bin/env bash
set -euo pipefail

FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3}"
MAX_JOBS="${MAX_JOBS:-4}"
export MAX_JOBS

if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/nvcc" ]]; then
  export CUDA_HOME="${CUDA_HOME:-${CONDA_PREFIX}}"
fi

python - <<'PY'
import sys

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "flash-attn must be installed after torch is available. "
        "Activate the conda env created from environment_internvideo_rerank.yml first."
    ) from exc

print(f"python={sys.executable}", flush=True)
print(f"torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
print(f"cuda_available={torch.cuda.is_available()}", flush=True)
PY

if command -v nvcc >/dev/null 2>&1; then
  nvcc --version
else
  echo "WARNING: nvcc was not found. flash-attn may need a prebuilt wheel or CUDA compiler." >&2
fi

python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation
