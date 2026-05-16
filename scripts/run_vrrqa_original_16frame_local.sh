#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-output/vrrqa_original_16frame_local}"
ANNOTATIONS="${ANNOTATIONS:-data/vrrqa/ImplicitQAv0.1.2.jsonl}"
VIDEO_DIR="${VIDEO_DIR:-data/vrrqa/videos}"
SEGMENT_DIR="${SEGMENT_DIR:-data/vrrqa/segments}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-}"
VRRQA_CONDA_ENV="${VRRQA_CONDA_ENV:-videorlm}"
PRECHECK_ONLY="${PRECHECK_ONLY:-0}"

if [[ -n "${PYTHONEXECUTABLE:-}" || -n "${PYTHONHOME:-}" || -n "${PYTHONPATH:-}" ]]; then
  echo "[VRR-QA] clearing inherited Python path overrides before using conda env ${VRRQA_CONDA_ENV}" >&2
  unset PYTHONEXECUTABLE PYTHONHOME PYTHONPATH
fi

CONTROLLER_REPO="${CONTROLLER_REPO:-Qwen/Qwen3-0.6B}"
VISUAL_REPO="${VISUAL_REPO:-Qwen/Qwen3-VL-2B-Instruct}"
SPEECH_REPO="${SPEECH_REPO:-Qwen/Qwen3-ASR-0.6B}"
CONTROLLER_DEVICE="${CONTROLLER_DEVICE:-mps}"
VISUAL_DEVICE="${VISUAL_DEVICE:-mps}"
SPEECH_DEVICE="${SPEECH_DEVICE:-mps}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
FRAME_COUNT="${FRAME_COUNT:-16}"
FRAME_WIDTH="${FRAME_WIDTH:-768}"
MAX_STEPS="${MAX_STEPS:-8}"
SEARCH_TOP_K="${SEARCH_TOP_K:-5}"
DOWNLOAD_MISSING="${DOWNLOAD_MISSING:-0}"
SKIP_UNAVAILABLE_VIDEOS="${SKIP_UNAVAILABLE_VIDEOS:-1}"
NO_PROGRESS="${NO_PROGRESS:-0}"
FORCE_CHOICE_FINALIZER="${FORCE_CHOICE_FINALIZER:-1}"
COOKIES_FROM_BROWSER="${COOKIES_FROM_BROWSER:-}"

if [[ -n "${CONDA_ENV:-}" || -n "${PYTHON_BIN:-}" || -n "${VRRQA_PYTHON_BIN:-}" ]]; then
  echo "[VRR-QA] using conda env ${VRRQA_CONDA_ENV}; ignoring generic CONDA_ENV/PYTHON_BIN overrides" >&2
fi

cmd=(
  conda run --no-capture-output -n "${VRRQA_CONDA_ENV}"
  python -u scripts/run_vrrqa_local.py
  --strategy original
  --annotations "${ANNOTATIONS}"
  --output "${OUTPUT_ROOT}/results.jsonl"
  --video-dir "${VIDEO_DIR}"
  --segment-dir "${SEGMENT_DIR}"
  --artifacts-dir "${OUTPUT_ROOT}/artifacts"
  --memory-dir "${OUTPUT_ROOT}/memories"
  --trace-dir "${OUTPUT_ROOT}/traces"
  --controller-repo "${CONTROLLER_REPO}"
  --visual-repo "${VISUAL_REPO}"
  --speech-repo "${SPEECH_REPO}"
  --controller-device "${CONTROLLER_DEVICE}"
  --visual-device "${VISUAL_DEVICE}"
  --speech-device "${SPEECH_DEVICE}"
  --torch-dtype "${TORCH_DTYPE}"
  --frame-count "${FRAME_COUNT}"
  --frame-width "${FRAME_WIDTH}"
  --max-steps "${MAX_STEPS}"
  --search-top-k "${SEARCH_TOP_K}"
  --verbose
)

if [[ -n "${SAMPLE_LIMIT}" ]]; then
  cmd+=(--sample-limit "${SAMPLE_LIMIT}")
fi
if [[ "${DOWNLOAD_MISSING}" == "1" || "${DOWNLOAD_MISSING}" == "true" || "${DOWNLOAD_MISSING}" == "yes" ]]; then
  cmd+=(--download-missing)
fi
if [[ "${SKIP_UNAVAILABLE_VIDEOS}" == "1" || "${SKIP_UNAVAILABLE_VIDEOS}" == "true" || "${SKIP_UNAVAILABLE_VIDEOS}" == "yes" ]]; then
  cmd+=(--skip-unavailable-videos)
fi
if [[ "${NO_PROGRESS}" == "1" || "${NO_PROGRESS}" == "true" || "${NO_PROGRESS}" == "yes" ]]; then
  cmd+=(--no-progress)
fi
if [[ "${FORCE_CHOICE_FINALIZER}" == "0" || "${FORCE_CHOICE_FINALIZER}" == "false" || "${FORCE_CHOICE_FINALIZER}" == "no" ]]; then
  cmd+=(--disable-forced-choice-finalizer)
fi
if [[ -n "${COOKIES_FROM_BROWSER}" ]]; then
  cmd+=(--cookies-from-browser "${COOKIES_FROM_BROWSER}")
fi

VRRQA_CONDA_ENV_NAME="${VRRQA_CONDA_ENV}" conda run --no-capture-output -n "${VRRQA_CONDA_ENV}" python - <<'PY'
import os
import sys

print(f"[VRR-QA] python: {sys.executable}", flush=True)
try:
    import PIL
except ModuleNotFoundError as exc:
    conda_env = os.environ.get("VRRQA_CONDA_ENV_NAME", "videorlm")
    raise SystemExit(
        "Pillow is not importable from the VRR-QA Python above. "
        f"Install it with: conda install -n {conda_env} -y pillow"
    ) from exc
print(f"[VRR-QA] PIL: {PIL.__file__}", flush=True)
PY
if [[ "${PRECHECK_ONLY}" == "1" || "${PRECHECK_ONLY}" == "true" || "${PRECHECK_ONLY}" == "yes" ]]; then
  exit 0
fi

KMP_DUPLICATE_LIB_OK=TRUE PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 "${cmd[@]}"
