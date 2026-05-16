#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-output/vrrqa_original_16frame_local}"
ANNOTATIONS="${ANNOTATIONS:-data/vrrqa/ImplicitQAv0.1.2.jsonl}"
VIDEO_DIR="${VIDEO_DIR:-data/vrrqa/videos}"
SEGMENT_DIR="${SEGMENT_DIR:-data/vrrqa/segments}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-}"

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
COOKIES_FROM_BROWSER="${COOKIES_FROM_BROWSER:-}"

cmd=(
  conda run --no-capture-output -n videorlm
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
if [[ -n "${COOKIES_FROM_BROWSER}" ]]; then
  cmd+=(--cookies-from-browser "${COOKIES_FROM_BROWSER}")
fi

KMP_DUPLICATE_LIB_OK=TRUE PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 "${cmd[@]}"
