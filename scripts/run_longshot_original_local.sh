#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-output/longshot_small_original_local}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-20}"
FRAME_COUNT="${FRAME_COUNT:-3}"
FRAME_WIDTH="${FRAME_WIDTH:-768}"
CONTROLLER_DEVICE="${CONTROLLER_DEVICE:-mps}"
VISUAL_DEVICE="${VISUAL_DEVICE:-mps}"
SPEECH_DEVICE="${SPEECH_DEVICE:-mps}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
SPEECH_CHUNK_DURATION_SECONDS="${SPEECH_CHUNK_DURATION_SECONDS:-60}"
SPEECH_MAX_NEW_TOKENS="${SPEECH_MAX_NEW_TOKENS:-512}"
SKIP_SPEECH_RECOGNITION="${SKIP_SPEECH_RECOGNITION:-0}"
MEMORY_CACHE_ONLY="${MEMORY_CACHE_ONLY:-0}"

MEMORY_CACHE_ONLY_ARG=""
if [[ "${MEMORY_CACHE_ONLY}" == "1" || "${MEMORY_CACHE_ONLY}" == "true" || "${MEMORY_CACHE_ONLY}" == "yes" ]]; then
  MEMORY_CACHE_ONLY_ARG="--memory-cache-only"
fi
SKIP_SPEECH_RECOGNITION_ARG=""
if [[ "${SKIP_SPEECH_RECOGNITION}" == "1" || "${SKIP_SPEECH_RECOGNITION}" == "true" || "${SKIP_SPEECH_RECOGNITION}" == "yes" ]]; then
  SKIP_SPEECH_RECOGNITION_ARG="--no-speech-recognition"
fi

PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 conda run --no-capture-output -n videorlm python -u -m rlm.video.cli run-longshot-local \
  --dataset-path MBZUAI/longshot-bench \
  --dataset-name postvalid_v1 \
  --split test \
  --output "${OUTPUT_ROOT}/results.jsonl" \
  --video-dir data/longshotbench/videos \
  --skip-unavailable-videos \
  ${MEMORY_CACHE_ONLY_ARG} \
  --artifacts-dir "${OUTPUT_ROOT}/artifacts" \
  --memory-dir "${OUTPUT_ROOT}/memories" \
  --trace-dir "${OUTPUT_ROOT}/traces" \
  --sample-limit "${SAMPLE_LIMIT}" \
  --controller-repo Qwen/Qwen3-0.6B \
  --visual-repo Qwen/Qwen3-VL-2B-Instruct \
  --speech-repo Qwen/Qwen3-ASR-0.6B \
  --no-forced-aligner \
  --controller-device "${CONTROLLER_DEVICE}" \
  --visual-device "${VISUAL_DEVICE}" \
  --speech-device "${SPEECH_DEVICE}" \
  --torch-dtype "${TORCH_DTYPE}" \
  ${SKIP_SPEECH_RECOGNITION_ARG} \
  --speech-chunk-duration-seconds "${SPEECH_CHUNK_DURATION_SECONDS}" \
  --speech-max-new-tokens "${SPEECH_MAX_NEW_TOKENS}" \
  --frame-count "${FRAME_COUNT}" \
  --frame-width "${FRAME_WIDTH}" \
  --verbose \
  --clip-duration-seconds 60
