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

PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 conda run --no-capture-output -n videorlm python -u -m rlm.video.cli run-longshot-local \
  --dataset-path MBZUAI/longshot-bench \
  --dataset-name postvalid_tools_v1 \
  --split test \
  --output "${OUTPUT_ROOT}/results.jsonl" \
  --video-dir data/longshotbench/videos \
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
  --frame-count "${FRAME_COUNT}" \
  --frame-width "${FRAME_WIDTH}" \
  --verbose \
  --clip-duration-seconds 60
