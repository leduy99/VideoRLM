#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-output/longshot_lazy_pitome_refinement_local_faiss}"
DATASET_PATH="${DATASET_PATH:-MBZUAI/longshot-bench}"
DATASET_NAME="${DATASET_NAME:-postvalid_v1}"
SPLIT="${SPLIT:-test}"
VIDEO_DIR="${VIDEO_DIR:-data/longshotbench/videos}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-20}"
HISTORY_MODE="${HISTORY_MODE:-gold}"

CONTROLLER_REPO="${CONTROLLER_REPO:-Qwen/Qwen3-0.6B}"
VISUAL_REPO="${VISUAL_REPO:-Qwen/Qwen3-VL-2B-Instruct}"
SPEECH_REPO="${SPEECH_REPO:-Qwen/Qwen3-ASR-0.6B}"
CONTROLLER_DEVICE="${CONTROLLER_DEVICE:-mps}"
VISUAL_DEVICE="${VISUAL_DEVICE:-mps}"
SPEECH_DEVICE="${SPEECH_DEVICE:-mps}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"

FRAME_COUNT="${FRAME_COUNT:-3}"
FRAME_WIDTH="${FRAME_WIDTH:-768}"
CLIP_DURATION_SECONDS="${CLIP_DURATION_SECONDS:-480}"
PITOME_DENSE_FRAME_RATE="${PITOME_DENSE_FRAME_RATE:-0.2}"
PITOME_MIN_FRAME_COUNT="${PITOME_MIN_FRAME_COUNT:-16}"
PITOME_EMBEDDING_BACKEND="${PITOME_EMBEDDING_BACKEND:-hybrid}"
PITOME_EMBEDDING_SIZE="${PITOME_EMBEDDING_SIZE:-32}"
PITOME_ANCHOR_FRAME_COUNT="${PITOME_ANCHOR_FRAME_COUNT:-8}"
PITOME_MAX_SELECTED_FRAMES="${PITOME_MAX_SELECTED_FRAMES:-8}"
PITOME_SCENE_THRESHOLD="${PITOME_SCENE_THRESHOLD:-0.35}"
PITOME_MAX_SCENE_BOUNDARY_FRAMES="${PITOME_MAX_SCENE_BOUNDARY_FRAMES:-6}"

SEMANTIC_FRAME_EMBEDDING_REPO="${SEMANTIC_FRAME_EMBEDDING_REPO:-google/siglip-base-patch16-224}"
SEMANTIC_FRAME_EMBEDDING_MODEL_PATH="${SEMANTIC_FRAME_EMBEDDING_MODEL_PATH:-}"
SEMANTIC_FRAME_EMBEDDING_DEVICE="${SEMANTIC_FRAME_EMBEDDING_DEVICE:-mps}"
SEMANTIC_FRAME_EMBEDDING_TORCH_DTYPE="${SEMANTIC_FRAME_EMBEDDING_TORCH_DTYPE:-float32}"
SEMANTIC_FRAME_EMBEDDING_BATCH_SIZE="${SEMANTIC_FRAME_EMBEDDING_BATCH_SIZE:-8}"

SKIP_SPEECH_RECOGNITION="${SKIP_SPEECH_RECOGNITION:-0}"
SPEECH_BACKEND="${SPEECH_BACKEND:-qwen}"
SPEECH_CHUNK_DURATION_SECONDS="${SPEECH_CHUNK_DURATION_SECONDS:-120}"
SPEECH_MAX_NEW_TOKENS="${SPEECH_MAX_NEW_TOKENS:-512}"
FASTER_WHISPER_MODEL="${FASTER_WHISPER_MODEL:-small}"
FASTER_WHISPER_DEVICE="${FASTER_WHISPER_DEVICE:-cpu}"
FASTER_WHISPER_COMPUTE_TYPE="${FASTER_WHISPER_COMPUTE_TYPE:-default}"

MEMORY_CACHE_ONLY="${MEMORY_CACHE_ONLY:-0}"
DOWNLOAD_MISSING="${DOWNLOAD_MISSING:-0}"
SKIP_UNAVAILABLE_VIDEOS="${SKIP_UNAVAILABLE_VIDEOS:-1}"
COOKIES_FROM_BROWSER="${COOKIES_FROM_BROWSER:-}"
NO_PROGRESS="${NO_PROGRESS:-0}"

cmd=(
  conda run --no-capture-output -n videorlm
  python -u -m rlm.video.cli run-longshot-local
  --dataset-path "${DATASET_PATH}"
  --dataset-name "${DATASET_NAME}"
  --split "${SPLIT}"
  --output "${OUTPUT_ROOT}/results.jsonl"
  --video-dir "${VIDEO_DIR}"
  --artifacts-dir "${OUTPUT_ROOT}/artifacts"
  --memory-dir "${OUTPUT_ROOT}/memories"
  --trace-dir "${OUTPUT_ROOT}/traces"
  --history-mode "${HISTORY_MODE}"
  --sample-limit "${SAMPLE_LIMIT}"
  --controller-repo "${CONTROLLER_REPO}"
  --visual-repo "${VISUAL_REPO}"
  --speech-repo "${SPEECH_REPO}"
  --no-forced-aligner
  --controller-device "${CONTROLLER_DEVICE}"
  --visual-device "${VISUAL_DEVICE}"
  --speech-device "${SPEECH_DEVICE}"
  --torch-dtype "${TORCH_DTYPE}"
  --speech-backend "${SPEECH_BACKEND}"
  --speech-chunk-duration-seconds "${SPEECH_CHUNK_DURATION_SECONDS}"
  --speech-max-new-tokens "${SPEECH_MAX_NEW_TOKENS}"
  --faster-whisper-model "${FASTER_WHISPER_MODEL}"
  --faster-whisper-device "${FASTER_WHISPER_DEVICE}"
  --faster-whisper-compute-type "${FASTER_WHISPER_COMPUTE_TYPE}"
  --semantic-frame-embedding-repo "${SEMANTIC_FRAME_EMBEDDING_REPO}"
  --semantic-frame-embedding-device "${SEMANTIC_FRAME_EMBEDDING_DEVICE}"
  --semantic-frame-embedding-torch-dtype "${SEMANTIC_FRAME_EMBEDDING_TORCH_DTYPE}"
  --semantic-frame-embedding-batch-size "${SEMANTIC_FRAME_EMBEDDING_BATCH_SIZE}"
  --frame-count "${FRAME_COUNT}"
  --frame-width "${FRAME_WIDTH}"
  --verbose
  --use-pitome
  --search-mode graph
  --clip-duration-seconds "${CLIP_DURATION_SECONDS}"
  --pitome-dense-frame-rate "${PITOME_DENSE_FRAME_RATE}"
  --pitome-min-frame-count "${PITOME_MIN_FRAME_COUNT}"
  --pitome-embedding-backend "${PITOME_EMBEDDING_BACKEND}"
  --pitome-embedding-size "${PITOME_EMBEDDING_SIZE}"
  --pitome-anchor-frame-count "${PITOME_ANCHOR_FRAME_COUNT}"
  --pitome-max-selected-frames "${PITOME_MAX_SELECTED_FRAMES}"
  --pitome-scene-threshold "${PITOME_SCENE_THRESHOLD}"
  --pitome-max-scene-boundary-frames "${PITOME_MAX_SCENE_BOUNDARY_FRAMES}"
)

if [[ "${SKIP_SPEECH_RECOGNITION}" == "1" || "${SKIP_SPEECH_RECOGNITION}" == "true" || "${SKIP_SPEECH_RECOGNITION}" == "yes" ]]; then
  cmd+=(--no-speech-recognition)
else
  cmd+=(--lazy-speech-refinement)
fi
cmd+=(--lazy-visual-refinement)
if [[ "${MEMORY_CACHE_ONLY}" == "1" || "${MEMORY_CACHE_ONLY}" == "true" || "${MEMORY_CACHE_ONLY}" == "yes" ]]; then
  cmd+=(--memory-cache-only)
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
if [[ -n "${SEMANTIC_FRAME_EMBEDDING_MODEL_PATH}" ]]; then
  cmd+=(--semantic-frame-embedding-model-path "${SEMANTIC_FRAME_EMBEDDING_MODEL_PATH}")
fi

KMP_DUPLICATE_LIB_OK=TRUE PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 "${cmd[@]}"
