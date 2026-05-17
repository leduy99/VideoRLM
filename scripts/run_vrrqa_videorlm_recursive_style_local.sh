#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-output/vrrqa_videorlm_longshot_style}"
ANNOTATIONS="${ANNOTATIONS:-data/vrrqa/val_set/ImplicitQAv0.1.2.jsonl}"
VIDEO_DIR="${VIDEO_DIR:-data/vrrqa/val_set/videos}"
SEGMENT_DIR="${SEGMENT_DIR:-data/vrrqa/val_set/segments}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-20}"
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
VISUAL_MAX_NEW_TOKENS="${VISUAL_MAX_NEW_TOKENS:-512}"

FRAME_COUNT="${FRAME_COUNT:-3}"
FRAME_WIDTH="${FRAME_WIDTH:-768}"
SCENE_DURATION_SECONDS="${SCENE_DURATION_SECONDS:-180}"
SEGMENT_DURATION_SECONDS="${SEGMENT_DURATION_SECONDS:-45}"
CLIP_DURATION_SECONDS="${CLIP_DURATION_SECONDS:-5}"
MAX_STEPS="${MAX_STEPS:-10}"
SEARCH_TOP_K="${SEARCH_TOP_K:-5}"
MAX_FRONTIER_ITEMS="${MAX_FRONTIER_ITEMS:-8}"

SEMANTIC_FRAME_EMBEDDING_REPO="${SEMANTIC_FRAME_EMBEDDING_REPO:-google/siglip-base-patch16-224}"
SEMANTIC_FRAME_EMBEDDING_MODEL_PATH="${SEMANTIC_FRAME_EMBEDDING_MODEL_PATH:-}"
SEMANTIC_FRAME_EMBEDDING_DEVICE="${SEMANTIC_FRAME_EMBEDDING_DEVICE:-mps}"
SEMANTIC_FRAME_EMBEDDING_TORCH_DTYPE="${SEMANTIC_FRAME_EMBEDDING_TORCH_DTYPE:-float32}"
SEMANTIC_FRAME_EMBEDDING_BATCH_SIZE="${SEMANTIC_FRAME_EMBEDDING_BATCH_SIZE:-8}"
USE_SEMANTIC_FRAME_EMBEDDINGS="${USE_SEMANTIC_FRAME_EMBEDDINGS:-1}"

PITOME_DENSE_FRAME_RATE="${PITOME_DENSE_FRAME_RATE:-0.2}"
PITOME_MIN_FRAME_COUNT="${PITOME_MIN_FRAME_COUNT:-16}"
PITOME_EMBEDDING_BACKEND="${PITOME_EMBEDDING_BACKEND:-hybrid}"
PITOME_EMBEDDING_SIZE="${PITOME_EMBEDDING_SIZE:-32}"
PITOME_ANCHOR_FRAME_COUNT="${PITOME_ANCHOR_FRAME_COUNT:-8}"
PITOME_MAX_SELECTED_FRAMES="${PITOME_MAX_SELECTED_FRAMES:-8}"
PITOME_SCENE_THRESHOLD="${PITOME_SCENE_THRESHOLD:-0.35}"
PITOME_MAX_SCENE_BOUNDARY_FRAMES="${PITOME_MAX_SCENE_BOUNDARY_FRAMES:-6}"

DOWNLOAD_MISSING="${DOWNLOAD_MISSING:-0}"
SKIP_UNAVAILABLE_VIDEOS="${SKIP_UNAVAILABLE_VIDEOS:-1}"
COOKIES_FROM_BROWSER="${COOKIES_FROM_BROWSER:-}"
NO_PROGRESS="${NO_PROGRESS:-0}"
VRRQA_GRAPH_REFINEMENT_EXPANSION="${VRRQA_GRAPH_REFINEMENT_EXPANSION:-1}"
VRRQA_GRAPH_REFINEMENT_NEIGHBOR_COUNT="${VRRQA_GRAPH_REFINEMENT_NEIGHBOR_COUNT:-1}"
MULTI_WINDOW_MEMORY="${MULTI_WINDOW_MEMORY:-1}"
FORCE_CHOICE_FINALIZER="${FORCE_CHOICE_FINALIZER:-0}"

cmd=(
  conda run --no-capture-output -n "${VRRQA_CONDA_ENV}"
  python -u scripts/run_vrrqa_local.py
  --strategy lazy-pitome
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
  --visual-max-new-tokens "${VISUAL_MAX_NEW_TOKENS}"
  --frame-count "${FRAME_COUNT}"
  --frame-width "${FRAME_WIDTH}"
  --scene-duration-seconds "${SCENE_DURATION_SECONDS}"
  --segment-duration-seconds "${SEGMENT_DURATION_SECONDS}"
  --clip-duration-seconds "${CLIP_DURATION_SECONDS}"
  --max-steps "${MAX_STEPS}"
  --search-top-k "${SEARCH_TOP_K}"
  --max-frontier-items "${MAX_FRONTIER_ITEMS}"
  --pitome-dense-frame-rate "${PITOME_DENSE_FRAME_RATE}"
  --pitome-min-frame-count "${PITOME_MIN_FRAME_COUNT}"
  --pitome-embedding-backend "${PITOME_EMBEDDING_BACKEND}"
  --pitome-embedding-size "${PITOME_EMBEDDING_SIZE}"
  --pitome-anchor-frame-count "${PITOME_ANCHOR_FRAME_COUNT}"
  --pitome-max-selected-frames "${PITOME_MAX_SELECTED_FRAMES}"
  --pitome-scene-threshold "${PITOME_SCENE_THRESHOLD}"
  --pitome-max-scene-boundary-frames "${PITOME_MAX_SCENE_BOUNDARY_FRAMES}"
  --vrrqa-graph-refinement-neighbor-count "${VRRQA_GRAPH_REFINEMENT_NEIGHBOR_COUNT}"
  --verbose
)

if [[ -n "${SAMPLE_LIMIT}" ]]; then
  cmd+=(--sample-limit "${SAMPLE_LIMIT}")
fi
if [[ "${USE_SEMANTIC_FRAME_EMBEDDINGS}" == "1" || "${USE_SEMANTIC_FRAME_EMBEDDINGS}" == "true" || "${USE_SEMANTIC_FRAME_EMBEDDINGS}" == "yes" ]]; then
  cmd+=(
    --semantic-frame-embedding-repo "${SEMANTIC_FRAME_EMBEDDING_REPO}"
    --semantic-frame-embedding-device "${SEMANTIC_FRAME_EMBEDDING_DEVICE}"
    --semantic-frame-embedding-torch-dtype "${SEMANTIC_FRAME_EMBEDDING_TORCH_DTYPE}"
    --semantic-frame-embedding-batch-size "${SEMANTIC_FRAME_EMBEDDING_BATCH_SIZE}"
  )
  if [[ -n "${SEMANTIC_FRAME_EMBEDDING_MODEL_PATH}" ]]; then
    cmd+=(--semantic-frame-embedding-model-path "${SEMANTIC_FRAME_EMBEDDING_MODEL_PATH}")
  fi
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
if [[ "${VRRQA_GRAPH_REFINEMENT_EXPANSION}" == "0" || "${VRRQA_GRAPH_REFINEMENT_EXPANSION}" == "false" || "${VRRQA_GRAPH_REFINEMENT_EXPANSION}" == "no" ]]; then
  cmd+=(--disable-vrrqa-graph-refinement-expansion)
fi
if [[ "${MULTI_WINDOW_MEMORY}" == "1" || "${MULTI_WINDOW_MEMORY}" == "true" || "${MULTI_WINDOW_MEMORY}" == "yes" ]]; then
  cmd+=(--multi-window-memory)
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

print(f"[VRR-QA VideoRLM] python: {sys.executable}", flush=True)
try:
    import PIL
except ModuleNotFoundError as exc:
    conda_env = os.environ.get("VRRQA_CONDA_ENV_NAME", "videorlm")
    raise SystemExit(
        "Pillow is not importable from the VRR-QA Python above. "
        f"Install it with: conda install -n {conda_env} -y pillow"
    ) from exc
print(f"[VRR-QA VideoRLM] PIL: {PIL.__file__}", flush=True)
PY
if [[ "${PRECHECK_ONLY}" == "1" || "${PRECHECK_ONLY}" == "true" || "${PRECHECK_ONLY}" == "yes" ]]; then
  exit 0
fi

KMP_DUPLICATE_LIB_OK=TRUE PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 "${cmd[@]}"
