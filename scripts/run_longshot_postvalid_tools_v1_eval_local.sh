#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-output/longshot_postvalid_tools_v1_cognitive_search}"
DATASET_PATH="${DATASET_PATH:-MBZUAI/longshot-bench}"
DATASET_NAME="${DATASET_NAME:-postvalid_tools_v1}"
SPLIT="${SPLIT:-test}"
VIDEO_DIR="${VIDEO_DIR:-data/LongShot/videos}"
SAMPLE_LIMIT="${SAMPLE_LIMIT-20}"
HISTORY_MODE="${HISTORY_MODE:-gold}"

RUN_INFERENCE="${RUN_INFERENCE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
SCORE_ONLY="${SCORE_ONLY:-0}"
ANSWER_ONLY_EVAL="${ANSWER_ONLY_EVAL:-1}"

CONTROLLER_REPO="${CONTROLLER_REPO:-Qwen/Qwen3-4B-Instruct-2507}"
CONTROLLER_MODEL_PATH="${CONTROLLER_MODEL_PATH:-}"
CONTROLLER_MAX_NEW_TOKENS="${CONTROLLER_MAX_NEW_TOKENS:-}"
CONTROLLER_TRUST_REMOTE_CODE="${CONTROLLER_TRUST_REMOTE_CODE:-0}"
CONTROLLER_API_BASE_URL="${CONTROLLER_API_BASE_URL:-}"
CONTROLLER_API_KEY="${CONTROLLER_API_KEY:-}"
CONTROLLER_API_MODEL="${CONTROLLER_API_MODEL:-}"
CONTROLLER_API_MAX_TOKENS="${CONTROLLER_API_MAX_TOKENS:-}"
CONTROLLER_API_TIMEOUT="${CONTROLLER_API_TIMEOUT:-300}"
VISUAL_REPO="${VISUAL_REPO:-Qwen/Qwen3-VL-4B-Instruct}"
SPEECH_REPO="${SPEECH_REPO:-Qwen/Qwen3-ASR-1.7B}"
CONTROLLER_DEVICE="${CONTROLLER_DEVICE:-cuda:0}"
VISUAL_DEVICE="${VISUAL_DEVICE:-cuda:0}"
SPEECH_DEVICE="${SPEECH_DEVICE:-cuda:0}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"

JUDGE_REPO="${JUDGE_REPO:-google/gemma-3-12b-it}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-output/models/google__gemma-3-12b-it}"
JUDGE_DEVICE="${JUDGE_DEVICE:-cuda:0}"
JUDGE_TORCH_DTYPE="${JUDGE_TORCH_DTYPE:-bfloat16}"
JUDGE_ATTN_IMPLEMENTATION="${JUDGE_ATTN_IMPLEMENTATION:-}"
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-96}"
EVAL_SAMPLE_LIMIT="${EVAL_SAMPLE_LIMIT:-}"

FRAME_COUNT="${FRAME_COUNT:-3}"
FRAME_WIDTH="${FRAME_WIDTH:-768}"
MAX_STEPS="${MAX_STEPS:-8}"
SEARCH_TOP_K="${SEARCH_TOP_K:-6}"
MAX_FRONTIER_ITEMS="${MAX_FRONTIER_ITEMS:-8}"
SEARCH_MODE="${SEARCH_MODE:-graph}"

CLIP_DURATION_SECONDS="${CLIP_DURATION_SECONDS:-480}"
SCENE_DURATION_SECONDS="${SCENE_DURATION_SECONDS:-180}"
SEGMENT_DURATION_SECONDS="${SEGMENT_DURATION_SECONDS:-45}"
PITOME_FRAME_WIDTH="${PITOME_FRAME_WIDTH:-224}"
PITOME_DENSE_FRAME_RATE="${PITOME_DENSE_FRAME_RATE:-0.2}"
PITOME_MIN_FRAME_COUNT="${PITOME_MIN_FRAME_COUNT:-16}"
PITOME_EMBEDDING_BACKEND="${PITOME_EMBEDDING_BACKEND:-hybrid}"
PITOME_EMBEDDING_SIZE="${PITOME_EMBEDDING_SIZE:-32}"
PITOME_EMBEDDING_DEVICE="${PITOME_EMBEDDING_DEVICE:-cuda:0}"
PITOME_FRAME_EXTRACTION_STRATEGY="${PITOME_FRAME_EXTRACTION_STRATEGY:-seek}"
PITOME_FRAME_EXTRACTION_WORKERS="${PITOME_FRAME_EXTRACTION_WORKERS:-4}"
PITOME_ANCHOR_FRAME_COUNT="${PITOME_ANCHOR_FRAME_COUNT:-8}"
PITOME_MAX_SELECTED_FRAMES="${PITOME_MAX_SELECTED_FRAMES:-8}"
PITOME_SCENE_THRESHOLD="${PITOME_SCENE_THRESHOLD:-0.35}"
PITOME_MAX_SCENE_BOUNDARY_FRAMES="${PITOME_MAX_SCENE_BOUNDARY_FRAMES:-6}"
PITOME_SCENE_SAMPLE_RATE="${PITOME_SCENE_SAMPLE_RATE:-1}"
PITOME_SCENE_KEYFRAMES_ONLY="${PITOME_SCENE_KEYFRAMES_ONLY:-1}"
VISUAL_INDEX_BATCH_SIZE="${VISUAL_INDEX_BATCH_SIZE:-12}"

ENABLE_PADDLE_OCR="${ENABLE_PADDLE_OCR:-1}"
PADDLE_OCR_LANG="${PADDLE_OCR_LANG:-en}"
PADDLE_OCR_VERSION="${PADDLE_OCR_VERSION:-PP-OCRv5}"
PADDLE_OCR_TEXT_DETECTION_MODEL_NAME="${PADDLE_OCR_TEXT_DETECTION_MODEL_NAME:-PP-OCRv5_server_det}"
PADDLE_OCR_TEXT_RECOGNITION_MODEL_NAME="${PADDLE_OCR_TEXT_RECOGNITION_MODEL_NAME:-PP-OCRv5_server_rec}"
PADDLE_OCR_TEXT_RECOGNITION_BATCH_SIZE="${PADDLE_OCR_TEXT_RECOGNITION_BATCH_SIZE:-16}"
PADDLE_OCR_DEVICE="${PADDLE_OCR_DEVICE:-gpu:0}"
PADDLE_OCR_WINDOW_SECONDS="${PADDLE_OCR_WINDOW_SECONDS:-45}"
PADDLE_OCR_FRAME_COUNT="${PADDLE_OCR_FRAME_COUNT:-6}"
PADDLE_OCR_FRAME_WIDTH="${PADDLE_OCR_FRAME_WIDTH:-960}"
PADDLE_OCR_MIN_CONFIDENCE="${PADDLE_OCR_MIN_CONFIDENCE:-0.35}"
PADDLE_OCR_ENABLE_MKLDNN="${PADDLE_OCR_ENABLE_MKLDNN:-0}"
PADDLE_OCR_CACHE_DIR="${PADDLE_OCR_CACHE_DIR:-${OUTPUT_ROOT}/paddlex_cache}"
PADDLE_OCR_FRAME_EXTRACTION_STRATEGY="${PADDLE_OCR_FRAME_EXTRACTION_STRATEGY:-seek}"
PADDLE_OCR_FRAME_EXTRACTION_WORKERS="${PADDLE_OCR_FRAME_EXTRACTION_WORKERS:-4}"
ENABLE_CONTROLLER_EVIDENCE_CLASSIFIER="${ENABLE_CONTROLLER_EVIDENCE_CLASSIFIER:-1}"

USE_SEMANTIC_FRAME_EMBEDDINGS="${USE_SEMANTIC_FRAME_EMBEDDINGS:-1}"
SEMANTIC_FRAME_EMBEDDING_REPO="${SEMANTIC_FRAME_EMBEDDING_REPO:-google/siglip-large-patch16-384}"
SEMANTIC_FRAME_EMBEDDING_MODEL_PATH="${SEMANTIC_FRAME_EMBEDDING_MODEL_PATH:-}"
SEMANTIC_FRAME_EMBEDDING_DEVICE="${SEMANTIC_FRAME_EMBEDDING_DEVICE:-cuda:0}"
SEMANTIC_FRAME_EMBEDDING_TORCH_DTYPE="${SEMANTIC_FRAME_EMBEDDING_TORCH_DTYPE:-float32}"
SEMANTIC_FRAME_EMBEDDING_BATCH_SIZE="${SEMANTIC_FRAME_EMBEDDING_BATCH_SIZE:-8}"

ENABLE_VIDEO_WINDOW_RERANKING="${ENABLE_VIDEO_WINDOW_RERANKING:-1}"
VIDEO_WINDOW_RERANKER_REPO="${VIDEO_WINDOW_RERANKER_REPO:-OpenGVLab/InternVideo2-Stage2_6B}"
VIDEO_WINDOW_RERANKER_MODEL_PATH="${VIDEO_WINDOW_RERANKER_MODEL_PATH:-output/models/OpenGVLab__InternVideo2-Stage2_6B}"
VIDEO_WINDOW_RERANKER_DEVICE="${VIDEO_WINDOW_RERANKER_DEVICE:-cuda:0}"
VIDEO_WINDOW_RERANKER_TORCH_DTYPE="${VIDEO_WINDOW_RERANKER_TORCH_DTYPE:-float32}"
VIDEO_WINDOW_RERANKER_FRAME_COUNT="${VIDEO_WINDOW_RERANKER_FRAME_COUNT:-4}"
VIDEO_WINDOW_RERANKER_FRAME_SIZE="${VIDEO_WINDOW_RERANKER_FRAME_SIZE:-224}"
VIDEO_WINDOW_RERANK_CANDIDATE_COUNT="${VIDEO_WINDOW_RERANK_CANDIDATE_COUNT:-24}"
VIDEO_WINDOW_RERANK_WEIGHT="${VIDEO_WINDOW_RERANK_WEIGHT:-0.75}"
VIDEO_WINDOW_RERANK_WINDOW_SECONDS="${VIDEO_WINDOW_RERANK_WINDOW_SECONDS:-}"
VIDEO_WINDOW_RERANK_MIN_SCORE="${VIDEO_WINDOW_RERANK_MIN_SCORE:-}"

SKIP_SPEECH_RECOGNITION="${SKIP_SPEECH_RECOGNITION:-0}"
SPEECH_BACKEND="${SPEECH_BACKEND:-qwen}"
SPEECH_CHUNK_DURATION_SECONDS="${SPEECH_CHUNK_DURATION_SECONDS:-120}"
SPEECH_MAX_NEW_TOKENS="${SPEECH_MAX_NEW_TOKENS:-512}"
FASTER_WHISPER_MODEL="${FASTER_WHISPER_MODEL:-small}"
FASTER_WHISPER_DEVICE="${FASTER_WHISPER_DEVICE:-cuda:0}"
FASTER_WHISPER_COMPUTE_TYPE="${FASTER_WHISPER_COMPUTE_TYPE:-default}"

MEMORY_CACHE_ONLY="${MEMORY_CACHE_ONLY:-0}"
DOWNLOAD_MISSING="${DOWNLOAD_MISSING:-0}"
SKIP_UNAVAILABLE_VIDEOS="${SKIP_UNAVAILABLE_VIDEOS:-1}"
COOKIES_FROM_BROWSER="${COOKIES_FROM_BROWSER:-}"
NO_PROGRESS="${NO_PROGRESS:-0}"
TASK_FILTER="${TASK_FILTER:-}"
SAMPLE_ID="${SAMPLE_ID:-}"
VIDEO_ID="${VIDEO_ID:-}"

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

is_true() {
  case "${1,,}" in
    1 | true | yes | on) return 0 ;;
    *) return 1 ;;
  esac
}

CONDA_ENV_PREFIX="$(
  "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
    python -c 'import os; print(os.environ["CONDA_PREFIX"])'
)"
CONDA_NVIDIA_LIBRARY_PATHS="${CONDA_NVIDIA_LIBRARY_PATHS:-}"
if [[ -z "${CONDA_NVIDIA_LIBRARY_PATHS}" ]]; then
  CONDA_NVIDIA_LIBRARY_PATHS="$(
    "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
      python -c 'import pathlib, nvidia; root=pathlib.Path(nvidia.__path__[0]); print(":".join(str(p) for p in sorted(root.glob("*/lib"))))' \
      2>/dev/null || true
  )"
fi
CUDA_LIBRARY_PATHS="${CUDA_LIBRARY_PATHS:-}"
EXTRA_LD_LIBRARY_PATH="${EXTRA_LD_LIBRARY_PATH:-}"
PRELOAD_CONDA_NATIVE_LIBS="${PRELOAD_CONDA_NATIVE_LIBS:-1}"
LD_LIBRARY_PATH="${CONDA_ENV_PREFIX}/lib"
if [[ -n "${CONDA_NVIDIA_LIBRARY_PATHS}" ]]; then
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CONDA_NVIDIA_LIBRARY_PATHS}"
fi
if [[ -n "${CUDA_LIBRARY_PATHS}" ]]; then
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDA_LIBRARY_PATHS}"
fi
if [[ -n "${EXTRA_LD_LIBRARY_PATH}" ]]; then
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${EXTRA_LD_LIBRARY_PATH}"
fi
export LD_LIBRARY_PATH
if is_true "${PRELOAD_CONDA_NATIVE_LIBS:-0}"; then
  LD_PRELOAD="${CONDA_ENV_PREFIX}/lib/libstdc++.so.6:${CONDA_ENV_PREFIX}/lib/libjpeg.so.8${LD_PRELOAD:+:${LD_PRELOAD}}"
  export LD_PRELOAD
fi

add_csv_args() {
  local flag="$1"
  local csv="$2"
  local item
  if [[ -z "${csv}" ]]; then
    return
  fi
  IFS=',' read -ra items <<<"${csv}"
  for item in "${items[@]}"; do
    if [[ -n "${item}" ]]; then
      cmd+=("${flag}" "${item}")
    fi
  done
}

mkdir -p "${OUTPUT_ROOT}"
PREDICTIONS_PATH="${PREDICTIONS_PATH:-${OUTPUT_ROOT}/results.jsonl}"
if is_true "${ANSWER_ONLY_EVAL}"; then
  EVAL_OUTPUT_PATH="${EVAL_OUTPUT_PATH:-${OUTPUT_ROOT}/answer_only_eval.jsonl}"
  SCORE_OUTPUT_PATH="${SCORE_OUTPUT_PATH:-${OUTPUT_ROOT}/answer_only_scores.txt}"
  SUMMARY_OUTPUT_PATH="${SUMMARY_OUTPUT_PATH:-${OUTPUT_ROOT}/answer_only_summary.json}"
else
  EVAL_OUTPUT_PATH="${EVAL_OUTPUT_PATH:-${OUTPUT_ROOT}/official_eval.jsonl}"
  SCORE_OUTPUT_PATH="${SCORE_OUTPUT_PATH:-${OUTPUT_ROOT}/official_scores.txt}"
  SUMMARY_OUTPUT_PATH="${SUMMARY_OUTPUT_PATH:-${OUTPUT_ROOT}/official_summary.json}"
fi

if is_true "${RUN_INFERENCE}"; then
  cmd=(
    "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}"
    python -u scripts/run_videorlm_cli_with_cv2_preload.py run-longshot-local
    --dataset-path "${DATASET_PATH}"
    --dataset-name "${DATASET_NAME}"
    --split "${SPLIT}"
    --output "${PREDICTIONS_PATH}"
    --video-dir "${VIDEO_DIR}"
    --artifacts-dir "${OUTPUT_ROOT}/artifacts"
    --memory-dir "${OUTPUT_ROOT}/memories"
    --trace-dir "${OUTPUT_ROOT}/traces"
    --history-mode "${HISTORY_MODE}"
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
    --frame-count "${FRAME_COUNT}"
    --frame-width "${FRAME_WIDTH}"
    --max-steps "${MAX_STEPS}"
    --search-top-k "${SEARCH_TOP_K}"
    --max-frontier-items "${MAX_FRONTIER_ITEMS}"
    --scene-duration-seconds "${SCENE_DURATION_SECONDS}"
    --segment-duration-seconds "${SEGMENT_DURATION_SECONDS}"
    --clip-duration-seconds "${CLIP_DURATION_SECONDS}"
    --search-mode "${SEARCH_MODE}"
    --verbose
    --use-pitome
    --pitome-dense-frame-rate "${PITOME_DENSE_FRAME_RATE}"
    --pitome-min-frame-count "${PITOME_MIN_FRAME_COUNT}"
    --pitome-embedding-backend "${PITOME_EMBEDDING_BACKEND}"
    --pitome-embedding-size "${PITOME_EMBEDDING_SIZE}"
    --pitome-embedding-device "${PITOME_EMBEDDING_DEVICE}"
    --pitome-frame-width "${PITOME_FRAME_WIDTH}"
    --pitome-frame-extraction-strategy "${PITOME_FRAME_EXTRACTION_STRATEGY}"
    --pitome-frame-extraction-workers "${PITOME_FRAME_EXTRACTION_WORKERS}"
    --pitome-anchor-frame-count "${PITOME_ANCHOR_FRAME_COUNT}"
    --pitome-max-selected-frames "${PITOME_MAX_SELECTED_FRAMES}"
    --pitome-scene-threshold "${PITOME_SCENE_THRESHOLD}"
    --pitome-max-scene-boundary-frames "${PITOME_MAX_SCENE_BOUNDARY_FRAMES}"
    --pitome-scene-sample-rate "${PITOME_SCENE_SAMPLE_RATE}"
    --visual-index-batch-size "${VISUAL_INDEX_BATCH_SIZE}"
  )
  if [[ -n "${SAMPLE_LIMIT}" ]]; then
    cmd+=(--sample-limit "${SAMPLE_LIMIT}")
  fi
  if [[ -n "${ATTN_IMPLEMENTATION}" ]]; then
    cmd+=(--attn-implementation "${ATTN_IMPLEMENTATION}")
  fi
  if [[ -n "${CONTROLLER_MODEL_PATH}" ]]; then
    cmd+=(--controller-model-path "${CONTROLLER_MODEL_PATH}")
  fi
  if [[ -n "${CONTROLLER_MAX_NEW_TOKENS}" ]]; then
    cmd+=(--controller-max-new-tokens "${CONTROLLER_MAX_NEW_TOKENS}")
  fi
  if is_true "${CONTROLLER_TRUST_REMOTE_CODE}"; then
    cmd+=(--controller-trust-remote-code)
  fi
  if [[ -n "${CONTROLLER_API_BASE_URL}" || -n "${CONTROLLER_API_MODEL}" ]]; then
    cmd+=(
      --controller-api-base-url "${CONTROLLER_API_BASE_URL}"
      --controller-api-model "${CONTROLLER_API_MODEL}"
      --controller-api-timeout "${CONTROLLER_API_TIMEOUT}"
    )
    if [[ -n "${CONTROLLER_API_KEY}" ]]; then
      cmd+=(--controller-api-key "${CONTROLLER_API_KEY}")
    fi
    if [[ -n "${CONTROLLER_API_MAX_TOKENS}" ]]; then
      cmd+=(--controller-api-max-tokens "${CONTROLLER_API_MAX_TOKENS}")
    fi
  fi
  if ! is_true "${PITOME_SCENE_KEYFRAMES_ONLY}"; then
    cmd+=(--no-pitome-scene-keyframes-only)
  fi
  if is_true "${SKIP_SPEECH_RECOGNITION}"; then
    cmd+=(--no-speech-recognition)
  else
    cmd+=(--lazy-speech-refinement)
  fi
  cmd+=(--lazy-visual-refinement)
  if is_true "${ENABLE_CONTROLLER_EVIDENCE_CLASSIFIER}"; then
    cmd+=(--enable-controller-evidence-classifier)
  fi
  if is_true "${ENABLE_PADDLE_OCR}"; then
    cmd+=(
      --enable-paddle-ocr
      --paddle-ocr-lang "${PADDLE_OCR_LANG}"
      --paddle-ocr-version "${PADDLE_OCR_VERSION}"
      --paddle-ocr-device "${PADDLE_OCR_DEVICE}"
      --paddle-ocr-window-seconds "${PADDLE_OCR_WINDOW_SECONDS}"
      --paddle-ocr-frame-count "${PADDLE_OCR_FRAME_COUNT}"
      --paddle-ocr-frame-width "${PADDLE_OCR_FRAME_WIDTH}"
      --paddle-ocr-min-confidence "${PADDLE_OCR_MIN_CONFIDENCE}"
      --paddle-ocr-cache-dir "${PADDLE_OCR_CACHE_DIR}"
      --paddle-ocr-frame-extraction-strategy "${PADDLE_OCR_FRAME_EXTRACTION_STRATEGY}"
      --paddle-ocr-frame-extraction-workers "${PADDLE_OCR_FRAME_EXTRACTION_WORKERS}"
    )
    if [[ -n "${PADDLE_OCR_TEXT_DETECTION_MODEL_NAME}" ]]; then
      cmd+=(--paddle-ocr-text-detection-model-name "${PADDLE_OCR_TEXT_DETECTION_MODEL_NAME}")
    fi
    if [[ -n "${PADDLE_OCR_TEXT_RECOGNITION_MODEL_NAME}" ]]; then
      cmd+=(--paddle-ocr-text-recognition-model-name "${PADDLE_OCR_TEXT_RECOGNITION_MODEL_NAME}")
    fi
    if [[ -n "${PADDLE_OCR_TEXT_RECOGNITION_BATCH_SIZE}" ]]; then
      cmd+=(--paddle-ocr-text-recognition-batch-size "${PADDLE_OCR_TEXT_RECOGNITION_BATCH_SIZE}")
    fi
    if is_true "${PADDLE_OCR_ENABLE_MKLDNN}"; then
      cmd+=(--paddle-ocr-enable-mkldnn)
    fi
  fi
  if is_true "${MEMORY_CACHE_ONLY}"; then
    cmd+=(--memory-cache-only)
  fi
  if is_true "${DOWNLOAD_MISSING}"; then
    cmd+=(--download-missing)
  fi
  if is_true "${SKIP_UNAVAILABLE_VIDEOS}"; then
    cmd+=(--skip-unavailable-videos)
  fi
  if is_true "${NO_PROGRESS}"; then
    cmd+=(--no-progress)
  fi
  if [[ -n "${COOKIES_FROM_BROWSER}" ]]; then
    cmd+=(--cookies-from-browser "${COOKIES_FROM_BROWSER}")
  fi
  add_csv_args --task-filter "${TASK_FILTER}"
  add_csv_args --sample-id "${SAMPLE_ID}"
  add_csv_args --video-id "${VIDEO_ID}"
  if is_true "${USE_SEMANTIC_FRAME_EMBEDDINGS}"; then
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
  if is_true "${ENABLE_VIDEO_WINDOW_RERANKING}"; then
    cmd+=(
      --enable-video-window-reranking
      --video-window-reranker-repo "${VIDEO_WINDOW_RERANKER_REPO}"
      --video-window-reranker-device "${VIDEO_WINDOW_RERANKER_DEVICE}"
      --video-window-reranker-torch-dtype "${VIDEO_WINDOW_RERANKER_TORCH_DTYPE}"
      --video-window-reranker-frame-count "${VIDEO_WINDOW_RERANKER_FRAME_COUNT}"
      --video-window-reranker-frame-size "${VIDEO_WINDOW_RERANKER_FRAME_SIZE}"
      --video-window-rerank-candidate-count "${VIDEO_WINDOW_RERANK_CANDIDATE_COUNT}"
      --video-window-rerank-weight "${VIDEO_WINDOW_RERANK_WEIGHT}"
    )
    if [[ -n "${VIDEO_WINDOW_RERANKER_MODEL_PATH}" ]]; then
      cmd+=(--video-window-reranker-model-path "${VIDEO_WINDOW_RERANKER_MODEL_PATH}")
    fi
    if [[ -n "${VIDEO_WINDOW_RERANK_WINDOW_SECONDS}" ]]; then
      cmd+=(--video-window-rerank-window-seconds "${VIDEO_WINDOW_RERANK_WINDOW_SECONDS}")
    fi
    if [[ -n "${VIDEO_WINDOW_RERANK_MIN_SCORE}" ]]; then
      cmd+=(--video-window-rerank-min-score "${VIDEO_WINDOW_RERANK_MIN_SCORE}")
    fi
  fi

  echo "[LongShOT] running ${DATASET_PATH}/${DATASET_NAME}:${SPLIT}" >&2
  echo "[LongShOT] predictions: ${PREDICTIONS_PATH}" >&2
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
    PADDLE_PDX_CACHE_HOME="${PADDLE_OCR_CACHE_DIR}" "${cmd[@]}"
fi

if is_true "${RUN_EVAL}"; then
  cmd=(
    "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}"
    python -u scripts/evaluate_longshot_official_metric.py
    --eval-output "${EVAL_OUTPUT_PATH}"
    --score-output "${SCORE_OUTPUT_PATH}"
    --summary-output "${SUMMARY_OUTPUT_PATH}"
  )
  if is_true "${SCORE_ONLY}"; then
    cmd+=(--score-only)
  else
    cmd+=(--predictions "${PREDICTIONS_PATH}")
    cmd+=(--judge-repo "${JUDGE_REPO}")
    cmd+=(--judge-device "${JUDGE_DEVICE}")
    cmd+=(--torch-dtype "${JUDGE_TORCH_DTYPE}")
    cmd+=(--max-new-tokens "${JUDGE_MAX_NEW_TOKENS}")
    if [[ -n "${JUDGE_MODEL_PATH}" ]]; then
      cmd+=(--judge-model-path "${JUDGE_MODEL_PATH}")
    fi
    if [[ -n "${JUDGE_ATTN_IMPLEMENTATION}" ]]; then
      cmd+=(--attn-implementation "${JUDGE_ATTN_IMPLEMENTATION}")
    fi
  fi
  if is_true "${ANSWER_ONLY_EVAL}"; then
    cmd+=(--answer-only)
  fi
  if [[ -n "${EVAL_SAMPLE_LIMIT}" ]]; then
    cmd+=(--sample-limit "${EVAL_SAMPLE_LIMIT}")
  fi

  if is_true "${ANSWER_ONLY_EVAL}"; then
    echo "[LongShOT] scoring answer-only final answers: ${SCORE_OUTPUT_PATH}" >&2
  else
    echo "[LongShOT] scoring official criteria: ${SCORE_OUTPUT_PATH}" >&2
  fi
  PYTHONUNBUFFERED=1 "${cmd[@]}"
fi

echo "[LongShOT] done"
echo "  predictions: ${PREDICTIONS_PATH}"
echo "  judged eval:  ${EVAL_OUTPUT_PATH}"
echo "  scores:       ${SCORE_OUTPUT_PATH}"
echo "  summary:      ${SUMMARY_OUTPUT_PATH}"
