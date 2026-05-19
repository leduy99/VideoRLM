#!/bin/bash
#SBATCH --job-name=timelogic_vrlm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

: "${DATASET_JSON:?Set DATASET_JSON to a TimeLogic validation or shard JSON file.}"
: "${VIDEOS_DIR:?Set VIDEOS_DIR to the extracted TimeLogic validation videos directory.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the run output directory.}"

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONDA_ENV="${CONDA_ENV:-videorlm}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MAX_STEPS="${MAX_STEPS:-8}"
SEARCH_TOP_K="${SEARCH_TOP_K:-5}"
MAX_FRONTIER_ITEMS="${MAX_FRONTIER_ITEMS:-8}"
SCENE_DURATION_SECONDS="${SCENE_DURATION_SECONDS:-120}"
SEGMENT_DURATION_SECONDS="${SEGMENT_DURATION_SECONDS:-30}"
CLIP_DURATION_SECONDS="${CLIP_DURATION_SECONDS:-10}"
CONTROLLER_REPO="${CONTROLLER_REPO:-Qwen/Qwen3-8B}"
VISUAL_REPO="${VISUAL_REPO:-Qwen/Qwen3-VL-8B-Instruct}"
SPEECH_REPO="${SPEECH_REPO:-Qwen/Qwen3-ASR-0.6B}"
FORCED_ALIGNER_REPO="${FORCED_ALIGNER_REPO:-Qwen/Qwen3-ForcedAligner-0.6B}"
VISUAL_ONLY="${VISUAL_ONLY:-1}"
NO_FORCED_ALIGNER="${NO_FORCED_ALIGNER:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "$REPO_ROOT"
mkdir -p logs "$OUTPUT_DIR"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
elif [ -f "$HOME/miniconda3/bin/activate" ]; then
  # Common cluster setup from the H200 quick guide.
  source "$HOME/miniconda3/bin/activate" "$CONDA_ENV"
fi

args=(
  scripts/timelogic/run_inference.py
  --dataset-json "$DATASET_JSON"
  --videos-dir "$VIDEOS_DIR"
  --output-dir "$OUTPUT_DIR"
  --all-samples
  --resume
  --max-steps "$MAX_STEPS"
  --search-top-k "$SEARCH_TOP_K"
  --max-frontier-items "$MAX_FRONTIER_ITEMS"
  --scene-duration-seconds "$SCENE_DURATION_SECONDS"
  --segment-duration-seconds "$SEGMENT_DURATION_SECONDS"
  --clip-duration-seconds "$CLIP_DURATION_SECONDS"
  --controller-device cuda:0
  --visual-device cuda:0
  --speech-device cuda:0
  --controller-repo "$CONTROLLER_REPO"
  --visual-repo "$VISUAL_REPO"
  --speech-repo "$SPEECH_REPO"
  --forced-aligner-repo "$FORCED_ALIGNER_REPO"
)

if [ "$VISUAL_ONLY" = "1" ]; then
  args+=(--visual-only)
fi
if [ "$NO_FORCED_ALIGNER" = "1" ]; then
  args+=(--no-forced-aligner)
fi

# shellcheck disable=SC2086
"$PYTHON_BIN" "${args[@]}" $EXTRA_ARGS
