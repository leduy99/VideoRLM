#!/usr/bin/env bash
set -euo pipefail

# LongShotBench postvalid_v1 run using the current VideoRLM pipeline.
# The updated MBZUAI/longshot-bench dataset exposes the benchmark as
# postvalid_v2. LONGSHOT_CONTEXT_DATASET_NAME keeps VideoRLM's postvalid_v1
# routing/prompt behavior enabled.
# Overrides can be passed as environment variables, for example:
#   SAMPLE_LIMIT=20 bash scripts/run_longshot_postvalid_v1_eval_local.sh
#   SAMPLE_LIMIT= SAMPLE_START_INDEX=1 SAMPLE_END_INDEX=200 bash scripts/run_longshot_postvalid_v1_eval_local.sh

export OUTPUT_ROOT="${OUTPUT_ROOT:-output/longshot_postvalid_v1_full_asr_cognitive_search}"
export DATASET_PATH="${DATASET_PATH:-MBZUAI/longshot-bench}"
export DATASET_NAME="${DATASET_NAME:-postvalid_v2}"
export LONGSHOT_CONTEXT_DATASET_NAME="${LONGSHOT_CONTEXT_DATASET_NAME:-postvalid_v1}"
export SPLIT="${SPLIT:-test}"
export VIDEO_DIR="${VIDEO_DIR:-data/videos}"
export LAZY_SPEECH_REFINEMENT="${LAZY_SPEECH_REFINEMENT:-0}"
export ENABLE_CONTROLLER_EVIDENCE_CLASSIFIER="${ENABLE_CONTROLLER_EVIDENCE_CLASSIFIER:-0}"
export SPEECH_CHUNK_DURATION_SECONDS="${SPEECH_CHUNK_DURATION_SECONDS:-300}"
export SPEECH_ASR_CHUNK_BATCH_SIZE="${SPEECH_ASR_CHUNK_BATCH_SIZE:-12}"
export SPEECH_MAX_NEW_TOKENS="${SPEECH_MAX_NEW_TOKENS:-1536}"

# Run 100 postvalid_v1 samples by default. Set SAMPLE_LIMIT=20 for a quick smoke test.
export SAMPLE_LIMIT="${SAMPLE_LIMIT-100}"
export SAMPLE_START_INDEX="${SAMPLE_START_INDEX:-}"
export SAMPLE_END_INDEX="${SAMPLE_END_INDEX:-}"

# postvalid_v1 has official criterion/task metadata, so default to official judging.
# Set ANSWER_ONLY_EVAL=1 if you only want final-answer correctness.
export ANSWER_ONLY_EVAL="${ANSWER_ONLY_EVAL:-0}"

exec scripts/run_longshot_postvalid_tools_v1_eval_local.sh "$@"
