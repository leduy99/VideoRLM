#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-output/longshot_postvalid_tools_v1_ocr_structured_classifier}"
SOURCE_EVAL_PATH="${SOURCE_EVAL_PATH:-${OUTPUT_ROOT}/answer_only_eval.jsonl}"
BASE_RUN_SCRIPT="${BASE_RUN_SCRIPT:-scripts/run_longshot_postvalid_tools_v1_eval_local.sh}"
RERUN_TAG="${RERUN_TAG:-wrong_samples_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "${SOURCE_EVAL_PATH}" ]]; then
  echo "Missing eval file: ${SOURCE_EVAL_PATH}" >&2
  exit 1
fi

if [[ ! -f "${BASE_RUN_SCRIPT}" ]]; then
  echo "Missing base run script: ${BASE_RUN_SCRIPT}" >&2
  exit 1
fi

readarray -t parsed_lines < <(
  python - "${SOURCE_EVAL_PATH}" <<'PY'
import json
import sys
from pathlib import Path

eval_path = Path(sys.argv[1])
wrong_samples: list[str] = []
wrong_turns: list[str] = []

for line in eval_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    sample_id = str(row.get("sample_id") or "")
    if not sample_id:
        continue
    conversations = row.get("conversations") or []
    for turn_index, message in enumerate(conversations):
        if message.get("role") != "assistant":
            continue
        criteria = [
            item
            for item in message.get("criteria", [])
            if item.get("name") == "answer_correctness"
            and item.get("evaluation_mode") == "answer_only"
        ]
        if not criteria or criteria[0].get("criteria_met"):
            continue
        if sample_id not in wrong_samples:
            wrong_samples.append(sample_id)
        wrong_turns.append(f"{sample_id}:turn_{turn_index:03d}")

print(",".join(wrong_samples))
print(",".join(wrong_turns))
print(str(len(wrong_samples)))
print(str(len(wrong_turns)))
PY
)

WRONG_SAMPLE_IDS="${parsed_lines[0]:-}"
WRONG_TURNS="${parsed_lines[1]:-}"
WRONG_SAMPLE_COUNT="${parsed_lines[2]:-0}"
WRONG_TURN_COUNT="${parsed_lines[3]:-0}"

if [[ -z "${WRONG_SAMPLE_IDS}" ]]; then
  echo "No wrong answer_only samples found in ${SOURCE_EVAL_PATH}."
  exit 0
fi

PREDICTIONS_PATH="${PREDICTIONS_PATH:-${OUTPUT_ROOT}/${RERUN_TAG}_results.jsonl}"
EVAL_OUTPUT_PATH="${EVAL_OUTPUT_PATH:-${OUTPUT_ROOT}/${RERUN_TAG}_answer_only_eval.jsonl}"
SCORE_OUTPUT_PATH="${SCORE_OUTPUT_PATH:-${OUTPUT_ROOT}/${RERUN_TAG}_answer_only_scores.txt}"
SUMMARY_OUTPUT_PATH="${SUMMARY_OUTPUT_PATH:-${OUTPUT_ROOT}/${RERUN_TAG}_answer_only_summary.json}"
WRONG_SAMPLE_LIST_PATH="${WRONG_SAMPLE_LIST_PATH:-${OUTPUT_ROOT}/${RERUN_TAG}_wrong_samples.txt}"

{
  echo "source_eval=${SOURCE_EVAL_PATH}"
  echo "wrong_sample_count=${WRONG_SAMPLE_COUNT}"
  echo "wrong_turn_count=${WRONG_TURN_COUNT}"
  echo "wrong_sample_ids=${WRONG_SAMPLE_IDS}"
  echo "wrong_turns=${WRONG_TURNS}"
} >"${WRONG_SAMPLE_LIST_PATH}"

echo "[LongShOT wrong-rerun] source eval: ${SOURCE_EVAL_PATH}"
echo "[LongShOT wrong-rerun] wrong samples (${WRONG_SAMPLE_COUNT}): ${WRONG_SAMPLE_IDS}"
echo "[LongShOT wrong-rerun] wrong turns (${WRONG_TURN_COUNT}): ${WRONG_TURNS}"
echo "[LongShOT wrong-rerun] sample list: ${WRONG_SAMPLE_LIST_PATH}"
echo "[LongShOT wrong-rerun] predictions: ${PREDICTIONS_PATH}"
echo "[LongShOT wrong-rerun] eval: ${EVAL_OUTPUT_PATH}"

if [[ "${DRY_RUN,,}" =~ ^(1|true|yes|on)$ ]]; then
  echo "[LongShOT wrong-rerun] dry run only; not launching inference."
  exit 0
fi

OUTPUT_ROOT="${OUTPUT_ROOT}" \
PREDICTIONS_PATH="${PREDICTIONS_PATH}" \
EVAL_OUTPUT_PATH="${EVAL_OUTPUT_PATH}" \
SCORE_OUTPUT_PATH="${SCORE_OUTPUT_PATH}" \
SUMMARY_OUTPUT_PATH="${SUMMARY_OUTPUT_PATH}" \
SAMPLE_LIMIT= \
SAMPLE_ID="${WRONG_SAMPLE_IDS}" \
bash "${BASE_RUN_SCRIPT}"
