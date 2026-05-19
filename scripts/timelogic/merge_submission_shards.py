from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


MC_ANSWERS = {"A", "B", "C", "D"}
BOOL_ANSWERS = {"Yes", "No"}
VALID_ANSWERS = MC_ANSWERS | BOOL_ANSWERS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge TimeLogic shard predictions into one official submission JSON."
    )
    parser.add_argument(
        "--dataset-json",
        required=True,
        help="Full TimeLogic validation JSON used to define output order and modes.",
    )
    parser.add_argument(
        "--predictions",
        action="append",
        required=True,
        help="Prediction JSON/JSONL from a shard run. May be repeated.",
    )
    parser.add_argument("--out", required=True, help="Output official submission JSON path.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any question is missing or invalid instead of using safe fallbacks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_rows = _load_dataset(Path(args.dataset_json))
    prediction_rows = _load_prediction_rows(args.predictions)
    merged, summary = build_submission(dataset_rows, prediction_rows, strict=args.strict)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path = out_path.parent / f"{out_path.stem}_merge_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(merged)} submission rows to {out_path}")
    print(f"Wrote merge summary to {summary_path}")


def build_submission(
    dataset_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    *,
    strict: bool = False,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    by_qid: dict[str, dict[str, Any]] = {}
    duplicate_qids: list[str] = []
    for row in prediction_rows:
        qid = _extract_qid(row)
        if qid is None:
            continue
        if qid in by_qid:
            duplicate_qids.append(qid)
        by_qid[qid] = row

    output_rows: list[dict[str, str]] = []
    missing_qids: list[str] = []
    invalid_qids: list[str] = []
    fallback_qids: list[str] = []
    answer_counts: Counter[str] = Counter()

    for sample in dataset_rows:
        qid = _require_qid(sample)
        mode = str(sample.get("mode", "")).strip().lower()
        row = by_qid.get(qid)
        answer = _extract_answer(row) if row else None
        if row is None:
            missing_qids.append(qid)
        if not _is_valid_for_mode(answer, mode):
            if row is not None:
                invalid_qids.append(qid)
            if strict:
                raise ValueError(f"Missing or invalid answer for qid {qid}: {answer!r}")
            answer = _fallback_answer(mode)
            fallback_qids.append(qid)
        output_rows.append({"question_id": qid, "answer_choice": answer})
        answer_counts[answer] += 1

    summary = {
        "dataset_count": len(dataset_rows),
        "prediction_count": len(prediction_rows),
        "submission_count": len(output_rows),
        "duplicate_prediction_qids": sorted(set(duplicate_qids)),
        "missing_qids": missing_qids,
        "invalid_qids": invalid_qids,
        "fallback_qids": fallback_qids,
        "fallback_count": len(fallback_qids),
        "answer_distribution": dict(sorted(answer_counts.items())),
    }
    return output_rows, summary


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return rows


def _load_prediction_rows(paths: list[str]) -> list[dict[str, Any]]:
    from rlm.video.timelogic_auditing import load_prediction_records

    rows: list[dict[str, Any]] = []
    for path in paths:
        loaded, _ = load_prediction_records(path)
        rows.extend(loaded)
    return rows


def _extract_qid(row: dict[str, Any] | None) -> str | None:
    if row is None:
        return None
    for key in ("question_id", "qid", "id"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return None


def _require_qid(row: dict[str, Any]) -> str:
    qid = _extract_qid(row)
    if qid is None:
        raise ValueError(f"Dataset row is missing question_id/qid/id: {row}")
    return qid


def _extract_answer(row: dict[str, Any] | None) -> str | None:
    if row is None:
        return None
    for key in ("answer_choice", "normalized_prediction", "prediction", "pred_answer", "answer"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _is_valid_for_mode(answer: str | None, mode: str) -> bool:
    if mode == "mc":
        return answer in MC_ANSWERS
    if mode in {"boolean", "bool", "yn", "yes_no"}:
        return answer in BOOL_ANSWERS
    return answer in VALID_ANSWERS


def _fallback_answer(mode: str) -> str:
    return "A" if mode == "mc" else "No"


if __name__ == "__main__":
    main()
