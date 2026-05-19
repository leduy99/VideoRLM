from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


VALID_ANSWERS = {"A", "B", "C", "D", "Yes", "No"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split TimeLogic validation JSON into video-balanced shards."
    )
    parser.add_argument("--dataset-json", required=True, help="Input TimeLogic validation JSON.")
    parser.add_argument("--out-dir", required=True, help="Directory for shard JSON files.")
    parser.add_argument("--num-shards", type=int, default=2, help="Number of shards to write.")
    parser.add_argument("--prefix", default="shard", help="Output file prefix.")
    parser.add_argument(
        "--completed-predictions",
        action="append",
        default=[],
        help="Optional completed prediction JSON/JSONL to skip. May be repeated.",
    )
    parser.add_argument(
        "--videos-dir",
        default=None,
        help="Optional video directory for missing-video checks.",
    )
    parser.add_argument(
        "--skip-missing-videos",
        action="store_true",
        help="If --videos-dir is set, omit samples whose video file is absent.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")

    dataset_path = Path(args.dataset_json)
    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list in {dataset_path}")

    completed_qids = _load_completed_qids(args.completed_predictions)
    videos_dir = Path(args.videos_dir) if args.videos_dir else None
    kept_rows: list[dict[str, Any]] = []
    skipped_completed: list[str] = []
    skipped_missing_video: list[str] = []

    for row in rows:
        qid = _extract_qid(row)
        video_id = _extract_video_id(row)
        if qid in completed_qids:
            skipped_completed.append(qid)
            continue
        if videos_dir and video_id and not (videos_dir / video_id).exists():
            skipped_missing_video.append(qid)
            if args.skip_missing_videos:
                continue
        kept_rows.append(row)

    shards = _video_balanced_split(kept_rows, args.num_shards)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_files = []
    for index, shard_rows in enumerate(shards):
        path = out_dir / f"{args.prefix}_{index}_dataset.json"
        path.write_text(json.dumps(shard_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        shard_files.append(str(path))
        print(f"Wrote {len(shard_rows)} rows to {path}")

    summary = {
        "dataset_json": str(dataset_path),
        "num_input_rows": len(rows),
        "num_shards": args.num_shards,
        "num_kept_rows": len(kept_rows),
        "num_skipped_completed": len(skipped_completed),
        "num_missing_video": len(skipped_missing_video),
        "skip_missing_videos": args.skip_missing_videos,
        "shard_files": shard_files,
        "shard_sizes": [len(shard) for shard in shards],
        "skipped_completed_qids": skipped_completed,
        "missing_video_qids": skipped_missing_video,
    }
    summary_path = out_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote split summary to {summary_path}")


def _video_balanced_split(rows: list[dict[str, Any]], num_shards: int) -> list[list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_extract_video_id(row) or f"qid:{_extract_qid(row)}"].append(row)

    shards: list[list[dict[str, Any]]] = [[] for _ in range(num_shards)]
    shard_sizes = [0] * num_shards
    for _, group in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        target = min(range(num_shards), key=lambda index: (shard_sizes[index], index))
        shards[target].extend(group)
        shard_sizes[target] += len(group)

    original_order = {_extract_qid(row): index for index, row in enumerate(rows)}
    for shard in shards:
        shard.sort(key=lambda row: original_order.get(_extract_qid(row), 10**12))
    return shards


def _load_completed_qids(paths: list[str]) -> set[str]:
    from rlm.video.timelogic_auditing import load_prediction_records

    completed: set[str] = set()
    for path in paths:
        rows, _ = load_prediction_records(path)
        for row in rows:
            qid = _extract_qid(row)
            answer = _extract_answer(row)
            if qid and answer in VALID_ANSWERS and not row.get("error"):
                completed.add(qid)
    return completed


def _extract_qid(row: dict[str, Any]) -> str:
    for key in ("question_id", "qid", "id"):
        if row.get(key) is not None:
            return str(row[key])
    raise ValueError(f"Missing question id in row: {row}")


def _extract_video_id(row: dict[str, Any]) -> str | None:
    value = row.get("video_id") or row.get("video")
    return str(value) if value else None


def _extract_answer(row: dict[str, Any]) -> str | None:
    for key in ("answer_choice", "normalized_prediction", "prediction", "pred_answer", "answer"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


if __name__ == "__main__":
    main()
