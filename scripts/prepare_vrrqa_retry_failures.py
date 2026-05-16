#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from rlm.video.vrrqa import safe_identifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove retryable skipped VRR-QA rows from a predictions JSONL file so the "
            "benchmark runner can resume them."
        )
    )
    parser.add_argument("--predictions", required=True, help="VRR-QA results JSONL to edit")
    parser.add_argument(
        "--reason-prefix",
        action="append",
        default=[],
        help="Retry skipped rows whose skip_reason starts with this prefix.",
    )
    parser.add_argument("--segment-dir", default="data/vrrqa/segments")
    parser.add_argument("--artifacts-dir")
    parser.add_argument("--memory-dir")
    parser.add_argument("--trace-dir")
    parser.add_argument("--retry-list")
    parser.add_argument(
        "--purge-caches",
        action="store_true",
        help="Delete artifacts, memory, segments, and traces for every selected row.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictions_path = Path(args.predictions)
    output_root = predictions_path.parent
    reason_prefixes = tuple(args.reason_prefix or ["KeyError", "CalledProcessError"])
    segment_dir = Path(args.segment_dir)
    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else output_root / "artifacts"
    memory_dir = Path(args.memory_dir) if args.memory_dir else output_root / "memories"
    trace_dir = Path(args.trace_dir) if args.trace_dir else output_root / "traces"
    retry_list = Path(args.retry_list) if args.retry_list else output_root / "retry_question_ids.txt"

    rows = _read_jsonl(predictions_path)
    retry_indexes = {
        index
        for index, row in enumerate(rows)
        if row.get("skipped") and _matches_reason(row.get("skip_reason"), reason_prefixes)
    }
    retry_rows = [row for index, row in enumerate(rows) if index in retry_indexes]
    kept_rows = [row for index, row in enumerate(rows) if index not in retry_indexes]

    cleanup_paths = _cleanup_paths(
        retry_rows,
        segment_dir=segment_dir,
        artifacts_dir=artifacts_dir,
        memory_dir=memory_dir,
        trace_dir=trace_dir,
        purge_caches=args.purge_caches,
    )

    print(f"predictions={predictions_path}")
    print(f"selected_retry_rows={len(retry_rows)}")
    print(f"kept_rows={len(kept_rows)}")
    print(f"cleanup_paths={len(cleanup_paths)}")
    print(f"retry_list={retry_list}")
    if args.dry_run:
        for path in cleanup_paths[:20]:
            print(f"would remove {path}")
        if len(cleanup_paths) > 20:
            print(f"... {len(cleanup_paths) - 20} more paths")
        return 0

    backup_path = predictions_path.with_suffix(predictions_path.suffix + ".bak")
    shutil.copy2(predictions_path, backup_path)
    _write_jsonl(predictions_path, kept_rows)
    retry_list.parent.mkdir(parents=True, exist_ok=True)
    retry_list.write_text(
        "".join(f"{row['question_id']}\n" for row in retry_rows),
        encoding="utf-8",
    )
    for path in cleanup_paths:
        _remove_path(path)
    print(f"backup={backup_path}")
    return 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=False)
            handle.write("\n")


def _matches_reason(reason: Any, prefixes: tuple[str, ...]) -> bool:
    reason_text = str(reason or "")
    return any(reason_text.startswith(prefix) for prefix in prefixes)


def _cleanup_paths(
    rows: list[dict[str, Any]],
    *,
    segment_dir: Path,
    artifacts_dir: Path,
    memory_dir: Path,
    trace_dir: Path,
    purge_caches: bool,
) -> list[Path]:
    paths: list[Path] = []
    for row in rows:
        cache_id = safe_identifier(f"{row['video_id']}_{row['question_id']}")
        reason = str(row.get("skip_reason") or "")
        full_cleanup = purge_caches or reason.startswith("CalledProcessError")
        if full_cleanup:
            paths.extend(
                [
                    segment_dir / f"{cache_id}.mp4",
                    artifacts_dir / cache_id,
                    memory_dir / f"{cache_id}.json",
                ]
            )
        paths.append(trace_dir / f"{cache_id}.json")
    return sorted({path for path in paths if path.exists()})


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
