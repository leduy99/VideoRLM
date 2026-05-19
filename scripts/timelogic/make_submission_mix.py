from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a mixed TimeLogic submission by replacing selected categories."
    )
    parser.add_argument(
        "--base",
        required=True,
        help="Base prediction JSON or JSONL file.",
    )
    parser.add_argument(
        "--replace-category",
        action="append",
        default=[],
        help="Replacement mapping in the form category=predictions.jsonl. May be repeated.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSON or JSONL file.",
    )
    return parser.parse_args()


def _parse_replacements(entries: list[str]) -> dict[str, str]:
    replacements: dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(
                f"Invalid --replace-category value {entry!r}; expected category=predictions.jsonl"
            )
        category, path = entry.split("=", 1)
        category = category.strip()
        path = path.strip()
        if not category or not path:
            raise ValueError(
                f"Invalid --replace-category value {entry!r}; expected category=predictions.jsonl"
            )
        replacements[category] = path
    return replacements


def _infer_output_format(base_fmt: str, output_path: Path) -> str:
    suffix = output_path.suffix.lower()
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".json":
        return "json"
    return base_fmt


def main() -> None:
    from rlm.video.timelogic_auditing import (
        load_prediction_records,
        mix_submission_by_category,
        write_records,
    )

    args = parse_args()
    base_rows, base_fmt = load_prediction_records(args.base)
    replacement_paths = _parse_replacements(args.replace_category)
    replacement_rows = {
        category: load_prediction_records(path)[0] for category, path in replacement_paths.items()
    }
    mixed_rows = mix_submission_by_category(base_rows=base_rows, replacements=replacement_rows)
    output_path = Path(args.out)
    output_fmt = _infer_output_format(base_fmt, output_path)
    write_records(output_path, mixed_rows, output_fmt)
    print(f"Wrote mixed submission to {output_path}")
    print(f"Rows: {len(mixed_rows)}")


if __name__ == "__main__":
    main()
