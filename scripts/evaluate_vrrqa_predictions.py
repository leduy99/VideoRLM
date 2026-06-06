#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlm.video.vrrqa import (
    VRRQA_ANNOTATION_FILENAME,
    VRRQA_DATASET_PATH,
    VRRQA_SPLIT,
    evaluate_vrrqa_predictions,
    load_vrrqa_samples,
    write_vrrqa_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VRR-QA answer-choice predictions.")
    parser.add_argument("--predictions", help="Path to a VRR-QA predictions JSONL file.")
    parser.add_argument(
        "--output-folder",
        help="Model run output folder containing results.jsonl.",
    )
    parser.add_argument("--annotations", default=f"data/vrrqa/{VRRQA_ANNOTATION_FILENAME}")
    parser.add_argument("--dataset-path", default=VRRQA_DATASET_PATH)
    parser.add_argument("--split", default=VRRQA_SPLIT)
    parser.add_argument(
        "--output-dir",
        help=(
            "Directory for evaluation outputs. When provided, omitted output paths "
            "default to <run-name>_summary.json and <run-name>_report.md."
        ),
    )
    parser.add_argument(
        "--run-name",
        help=(
            "Name used for auto-generated output files. Defaults to the predictions "
            "run directory name."
        ),
    )
    parser.add_argument("--summary-output")
    parser.add_argument("--report-output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictions_path = resolve_predictions_path(args)
    run_output_dir = Path(args.output_folder) if args.output_folder else None
    summary_path, report_output = resolve_output_paths(
        args,
        predictions_path,
        run_output_dir,
    )
    predictions = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    annotation_path = Path(args.annotations)
    samples = load_vrrqa_samples(
        annotation_path=annotation_path if annotation_path.exists() else None,
        dataset_path=args.dataset_path,
        split=args.split,
    )
    summary = evaluate_vrrqa_predictions(predictions, samples)
    summary["run_name"] = infer_run_name(predictions_path, args.run_name)
    summary["predictions_path"] = str(predictions_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path = write_vrrqa_report(summary, report_output)
    print(f"Saved VRR-QA summary to {summary_path}")
    print(f"Saved VRR-QA report to {report_path}")
    print(
        "overall={overall:.2%} macro={macro:.2%} evaluated={evaluated}/{total}".format(
            overall=summary["overall_accuracy"],
            macro=summary["macro_average_accuracy"],
            evaluated=summary["evaluated_rows"],
            total=summary["total_rows"],
        )
    )
    return 0


def resolve_predictions_path(args: argparse.Namespace) -> Path:
    if args.predictions and args.output_folder:
        raise ValueError("Provide only one of --predictions or --output-folder.")
    if args.predictions:
        return Path(args.predictions)
    if args.output_folder:
        return Path(args.output_folder) / "results.jsonl"
    raise ValueError("Provide --predictions or --output-folder.")


def resolve_output_paths(
    args: argparse.Namespace,
    predictions_path: Path,
    run_output_dir: Path | None,
) -> tuple[Path, Path]:
    default_output_dir = Path(args.output_dir) if args.output_dir else run_output_dir
    if default_output_dir is None and (not args.summary_output or not args.report_output):
        raise ValueError(
            "Provide --output-dir, --output-folder, or both --summary-output and "
            "--report-output."
        )
    if default_output_dir is None:
        default_output_dir = Path(".")

    run_name = infer_run_name(predictions_path, args.run_name)
    summary_path = (
        Path(args.summary_output)
        if args.summary_output
        else default_output_dir / f"{run_name}_summary.json"
    )
    report_path = (
        Path(args.report_output)
        if args.report_output
        else default_output_dir / f"{run_name}_report.md"
    )
    return summary_path, report_path


def infer_run_name(predictions_path: Path, explicit_run_name: str | None = None) -> str:
    if explicit_run_name:
        return sanitize_run_name(explicit_run_name)
    if predictions_path.parent.name:
        return sanitize_run_name(predictions_path.parent.name)
    return sanitize_run_name(predictions_path.stem)


def sanitize_run_name(value: str) -> str:
    sanitized = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    )
    return sanitized.strip("._-") or "vrrqa_run"


if __name__ == "__main__":
    raise SystemExit(main())
