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
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--annotations", default=f"data/vrrqa/{VRRQA_ANNOTATION_FILENAME}")
    parser.add_argument("--dataset-path", default=VRRQA_DATASET_PATH)
    parser.add_argument("--split", default=VRRQA_SPLIT)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--report-output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictions = [
        json.loads(line)
        for line in Path(args.predictions).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    annotation_path = Path(args.annotations)
    samples = load_vrrqa_samples(
        annotation_path=annotation_path if annotation_path.exists() else None,
        dataset_path=args.dataset_path,
        split=args.split,
    )
    summary = evaluate_vrrqa_predictions(predictions, samples)
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path = write_vrrqa_report(summary, args.report_output)
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


if __name__ == "__main__":
    raise SystemExit(main())
