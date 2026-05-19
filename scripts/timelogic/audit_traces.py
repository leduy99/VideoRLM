from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit TimeLogic predictions and traces without ground truth."
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Prediction JSON or JSONL file to audit.",
    )
    parser.add_argument(
        "--traces",
        default=None,
        help="Optional trace folder or single trace file. If omitted, uses trace_path fields when present.",
    )
    parser.add_argument(
        "--direct-baseline",
        default=None,
        help="Optional direct baseline prediction JSON/JSONL for disagreement analysis.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for CSV/JSON audit outputs. Defaults to <predictions_stem>_audit/ next to the input file.",
    )
    return parser.parse_args()


def main() -> None:
    from rlm.video.timelogic_auditing import (
        build_audit_samples,
        load_prediction_records,
        write_audit_outputs,
    )

    args = parse_args()
    prediction_path = Path(args.predictions)
    prediction_rows, _ = load_prediction_records(prediction_path)
    baseline_rows = None
    if args.direct_baseline:
        baseline_rows, _ = load_prediction_records(args.direct_baseline)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else prediction_path.parent / f"{prediction_path.stem}_audit"
    )
    rows = build_audit_samples(
        prediction_rows,
        trace_source=args.traces,
        direct_baseline_rows=baseline_rows,
    )
    write_audit_outputs(rows, out_dir)
    print(f"Wrote audit outputs to {out_dir}")
    print(f"Audited {len(rows)} samples")


if __name__ == "__main__":
    main()
