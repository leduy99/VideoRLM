#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rlm.video.vrrqa import load_vrrqa_samples, normalize_answer_choice, parse_choice_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert VRR-QA prediction JSONL into EvalAI submission JSON."
    )
    parser.add_argument("--predictions", required=True, help="VideoRLM VRR-QA results.jsonl")
    parser.add_argument("--output", required=True, help="Submission JSON output path")
    parser.add_argument(
        "--annotations",
        help="Optional VRR-QA annotation file used to preserve question order and check coverage.",
    )
    parser.add_argument(
        "--default-choice",
        help="Fallback answer choice for missing or unparseable predictions, e.g. A.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    default_choice = normalize_answer_choice(args.default_choice)
    if args.default_choice is not None and default_choice is None:
        raise ValueError(f"Invalid --default-choice value: {args.default_choice!r}")

    predictions = _load_records(Path(args.predictions))
    prediction_by_question_id = {
        str(record["question_id"]): record for record in predictions if record.get("question_id")
    }
    if args.annotations:
        samples = load_vrrqa_samples(annotation_path=args.annotations)
        question_ids = [str(sample["question_id"]) for sample in samples]
    else:
        question_ids = list(prediction_by_question_id)

    submission: list[dict[str, str]] = []
    missing: list[str] = []
    unparseable: list[str] = []
    for question_id in question_ids:
        prediction = prediction_by_question_id.get(question_id)
        if prediction is None:
            if default_choice is None:
                missing.append(question_id)
                continue
            answer_choice = default_choice
        else:
            answer_choice = _prediction_choice(prediction)
            if answer_choice is None:
                if default_choice is None:
                    unparseable.append(question_id)
                    continue
                answer_choice = default_choice
        submission.append({"question_id": question_id, "answer_choice": answer_choice})

    if missing or unparseable:
        details = []
        if missing:
            details.append(f"missing={len(missing)} first={missing[:5]}")
        if unparseable:
            details.append(f"unparseable={len(unparseable)} first={unparseable[:5]}")
        raise ValueError(
            "Could not create complete VRR-QA submission without --default-choice: "
            + "; ".join(details)
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(submission, indent=2), encoding="utf-8")
    print(f"Saved {len(submission)} VRR-QA submission rows to {output_path}")
    return 0


def _load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON array in {path}, got {type(payload).__name__}")
        return [dict(item) for item in payload]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _prediction_choice(prediction: dict[str, Any]) -> str | None:
    choice = normalize_answer_choice(prediction.get("predicted_choice"))
    if choice is not None:
        return choice
    options = prediction.get("options")
    text = str(prediction.get("prediction") or "")
    if options and text:
        return parse_choice_prediction(text, options)
    return normalize_answer_choice(text)


if __name__ == "__main__":
    raise SystemExit(main())
