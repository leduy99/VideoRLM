#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

DEFAULT_RESULTS_DIR = Path("output/timelogic_full_4b_9b")
DEFAULT_SUBMISSION_NAME = "submission.json"
TIMELOGIC_ANNOTATION_PATH = "data/TimeLogic/timelogic_test_data.json"
TIMELOGIC_BOOL_OPTIONS = {"A": "yes", "B": "no"}
BOOL_ANSWER_TEXT = {"yes": "Yes", "no": "No"}
DEFAULT_MISSING_MC_CHOICE = "A"
DEFAULT_MISSING_BOOL_ANSWER = "No"
OPTION_PATTERN = re.compile(
    r"Option\s+([A-Z])\s*:\s*(.*?)(?=,\s*Option\s+[A-Z]\s*:|\.?\s*Reply with|$)",
    flags=re.IGNORECASE | re.DOTALL,
)
DIRECT_CHOICE_PATTERN = re.compile(r"^(?:OPTION[\s_-]*)?([A-Z])[\).]?$")
LABELED_CHOICE_PATTERN = re.compile(
    r"\b(?:ANSWER|CHOICE)\s*(?:IS|=|:)\s*\(?([A-Z])\)?\b"
    r"|\bOPTION[\s_:=.-]*\(?([A-Z])\)?\b",
    flags=re.IGNORECASE,
)
LEADING_CHOICE_PATTERN = re.compile(r"^\s*([A-Z])\s*(?:[).:]|-)\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert TimeLogic prediction results into a submission JSON list."
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Run output directory containing results.jsonl.",
    )
    parser.add_argument(
        "--predictions",
        help="Prediction JSONL/JSON path. Defaults to <results-dir>/results.jsonl.",
    )
    parser.add_argument(
        "--output",
        help="Submission JSON path. Defaults to <results-dir>/submission.json.",
    )
    parser.add_argument(
        "--annotations",
        default=TIMELOGIC_ANNOTATION_PATH,
        help="Optional TimeLogic annotations used to preserve dataset order for completed rows.",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Require every annotation question_id to have a prediction.",
    )
    parser.add_argument(
        "--fill-missing-defaults",
        action="store_true",
        help=(
            "Write every annotation question and fill missing or unparseable predictions "
            "with task-type defaults: A for multiple-choice, No for boolean."
        ),
    )
    parser.add_argument(
        "--default-mc-choice",
        choices=["A", "B", "C", "D"],
        help=(
            "Fallback answer choice for missing or unparseable multiple-choice predictions. "
            "Also overrides the multiple-choice default used by --fill-missing-defaults."
        ),
    )
    parser.add_argument(
        "--default-bool-answer",
        choices=["Yes", "No", "yes", "no"],
        help=(
            "Fallback answer for missing or unparseable boolean predictions. "
            "Also overrides the boolean default used by --fill-missing-defaults."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    predictions_path = Path(args.predictions) if args.predictions else results_dir / "results.jsonl"
    output_path = Path(args.output) if args.output else results_dir / DEFAULT_SUBMISSION_NAME

    records = load_records(predictions_path)
    records_by_question_id = index_records(records)
    samples_by_question_id = load_samples_by_question_id(Path(args.annotations))
    if args.fill_missing_defaults and not samples_by_question_id:
        raise ValueError("--fill-missing-defaults requires a valid --annotations file")
    question_ids = ordered_question_ids(
        records_by_question_id,
        samples_by_question_id,
        args.require_all or args.fill_missing_defaults,
    )

    default_mc_choice = args.default_mc_choice
    default_bool_answer = normalize_default_bool_answer(args.default_bool_answer)
    if args.fill_missing_defaults:
        default_mc_choice = default_mc_choice or DEFAULT_MISSING_MC_CHOICE
        default_bool_answer = default_bool_answer or DEFAULT_MISSING_BOOL_ANSWER

    submission: list[dict[str, str]] = []
    missing: list[str] = []
    unparseable: list[str] = []

    for question_id in question_ids:
        record = records_by_question_id.get(question_id)
        sample = samples_by_question_id.get(question_id)
        if record is None:
            if sample is None:
                missing.append(question_id)
                continue
            fallback = fallback_answer(sample["mode"], default_mc_choice, default_bool_answer)
            if fallback is None:
                missing.append(question_id)
                continue
            submission.append({"question_id": question_id, "answer_choice": fallback})
            continue

        answer_choice = submission_answer_choice(record, sample)
        if answer_choice is None:
            mode = sample["mode"] if sample is not None else str(record.get("mode") or "")
            answer_choice = fallback_answer(mode, default_mc_choice, default_bool_answer)
        if answer_choice is None:
            unparseable.append(question_id)
            continue
        submission.append({"question_id": question_id, "answer_choice": answer_choice})

    if missing or unparseable:
        details = []
        if missing:
            details.append(f"missing={len(missing)} first={missing[:5]}")
        if unparseable:
            details.append(f"unparseable={len(unparseable)} first={unparseable[:5]}")
        raise ValueError(
            "Could not create complete TimeLogic submission without fallback arguments: "
            + "; ".join(details)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(submission, indent=2), encoding="utf-8")
    print(f"Saved {len(submission)} TimeLogic submission rows to {output_path}")
    return 0


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON array in {path}, got {type(payload).__name__}")
        return [dict(item) for item in payload]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def index_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for record in records:
        question_id = str(record.get("question_id") or "")
        if not question_id:
            raise ValueError(f"Prediction record is missing question_id: {record}")
        if question_id in indexed:
            duplicates.append(question_id)
            continue
        indexed[question_id] = record
    if duplicates:
        raise ValueError(f"Duplicate prediction question_id values: {duplicates[:10]}")
    return indexed


def load_samples_by_question_id(annotation_path: Path) -> dict[str, dict[str, Any]]:
    if not annotation_path.exists():
        return {}
    samples = load_timelogic_samples(annotation_path)
    return {str(sample["question_id"]): sample for sample in samples}


def ordered_question_ids(
    records_by_question_id: dict[str, dict[str, Any]],
    samples_by_question_id: dict[str, dict[str, Any]],
    require_all: bool,
) -> list[str]:
    if not samples_by_question_id:
        return list(records_by_question_id)
    if require_all:
        return list(samples_by_question_id)
    annotation_order = [
        question_id
        for question_id in samples_by_question_id
        if question_id in records_by_question_id
    ]
    extra_question_ids = [
        question_id
        for question_id in records_by_question_id
        if question_id not in samples_by_question_id
    ]
    return annotation_order + extra_question_ids


def submission_answer_choice(
    record: dict[str, Any],
    sample: dict[str, Any] | None,
) -> str | None:
    mode = resolve_mode(record, sample)
    options = resolve_options(record, sample)
    if mode == "bool":
        return bool_submission_answer(record, options)
    return choice_submission_answer(record, options)


def resolve_mode(record: dict[str, Any], sample: dict[str, Any] | None) -> str:
    if sample is not None:
        return str(sample["mode"])
    mode = str(record.get("mode") or "").strip().lower()
    if mode in {"bool", "boolean"}:
        return "bool"
    if mode in {"mc", "multiple_choice", "multiple-choice"}:
        return "mc"
    options = record.get("options")
    if isinstance(options, dict) and _normalized_options(options) == TIMELOGIC_BOOL_OPTIONS:
        return "bool"
    raise ValueError(
        f"Could not resolve TimeLogic mode for question_id={record.get('question_id')}"
    )


def resolve_options(record: dict[str, Any], sample: dict[str, Any] | None) -> dict[str, str]:
    if sample is not None:
        return timelogic_options(sample)
    options = record.get("options")
    if not isinstance(options, dict) or not options:
        raise ValueError(
            f"Could not resolve TimeLogic options for question_id={record.get('question_id')}"
        )
    return _normalized_options(options)


def bool_submission_answer(record: dict[str, Any], options: dict[str, str]) -> str | None:
    for candidate in prediction_candidates(record):
        bool_answer = normalize_bool_prediction(str(candidate))
        if bool_answer is not None:
            return BOOL_ANSWER_TEXT[bool_answer]
        choice = parse_choice_prediction(str(candidate), options)
        if choice is not None:
            option_text = normalize_bool_prediction(options[choice])
            if option_text is not None:
                return BOOL_ANSWER_TEXT[option_text]
    return None


def choice_submission_answer(record: dict[str, Any], options: dict[str, str]) -> str | None:
    for candidate in prediction_candidates(record):
        choice = parse_choice_prediction(str(candidate), options)
        if choice is not None:
            return choice
    return None


def prediction_candidates(record: dict[str, Any]) -> list[Any]:
    candidates = [
        record.get("prediction"),
        record.get("predicted_choice"),
        record.get("finalizer_prediction"),
        record.get("raw_prediction"),
    ]
    return [candidate for candidate in candidates if candidate is not None]


def fallback_answer(
    mode: str,
    default_mc_choice: str | None,
    default_bool_answer: str | None,
) -> str | None:
    normalized_mode = mode.strip().lower()
    if normalized_mode in {"bool", "boolean"}:
        return default_bool_answer
    if normalized_mode in {"mc", "multiple_choice", "multiple-choice"}:
        return default_mc_choice
    return None


def normalize_default_bool_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    normalized = normalize_bool_prediction(answer)
    if normalized is None:
        raise ValueError(f"Invalid --default-bool-answer value: {answer!r}")
    return BOOL_ANSWER_TEXT[normalized]


def _normalized_options(options: dict[Any, Any]) -> dict[str, str]:
    return {str(key).upper(): str(value) for key, value in options.items()}


def load_timelogic_samples(annotation_path: Path) -> list[dict[str, str]]:
    samples = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        raise ValueError(f"TimeLogic annotations must be a JSON list: {annotation_path}")
    normalized = [normalize_sample(sample) for sample in samples]
    normalized.sort(key=lambda item: (_sort_key(item["question_id"]), item["video_id"]))
    return normalized


def normalize_sample(sample: dict[str, Any]) -> dict[str, str]:
    required = {"question_id", "video_id", "mode", "question"}
    missing = required - set(sample)
    if missing:
        raise ValueError(f"TimeLogic sample missing fields: {sorted(missing)}")
    return {
        "question_id": str(sample["question_id"]),
        "video_id": str(sample["video_id"]),
        "mode": normalize_mode(sample["mode"]),
        "question": str(sample["question"]).strip(),
    }


def normalize_mode(value: Any) -> str:
    mode = str(value).strip().lower()
    if mode in {"bool", "boolean"}:
        return "bool"
    if mode in {"mc", "multiple_choice", "multiple-choice"}:
        return "mc"
    raise ValueError(f"Unsupported TimeLogic mode: {value!r}")


def timelogic_options(sample: dict[str, Any]) -> dict[str, str]:
    if normalize_mode(sample["mode"]) == "bool":
        return dict(TIMELOGIC_BOOL_OPTIONS)
    return parse_timelogic_options(str(sample["question"]))


def parse_timelogic_options(question: str) -> dict[str, str]:
    options: dict[str, str] = {}
    for match in OPTION_PATTERN.finditer(question):
        letter = match.group(1).upper()
        text = " ".join(match.group(2).split()).strip(" .,")
        if text:
            options[letter] = text
    if not options:
        raise ValueError(f"Could not parse TimeLogic options from question: {question}")
    return dict(sorted(options.items()))


def parse_choice_prediction(prediction: str, options: dict[str, str]) -> str | None:
    valid_choices = set(options)
    normalized = prediction.strip().upper()
    if normalized in valid_choices:
        return normalized
    direct_match = DIRECT_CHOICE_PATTERN.fullmatch(normalized)
    if direct_match is not None and direct_match.group(1) in valid_choices:
        return direct_match.group(1)
    leading_match = LEADING_CHOICE_PATTERN.match(normalized)
    if leading_match is not None and leading_match.group(1) in valid_choices:
        return leading_match.group(1)
    for match in LABELED_CHOICE_PATTERN.finditer(prediction):
        choice = match.group(1) or match.group(2)
        if choice and choice.upper() in valid_choices:
            return choice.upper()

    normalized_text = normalize_text(prediction)
    for choice, option_text in options.items():
        if normalized_text == normalize_text(option_text):
            return choice
    contained_choices = [
        choice
        for choice, option_text in options.items()
        if normalize_text(option_text) and normalize_text(option_text) in normalized_text
    ]
    if len(contained_choices) == 1:
        return contained_choices[0]
    return None


def normalize_bool_prediction(prediction: str | None) -> str | None:
    if prediction is None:
        return None
    text = " ".join(str(prediction).strip().lower().split())
    if not text:
        return None
    labeled = re.search(
        r"\b(?:answer|final answer|choice)\s*(?:is|=|:)?\s*(yes|no|true|false)\b",
        text,
    )
    if labeled is not None:
        return bool_word_to_answer(labeled.group(1))
    first_token = re.match(r"^(yes|no|true|false)\b", text)
    if first_token is not None:
        return bool_word_to_answer(first_token.group(1))
    if text in {"a", "option a"}:
        return "yes"
    if text in {"b", "option b"}:
        return "no"
    return None


def bool_word_to_answer(value: str) -> str:
    if value.lower() in {"yes", "true"}:
        return "yes"
    return "no"


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _sort_key(question_id: str) -> tuple[int, int | str]:
    if str(question_id).isdigit():
        return (0, int(question_id))
    return (1, str(question_id))


if __name__ == "__main__":
    raise SystemExit(main())
