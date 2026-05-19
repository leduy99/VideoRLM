import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlm.video import QwenLocalVideoStackConfig, VideoMemoryBuilder  # noqa: E402
from rlm.video.media import probe_media_duration  # noqa: E402

DEFAULT_SAMPLE_IDS = ["1", "2", "8", "20"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VideoRLM inference for TimeLogic validation.")
    parser.add_argument(
        "--dataset-json",
        default="output/timelogic/timelogic_val_data.json",
        help="Path to timelogic_val_data.json",
    )
    parser.add_argument(
        "--videos-dir",
        default="output/timelogic/combined_2k_videos",
        help="Directory containing extracted validation videos",
    )
    parser.add_argument(
        "--output-dir",
        default="output/timelogic/infer_local_qwen3_8b",
        help="Directory where predictions, traces, artifacts, and memories will be written",
    )
    parser.add_argument(
        "--sample-id",
        action="append",
        default=[],
        help="Specific question_id to run. May be repeated.",
    )
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="Run every sample in the validation JSON instead of the default small slice.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing predictions.jsonl file in output-dir.",
    )
    parser.add_argument(
        "--submission-json",
        default=None,
        help="Optional explicit path for submission JSON. Defaults to <output-dir>/submission.json",
    )
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--search-top-k", type=int, default=5)
    parser.add_argument("--max-frontier-items", type=int, default=8)
    parser.add_argument("--scene-duration-seconds", type=float, default=120.0)
    parser.add_argument("--segment-duration-seconds", type=float, default=30.0)
    parser.add_argument("--clip-duration-seconds", type=float, default=10.0)
    parser.add_argument("--controller-device", default="cuda:0")
    parser.add_argument("--visual-device", default="cuda:0")
    parser.add_argument("--speech-device", default="cuda:0")
    parser.add_argument("--controller-repo", default="Qwen/Qwen3-8B")
    parser.add_argument("--visual-repo", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--speech-repo", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument(
        "--forced-aligner-repo",
        default="Qwen/Qwen3-ForcedAligner-0.6B",
    )
    parser.add_argument("--no-forced-aligner", action="store_true")
    parser.add_argument(
        "--visual-only",
        action="store_true",
        help="Disable ASR and build video memory from visual summaries only.",
    )
    return parser.parse_args()


def load_samples(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def choose_samples(rows: list[dict[str, Any]], sample_ids: list[str]) -> list[dict[str, Any]]:
    wanted = sample_ids or DEFAULT_SAMPLE_IDS
    by_id = {str(row["question_id"]): row for row in rows}
    missing = [sample_id for sample_id in wanted if sample_id not in by_id]
    if missing:
        raise ValueError(f"Unknown sample_id(s): {missing}")
    return [by_id[sample_id] for sample_id in wanted]


def choose_all_samples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def normalize_prediction(mode: str, answer: str) -> str | None:
    text = answer.strip()
    if not text:
        return None
    if mode == "mc":
        compact = text.strip()
        if compact in {"A", "B", "C", "D"}:
            return compact
        match = re.search(
            r"\b(?:option|chosen option|correct option)\s*[:is]*\s*([ABCD])\b",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).upper()
        match = re.search(r"\bthe answer is\s+([ABCD])\b", text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return None
    lowered = text.lower()
    if re.search(r"\b(yes|true)\b", lowered):
        return "Yes"
    if re.search(r"\b(no|false)\b", lowered):
        return "No"
    return None


def format_question(sample: dict[str, Any]) -> str:
    question = sample["question"].strip()
    if sample["mode"] == "mc":
        if "one character" not in question.lower():
            return question + " Answer with exactly one character: A, B, C, or D."
        return question
    return question + " Answer with exactly one word: Yes or No."


def build_answer_choice_prompt(
    *,
    sample: dict[str, Any],
    raw_answer: str,
    evidence_claims: list[str],
) -> str:
    mode = sample["mode"]
    target = "exactly one character: A, B, C, or D" if mode == "mc" else "exactly one word: Yes or No"
    evidence_blob = "\n".join(f"- {claim}" for claim in evidence_claims[:6]) or "- No strong evidence extracted."
    return (
        "You are converting a VideoRLM result into a TimeLogic validation submission label.\n"
        f"Question mode: {mode}\n"
        f"Question: {sample['question']}\n"
        f"Current model answer: {raw_answer}\n"
        "Evidence summary:\n"
        f"{evidence_blob}\n\n"
        f"Return {target}.\n"
        "Do not explain. Output only the final label."
    )


def build_bundle(args: argparse.Namespace):
    config = QwenLocalVideoStackConfig.default(
        controller_device=args.controller_device,
        visual_device=args.visual_device,
        speech_device=args.speech_device,
        controller_model=args.controller_repo,
        visual_model=args.visual_repo,
        speech_model=args.speech_repo,
        forced_aligner_model=None if args.no_forced_aligner else args.forced_aligner_repo,
    )
    config.scene_duration_seconds = args.scene_duration_seconds
    config.segment_duration_seconds = args.segment_duration_seconds
    config.clip_duration_seconds = args.clip_duration_seconds
    config.controller_enable_thinking = False
    bundle = config.build_bundle(
        max_steps=args.max_steps,
        search_top_k=args.search_top_k,
        max_frontier_items=args.max_frontier_items,
    )
    if args.visual_only:
        bundle.memory_builder.speech_recognizer = None
    return bundle


def artifact_dir_name(video_id: str) -> str:
    return video_id.removesuffix(".mp4")


def run_sample(
    sample: dict[str, Any],
    *,
    bundle,
    builder: VideoMemoryBuilder,
    videos_dir: Path,
    artifacts_root: Path,
    memories_root: Path,
    traces_root: Path,
) -> dict[str, Any]:
    video_path = videos_dir / sample["video_id"]
    if not video_path.exists():
        raise FileNotFoundError(f"Missing video: {video_path}")

    duration_seconds = probe_media_duration(video_path)
    artifact_dir = artifacts_root / artifact_dir_name(sample["video_id"])
    memory_path = memories_root / f"{video_path.stem}.json"

    if memory_path.exists():
        memory = builder.load_memory(memory_path)
    else:
        if artifact_dir.exists():
            artifacts = builder.load_artifacts_dir(artifact_dir)
        else:
            artifacts = builder.prepare_artifacts(
                video_path=str(video_path),
                duration_seconds=duration_seconds,
                video_id=video_path.stem,
            )
            builder.save_artifacts_dir(artifacts, artifact_dir)
        memory = builder.build_from_artifacts(artifacts)
        builder.save_memory(memory, memory_path)

    result = bundle.controller.run(
        format_question(sample),
        memory,
        task_type="timelogic_temporal_reasoning",
    )
    trace_path = traces_root / f"sample_{sample['question_id']}.json"
    trace_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    normalized_prediction = normalize_prediction(sample["mode"], result.answer)
    answer_choice_source = "raw_answer"
    if normalized_prediction is None:
        evidence_claims = [item.claim for item in result.state.evidence_ledger[:6]]
        choice_prompt = build_answer_choice_prompt(
            sample=sample,
            raw_answer=result.answer,
            evidence_claims=evidence_claims,
        )
        choice_text = bundle.controller.controller_client.completion(choice_prompt)
        normalized_prediction = normalize_prediction(sample["mode"], choice_text)
        answer_choice_source = "choice_head"
        if normalized_prediction is None:
            normalized_prediction = "A" if sample["mode"] == "mc" else "No"
            answer_choice_source = "default_fallback"

    return {
        "question_id": str(sample["question_id"]),
        "video_id": sample["video_id"],
        "mode": sample["mode"],
        "question": sample["question"],
        "formatted_question": format_question(sample),
        "raw_answer": result.answer,
        "normalized_prediction": normalized_prediction,
        "answer_choice": normalized_prediction,
        "answer_choice_source": answer_choice_source,
        "duration_seconds": duration_seconds,
        "execution_time": result.execution_time,
        "steps_used": len(result.trace),
        "tool_calls_used": sum(
            1
            for step in result.trace
            if step.get("action", {}).get("action_type") not in {None, "STOP"}
        ),
        "trace_path": str(trace_path),
        "memory_path": str(memory_path),
        "artifact_dir": str(artifact_dir),
    }


def build_submission_rows(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for record in records:
        choice = record.get("answer_choice")
        if not choice:
            choice = "A" if record.get("mode") == "mc" else "No"
        rows.append({"question_id": str(record["question_id"]), "answer_choice": str(choice)})
    return rows


def write_submission_json(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        json.dumps(build_submission_rows(records), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_existing_records(predictions_path: Path) -> list[dict[str, Any]]:
    if not predictions_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in predictions_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset_json)
    videos_dir = Path(args.videos_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_root = output_dir / "artifacts"
    memories_root = output_dir / "memories"
    traces_root = output_dir / "traces"
    for path in [artifacts_root, memories_root, traces_root]:
        path.mkdir(parents=True, exist_ok=True)

    rows = load_samples(dataset_path)
    samples = choose_all_samples(rows) if args.all_samples else choose_samples(rows, args.sample_id)

    bundle = build_bundle(args)
    builder = bundle.memory_builder

    predictions_path = output_dir / "predictions.jsonl"
    submission_path = Path(args.submission_json) if args.submission_json else output_dir / "submission.json"
    existing_records = load_existing_records(predictions_path) if args.resume else []
    existing_ids = {str(record["question_id"]) for record in existing_records}
    if predictions_path.exists() and not args.resume:
        predictions_path.unlink()
    records: list[dict[str, Any]] = list(existing_records)
    for sample in samples:
        if str(sample["question_id"]) in existing_ids:
            continue
        try:
            record = run_sample(
                sample,
                bundle=bundle,
                builder=builder,
                videos_dir=videos_dir,
                artifacts_root=artifacts_root,
                memories_root=memories_root,
                traces_root=traces_root,
            )
        except Exception as exc:  # noqa: BLE001
            record = {
                "question_id": str(sample["question_id"]),
                "video_id": sample["video_id"],
                "mode": sample["mode"],
                "question": sample["question"],
                "formatted_question": format_question(sample),
                "error": f"{type(exc).__name__}: {exc}",
            }
        records.append(record)
        with predictions_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        write_submission_json(submission_path, records)
        if "error" in record:
            print(f"[error] qid={record['question_id']} {record['error']}")
        else:
            print(
                f"[done] qid={record['question_id']} mode={record['mode']} "
                f"pred={record['answer_choice']} source={record['answer_choice_source']} raw={record['raw_answer']!r}"
            )

    summary = {
        "dataset_json": str(dataset_path),
        "videos_dir": str(videos_dir),
        "sample_ids": [record["question_id"] for record in records],
        "prediction_count": len(records),
        "predictions_path": str(predictions_path),
        "submission_path": str(submission_path),
        "all_samples": args.all_samples,
        "resume": args.resume,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
