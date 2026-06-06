from __future__ import annotations

import json
import math
import re
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlm.video.controller import VideoRLM
from rlm.video.longshot import VIDEO_EXTENSIONS
from rlm.video.media import probe_media_duration
from rlm.video.memory import VideoMemoryBuilder
from rlm.video.prompt_plugins import (
    BenchmarkPromptPlugin,
    prompt_plugin_context,
    render_prompt_plugin_section,
)
from rlm.video.types import EventMemory, TimeSpan, VideoMemory

TIMELOGIC_ANNOTATION_PATH = "data/TimeLogic/timelogic_test_data.json"
TIMELOGIC_VIDEO_DIR = "data/TimeLogic/benchmark_test_videos_json"
TIMELOGIC_BOOL_OPTIONS = {"A": "yes", "B": "no"}
TIMELOGIC_TASK_TYPE = "multiple_choice_visual_qa"
SPEECH_PROGRESS_UNIT_WEIGHT = 1
GENERIC_VISUAL_PROGRESS_UNIT_WEIGHT = 1
LOCAL_QWEN_VISUAL_PROGRESS_UNIT_WEIGHT = 6
TIMELOGIC_OPERATOR_GUIDE = [
    {
        "operator": "eventual",
        "pattern": "Does the person eventually X?",
        "verification": "Answer yes iff X has at least one localized interval.",
    },
    {
        "operator": "always",
        "pattern": "Is the person always X?",
        "verification": "Answer yes only when X is supported throughout the checked video span.",
    },
    {
        "operator": "until",
        "pattern": "Did the person X until Y?",
        "verification": "Find Y, then verify X continues before Y happens.",
    },
    {
        "operator": "since",
        "pattern": "Has the person been X since they Y?",
        "verification": "Find Y, then verify X holds after Y.",
    },
    {
        "operator": "disjoint",
        "pattern": "Is it true that X does not overlap with Y?",
        "verification": "Answer yes iff no localized X interval overlaps a Y interval.",
    },
    {
        "operator": "imply",
        "pattern": "Does X imply Y?",
        "verification": "Answer yes iff every visible X occurrence has supporting Y.",
    },
    {
        "operator": "before",
        "pattern": "Did the person X before Y?",
        "verification": "Answer yes iff a localized X interval starts before Y.",
    },
    {
        "operator": "after",
        "pattern": "Did the person X after Y?",
        "verification": "Treat as Y before X.",
    },
    {
        "operator": "cooccur",
        "pattern": "Do X and Y co-occur?",
        "verification": "Answer yes iff an X interval overlaps a Y interval.",
    },
    {
        "operator": "immediate_next",
        "pattern": "Did X immediately after Y?",
        "verification": "Answer yes iff X starts right after Y with no unrelated gap.",
    },
    {
        "operator": "always_before",
        "pattern": "Did X always before Y?",
        "verification": "Answer yes iff every Y is preceded by X.",
    },
    {
        "operator": "always_after",
        "pattern": "Did X always after Y?",
        "verification": "Answer yes iff every Y is followed by X.",
    },
    {
        "operator": "always_cooccur",
        "pattern": "Does X always co-occur with Y?",
        "verification": "Answer yes iff every X occurrence overlaps Y.",
    },
    {
        "operator": "strict_chain",
        "pattern": "Does A always occur before B, which always occurs before C?",
        "verification": "Verify strict repeated ordering A -> B -> C.",
    },
    {
        "operator": "loose_chain",
        "pattern": "Does A occur before B, which occurs before C?",
        "verification": "Verify at least one localized A -> B -> C chain.",
    },
    {
        "operator": "one_before_two",
        "pattern": "Does A always occur before B and C?",
        "verification": "Answer yes iff A precedes both B and C.",
    },
]

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
SAFE_IDENTIFIER_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")
SKIPPABLE_SAMPLE_ERRORS = (
    FileNotFoundError,
    KeyError,
    OSError,
    RuntimeError,
    ValueError,
    subprocess.CalledProcessError,
)


@dataclass
class TimeLogicResult:
    question_id: str
    video_id: str
    mode: str
    prediction: str
    raw_prediction: str
    predicted_choice: str | None


def load_timelogic_samples(
    annotation_path: str | Path = TIMELOGIC_ANNOTATION_PATH,
    *,
    sample_limit: int | None = None,
    question_ids: Sequence[str] | None = None,
    video_ids: Sequence[str] | None = None,
    modes: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    path = Path(annotation_path)
    samples = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        raise ValueError(f"TimeLogic annotations must be a JSON list: {path}")

    normalized = [_normalize_sample(sample) for sample in samples]
    question_filter = {str(item) for item in question_ids or []}
    video_filter = {str(item) for item in video_ids or []}
    mode_filter = {_normalize_mode(item) for item in modes or []}

    if question_filter:
        normalized = [item for item in normalized if item["question_id"] in question_filter]
    if video_filter:
        normalized = [item for item in normalized if item["video_id"] in video_filter]
    if mode_filter:
        normalized = [item for item in normalized if item["mode"] in mode_filter]

    normalized.sort(key=lambda item: (_sort_key(item["question_id"]), item["video_id"]))
    if sample_limit is not None:
        return normalized[:sample_limit]
    return normalized


class TimeLogicVideoResolver:
    def __init__(self, video_dir: str | Path):
        self.video_dir = Path(video_dir)

    def resolve(self, sample_or_video_id: dict[str, Any] | str) -> Path:
        video_id = (
            str(sample_or_video_id["video_id"])
            if isinstance(sample_or_video_id, dict)
            else str(sample_or_video_id)
        )
        existing = self.find(video_id)
        if existing is None:
            raise FileNotFoundError(
                f"Could not find local TimeLogic video_id={video_id} under {self.video_dir}"
            )
        return existing

    def find(self, video_id: str) -> Path | None:
        direct_name = Path(video_id).name
        direct = self.video_dir / direct_name
        if direct.exists() and direct.is_file():
            return direct

        stem = Path(direct_name).stem
        if Path(direct_name).suffix:
            matches = [
                path
                for path in self.video_dir.rglob(direct_name)
                if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
            ]
        else:
            matches = [
                path
                for path in self.video_dir.rglob(f"{stem}.*")
                if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
            ]
        return sorted(matches)[0] if matches else None


class TimeLogicBenchmarkRunner:
    def __init__(
        self,
        *,
        video_rlm: VideoRLM,
        memory_builder: VideoMemoryBuilder,
        video_resolver: TimeLogicVideoResolver,
        artifact_cache_dir: str | Path | None = None,
        memory_cache_dir: str | Path | None = None,
        trace_dir: str | Path | None = None,
        verbose: bool = False,
        show_progress: bool = True,
        skip_unavailable_videos: bool = False,
        force_answer_normalization: bool = True,
        prompt_plugin: BenchmarkPromptPlugin | None = None,
    ):
        self.video_rlm = video_rlm
        self.memory_builder = memory_builder
        self.video_resolver = video_resolver
        self.artifact_cache_dir = Path(artifact_cache_dir) if artifact_cache_dir else None
        self.memory_cache_dir = Path(memory_cache_dir) if memory_cache_dir else None
        self.trace_dir = Path(trace_dir) if trace_dir else None
        self.verbose = verbose
        self.show_progress = show_progress
        self.skip_unavailable_videos = skip_unavailable_videos
        self.force_answer_normalization = force_answer_normalization
        self.prompt_plugin = prompt_plugin
        self._memory_cache: dict[str, tuple[VideoMemory, Path | None]] = {}

        for directory in (self.artifact_cache_dir, self.memory_cache_dir, self.trace_dir):
            if directory is not None:
                directory.mkdir(parents=True, exist_ok=True)

    def run_samples(
        self,
        samples: Sequence[dict[str, Any]],
        *,
        output_path: str | Path,
    ) -> list[dict[str, Any]]:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        completed = self._load_completed_question_ids(output_file)
        results: list[dict[str, Any]] = []
        self._log(
            f"run_samples start total={len(samples)} completed={len(completed)} "
            f"output={output_file}"
        )

        planned_memory_video_ids: set[str] = set()
        sample_units = [
            self._estimate_progress_units(sample, planned_memory_video_ids)
            for sample in samples
        ]
        progress_total = sum(sample_units)
        progress_completed = sum(
            units
            for sample, units in zip(samples, sample_units, strict=True)
            if str(sample.get("question_id")) in completed
        )
        completed_question_ids = {
            str(sample.get("question_id"))
            for sample in samples
            if str(sample.get("question_id")) in completed
        }
        progress = self._build_progress()
        progress_task_id = None
        if progress is not None:
            progress.start()
            progress_task_id = progress.add_task(
                "TimeLogic",
                total=progress_total,
                completed=progress_completed,
                status="starting",
                progress_label=f"{len(completed_question_ids)}/{len(samples)} samples",
            )
        try:
            for index, sample in enumerate(samples, start=1):
                question_id = str(sample["question_id"])
                progress_label = f"{index}/{len(samples)} samples"
                if question_id in completed:
                    self._progress_update(
                        progress,
                        progress_task_id,
                        description=f"TimeLogic {index}/{len(samples)}",
                        status=f"skipped completed {question_id}",
                        progress_label=progress_label,
                    )
                    self._log(f"{index}/{len(samples)} skip completed question_id={question_id}")
                    continue
                sample_progress = _TimeLogicProgressReporter(
                    runner=self,
                    progress=progress,
                    task_id=progress_task_id,
                    question_id=question_id,
                    estimated_units=sample_units[index - 1],
                    progress_label=progress_label,
                )
                self._progress_update(
                    progress,
                    progress_task_id,
                    description=f"TimeLogic {index}/{len(samples)}",
                    status=f"running {question_id}",
                    progress_label=progress_label,
                )
                start_time = time.perf_counter()
                try:
                    result = self.run_sample(sample, progress_callback=sample_progress)
                except SKIPPABLE_SAMPLE_ERRORS as exc:
                    if not self.skip_unavailable_videos:
                        self._progress_update(
                            progress,
                            progress_task_id,
                            status=f"failed {question_id}",
                        )
                        raise
                    result = self._skipped_record(sample, self._format_skip_reason(exc))
                except Exception:
                    self._progress_update(
                        progress,
                        progress_task_id,
                        status=f"failed {question_id}",
                    )
                    raise
                result.setdefault("execution_time", round(time.perf_counter() - start_time, 4))
                results.append(result)
                with output_file.open("a", encoding="utf-8") as handle:
                    json.dump(result, handle, ensure_ascii=False)
                    handle.write("\n")
                elapsed = time.perf_counter() - start_time
                status = (
                    f"skipped {question_id}"
                    if result.get("skipped")
                    else f"done {question_id} in {_format_duration(elapsed)}"
                )
                sample_progress.finish(status=status)
                log_status = "skipped" if result.get("skipped") else "done"
                self._log(
                    f"{index}/{len(samples)} {log_status} question_id={question_id} "
                    f"prediction={result.get('prediction')}"
                )
        finally:
            if progress is not None:
                progress.stop()
        self._log(f"run_samples done new_results={len(results)}")
        return results

    def run_sample(
        self,
        sample: dict[str, Any],
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        start_time = time.perf_counter()
        video_path = self.video_resolver.resolve(sample)
        memory, memory_path = self._load_or_build_memory(
            sample,
            video_path,
            progress_callback=progress_callback,
        )
        prompt = build_timelogic_prompt(sample)
        _emit_progress_status(progress_callback, f"controller starting {sample['question_id']}")
        result = self.video_rlm.run(
            prompt,
            memory,
            task_type=TIMELOGIC_TASK_TYPE,
            progress_callback=progress_callback,
            **self._prompt_plugin_run_kwargs(sample),
        )
        _emit_progress_status(progress_callback, f"normalizing answer {sample['question_id']}")
        options = timelogic_options(sample)
        raw_prediction = result.answer.strip()
        predicted_choice = parse_choice_prediction(raw_prediction, options)
        forced_info = None
        if self.force_answer_normalization:
            symbolic_choice, symbolic_info = self._symbolic_choice_from_result(
                sample,
                options,
                result,
            )
            if symbolic_choice is not None:
                predicted_choice = symbolic_choice
                forced_info = symbolic_info
            elif predicted_choice is not None:
                grounded_info = self._grounded_early_stop_info(
                    sample,
                    options,
                    result,
                    predicted_choice,
                )
                if grounded_info is not None:
                    forced_info = grounded_info
        should_finalize = self.force_answer_normalization and (
            forced_info is None
            and (predicted_choice is None or self._should_verify_timelogic_choice(result))
        )
        if should_finalize:
            _emit_progress_status(
                progress_callback,
                f"timelogic finalizer {sample['question_id']}",
            )
            finalized_choice, finalized_info = self._finalize_choice(sample, options, result)
            if not finalized_info.get("forced_choice_fallback") or predicted_choice is None:
                predicted_choice = finalized_choice
                forced_info = finalized_info

        prediction = _prediction_from_choice(sample["mode"], predicted_choice, options)
        if sample["mode"] == "bool" and prediction is None:
            prediction = normalize_bool_prediction(raw_prediction)
        if prediction is None:
            prediction = predicted_choice or raw_prediction

        _emit_progress_status(progress_callback, f"writing trace {sample['question_id']}")
        trace_path = self._write_trace(sample, result.to_dict())
        record = {
            "question_id": str(sample["question_id"]),
            "video_id": str(sample["video_id"]),
            "mode": sample["mode"],
            "question": sample["question"],
            "options": options,
            "prediction": prediction,
            "raw_prediction": raw_prediction,
            "predicted_choice": predicted_choice,
            "video_path": str(video_path),
            "memory_path": str(memory_path) if memory_path else None,
            "trace_path": str(trace_path) if trace_path else None,
            "execution_time": round(time.perf_counter() - start_time, 4),
            "steps_used": result.state.budget.steps_used,
            "tool_calls_used": result.state.budget.tool_calls_used,
        }
        if forced_info is not None:
            record.update(forced_info)
        enforce_timelogic_prediction_mode(record, options)
        return record

    def _finalize_choice(
        self,
        sample: dict[str, Any],
        options: dict[str, str],
        result,
    ) -> tuple[str, dict[str, Any]]:
        controller_client = getattr(self.video_rlm, "controller_client", None)
        if controller_client is None:
            raise ValueError("TimeLogic forced answer normalization requires controller_client")
        prompt = build_timelogic_forced_choice_prompt(
            sample,
            options,
            result,
            prompt_plugin=self._prompt_plugin_payload(sample),
        )
        finalizer_prediction = controller_client.completion(prompt).strip()
        predicted_choice = parse_choice_prediction(finalizer_prediction, options)
        fallback_choice = False
        if predicted_choice is None:
            predicted_choice = sorted(options)[0]
            fallback_choice = True
        return predicted_choice, {
            "prediction": _prediction_from_choice(sample["mode"], predicted_choice, options),
            "raw_prediction": result.answer,
            "finalizer_prediction": finalizer_prediction,
            "forced_choice_used": True,
            "forced_choice_fallback": fallback_choice,
        }

    def _symbolic_choice_from_result(
        self,
        sample: dict[str, Any],
        options: dict[str, str],
        result,
    ) -> tuple[str | None, dict[str, Any] | None]:
        symbolic = timelogic_symbolic_choice_from_event_memory(
            getattr(result.state, "event_memory", None),
            options,
        )
        if symbolic is None:
            return None, None
        predicted_choice, metadata = symbolic
        return predicted_choice, {
            "prediction": _prediction_from_choice(sample["mode"], predicted_choice, options),
            "raw_prediction": result.answer,
            "symbolic_choice_used": True,
            "symbolic_choice_source": "timelogic_event_memory",
            "symbolic_choice_metadata": metadata,
        }

    def _grounded_early_stop_info(
        self,
        sample: dict[str, Any],
        options: dict[str, str],
        result,
        predicted_choice: str,
    ) -> dict[str, Any] | None:
        state = getattr(result, "state", None)
        global_context = getattr(state, "global_context", {})
        early_stop = global_context.get("early_stop") if isinstance(global_context, dict) else None
        if not isinstance(early_stop, dict):
            return None
        if early_stop.get("source") != "grounded_multiple_choice_completion":
            return None
        if predicted_choice not in options:
            return None
        return {
            "prediction": _prediction_from_choice(sample["mode"], predicted_choice, options),
            "raw_prediction": result.answer,
            "grounded_choice_used": True,
            "grounded_choice_source": "controller_early_stop",
            "grounded_choice_metadata": dict(early_stop),
        }

    def _should_verify_timelogic_choice(self, result) -> bool:
        controller_client = getattr(self.video_rlm, "controller_client", None)
        if controller_client is None:
            return False
        state = getattr(result, "state", None)
        event_memory = getattr(state, "event_memory", None)
        evidence_ledger = getattr(state, "evidence_ledger", [])
        return event_memory is not None and bool(evidence_ledger)

    def _prompt_plugin_run_kwargs(self, sample: dict[str, Any]) -> dict[str, Any]:
        context = build_timelogic_global_context(sample)
        plugin_context = prompt_plugin_context(self.prompt_plugin, sample)
        if plugin_context is not None:
            context.update(plugin_context)
        return {"global_context_overrides": context}

    def _prompt_plugin_payload(self, sample: dict[str, Any]) -> dict[str, Any] | None:
        context = prompt_plugin_context(self.prompt_plugin, sample)
        if context is None:
            return None
        payload = context.get("benchmark_prompt_plugin")
        return payload if isinstance(payload, dict) else None

    def _load_or_build_memory(
        self,
        sample: dict[str, Any],
        video_path: Path,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[VideoMemory, Path | None]:
        video_id = str(sample["video_id"])
        if video_id in self._memory_cache:
            return self._memory_cache[video_id]

        memory_path = self._memory_path(video_id)
        if memory_path is not None and memory_path.exists():
            memory = self.memory_builder.load_memory(memory_path)
            self._mark_timelogic_memory(memory, video_path)
            cached = (memory, memory_path)
            self._memory_cache[video_id] = cached
            return cached

        duration_seconds = probe_media_duration(video_path)
        artifact_dir = self._artifact_dir(video_id)
        artifacts = None
        if artifact_dir is not None and artifact_dir.exists():
            artifacts = self.memory_builder.load_artifacts_dir(artifact_dir)
            artifacts.metadata.setdefault("timelogic_visual_only", True)
            artifacts.metadata.setdefault("vrrqa_visual_only", True)
        if artifacts is None:
            artifacts = self.memory_builder.prepare_artifacts(
                video_path=str(video_path),
                duration_seconds=duration_seconds,
                video_id=safe_identifier(Path(video_id).stem),
                metadata={
                    "source_video_path": str(video_path),
                    "timelogic_video_id": video_id,
                    "timelogic_visual_only": True,
                    "timelogic_event_graph": True,
                    "vrrqa_visual_only": True,
                },
                progress_callback=progress_callback,
            )
            if artifact_dir is not None:
                self.memory_builder.save_artifacts_dir(artifacts, artifact_dir)
        memory = self.memory_builder.build_from_artifacts(artifacts)
        self._mark_timelogic_memory(memory, video_path)
        if memory_path is not None:
            self.memory_builder.save_memory(memory, memory_path)
        cached = (memory, memory_path)
        self._memory_cache[video_id] = cached
        return cached

    def _mark_timelogic_memory(self, memory: VideoMemory, video_path: Path) -> None:
        memory.metadata.setdefault("source_video_path", str(video_path))
        memory.metadata.setdefault("timelogic_visual_only", True)
        memory.metadata.setdefault("timelogic_event_graph", True)
        memory.metadata.setdefault("vrrqa_visual_only", True)

    def _memory_path(self, video_id: str) -> Path | None:
        if self.memory_cache_dir is None:
            return None
        return self.memory_cache_dir / f"{safe_identifier(video_id)}.json"

    def _artifact_dir(self, video_id: str) -> Path | None:
        if self.artifact_cache_dir is None:
            return None
        return self.artifact_cache_dir / safe_identifier(video_id)

    def _estimate_progress_units(
        self,
        sample: dict[str, Any],
        planned_memory_video_ids: set[str] | None = None,
    ) -> int:
        units = 1
        video_id = str(sample.get("video_id"))
        if video_id in self._memory_cache:
            return units
        if planned_memory_video_ids is not None and video_id in planned_memory_video_ids:
            return units
        if not self._will_prepare_artifacts_for_progress(video_id):
            return units
        try:
            video_path = self.video_resolver.resolve(sample)
            duration_seconds = probe_media_duration(video_path)
        except (FileNotFoundError, OSError, RuntimeError, ValueError, subprocess.CalledProcessError):
            return units

        if self.memory_builder.speech_recognizer is not None:
            units += self._estimate_speech_progress_units(duration_seconds)
        if self.memory_builder.visual_summarizer is not None:
            units += self._estimate_visual_progress_units(duration_seconds)
        if planned_memory_video_ids is not None:
            planned_memory_video_ids.add(video_id)
        return max(1, units)

    def _will_prepare_artifacts_for_progress(self, video_id: str) -> bool:
        if video_id in self._memory_cache:
            return False
        memory_path = self._memory_path(video_id)
        if memory_path is not None and memory_path.exists():
            return False
        artifact_dir = self._artifact_dir(video_id)
        return not (artifact_dir is not None and artifact_dir.exists())

    def _estimate_speech_progress_units(self, duration_seconds: float) -> int:
        recognizer = self.memory_builder.speech_recognizer
        if recognizer is None:
            return 0
        chunk_duration = getattr(recognizer, "chunk_duration_seconds", None)
        if isinstance(chunk_duration, (int, float)) and chunk_duration > 0:
            return (
                max(1, math.ceil(duration_seconds / chunk_duration)) * SPEECH_PROGRESS_UNIT_WEIGHT
            )
        return SPEECH_PROGRESS_UNIT_WEIGHT

    def _estimate_visual_progress_units(self, duration_seconds: float) -> int:
        if self.memory_builder.visual_summarizer is None:
            return 0
        span_count = len(self.memory_builder._visual_spans(TimeSpan(0.0, duration_seconds)))
        return span_count * self._visual_progress_unit_weight()

    def _visual_progress_unit_weight(self) -> int:
        visual_summarizer = self.memory_builder.visual_summarizer
        if visual_summarizer is None:
            return 0
        explicit_weight = getattr(visual_summarizer, "progress_unit_weight", None)
        if isinstance(explicit_weight, int) and explicit_weight > 0:
            return explicit_weight
        if visual_summarizer.__class__.__name__ == "LocalQwenVisualSummarizer":
            return LOCAL_QWEN_VISUAL_PROGRESS_UNIT_WEIGHT
        return GENERIC_VISUAL_PROGRESS_UNIT_WEIGHT

    def _progress_event_units(self, event: dict[str, Any], advance: int) -> int:
        if advance <= 0:
            return 0
        phase = event.get("phase")
        if phase == "visual":
            return advance * self._visual_progress_unit_weight()
        if phase == "visual_refinement":
            return advance * LOCAL_QWEN_VISUAL_PROGRESS_UNIT_WEIGHT
        if phase in {"asr", "speech_refinement"}:
            return advance * SPEECH_PROGRESS_UNIT_WEIGHT
        return advance

    def _write_trace(self, sample: dict[str, Any], payload: dict[str, Any]) -> Path | None:
        if self.trace_dir is None:
            return None
        trace_path = self.trace_dir / f"{safe_identifier(sample['question_id'])}.json"
        trace_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return trace_path

    def _skipped_record(self, sample: dict[str, Any], reason: str) -> dict[str, Any]:
        return {
            "question_id": str(sample["question_id"]),
            "video_id": str(sample["video_id"]),
            "mode": sample.get("mode"),
            "question": sample.get("question"),
            "options": timelogic_options(sample) if sample.get("mode") in {"bool", "mc"} else {},
            "prediction": "",
            "raw_prediction": "",
            "predicted_choice": None,
            "skipped": True,
            "skip_reason": reason,
        }

    def _load_completed_question_ids(self, output_file: Path) -> set[str]:
        if not output_file.exists():
            return set()
        completed: set[str] = set()
        for line in output_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            completed.add(str(json.loads(line)["question_id"]))
        return completed

    def _format_skip_reason(self, exc: Exception) -> str:
        message = str(exc)
        if message:
            return f"{exc.__class__.__name__}: {message}"
        return exc.__class__.__name__

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[TimeLogic] {message}", flush=True)

    def _build_progress(self):
        if not self.show_progress:
            return None
        try:
            from rich.console import Console
            from rich.progress import (
                BarColumn,
                Progress,
                TaskProgressColumn,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )
        except ImportError:
            return None

        return Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("{task.fields[progress_label]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[status]}"),
            console=Console(stderr=True),
            transient=False,
        )

    def _progress_update(
        self,
        progress,
        task_id,
        *,
        description: str | None = None,
        status: str | None = None,
        progress_label: str | None = None,
    ) -> None:
        if progress is None or task_id is None:
            return
        kwargs: dict[str, Any] = {}
        if description is not None:
            kwargs["description"] = description
        if status is not None:
            kwargs["status"] = status
        if progress_label is not None:
            kwargs["progress_label"] = progress_label
        if kwargs:
            progress.update(task_id, **kwargs)

    def _progress_advance_units(
        self,
        progress,
        task_id,
        *,
        units: int,
        status: str | None = None,
        progress_label: str | None = None,
    ) -> None:
        if progress is None or task_id is None or units <= 0:
            return
        kwargs: dict[str, Any] = {"advance": units}
        if status is not None:
            kwargs["status"] = status
        if progress_label is not None:
            kwargs["progress_label"] = progress_label
        progress.update(task_id, **kwargs)

    def _progress_increase_total(self, progress, task_id, *, units: int) -> None:
        if progress is None or task_id is None or units <= 0:
            return
        for task in progress.tasks:
            if task.id == task_id:
                total = (task.total or 0) + units
                progress.update(task_id, total=total)
                return


class _TimeLogicProgressReporter:
    def __init__(
        self,
        *,
        runner: TimeLogicBenchmarkRunner,
        progress,
        task_id,
        question_id: str,
        estimated_units: int,
        progress_label: str,
    ):
        self.runner = runner
        self.progress = progress
        self.task_id = task_id
        self.question_id = question_id
        self.estimated_units = max(1, estimated_units)
        self.completed_units = 0
        self.progress_label = progress_label

    def __call__(self, event: dict[str, Any]) -> None:
        advance = int(event.get("advance") or 0)
        status = str(event.get("status") or self._format_status(event))
        if advance > 0:
            self.advance(self.runner._progress_event_units(event, advance), status=status)
        elif status:
            self.runner._progress_update(
                self.progress,
                self.task_id,
                status=f"{status} {self.question_id}",
                progress_label=self.progress_label,
            )

    def advance(self, units: int, *, status: str | None = None) -> None:
        if units <= 0:
            return
        overflow = max(0, self.completed_units + units - self.estimated_units)
        if overflow:
            self.runner._progress_increase_total(self.progress, self.task_id, units=overflow)
            self.estimated_units += overflow
        self.completed_units += units
        self.runner._progress_advance_units(
            self.progress,
            self.task_id,
            units=units,
            status=f"{status} {self.question_id}" if status else None,
            progress_label=self.progress_label,
        )

    def finish(self, *, status: str) -> None:
        remaining = max(0, self.estimated_units - self.completed_units)
        if remaining:
            self.advance(remaining, status=status)
            return
        self.runner._progress_update(
            self.progress,
            self.task_id,
            status=status,
            progress_label=self.progress_label,
        )

    def _format_status(self, event: dict[str, Any]) -> str:
        phase = event.get("phase")
        index = event.get("index")
        total = event.get("total")
        if phase and index is not None and total is not None:
            return f"{phase} {index}/{total}"
        if phase and total is not None:
            return f"{phase} 0/{total}"
        return str(phase or "running")


def build_timelogic_prompt(sample: dict[str, Any]) -> str:
    options = timelogic_options(sample)
    option_lines = [f"{letter}. {text}" for letter, text in options.items()]
    question = (
        clean_timelogic_multiple_choice_question(sample["question"])
        if sample["mode"] == "mc"
        else str(sample["question"]).strip()
    )
    event_phrases = extract_timelogic_event_phrases(sample["question"], options)
    event_lines = [f"- {phrase}" for phrase in event_phrases]

    lines = [
        f"Question: {question}",
        "Options:",
        *option_lines,
        "Use the options above to answer TimeLogic TLQA: localize each relevant action as "
        "start-end seconds, then apply before/after/overlap/imply/always/immediately "
        "relations over those intervals.",
    ]
    if event_lines:
        lines.extend(["Use the options above with these action phrases to localize:", *event_lines])
    lines.extend(
        [
            "When you stop, return exactly one option letter.",
            "Do not answer with None.",
        ]
    )
    return "\n".join(lines)


def build_timelogic_global_context(sample: dict[str, Any]) -> dict[str, Any]:
    options = timelogic_options(sample)
    event_specs = build_timelogic_event_specs(sample)
    mode = _normalize_mode(sample["mode"])
    clean_question = (
        clean_timelogic_multiple_choice_question(sample["question"])
        if mode == "mc"
        else str(sample["question"]).strip()
    )
    return {
        "benchmark": "timelogic",
        "clean_question": clean_question,
        "answer_options": options,
        "valid_answer_letters": sorted(options),
        "timelogic": {
            "question_id": str(sample["question_id"]),
            "video_id": str(sample["video_id"]),
            "mode": mode,
            "event_phrases": [item["phrase"] for item in event_specs],
            "operator_guide": TIMELOGIC_OPERATOR_GUIDE,
            "stop_policy": (
                "Once all events needed by the matched operator are localized and the "
                "relation has a unique verified answer, STOP with the option letter."
            ),
        },
        "timelogic_operator_guide": TIMELOGIC_OPERATOR_GUIDE,
        "event_memory_spec": {
            "task_name": "timelogic",
            "question": str(sample["question"]).strip(),
            "mode": mode,
            "events": event_specs,
            "relations": build_timelogic_relation_specs(sample, event_specs),
        },
    }


def build_timelogic_event_specs(sample: dict[str, Any]) -> list[dict[str, Any]]:
    options = timelogic_options(sample)
    mode = _normalize_mode(sample["mode"])
    formula = parse_timelogic_formula(sample["question"], mode=mode)
    phrases = []
    if mode == "mc":
        phrases.extend(
            option
            for option in options.values()
            if _normalize_text(option) not in {"yes", "no"}
        )
    phrases.extend(formula["target_phrases"])
    specs: list[dict[str, Any]] = []
    target_index = 1
    for phrase in phrases:
        option_letter = _option_letter_for_event_phrase(phrase, options) if mode == "mc" else None
        if option_letter is not None:
            specs.append(
                {
                    "event_id": f"option_{option_letter}",
                    "phrase": phrase,
                    "source": "option",
                    "option_letter": option_letter,
                }
            )
            continue
        specs.append(
            {
                "event_id": f"target_{target_index:02d}",
                "phrase": phrase,
                "source": "question",
                "option_letter": None,
            }
        )
        target_index += 1
    return specs


def build_timelogic_relation_specs(
    sample: dict[str, Any],
    event_specs: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    mode = _normalize_mode(sample["mode"])
    formula = parse_timelogic_formula(sample["question"], mode=mode)
    option_event_ids = [
        str(item["event_id"]) for item in event_specs if item.get("source") == "option"
    ]
    target_event_ids = [
        str(item["event_id"]) for item in event_specs if item.get("source") != "option"
    ]
    quantifier = "always" if "always" in str(sample["question"]).lower() else None
    relations: list[dict[str, Any]] = []

    def add(left: str, operator: str, right: str, *, include_quantifier: bool = True) -> None:
        relation: dict[str, Any] = {"left": left, "operator": operator, "right": right}
        if include_quantifier and quantifier is not None:
            relation["quantifier"] = quantifier
        relations.append(relation)

    for template in formula["relations"]:
        left_refs = _resolve_timelogic_relation_ref(
            template["left"],
            target_event_ids,
            option_event_ids,
        )
        right_refs = _resolve_timelogic_relation_ref(
            template["right"],
            target_event_ids,
            option_event_ids,
        )
        for left in left_refs:
            for right in right_refs:
                add(
                    left,
                    str(template["operator"]),
                    right,
                    include_quantifier=str(template["operator"]) != "imply",
                )
    return relations


def parse_timelogic_formula(question: str, *, mode: str) -> dict[str, Any]:
    stem = _timelogic_formula_stem(question)
    if mode == "mc" and stem.lower().startswith("which action"):
        parsed = _parse_which_action_formula(stem)
    elif mode == "mc":
        parsed = _parse_multiple_choice_formula(stem)
    else:
        parsed = _parse_boolean_formula(stem)

    if parsed["target_phrases"]:
        return parsed
    return {
        "target_phrases": _fallback_timelogic_target_phrases(question, mode=mode),
        "relations": [],
    }


def _parse_multiple_choice_formula(stem: str) -> dict[str, Any]:
    match = re.match(
        r"what did (?:the )?person do before\s+(.+)$",
        stem,
        flags=re.IGNORECASE,
    )
    if match is not None:
        return {
            "target_phrases": [_clean_event_phrase(match.group(1))],
            "relations": [{"left": "option", "operator": "before", "right": 0}],
        }

    match = re.match(
        r"what did (?:the )?person do (?:immediately\s+)?after\s+(.+)$",
        stem,
        flags=re.IGNORECASE,
    )
    if match is not None:
        return {
            "target_phrases": [_clean_event_phrase(match.group(1))],
            "relations": [{"left": 0, "operator": "before", "right": "option"}],
        }

    match = re.match(
        r"what does (?:the )?person do when\s+(.+)$",
        stem,
        flags=re.IGNORECASE,
    )
    if match is not None:
        return {
            "target_phrases": [_clean_event_phrase(match.group(1))],
            "relations": [{"left": "option", "operator": "overlap", "right": 0}],
        }

    match = re.match(
        r"while (?:the )?person is\s+(.+?)\s*,?\s*what does this imply about "
        r"(?:person\s+)?action$",
        stem,
        flags=re.IGNORECASE,
    )
    if match is not None:
        return {
            "target_phrases": [_clean_event_phrase(match.group(1))],
            "relations": [],
        }

    return _parse_boolean_formula(stem)


def _parse_which_action_formula(stem: str) -> dict[str, Any]:
    match = re.match(
        r"which action\s+(?:always\s+)?occurs before\s+(.+?)\s+which in turn\s+"
        r"(?:always\s+)?occurs before\s+(.+)$",
        stem,
        flags=re.IGNORECASE,
    )
    if match is not None:
        targets = [_clean_event_phrase(match.group(1)), _clean_event_phrase(match.group(2))]
        return {
            "target_phrases": targets,
            "relations": [
                {"left": "option", "operator": "before", "right": 0},
                {"left": 0, "operator": "before", "right": 1},
            ],
        }

    match = re.match(
        r"which action\s+(?:always\s+)?occurs before\s+(.+?)\s+and\s+(.+)$",
        stem,
        flags=re.IGNORECASE,
    )
    if match is not None:
        targets = [_clean_event_phrase(match.group(1)), _clean_event_phrase(match.group(2))]
        return {
            "target_phrases": targets,
            "relations": [
                {"left": "option", "operator": "before", "right": 0},
                {"left": "option", "operator": "before", "right": 1},
            ],
        }

    match = re.match(
        r"which action\s+(?:always\s+)?occurs before\s+(.+)$",
        stem,
        flags=re.IGNORECASE,
    )
    if match is not None:
        return {
            "target_phrases": [_clean_event_phrase(match.group(1))],
            "relations": [{"left": "option", "operator": "before", "right": 0}],
        }
    return {"target_phrases": [], "relations": []}


def _parse_boolean_formula(stem: str) -> dict[str, Any]:
    match = re.match(
        r"(.+?)\s+(?:always\s+)?occurs before\s+(.+?)\s+which in turn\s+"
        r"(?:always\s+)?occurs before\s+(.+)$",
        stem,
        flags=re.IGNORECASE,
    )
    if match is not None:
        targets = [
            _clean_event_phrase(match.group(1)),
            _clean_event_phrase(match.group(2)),
            _clean_event_phrase(match.group(3)),
        ]
        return {
            "target_phrases": targets,
            "relations": [
                {"left": 0, "operator": "before", "right": 1},
                {"left": 1, "operator": "before", "right": 2},
            ],
        }

    match = re.match(
        r"(.+?)\s+(?:always\s+)?occurs before\s+(.+?)\s+and\s+(.+)$",
        stem,
        flags=re.IGNORECASE,
    )
    if match is not None:
        targets = [
            _clean_event_phrase(match.group(1)),
            _clean_event_phrase(match.group(2)),
            _clean_event_phrase(match.group(3)),
        ]
        return {
            "target_phrases": targets,
            "relations": [
                {"left": 0, "operator": "before", "right": 1},
                {"left": 0, "operator": "before", "right": 2},
            ],
        }

    match = re.match(r"(.+?)\s+imply\s+(.+)$", stem, flags=re.IGNORECASE)
    if match is not None:
        return {
            "target_phrases": [
                _clean_event_phrase(match.group(1)),
                _clean_event_phrase(match.group(2)),
            ],
            "relations": [{"left": 0, "operator": "imply", "right": 1}],
        }

    match = re.match(r"(.+?)\s+does not overlap with\s+(.+)$", stem, flags=re.IGNORECASE)
    if match is not None:
        return {
            "target_phrases": [
                _clean_event_phrase(match.group(1)),
                _clean_event_phrase(match.group(2)),
            ],
            "relations": [{"left": 0, "operator": "disjoint", "right": 1}],
        }

    match = re.match(
        r"(.+?)\s+(?:co-?occur(?:s|red)? with|overlap(?:s|ped)? with)\s+(.+)$",
        stem,
        flags=re.IGNORECASE,
    )
    if match is not None:
        return {
            "target_phrases": [
                _clean_event_phrase(match.group(1)),
                _clean_event_phrase(match.group(2)),
            ],
            "relations": [{"left": 0, "operator": "overlap", "right": 1}],
        }

    match = re.match(r"(.+?)\s+immediately after\s+(.+)$", stem, flags=re.IGNORECASE)
    if match is not None:
        return {
            "target_phrases": [
                _clean_event_phrase(match.group(1)),
                _clean_event_phrase(match.group(2)),
            ],
            "relations": [{"left": 1, "operator": "before", "right": 0}],
        }

    match = re.match(r"(.+?)\s+before\s+(.+)$", stem, flags=re.IGNORECASE)
    if match is not None:
        return {
            "target_phrases": [
                _clean_event_phrase(match.group(1)),
                _clean_event_phrase(match.group(2)),
            ],
            "relations": [{"left": 0, "operator": "before", "right": 1}],
        }

    match = re.match(r"(.+?)\s+after\s+(.+)$", stem, flags=re.IGNORECASE)
    if match is not None:
        return {
            "target_phrases": [
                _clean_event_phrase(match.group(1)),
                _clean_event_phrase(match.group(2)),
            ],
            "relations": [{"left": 1, "operator": "before", "right": 0}],
        }

    return {"target_phrases": [], "relations": []}


def _resolve_timelogic_relation_ref(
    ref: Any,
    target_event_ids: Sequence[str],
    option_event_ids: Sequence[str],
) -> list[str]:
    if ref == "option":
        return list(option_event_ids)
    if isinstance(ref, int) and 0 <= ref < len(target_event_ids):
        return [target_event_ids[ref]]
    return []


def _timelogic_formula_stem(question: str) -> str:
    stem = clean_timelogic_multiple_choice_question(question)
    stem = " ".join(stem.split()).strip(" .?")
    stem = re.sub(r"^is it true that\s+", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"^did\s+", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"^does\s+", "", stem, flags=re.IGNORECASE)
    return stem.strip(" .?")


def _fallback_timelogic_target_phrases(question: str, *, mode: str) -> list[str]:
    phrases = []
    for phrase in extract_timelogic_event_phrases(question, options=None):
        cleaned = _clean_event_phrase(phrase)
        if cleaned and not cleaned.lower().startswith("which action"):
            phrases.append(cleaned)
    if mode == "bool" and not phrases:
        phrases.append(_clean_event_phrase(question))
    return phrases[:8]


def build_timelogic_forced_choice_prompt(
    sample: dict[str, Any],
    options: dict[str, str],
    result,
    prompt_plugin: dict[str, Any] | None = None,
) -> str:
    option_lines = [f"{letter}. {text}" for letter, text in options.items()]
    event_memory_lines = _timelogic_event_memory_lines(result)
    evidence_lines = _timelogic_evidence_lines(result)
    trace_lines = _timelogic_trace_lines(result)
    plugin_section = render_prompt_plugin_section(prompt_plugin)
    output_rule = (
        "Return exactly one option letter."
        if sample["mode"] == "mc"
        else "Return exactly A for yes or B for no."
    )
    sections = [
        "You are a strict TimeLogic TLQA finalizer.",
        "Use only the collected evidence below.",
        "Convert visible actions into timestamped event intervals and evaluate the temporal "
        "formula symbolically.",
        output_rule,
    ]
    if plugin_section:
        sections.extend(["", plugin_section])
    sections.extend(
        [
            "",
            f"Question: {sample['question']}",
            "Options:",
            *option_lines,
            "",
            f"Initial VideoRLM answer: {result.answer}",
        ]
    )
    if event_memory_lines:
        sections.extend(["", "Timestamped event table:", *event_memory_lines])
    if evidence_lines:
        sections.extend(["", "Collected evidence:", *evidence_lines])
    if trace_lines:
        sections.extend(["", "Recent observations:", *trace_lines])
    sections.extend(["", "Final answer letter:"])
    return "\n".join(sections)


def timelogic_symbolic_choice_from_event_memory(
    event_memory: EventMemory | None,
    options: dict[str, str],
) -> tuple[str, dict[str, Any]] | None:
    if event_memory is None or event_memory.task_name != "timelogic":
        return None
    if not event_memory.relations:
        return None

    if event_memory.mode == "bool":
        verdicts = [
            _evaluate_timelogic_event_relation(event_memory, relation)
            for relation in event_memory.relations
        ]
        if any(verdict is None for verdict in verdicts):
            return None
        answer_text = "yes" if all(verdicts) else "no"
        choice = parse_choice_prediction(answer_text, options)
        if choice is None:
            return None
        return choice, {
            "mode": "bool",
            "answer": answer_text,
            "verdicts": verdicts,
        }

    if event_memory.mode == "mc":
        target_relations = [
            relation
            for relation in event_memory.relations
            if not _timelogic_relation_mentions_option(relation)
        ]
        target_verdicts = [
            _evaluate_timelogic_event_relation(event_memory, relation)
            for relation in target_relations
        ]
        if any(verdict is False for verdict in target_verdicts):
            return None

        supported: list[str] = []
        option_verdicts_by_letter: dict[str, list[bool | None]] = {}
        for letter in sorted(options):
            option_event_id = f"option_{letter}"
            option_relations = [
                relation
                for relation in event_memory.relations
                if relation.get("left") == option_event_id
                or relation.get("right") == option_event_id
            ]
            if not option_relations:
                continue
            option_verdicts = [
                _evaluate_timelogic_event_relation(event_memory, relation)
                for relation in option_relations
            ]
            option_verdicts_by_letter[letter] = option_verdicts
            if option_verdicts and all(verdict is True for verdict in option_verdicts):
                supported.append(letter)
        if len(supported) != 1:
            return None
        return supported[0], {
            "mode": "mc",
            "best_option": supported[0],
            "target_verdicts": target_verdicts,
            "option_verdicts": option_verdicts_by_letter,
        }
    return None


def timelogic_options(sample: dict[str, Any]) -> dict[str, str]:
    mode = _normalize_mode(sample["mode"])
    if mode == "bool":
        return dict(TIMELOGIC_BOOL_OPTIONS)
    return parse_timelogic_options(str(sample["question"]))


def _timelogic_relation_mentions_option(relation: dict[str, Any]) -> bool:
    return str(relation.get("left", "")).startswith("option_") or str(
        relation.get("right", "")
    ).startswith("option_")


def _evaluate_timelogic_event_relation(
    event_memory: EventMemory,
    relation: dict[str, Any],
) -> bool | None:
    left = event_memory.events.get(str(relation.get("left")))
    right = event_memory.events.get(str(relation.get("right")))
    if left is None or right is None:
        return None

    left_intervals = left.intervals
    right_intervals = right.intervals
    operator = str(relation.get("operator") or "").lower()
    if operator == "imply":
        if left_intervals and not right_intervals:
            return False
        if not left_intervals:
            return None
        return bool(right_intervals)
    if not left_intervals or not right_intervals:
        return None
    if operator == "before":
        if relation.get("quantifier") == "always":
            return max(interval.time_span.end for interval in left_intervals) <= min(
                interval.time_span.start for interval in right_intervals
            )
        return min(interval.time_span.start for interval in left_intervals) <= min(
            interval.time_span.start for interval in right_intervals
        )
    if operator == "overlap":
        return any(
            left_interval.time_span.overlaps(right_interval.time_span)
            for left_interval in left_intervals
            for right_interval in right_intervals
        )
    if operator == "disjoint":
        return not any(
            left_interval.time_span.overlaps(right_interval.time_span)
            for left_interval in left_intervals
            for right_interval in right_intervals
        )
    return None


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


def clean_timelogic_multiple_choice_question(question: str) -> str:
    text = " ".join(question.split()).strip()
    text = re.sub(
        r"^The following is a multiple choice question with four possible answer choices: "
        r"A, B, C, D\.\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*Reply with\b.*$", "", text, flags=re.IGNORECASE).strip()
    split_match = re.search(r"\s+Is it Option\s+A\s*:", text, flags=re.IGNORECASE)
    if split_match is not None:
        text = text[: split_match.start()].strip()
    return text


def extract_timelogic_event_phrases(
    question: str,
    options: dict[str, str] | None = None,
) -> list[str]:
    phrases: list[str] = []
    for option in (options or {}).values():
        if _normalize_text(option) in {"yes", "no"}:
            continue
        _append_phrase(phrases, option)

    stem = _timelogic_formula_stem(question)
    parsed = (
        _parse_which_action_formula(stem)
        if stem.lower().startswith("which action")
        else _parse_boolean_formula(stem)
    )
    for target in parsed["target_phrases"]:
        _append_phrase(phrases, target)
    if parsed["target_phrases"]:
        return phrases[:12]

    return _legacy_extract_timelogic_event_phrases(question, phrases)


def _legacy_extract_timelogic_event_phrases(
    question: str,
    phrases: list[str],
) -> list[str]:

    stem = clean_timelogic_multiple_choice_question(question)
    if stem == question:
        stem = " ".join(question.split()).strip()
    extraction_patterns = [
        r"before\s+(.+?)\s*,?\s+which in turn(?:\s+always)?\s+occurs before\s+(.+?)\s*\?",
        r"before\s+(.+?)\s+and\s+(.+?)\s*\?",
        r"after\s+(.+?)\s*\?",
        r"when\s+(.+?)\s*\?",
        r"imply\s+(.+?)\s*\?",
        r"(.+?)\s+before\s+(.+?)\s*\?",
        r"(.+?)\s+after\s+(.+?)\s*\?",
        r"(.+?)\s+does not overlap with\s+(.+?)\s*\?",
        r"(.+?)\s+overlap(?:s|ped)? with\s+(.+?)\s*\?",
    ]
    for pattern in extraction_patterns:
        match = re.search(pattern, stem, flags=re.IGNORECASE)
        if match is None:
            continue
        for group in match.groups():
            _append_phrase(phrases, _clean_event_phrase(group))
    return phrases[:12]


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

    normalized_text = _normalize_text(prediction)
    for choice, option_text in options.items():
        if normalized_text == _normalize_text(option_text):
            return choice
    contained_choices = [
        choice
        for choice, option_text in options.items()
        if _normalize_text(option_text) and _normalize_text(option_text) in normalized_text
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
        return _bool_word_to_answer(labeled.group(1))
    first_token = re.match(r"^(yes|no|true|false)\b", text)
    if first_token is not None:
        return _bool_word_to_answer(first_token.group(1))
    if text in {"a", "option a"}:
        return "yes"
    if text in {"b", "option b"}:
        return "no"
    return None


def safe_identifier(value: str) -> str:
    cleaned = SAFE_IDENTIFIER_PATTERN.sub("_", str(value)).strip("._-")
    return cleaned or "timelogic_sample"


def _prediction_from_choice(
    mode: str,
    predicted_choice: str | None,
    options: dict[str, str],
) -> str | None:
    if predicted_choice is None:
        return None
    if mode == "bool":
        return options[predicted_choice]
    return predicted_choice


def enforce_timelogic_prediction_mode(
    record: dict[str, Any],
    options: dict[str, str],
) -> None:
    mode = _normalize_mode(record["mode"])
    original_prediction = record.get("prediction")
    original_predicted_choice = record.get("predicted_choice")
    candidates = [
        original_predicted_choice,
        original_prediction,
        record.get("finalizer_prediction"),
        record.get("raw_prediction"),
    ]

    predicted_choice = _first_choice_prediction(candidates, options)
    fallback_used = False
    if mode == "bool":
        prediction = _prediction_from_choice(mode, predicted_choice, options)
        if prediction is None:
            prediction = _first_bool_prediction(candidates)
        if prediction is None:
            predicted_choice = sorted(options)[0]
            prediction = options[predicted_choice]
            fallback_used = True
        else:
            predicted_choice = predicted_choice or _choice_from_bool_prediction(prediction, options)
    else:
        if predicted_choice is None:
            predicted_choice = sorted(options)[0]
            fallback_used = True
        prediction = predicted_choice

    record["prediction"] = prediction
    record["predicted_choice"] = predicted_choice
    if original_prediction != prediction or original_predicted_choice != predicted_choice:
        record["mode_enforced_prediction"] = True
        record["pre_enforced_prediction"] = original_prediction
        record["pre_enforced_predicted_choice"] = original_predicted_choice
    if fallback_used:
        record["mode_enforced_fallback"] = True


def _first_choice_prediction(candidates: Sequence[Any], options: dict[str, str]) -> str | None:
    for candidate in candidates:
        if candidate is None:
            continue
        predicted_choice = parse_choice_prediction(str(candidate), options)
        if predicted_choice is not None:
            return predicted_choice
    return None


def _first_bool_prediction(candidates: Sequence[Any]) -> str | None:
    for candidate in candidates:
        prediction = normalize_bool_prediction(candidate)
        if prediction is not None:
            return prediction
    return None


def _choice_from_bool_prediction(prediction: str, options: dict[str, str]) -> str | None:
    normalized_prediction = _normalize_text(prediction)
    for choice, option_text in options.items():
        if _normalize_text(option_text) == normalized_prediction:
            return choice
    return None


def _timelogic_evidence_lines(result, max_items: int = 10) -> list[str]:
    evidence = getattr(getattr(result, "state", None), "evidence_ledger", [])
    ordered = sorted(evidence, key=lambda item: (item.time_span.start, -item.confidence))
    lines = []
    for item in ordered[:max_items]:
        lines.append(
            "- "
            + json.dumps(
                {
                    "evidence_id": item.evidence_id,
                    "time_span": item.time_span.to_dict(),
                    "claim": item.claim,
                    "detail": (item.detail or item.claim)[:700],
                },
                ensure_ascii=True,
            )
        )
    return lines


def _timelogic_event_memory_lines(result, max_events: int = 16) -> list[str]:
    event_memory = getattr(getattr(result, "state", None), "event_memory", None)
    if event_memory is None:
        return []
    lines = []
    for event in list(event_memory.events.values())[:max_events]:
        intervals = [
            {
                "time_span": interval.time_span.to_dict(),
                "evidence_id": interval.evidence_id,
                "confidence": interval.confidence,
                "match_score": interval.match_score,
            }
            for interval in event.intervals[:3]
        ]
        lines.append(
            "- "
            + json.dumps(
                {
                    "event_id": event.event_id,
                    "phrase": event.phrase,
                    "source": event.source,
                    "option_letter": event.option_letter,
                    "status": event.status,
                    "intervals": intervals,
                },
                ensure_ascii=True,
            )
        )
    if event_memory.relations:
        lines.append(
            "- "
            + json.dumps(
                {"relations": event_memory.relations[:12]},
                ensure_ascii=True,
            )
        )
    return lines


def _timelogic_trace_lines(result, max_items: int = 6) -> list[str]:
    lines = []
    for step in list(getattr(result, "trace", []))[-max_items:]:
        observation = step.get("observation") or {}
        summary = observation.get("summary")
        if summary:
            lines.append(f"- {summary}")
    return lines


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remainder = seconds - (minutes * 60)
    return f"{minutes}m{remainder:04.1f}s"


def _emit_progress_status(
    progress_callback: Callable[[dict[str, Any]], None] | None,
    status: str,
) -> None:
    if progress_callback is None:
        return
    progress_callback({"phase": "controller", "event": "status", "status": status})


def _normalize_sample(sample: dict[str, Any]) -> dict[str, Any]:
    required = {"question_id", "video_id", "mode", "question"}
    missing = required - set(sample)
    if missing:
        raise ValueError(f"TimeLogic sample missing fields: {sorted(missing)}")
    mode = _normalize_mode(sample["mode"])
    return {
        "question_id": str(sample["question_id"]),
        "video_id": str(sample["video_id"]),
        "mode": mode,
        "question": str(sample["question"]).strip(),
    }


def _normalize_mode(value: Any) -> str:
    mode = str(value).strip().lower()
    if mode in {"bool", "boolean"}:
        return "bool"
    if mode in {"mc", "multiple_choice", "multiple-choice"}:
        return "mc"
    raise ValueError(f"Unsupported TimeLogic mode: {value!r}")


def _sort_key(question_id: str) -> tuple[int, int | str]:
    if str(question_id).isdigit():
        return (0, int(question_id))
    return (1, str(question_id))


def _append_phrase(phrases: list[str], phrase: str) -> None:
    normalized = " ".join(str(phrase).split()).strip(" .,?")
    if not normalized:
        return
    key = normalized.lower()
    if any(existing.lower() == key for existing in phrases):
        return
    phrases.append(normalized)


def _option_letter_for_event_phrase(
    phrase: str,
    options: dict[str, str],
) -> str | None:
    normalized_phrase = _normalize_text(phrase)
    for letter, option_text in options.items():
        if normalized_phrase == _normalize_text(option_text):
            return letter
    return None


def _clean_event_phrase(phrase: str) -> str:
    cleaned = " ".join(str(phrase).split()).strip(" .,?")
    previous = None
    while previous != cleaned:
        previous = cleaned
        cleaned = re.sub(
            r"^(did|does|is it true that|the|person)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
    return cleaned.strip(" .,?")


def _normalize_text(text: str) -> str:
    return " ".join(re.findall(r"\b\w+\b", str(text).lower()))


def _bool_word_to_answer(value: str) -> str:
    return "yes" if value.lower() in {"yes", "true"} else "no"
