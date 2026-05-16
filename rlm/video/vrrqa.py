from __future__ import annotations

import json
import math
import re
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlm.video.controller import VideoRLM
from rlm.video.longshot import VIDEO_EXTENSIONS
from rlm.video.media import extract_video_segment, probe_media_duration
from rlm.video.memory import VideoMemoryBuilder
from rlm.video.types import TimeSpan, VideoMemory

VRRQA_DATASET_PATH = "ucf-crcv/ImplicitQA"
VRRQA_SPLIT = "eval"
VRRQA_ANNOTATION_FILENAME = "ImplicitQAv0.1.2.jsonl"
DIRECT_CHOICE_PATTERN = re.compile(r"^(?:OPTION[\s_-]*)?([A-Z])[\).]?$")
LABELED_CHOICE_PATTERN = re.compile(
    r"\b(?:ANSWER|CHOICE)\s*(?:IS|=|:)\s*\(?([A-Z])\)?\b"
    r"|\bOPTION[\s_:=.-]*\(?([A-Z])\)?\b"
)
LEADING_CHOICE_PATTERN = re.compile(r"^\s*([A-Z])\s*(?:[).:]|-)\s+")
DIAGNOSTIC_PREDICTION_MARKERS = (
    "could not fill all required evidence slots",
    "could not collect enough grounded evidence",
    "found related background evidence",
    "controller exhausted its budget",
)
SPEECH_PROGRESS_UNIT_WEIGHT = 1
GENERIC_VISUAL_PROGRESS_UNIT_WEIGHT = 1
LOCAL_QWEN_VISUAL_PROGRESS_UNIT_WEIGHT = 6
SKIPPABLE_SAMPLE_ERRORS = (
    FileNotFoundError,
    KeyError,
    OSError,
    RuntimeError,
    ValueError,
    subprocess.CalledProcessError,
)


@dataclass
class VRRQAResult:
    question_id: str
    video_id: str
    prediction: str
    predicted_choice: str | None
    answer_choice: str | None
    correct: bool


def load_vrrqa_samples(
    *,
    annotation_path: str | Path | None = None,
    dataset_path: str = VRRQA_DATASET_PATH,
    split: str = VRRQA_SPLIT,
    sample_limit: int | None = None,
    question_ids: Sequence[str] | None = None,
    video_ids: Sequence[str] | None = None,
    categories: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    if annotation_path is not None and Path(annotation_path).exists():
        samples = _load_jsonl(Path(annotation_path))
    else:
        samples = _load_hf_samples(dataset_path, split)

    normalized = [_normalize_sample(sample) for sample in samples]
    question_filter = set(question_ids or [])
    video_filter = set(video_ids or [])
    category_filter = {item.lower() for item in categories or []}
    if question_filter:
        normalized = [item for item in normalized if item["question_id"] in question_filter]
    if video_filter:
        normalized = [item for item in normalized if item["video_id"] in video_filter]
    if category_filter:
        normalized = [
            item
            for item in normalized
            if str(item.get("category", "")).lower() in category_filter
            or str(item.get("category_id", "")).lower() in category_filter
        ]
    normalized.sort(key=lambda item: (item["video_id"], item["question_id"]))
    if sample_limit is not None:
        return normalized[:sample_limit]
    return normalized


def ensure_vrrqa_annotations(
    annotation_path: str | Path,
    *,
    dataset_path: str = VRRQA_DATASET_PATH,
    filename: str = VRRQA_ANNOTATION_FILENAME,
) -> Path:
    output_path = Path(annotation_path)
    if output_path.exists():
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "huggingface-cli",
        "download",
        dataset_path,
        filename,
        "--repo-type",
        "dataset",
        "--local-dir",
        str(output_path.parent),
    ]
    subprocess.run(command, check=True)
    downloaded_path = output_path.parent / filename
    if downloaded_path != output_path and downloaded_path.exists():
        output_path.write_bytes(downloaded_path.read_bytes())
    if not output_path.exists():
        raise FileNotFoundError(f"Failed to download VRR-QA annotations to {output_path}")
    return output_path


def unique_vrrqa_videos(samples: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    videos: dict[str, str] = {}
    for sample in samples:
        video_id = str(sample["video_id"])
        video_url = str(sample.get("video_url") or "").strip()
        if not video_url:
            continue
        videos.setdefault(video_id, video_url)
    return [
        {"video_id": video_id, "video_url": video_url}
        for video_id, video_url in sorted(videos.items())
    ]


class VRRQAVideoResolver:
    def __init__(
        self,
        video_dir: str | Path,
        *,
        download_missing: bool = False,
        yt_dlp_bin: str = "yt-dlp",
        cookies_from_browser: str | None = None,
        extra_ytdlp_args: Sequence[str] | None = None,
    ):
        self.video_dir = Path(video_dir)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.download_missing = download_missing
        self.yt_dlp_bin = yt_dlp_bin
        self.cookies_from_browser = cookies_from_browser
        self.extra_ytdlp_args = list(extra_ytdlp_args or [])

    def resolve(self, sample: dict[str, Any]) -> Path:
        video_id = str(sample["video_id"])
        question_id = str(sample["question_id"])
        clipped = self.find(question_id)
        if clipped is not None:
            return clipped
        cached = self.find(video_id)
        if cached is not None:
            return cached
        if not self.download_missing:
            raise FileNotFoundError(
                f"VRR-QA video {video_id} or question clip {question_id} "
                f"was not found in {self.video_dir}"
            )
        return self.download(video_id, str(sample["video_url"]))

    def find(self, identifier: str) -> Path | None:
        for extension in VIDEO_EXTENSIONS:
            path = self.video_dir / f"{identifier}{extension}"
            if path.exists():
                return path
        matches = [
            path
            for path in self.video_dir.rglob(f"{identifier}.*")
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ]
        return sorted(matches)[0] if matches else None

    def download(self, video_id: str, video_url: str) -> Path:
        if not video_url:
            raise ValueError(f"Missing video_url for video_id={video_id}")
        output_template = str(self.video_dir / f"{video_id}.%(ext)s")
        command = [
            self.yt_dlp_bin,
            "--no-progress",
            "--merge-output-format",
            "mp4",
            "-o",
            output_template,
        ]
        if self.cookies_from_browser:
            command.extend(["--cookies-from-browser", self.cookies_from_browser])
        command.extend(self.extra_ytdlp_args)
        command.append(video_url)
        subprocess.run(command, check=True)
        cached = self.find(video_id)
        if cached is None:
            raise FileNotFoundError(f"yt-dlp completed but no video file appeared for {video_id}")
        return cached


class VRRQABenchmarkRunner:
    def __init__(
        self,
        *,
        video_rlm: VideoRLM,
        memory_builder: VideoMemoryBuilder,
        video_resolver: VRRQAVideoResolver,
        segment_dir: str | Path,
        artifact_cache_dir: str | Path | None = None,
        memory_cache_dir: str | Path | None = None,
        trace_dir: str | Path | None = None,
        ffmpeg_bin: str = "ffmpeg",
        verbose: bool = False,
        show_progress: bool = True,
        skip_unavailable_videos: bool = False,
        single_window_memory: bool = True,
        force_choice_finalizer: bool = True,
    ):
        self.video_rlm = video_rlm
        self.memory_builder = memory_builder
        self.video_resolver = video_resolver
        self.segment_dir = Path(segment_dir)
        self.artifact_cache_dir = Path(artifact_cache_dir) if artifact_cache_dir else None
        self.memory_cache_dir = Path(memory_cache_dir) if memory_cache_dir else None
        self.trace_dir = Path(trace_dir) if trace_dir else None
        self.ffmpeg_bin = ffmpeg_bin
        self.verbose = verbose
        self.show_progress = show_progress
        self.skip_unavailable_videos = skip_unavailable_videos
        self.single_window_memory = single_window_memory
        self.force_choice_finalizer = force_choice_finalizer
        self._memory_cache: dict[str, VideoMemory] = {}

        for directory in (
            self.segment_dir,
            self.artifact_cache_dir,
            self.memory_cache_dir,
            self.trace_dir,
        ):
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
        completed_question_ids = {
            str(sample.get("question_id"))
            for sample in samples
            if str(sample.get("question_id")) in completed
        }
        sample_units = [self._estimate_progress_units(sample) for sample in samples]
        progress_total = sum(sample_units)
        progress_completed = sum(
            units
            for sample, units in zip(samples, sample_units, strict=True)
            if str(sample.get("question_id")) in completed
        )
        progress = self._build_progress()
        progress_task_id = None
        if progress is not None:
            progress.start()
            progress_task_id = progress.add_task(
                "VRR-QA",
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
                        description=f"VRR-QA {index}/{len(samples)}",
                        status=f"skipped completed {question_id}",
                        progress_label=progress_label,
                    )
                    self._log(f"{index}/{len(samples)} skip completed question_id={question_id}")
                    continue
                sample_progress = _VRRQAProgressReporter(
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
                    description=f"VRR-QA {index}/{len(samples)}",
                    status=f"running {question_id}",
                    progress_label=progress_label,
                )
                sample_start = time.perf_counter()
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
                    reason = self._format_skip_reason(exc)
                    self._log(f"{index}/{len(samples)} skip question_id={question_id}: {reason}")
                    result = self._skipped_record(sample, reason)
                except Exception:
                    self._progress_update(
                        progress,
                        progress_task_id,
                        status=f"failed {question_id}",
                    )
                    raise
                results.append(result)
                with output_file.open("a", encoding="utf-8") as handle:
                    json.dump(result, handle, ensure_ascii=False)
                    handle.write("\n")
                elapsed = time.perf_counter() - sample_start
                if result.get("skipped"):
                    status = f"skipped {question_id}"
                else:
                    status = f"done {question_id} in {_format_duration(elapsed)}"
                sample_progress.finish(status=status)
                log_status = "skipped" if result.get("skipped") else "done"
                self._log(
                    f"{index}/{len(samples)} {log_status} question_id={question_id} "
                    f"pred={result.get('predicted_choice')} gold={result.get('answer_choice')}"
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
        segment_path = self._segment_path(sample, video_path)
        segment_duration = self._ensure_segment(sample, video_path, segment_path)
        memory = self._load_or_build_memory(
            sample,
            segment_path,
            progress_callback,
            duration_seconds=segment_duration,
        )
        prompt = build_vrrqa_prompt(sample)
        result = self.video_rlm.run(
            prompt,
            memory,
            task_type="multiple_choice_visual_qa",
            progress_callback=progress_callback,
        )
        options = non_null_options(sample)
        predicted_choice = parse_choice_prediction(result.answer, options)
        prediction = result.answer
        forced_choice_info = None
        if self.force_choice_finalizer and predicted_choice is None:
            predicted_choice, forced_choice_info = self._finalize_choice(sample, options, result)
            prediction = forced_choice_info["prediction"]
        answer_choice = normalize_answer_choice(sample.get("answer_choice"))
        correct = predicted_choice is not None and predicted_choice == answer_choice
        trace_path = self._write_trace(sample, result.to_dict())
        record = {
            "question_id": str(sample["question_id"]),
            "video_id": str(sample["video_id"]),
            "category": sample.get("category"),
            "category_id": sample.get("category_id"),
            "question_start_time": float(sample["question_start_time"]),
            "question_stop_time": float(sample["question_stop_time"]),
            "question_text": sample["question_text"],
            "options": options,
            "prediction": prediction,
            "predicted_choice": predicted_choice,
            "answer_choice": answer_choice,
            "answer_text": sample.get("answer_text"),
            "correct": correct,
            "video_path": str(video_path),
            "segment_path": str(segment_path),
            "trace_path": str(trace_path) if trace_path else None,
            "execution_time": round(time.perf_counter() - start_time, 4),
            "steps_used": result.state.budget.steps_used,
            "tool_calls_used": result.state.budget.tool_calls_used,
        }
        if forced_choice_info is not None:
            record.update(forced_choice_info)
        return record

    def _finalize_choice(
        self,
        sample: dict[str, Any],
        options: dict[str, str],
        result,
    ) -> tuple[str, dict[str, Any]]:
        if not options:
            raise ValueError(f"VRR-QA sample {sample['question_id']} has no answer options")
        controller_client = getattr(self.video_rlm, "controller_client", None)
        if controller_client is None:
            raise ValueError("VRR-QA forced-choice finalizer requires video_rlm.controller_client")
        prompt = build_vrrqa_forced_choice_prompt(sample, options, result)
        finalizer_prediction = controller_client.completion(prompt).strip()
        predicted_choice = parse_choice_prediction(finalizer_prediction, options)
        fallback_choice = False
        if predicted_choice is None:
            predicted_choice = sorted(options)[0]
            fallback_choice = True
        return predicted_choice, {
            "prediction": finalizer_prediction,
            "rlm_prediction": result.answer,
            "forced_choice_used": True,
            "forced_choice_fallback": fallback_choice,
        }

    def _load_or_build_memory(
        self,
        sample: dict[str, Any],
        segment_path: Path,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        duration_seconds: float | None = None,
    ) -> VideoMemory:
        question_id = str(sample["question_id"])
        if question_id in self._memory_cache:
            return self._memory_cache[question_id]
        memory_path = self._memory_path(sample)
        if memory_path is not None and memory_path.exists():
            memory = self.memory_builder.load_memory(memory_path)
            self._memory_cache[question_id] = memory
            return memory

        if duration_seconds is None:
            duration_seconds = self._resolve_duration_seconds(sample)
        artifact_dir = self._artifact_dir(sample)
        artifacts = None
        if artifact_dir is not None and artifact_dir.exists():
            artifacts = self.memory_builder.load_artifacts_dir(artifact_dir)
        if artifacts is None:
            with self._single_window_overrides(duration_seconds):
                artifacts = self.memory_builder.prepare_artifacts(
                    video_path=str(segment_path),
                    duration_seconds=duration_seconds,
                    video_id=self._sample_cache_id(sample),
                    metadata={
                        "source_video_path": str(segment_path),
                        "vrrqa_question_id": sample["question_id"],
                        "vrrqa_video_id": sample["video_id"],
                        "vrrqa_video_url": sample.get("video_url"),
                        "vrrqa_original_start_time": sample["question_start_time"],
                        "vrrqa_original_stop_time": sample["question_stop_time"],
                        "vrrqa_visual_only": True,
                    },
                    progress_callback=progress_callback,
                )
            if artifact_dir is not None:
                self.memory_builder.save_artifacts_dir(artifacts, artifact_dir)
        with self._single_window_overrides(duration_seconds):
            memory = self.memory_builder.build_from_artifacts(artifacts)
        if memory_path is not None:
            self.memory_builder.save_memory(memory, memory_path)
        self._memory_cache[question_id] = memory
        return memory

    def _ensure_segment(
        self,
        sample: dict[str, Any],
        video_path: Path,
        segment_path: Path,
    ) -> float:
        if segment_path == video_path:
            return probe_media_duration(video_path)

        if segment_path.exists():
            try:
                return probe_media_duration(segment_path)
            except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError):
                segment_path.unlink(missing_ok=True)

        requested_span = TimeSpan(
            float(sample["question_start_time"]),
            float(sample["question_stop_time"]),
        )
        source_duration = probe_media_duration(video_path)
        if requested_span.start >= source_duration:
            raise ValueError(
                "VRR-QA segment starts after the local video ends: "
                f"question_id={sample['question_id']} video_id={sample['video_id']} "
                f"start={requested_span.start:.3f}s duration={source_duration:.3f}s"
            )
        clipped_span = TimeSpan(requested_span.start, min(requested_span.end, source_duration))
        extract_video_segment(
            video_path,
            clipped_span,
            segment_path,
            ffmpeg_bin=self.ffmpeg_bin,
            reencode=True,
        )
        return probe_media_duration(segment_path)

    def _is_preclipped_question_video(self, sample: dict[str, Any], video_path: Path) -> bool:
        question_id = str(sample["question_id"])
        return video_path.stem in {question_id, safe_identifier(question_id)}

    def _single_window_overrides(self, duration_seconds: float):
        return _TemporaryMemoryWindow(
            self.memory_builder, duration_seconds, self.single_window_memory
        )

    def _sample_cache_id(self, sample: dict[str, Any]) -> str:
        return safe_identifier(f"{sample['video_id']}_{sample['question_id']}")

    def _segment_path(self, sample: dict[str, Any], video_path: Path | None = None) -> Path:
        if video_path is not None and self._is_preclipped_question_video(sample, video_path):
            return video_path
        return self.segment_dir / f"{self._sample_cache_id(sample)}.mp4"

    def _memory_path(self, sample: dict[str, Any]) -> Path | None:
        if self.memory_cache_dir is None:
            return None
        return self.memory_cache_dir / f"{self._sample_cache_id(sample)}.json"

    def _artifact_dir(self, sample: dict[str, Any]) -> Path | None:
        if self.artifact_cache_dir is None:
            return None
        return self.artifact_cache_dir / self._sample_cache_id(sample)

    def _resolve_duration_seconds(self, sample: dict[str, Any]) -> float:
        return float(sample["question_stop_time"]) - float(sample["question_start_time"])

    def _estimate_progress_units(self, sample: dict[str, Any]) -> int:
        units = 1
        if not self._will_prepare_artifacts_for_progress(sample):
            return units
        try:
            duration_seconds = self._resolve_duration_seconds(sample)
        except (KeyError, TypeError, ValueError):
            return units

        if self.memory_builder.speech_recognizer is not None:
            units += self._estimate_speech_progress_units(duration_seconds)
        if self.memory_builder.visual_summarizer is not None:
            units += self._estimate_visual_progress_units(duration_seconds)
        return max(1, units)

    def _will_prepare_artifacts_for_progress(self, sample: dict[str, Any]) -> bool:
        question_id = str(sample.get("question_id"))
        if question_id in self._memory_cache:
            return False
        memory_path = self._memory_path(sample)
        if memory_path is not None and memory_path.exists():
            return False
        artifact_dir = self._artifact_dir(sample)
        return not (artifact_dir is not None and artifact_dir.exists())

    def _estimate_speech_progress_units(self, duration_seconds: float) -> int:
        recognizer = self.memory_builder.speech_recognizer
        if recognizer is None:
            return 0
        if getattr(recognizer, "forced_aligner_name", None) or getattr(
            recognizer,
            "forced_aligner_path",
            None,
        ):
            return 1
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
        if phase == "asr":
            return advance * SPEECH_PROGRESS_UNIT_WEIGHT
        if phase == "speech_refinement":
            return advance * SPEECH_PROGRESS_UNIT_WEIGHT
        return advance

    def _write_trace(self, sample: dict[str, Any], payload: dict[str, Any]) -> Path | None:
        if self.trace_dir is None:
            return None
        trace_path = self.trace_dir / f"{self._sample_cache_id(sample)}.json"
        trace_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return trace_path

    def _skipped_record(self, sample: dict[str, Any], reason: str) -> dict[str, Any]:
        return {
            "question_id": str(sample["question_id"]),
            "video_id": str(sample["video_id"]),
            "category": sample.get("category"),
            "category_id": sample.get("category_id"),
            "question_text": sample.get("question_text"),
            "options": non_null_options(sample),
            "answer_choice": normalize_answer_choice(sample.get("answer_choice")),
            "predicted_choice": None,
            "prediction": "",
            "correct": False,
            "skipped": True,
            "skip_reason": reason,
        }

    def _format_skip_reason(self, exc: Exception) -> str:
        message = str(exc)
        if message:
            return f"{exc.__class__.__name__}: {message}"
        return exc.__class__.__name__

    def _load_completed_question_ids(self, output_file: Path) -> set[str]:
        if not output_file.exists():
            return set()
        completed: set[str] = set()
        for line in output_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            completed.add(str(payload["question_id"]))
        return completed

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[VRR-QA] {message}", flush=True)

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


class _VRRQAProgressReporter:
    def __init__(
        self,
        *,
        runner: VRRQABenchmarkRunner,
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


class _TemporaryMemoryWindow:
    def __init__(
        self,
        builder: VideoMemoryBuilder,
        duration_seconds: float,
        enabled: bool,
    ):
        self.builder = builder
        self.duration_seconds = duration_seconds
        self.enabled = enabled
        self.previous: tuple[float, float, float] | None = None

    def __enter__(self):
        if not self.enabled:
            return self
        self.previous = (
            self.builder.scene_duration_seconds,
            self.builder.segment_duration_seconds,
            self.builder.clip_duration_seconds,
        )
        window = max(0.001, self.duration_seconds) + 0.001
        self.builder.scene_duration_seconds = window
        self.builder.segment_duration_seconds = window
        self.builder.clip_duration_seconds = window
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.previous is None:
            return False
        (
            self.builder.scene_duration_seconds,
            self.builder.segment_duration_seconds,
            self.builder.clip_duration_seconds,
        ) = self.previous
        return False


def build_vrrqa_prompt(sample: dict[str, Any]) -> str:
    options = non_null_options(sample)
    option_lines = [f"{letter}. {text}" for letter, text in options.items()]
    return "\n".join(
        [
            f"Question: {sample['question_text']}",
            "Options:",
            *option_lines,
            "Answer with only the option letter.",
        ]
    )


def build_vrrqa_forced_choice_prompt(
    sample: dict[str, Any],
    options: dict[str, str],
    result,
) -> str:
    option_lines = [f"{letter}. {text}" for letter, text in options.items()]
    evidence_lines = _vrrqa_evidence_lines(result)
    trace_lines = _vrrqa_trace_lines(result)
    sections = [
        "You are a strict multiple-choice evaluator for VRR-QA.",
        "Choose the best option from the listed choices using the available evidence.",
        "Return only one option letter. Do not explain.",
        "",
        f"Question: {sample['question_text']}",
        "Options:",
        *option_lines,
        "",
        f"Initial VideoRLM answer: {result.answer}",
    ]
    if evidence_lines:
        sections.extend(["", "Collected evidence:", *evidence_lines])
    if trace_lines:
        sections.extend(["", "Recent observations:", *trace_lines])
    sections.extend(["", "Final answer letter:"])
    return "\n".join(sections)


def _vrrqa_evidence_lines(result, max_items: int = 8) -> list[str]:
    evidence = getattr(getattr(result, "state", None), "evidence_ledger", [])
    ordered = sorted(evidence, key=lambda item: (-item.confidence, item.time_span.start))
    lines = []
    for item in ordered[:max_items]:
        detail = item.detail or item.claim
        lines.append(
            "- "
            + json.dumps(
                {
                    "modality": item.modality,
                    "time_span": item.time_span.to_dict(),
                    "claim": item.claim,
                    "detail": detail[:500],
                },
                ensure_ascii=True,
            )
        )
    return lines


def _vrrqa_trace_lines(result, max_items: int = 4) -> list[str]:
    lines = []
    for step in list(getattr(result, "trace", []))[-max_items:]:
        observation = step.get("observation") or {}
        summary = observation.get("summary")
        if summary:
            lines.append(f"- {summary}")
    return lines


def non_null_options(sample: dict[str, Any]) -> dict[str, str]:
    raw_options = sample.get("options") or {}
    if isinstance(raw_options, str):
        raw_options = json.loads(raw_options)
    if isinstance(raw_options, list):
        raw_options = {chr(ord("A") + index): value for index, value in enumerate(raw_options)}
    if not isinstance(raw_options, dict):
        raise ValueError(f"Unsupported options payload: {raw_options!r}")

    options: dict[str, str] = {}
    for raw_key, raw_value in raw_options.items():
        if raw_value is None:
            continue
        value = str(raw_value).strip()
        if not value or value.lower() == "null":
            continue
        letter = normalize_answer_choice(raw_key)
        if letter is None:
            continue
        options[letter] = value
    return dict(sorted(options.items()))


def normalize_answer_choice(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, int):
        if 0 <= value < 26:
            return chr(ord("A") + value)
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    if text.isdigit():
        index = int(text)
        if 0 <= index < 26:
            return chr(ord("A") + index)
    if len(text) == 1 and "A" <= text <= "Z":
        return text
    match = DIRECT_CHOICE_PATTERN.fullmatch(text)
    return match.group(1) if match else None


def parse_choice_prediction(prediction: str, options: dict[str, str]) -> str | None:
    valid_choices = set(options)
    if is_diagnostic_vrrqa_prediction(prediction):
        return None
    normalized = prediction.strip().upper()
    if normalized in valid_choices:
        return normalized
    direct_match = DIRECT_CHOICE_PATTERN.fullmatch(normalized)
    if direct_match is not None and direct_match.group(1) in valid_choices:
        return direct_match.group(1)
    leading_match = LEADING_CHOICE_PATTERN.match(normalized)
    if leading_match is not None and leading_match.group(1) in valid_choices:
        return leading_match.group(1)
    for match in LABELED_CHOICE_PATTERN.finditer(normalized):
        choice = match.group(1) or match.group(2)
        if choice in valid_choices:
            return choice

    normalized_text = _normalize_text(prediction)
    for choice, option_text in options.items():
        if normalized_text == _normalize_text(option_text):
            return choice
    return None


def is_diagnostic_vrrqa_prediction(prediction: str | None) -> bool:
    if prediction is None:
        return False
    normalized = " ".join(str(prediction).lower().split())
    return any(marker in normalized for marker in DIAGNOSTIC_PREDICTION_MARKERS)


def evaluate_vrrqa_predictions(
    predictions: Sequence[dict[str, Any]],
    samples: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    sample_by_question_id = {str(sample["question_id"]): sample for sample in samples or []}
    total = 0
    correct = 0
    skipped = 0
    excluded_question_ids: list[str] = []
    categories: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})

    for raw_prediction in predictions:
        question_id = str(raw_prediction["question_id"])
        sample = sample_by_question_id.get(question_id, raw_prediction)
        if raw_prediction.get("skipped"):
            skipped += 1
            excluded_question_ids.append(question_id)
            continue
        options = non_null_options(sample) if "options" in sample else {}
        prediction_text = str(raw_prediction.get("prediction") or "")
        predicted_choice = None
        if prediction_text and options:
            predicted_choice = parse_choice_prediction(prediction_text, options)
        if predicted_choice is None and not is_diagnostic_vrrqa_prediction(prediction_text):
            predicted_choice = normalize_answer_choice(raw_prediction.get("predicted_choice"))
        answer_choice = normalize_answer_choice(
            sample.get("answer_choice", raw_prediction.get("answer_choice"))
        )
        is_correct = predicted_choice is not None and predicted_choice == answer_choice
        total += 1
        correct += int(is_correct)
        category = str(sample.get("category") or sample.get("category_id") or "unknown")
        categories[category]["total"] += 1
        categories[category]["correct"] += int(is_correct)

    per_category = {}
    for category, counts in sorted(categories.items()):
        category_total = counts["total"]
        category_correct = counts["correct"]
        per_category[category] = {
            "total": category_total,
            "correct": category_correct,
            "accuracy": category_correct / category_total if category_total else 0.0,
        }
    macro_average = (
        sum(item["accuracy"] for item in per_category.values()) / len(per_category)
        if per_category
        else 0.0
    )
    return {
        "total_rows": len(samples) if samples is not None else len(predictions),
        "evaluated_rows": total,
        "skipped_rows": skipped,
        "excluded_question_ids": excluded_question_ids,
        "overall_accuracy": correct / total if total else 0.0,
        "correct": correct,
        "macro_average_accuracy": macro_average,
        "category_count": len(per_category),
        "per_category": per_category,
    }


def write_vrrqa_report(summary: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# VRR-QA Evaluation",
        "",
        f"- Total rows: {summary['total_rows']}",
        f"- Evaluated rows: {summary['evaluated_rows']}",
        f"- Skipped rows: {summary['skipped_rows']}",
        f"- Overall accuracy: {summary['overall_accuracy']:.2%}",
        f"- Macro-average accuracy: {summary['macro_average_accuracy']:.2%}",
        "",
        "## Categories",
        "",
        "| Category | Accuracy | Correct | Total |",
        "| --- | ---: | ---: | ---: |",
    ]
    for category, item in summary["per_category"].items():
        lines.append(
            f"| {category} | {item['accuracy']:.2%} | {item['correct']} | {item['total']} |"
        )
    if summary["excluded_question_ids"]:
        lines.extend(["", "## Excluded Question IDs", ""])
        lines.extend(f"- {question_id}" for question_id in summary["excluded_question_ids"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def safe_identifier(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "item"


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining_seconds = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{remaining_seconds:02d}s"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{hours}h{remaining_minutes:02d}m"


def _load_hf_samples(dataset_path: str, split: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("VRR-QA dataset loading requires the 'datasets' package.") from exc
    dataset = load_dataset(dataset_path, split=split)
    return [dict(sample) for sample in dataset]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON array in {path}, got {type(payload).__name__}")
        return [dict(item) for item in payload]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _normalize_sample(sample: dict[str, Any]) -> dict[str, Any]:
    payload = dict(sample)
    payload["video_id"] = str(payload["video_id"])
    payload["question_id"] = str(payload["question_id"])
    payload["question_start_time"] = float(payload["question_start_time"])
    payload["question_stop_time"] = float(payload["question_stop_time"])
    payload["question_text"] = str(payload["question_text"])
    payload["answer_choice"] = normalize_answer_choice(payload.get("answer_choice"))
    payload["options"] = non_null_options(payload)
    return payload


def _normalize_text(value: str) -> str:
    return " ".join(re.findall(r"\w+", value.lower()))
