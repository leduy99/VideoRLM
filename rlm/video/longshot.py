import contextlib
import copy
import json
import math
import subprocess
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, Literal

from rlm.video.controller import VideoRLM
from rlm.video.memory import VideoMemoryBuilder
from rlm.video.timing import TimingRecorder, merge_timing_summaries
from rlm.video.types import Evidence, TimeSpan, VideoMemory, VideoRLMResult

LongShOTHistoryMode = Literal["gold", "candidate"]

LONGSHOT_DATASET_PATH = "MBZUAI/longshot-bench"
# MBZUAI/longshot-bench moved the public benchmark config from postvalid_v1 to
# postvalid_v2. Keep the VideoRLM context name separate below so existing
# postvalid routing and prompt behavior remains enabled.
LONGSHOT_DATASET_NAME = "postvalid_v2"
LONGSHOT_CONTEXT_DATASET_NAME = "postvalid_v1"
LONGSHOT_DATASET_SPLIT = "test"
LONGSHOT_VIDEO_URL_TEMPLATE = "https://www.youtube.com/watch?v={video_id}"
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".mov", ".webm", ".m4v")
UNSUPPORTED_VIDEO_EXTENSIONS = {".webm"}
SPEECH_PROGRESS_UNIT_WEIGHT = 1
GENERIC_VISUAL_PROGRESS_UNIT_WEIGHT = 1
LOCAL_QWEN_VISUAL_PROGRESS_UNIT_WEIGHT = 6


def _load_hf_dataset(path: str, name: str | None, split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "LongShOT dataset loading requires the optional 'datasets' package."
        ) from exc

    if _normalize_dataset_name(name) is None:
        return load_dataset(path, split=split)
    return load_dataset(path, name=_normalize_dataset_name(name), split=split)


def _normalize_dataset_name(name: str | None) -> str | None:
    if name is None:
        return None
    normalized = str(name).strip()
    if normalized.lower() in {"", "none", "null", "default"}:
        return None
    return normalized


def load_longshot_samples(
    dataset_path: str = LONGSHOT_DATASET_PATH,
    dataset_name: str | None = LONGSHOT_DATASET_NAME,
    split: str = LONGSHOT_DATASET_SPLIT,
    *,
    sample_limit: int | None = None,
    sample_start_index: int | None = None,
    sample_end_index: int | None = None,
    sample_ids: Sequence[str] | None = None,
    video_ids: Sequence[str] | None = None,
    task_filters: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    if sample_start_index is not None and sample_start_index < 1:
        raise ValueError("sample_start_index is 1-based and must be >= 1")
    if sample_end_index is not None and sample_end_index < 1:
        raise ValueError("sample_end_index is 1-based and must be >= 1")
    if (
        sample_start_index is not None
        and sample_end_index is not None
        and sample_end_index < sample_start_index
    ):
        raise ValueError("sample_end_index must be >= sample_start_index")

    dataset = _load_hf_dataset(dataset_path, dataset_name, split)
    samples = [dict(sample) for sample in dataset]

    sample_id_filter = set(sample_ids or [])
    video_id_filter = set(video_ids or [])
    task_filter = set(task_filters or [])

    if sample_id_filter:
        samples = [sample for sample in samples if sample.get("sample_id") in sample_id_filter]
    if video_id_filter:
        samples = [sample for sample in samples if sample.get("video_id") in video_id_filter]
    if task_filter:
        samples = [sample for sample in samples if sample.get("task") in task_filter]

    samples.sort(key=lambda sample: (sample.get("video_id", ""), sample.get("sample_id", "")))
    if sample_start_index is not None or sample_end_index is not None:
        start = (sample_start_index or 1) - 1
        end = sample_end_index
        samples = samples[start:end]
    if sample_limit is not None:
        return samples[:sample_limit]
    return samples


def _longshot_user_turn_context(turn: dict[str, Any]) -> dict[str, Any]:
    return _drop_none_values(
        {
            "expected_modalities": turn.get("modalities"),
            "required_tools": turn.get("required_tools"),
            "difficulty": turn.get("difficulty"),
            "conversation_role": turn.get("conversation_role"),
        }
    )


def _longshot_global_context(
    sample: dict[str, Any],
    turn_context: dict[str, Any] | None,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    context = _drop_none_values(
        {
            "sample_id": sample.get("sample_id"),
            "video_id": sample.get("video_id"),
            "dataset_name": dataset_name,
            "task": sample.get("task"),
            "sample_type": sample.get("sample_type"),
            "scenario": sample.get("scenario"),
            **(turn_context or {}),
        }
    )
    return {"benchmark": "longshotbench", "longshot": context}


def _drop_none_values(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _preferred_video_match(paths: Sequence[Path]) -> Path:
    return sorted(
        paths,
        key=lambda path: (
            path.suffix.lower() in UNSUPPORTED_VIDEO_EXTENSIONS,
            str(path),
        ),
    )[0]


class LongShOTVideoResolver:
    def __init__(
        self,
        video_dir: str | Path,
        *,
        download_missing: bool = False,
        yt_dlp_bin: str = "yt-dlp",
        cookies_from_browser: str | None = None,
        extra_ytdlp_args: Sequence[str] | None = None,
        url_template: str = LONGSHOT_VIDEO_URL_TEMPLATE,
    ):
        self.video_dir = Path(video_dir)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.download_missing = download_missing
        self.yt_dlp_bin = yt_dlp_bin
        self.cookies_from_browser = cookies_from_browser
        self.extra_ytdlp_args = tuple(extra_ytdlp_args or [])
        self.url_template = url_template

    def resolve(self, video_id: str) -> Path:
        existing = self.find(video_id)
        if existing is not None:
            return existing
        if not self.download_missing:
            raise FileNotFoundError(
                f"Could not find local video for LongShOT video_id={video_id} under {self.video_dir}"
            )
        return self.download(video_id)

    def find(self, video_id: str) -> Path | None:
        direct_matches = []
        for extension in VIDEO_EXTENSIONS:
            candidate = self.video_dir / f"{video_id}{extension}"
            if candidate.exists():
                direct_matches.append(candidate)
        if direct_matches:
            return _preferred_video_match(direct_matches)

        recursive_matches = [
            path
            for path in self.video_dir.rglob(f"{video_id}.*")
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ]
        if recursive_matches:
            return _preferred_video_match(recursive_matches)
        return None

    def download(self, video_id: str) -> Path:
        output_template = self.video_dir / f"{video_id}.%(ext)s"
        command = [
            self.yt_dlp_bin,
            "--no-progress",
            "--merge-output-format",
            "mp4",
            "-o",
            str(output_template),
        ]
        if self.cookies_from_browser:
            command.extend(["--cookies-from-browser", self.cookies_from_browser])
        command.extend(self.extra_ytdlp_args)
        command.append(self.url_template.format(video_id=video_id))
        self._run_yt_dlp(command)

        resolved = self.find(video_id)
        if resolved is None:
            raise FileNotFoundError(
                f"yt-dlp completed but no video file was created for LongShOT video_id={video_id}"
            )
        return resolved

    def _run_yt_dlp(self, command: list[str]) -> None:
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip()
            raise RuntimeError(
                "Failed to download LongShOT video with yt-dlp. "
                f"Command: {' '.join(command)}. Error: {stderr}"
            ) from exc


class LongShOTVideoUnavailableError(RuntimeError):
    def __init__(self, *, sample_id: str, video_id: str, reason: str):
        super().__init__(
            f"LongShOT video unavailable for sample_id={sample_id} video_id={video_id}: {reason}"
        )
        self.sample_id = sample_id
        self.video_id = video_id
        self.reason = reason


class LongShOTMemoryUnavailableError(RuntimeError):
    def __init__(self, *, sample_id: str, video_id: str, memory_path: Path | None):
        location = str(memory_path) if memory_path is not None else "no memory cache directory"
        super().__init__(
            f"LongShOT memory unavailable for sample_id={sample_id} video_id={video_id}: {location}"
        )
        self.sample_id = sample_id
        self.video_id = video_id
        self.memory_path = memory_path


class LongShOTBenchmarkRunner:
    def __init__(
        self,
        *,
        video_rlm: VideoRLM,
        memory_builder: VideoMemoryBuilder,
        video_resolver: LongShOTVideoResolver,
        artifact_cache_dir: str | Path | None = None,
        memory_cache_dir: str | Path | None = None,
        trace_dir: str | Path | None = None,
        dataset_name: str | None = None,
        context_dataset_name: str | None = LONGSHOT_CONTEXT_DATASET_NAME,
        history_mode: LongShOTHistoryMode = "gold",
        verbose: bool = False,
        show_progress: bool = True,
        skip_unavailable_videos: bool = False,
        memory_cache_only: bool = False,
    ):
        if history_mode not in {"gold", "candidate"}:
            raise ValueError(f"Unsupported LongShOT history mode: {history_mode}")

        self.video_rlm = video_rlm
        self.memory_builder = memory_builder
        self.video_resolver = video_resolver
        self.artifact_cache_dir = Path(artifact_cache_dir) if artifact_cache_dir else None
        self.memory_cache_dir = Path(memory_cache_dir) if memory_cache_dir else None
        self.trace_dir = Path(trace_dir) if trace_dir else None
        self.dataset_name = _normalize_dataset_name(context_dataset_name) or _normalize_dataset_name(
            dataset_name
        )
        self.history_mode = history_mode
        self.verbose = verbose
        self.show_progress = show_progress
        self.skip_unavailable_videos = skip_unavailable_videos
        self.memory_cache_only = memory_cache_only
        self._memory_cache: dict[str, tuple[VideoMemory, Path | None]] = {}

        for directory in (
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
        output_path: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        output_file = Path(output_path) if output_path else None
        if output_file is not None:
            output_file.parent.mkdir(parents=True, exist_ok=True)
        completed_ids = self._load_completed_ids(output_file) if output_file else set()
        results: list[dict[str, Any]] = []
        self._log(
            f"run_samples start total={len(samples)} completed={len(completed_ids)} "
            f"output={output_file}"
        )

        completed_sample_ids = {
            sample.get("sample_id")
            for sample in samples
            if sample.get("sample_id") in completed_ids
        }
        sample_units = [self._estimate_progress_units(sample) for sample in samples]
        progress_total = sum(sample_units)
        progress_completed = sum(
            units
            for sample, units in zip(samples, sample_units, strict=True)
            if sample.get("sample_id") in completed_ids
        )
        progress = self._build_progress()
        progress_task_id = None
        if progress is not None:
            progress.start()
            progress_task_id = progress.add_task(
                "LongShOT",
                total=progress_total,
                completed=progress_completed,
                status="starting",
                progress_label=f"{len(completed_sample_ids)}/{len(samples)} samples",
            )
        try:
            for sample_index, sample in enumerate(samples, start=1):
                sample_id = sample.get("sample_id")
                progress_label = f"{sample_index}/{len(samples)} samples"
                if sample_id in completed_ids:
                    self._progress_update(
                        progress,
                        progress_task_id,
                        description=f"LongShOT {sample_index}/{len(samples)}",
                        status=f"skipped {sample_id}",
                        progress_label=progress_label,
                    )
                    self._log(
                        f"sample {sample_index}/{len(samples)} skip completed sample_id={sample_id}"
                    )
                    continue
                sample_progress = _LongShOTProgressReporter(
                    runner=self,
                    progress=progress,
                    task_id=progress_task_id,
                    sample_id=str(sample_id),
                    estimated_units=sample_units[sample_index - 1],
                    progress_label=progress_label,
                )
                self._progress_update(
                    progress,
                    progress_task_id,
                    description=f"LongShOT {sample_index}/{len(samples)}",
                    status=f"running {sample_id}",
                    progress_label=progress_label,
                )
                self._log(f"sample {sample_index}/{len(samples)} start sample_id={sample_id}")
                sample_start = time.perf_counter()
                try:
                    result = self.run_sample(sample, progress_callback=sample_progress)
                except LongShOTMemoryUnavailableError as exc:
                    if not self.memory_cache_only:
                        self._progress_update(
                            progress,
                            progress_task_id,
                            status=f"failed {sample_id}",
                        )
                        raise
                    sample_progress.finish(status=f"skipped uncached memory {sample_id}")
                    self._log(str(exc))
                    continue
                except LongShOTVideoUnavailableError as exc:
                    if not self.skip_unavailable_videos:
                        self._progress_update(
                            progress,
                            progress_task_id,
                            status=f"failed {sample_id}",
                        )
                        raise
                    sample_progress.finish(status=f"skipped unavailable video {sample_id}")
                    self._log(str(exc))
                    continue
                except Exception:
                    self._progress_update(
                        progress,
                        progress_task_id,
                        status=f"failed {sample_id}",
                    )
                    raise
                results.append(result)
                if output_file is not None:
                    with output_file.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    self._log(f"sample {sample_id} appended output={output_file}")
                elapsed = time.perf_counter() - sample_start
                sample_progress.finish(status=f"done {sample_id} in {_format_duration(elapsed)}")
                self._log(f"sample {sample_id} done")
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
        sample_start = time.perf_counter()
        timing_recorder = TimingRecorder()
        payload = copy.deepcopy(sample)
        video_id = str(payload["video_id"])
        sample_id = str(payload.get("sample_id", video_id))
        with timing_recorder.record("longshot.memory.load_cached"):
            cached = self._load_cached_memory(video_id)
        if cached is not None:
            memory, memory_path = cached
            source_video_path = memory.metadata.get("source_video_path")
            video_path = Path(source_video_path) if source_video_path else None
            self._raise_if_unsupported_video_path(
                sample_id=sample_id,
                video_id=video_id,
                video_path=video_path,
            )
        else:
            if self.memory_cache_only:
                raise LongShOTMemoryUnavailableError(
                    sample_id=sample_id,
                    video_id=video_id,
                    memory_path=self._expected_memory_path(video_id),
                )
            self._log(f"resolve video start video_id={video_id}")
            try:
                with timing_recorder.record("longshot.video.resolve"):
                    video_path = self.video_resolver.resolve(video_id)
            except (FileNotFoundError, RuntimeError) as exc:
                raise LongShOTVideoUnavailableError(
                    sample_id=sample_id,
                    video_id=video_id,
                    reason=str(exc),
                ) from exc
            self._log(f"resolve video done path={video_path}")
            self._raise_if_unsupported_video_path(
                sample_id=sample_id,
                video_id=video_id,
                video_path=video_path,
            )
            memory, memory_path = self._load_or_build_memory(
                payload,
                video_path,
                progress_callback=progress_callback,
                timing_recorder=timing_recorder,
            )
        self._log(f"memory ready video_id={video_id} memory_path={memory_path}")

        dialogue_context: list[dict[str, str]] = []
        carried_evidence: list[Evidence] = []
        turn_results: list[dict[str, Any]] = []
        pending_question: str | None = None
        pending_turn_context: dict[str, Any] | None = None

        for index, turn in enumerate(payload.get("conversations", [])):
            role = turn.get("role")
            content = str(turn.get("content", ""))
            self._log(f"turn index={index} role={role}")

            if role == "user":
                pending_question = content
                pending_turn_context = _longshot_user_turn_context(turn)
                dialogue_context.append({"role": "user", "content": content})
                self._log(f"user question queued={_truncate_for_log(content)}")
                continue

            if role != "assistant":
                dialogue_context.append({"role": str(role), "content": content})
                continue

            if pending_question is None:
                self._log(
                    f"assistant turn skipped sample_id={sample_id} turn={index} "
                    "reason=missing_user_question"
                )
                turn["candidate_response"] = ""
                dialogue_context.append({"role": "assistant", "content": content})
                turn_results.append(
                    {
                        "turn_index": index,
                        "question": None,
                        "answer": "",
                        "execution_time": 0.0,
                        "steps_used": 0,
                        "tool_calls_used": 0,
                        "trace_path": None,
                        "skipped": True,
                        "skip_reason": "missing_user_question",
                    }
                )
                continue

            self._log(f"VideoRLM run start sample_id={sample_id} turn={index}")
            controller_start = time.perf_counter()
            result = self.video_rlm.run(
                pending_question,
                memory,
                dialogue_context=list(dialogue_context),
                task_type=payload.get("task"),
                progress_callback=progress_callback,
                global_context_overrides=_longshot_global_context(
                    payload,
                    pending_turn_context,
                    self.dataset_name,
                ),
                prior_evidence=carried_evidence,
            )
            controller_seconds = time.perf_counter() - controller_start
            timing_recorder.add("longshot.controller.run", controller_seconds)
            self._log(
                f"VideoRLM run done sample_id={sample_id} turn={index} "
                f"steps={result.state.budget.steps_used} "
                f"tool_calls={result.state.budget.tool_calls_used} "
                f"answer={_truncate_for_log(result.answer)}"
            )
            ground_truth_response = content
            turn["ground_truth_response"] = ground_truth_response
            turn["candidate_response"] = result.answer
            turn["content"] = result.answer
            trace_write_start = time.perf_counter()
            trace_path = self._write_trace(sample_id, index, result)
            timing_recorder.add(
                "longshot.trace.write",
                time.perf_counter() - trace_write_start,
            )
            if trace_path is not None:
                self._log(f"trace written path={trace_path}")
            turn_timing = merge_timing_summaries(
                result.timing,
                {
                    "components": {
                        "longshot.controller.run": {
                            "seconds": round(controller_seconds, 6),
                            "calls": 1,
                        }
                    }
                },
            )
            turn_timing["controller_wall_seconds"] = round(controller_seconds, 6)
            turn_results.append(
                {
                    "turn_index": index,
                    "question": pending_question,
                    "answer": result.answer,
                    "execution_time": result.execution_time,
                    "steps_used": result.state.budget.steps_used,
                    "tool_calls_used": result.state.budget.tool_calls_used,
                    "trace_path": str(trace_path) if trace_path else None,
                    "prior_evidence_count": len(carried_evidence),
                    "timing": turn_timing,
                }
            )

            carried_evidence = _update_longshot_carried_evidence(
                carried_evidence,
                result,
                turn_index=index,
                question=pending_question,
                answer=result.answer,
            )
            assistant_history = content if self.history_mode == "gold" else result.answer
            dialogue_context.append({"role": "assistant", "content": assistant_history})
            pending_question = None
            pending_turn_context = None

        timing_recorder.add("longshot.sample.total_wall", time.perf_counter() - sample_start)
        sample_timing = timing_recorder.snapshot()
        payload["video_rlm_metadata"] = {
            "video_path": str(video_path) if video_path is not None else None,
            "memory_path": str(memory_path) if memory_path else None,
            "history_mode": self.history_mode,
            "turn_results": turn_results,
            "timing": sample_timing,
        }
        return payload

    def _load_or_build_memory(
        self,
        sample: dict[str, Any],
        video_path: Path,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        timing_recorder: TimingRecorder | None = None,
    ) -> tuple[VideoMemory, Path | None]:
        video_id = str(sample["video_id"])
        cached_start = time.perf_counter()
        cached = self._memory_cache.get(video_id)
        if timing_recorder is not None:
            timing_recorder.add("longshot.memory.load_cached", time.perf_counter() - cached_start)
        if cached is not None:
            self._log(f"memory cache hit video_id={video_id}")
            return cached

        lock_wait_start = time.perf_counter()
        with self._video_cache_lock(video_id):
            if timing_recorder is not None:
                timing_recorder.add(
                    "longshot.memory.cache_lock_wait",
                    time.perf_counter() - lock_wait_start,
                )
            return self._load_or_build_memory_locked(
                sample=sample,
                video_path=video_path,
                progress_callback=progress_callback,
                timing_recorder=timing_recorder,
            )

    def _load_or_build_memory_locked(
        self,
        sample: dict[str, Any],
        video_path: Path,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        timing_recorder: TimingRecorder | None = None,
    ) -> tuple[VideoMemory, Path | None]:
        video_id = str(sample["video_id"])
        cached_start = time.perf_counter()
        cached = self._load_cached_memory(video_id)
        if timing_recorder is not None:
            timing_recorder.add("longshot.memory.load_cached_after_lock", time.perf_counter() - cached_start)
        if cached is not None:
            return cached

        if self.memory_cache_only:
            raise LongShOTMemoryUnavailableError(
                sample_id=str(sample.get("sample_id", video_id)),
                video_id=video_id,
                memory_path=self._expected_memory_path(video_id),
            )

        memory_path = self._expected_memory_path(video_id)
        artifact_dir = self.artifact_cache_dir / video_id if self.artifact_cache_dir else None
        artifacts = None
        if artifact_dir is not None and artifact_dir.exists():
            self._log(f"loading artifacts cache dir={artifact_dir}")
            artifacts_start = time.perf_counter()
            artifacts = self.memory_builder.load_artifacts_dir(artifact_dir)
            if timing_recorder is not None:
                timing_recorder.add(
                    "longshot.artifacts.load_cache",
                    time.perf_counter() - artifacts_start,
                )
        if artifacts is None:
            self._log(f"preparing artifacts video_id={video_id}")
            prepare_start = time.perf_counter()
            artifacts = self.memory_builder.prepare_artifacts(
                video_path=str(video_path),
                duration_seconds=self._resolve_duration_seconds(sample),
                video_id=video_id,
                metadata={
                    "longshot_sample_id": sample.get("sample_id"),
                    "longshot_task": sample.get("task"),
                    "longshot_dataset_name": self.dataset_name,
                },
                progress_callback=progress_callback,
            )
            if timing_recorder is not None:
                timing_recorder.add(
                    "longshot.artifacts.prepare",
                    time.perf_counter() - prepare_start,
                )
            if artifact_dir is not None:
                save_artifacts_start = time.perf_counter()
                self.memory_builder.save_artifacts_dir(artifacts, artifact_dir)
                if timing_recorder is not None:
                    timing_recorder.add(
                        "longshot.artifacts.save_cache",
                        time.perf_counter() - save_artifacts_start,
                    )
                self._log(f"artifacts saved dir={artifact_dir}")

        self._log(f"building memory video_id={video_id}")
        build_start = time.perf_counter()
        memory = self.memory_builder.build_from_artifacts(artifacts)
        if timing_recorder is not None:
            timing_recorder.add("longshot.memory.build", time.perf_counter() - build_start)
        if memory_path is not None:
            save_memory_start = time.perf_counter()
            self.memory_builder.save_memory(memory, memory_path)
            if timing_recorder is not None:
                timing_recorder.add(
                    "longshot.memory.save",
                    time.perf_counter() - save_memory_start,
                )
            self._log(f"memory saved path={memory_path}")

        self._memory_cache[video_id] = (memory, memory_path)
        return memory, memory_path

    @contextlib.contextmanager
    def _video_cache_lock(self, video_id: str):
        lock_root = self.memory_cache_dir or self.artifact_cache_dir
        if lock_root is None:
            yield
            return
        lock_dir = lock_root / ".locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / f"{video_id}.lock"
        with lock_path.open("w", encoding="utf-8") as handle:
            try:
                import fcntl
            except ImportError:
                yield
                return
            self._log(f"waiting memory cache lock video_id={video_id} path={lock_path}")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            self._log(f"acquired memory cache lock video_id={video_id}")
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                self._log(f"released memory cache lock video_id={video_id}")

    def _load_cached_memory(self, video_id: str) -> tuple[VideoMemory, Path | None] | None:
        if video_id in self._memory_cache:
            self._log(f"memory cache hit video_id={video_id}")
            return self._memory_cache[video_id]

        memory_path = self._expected_memory_path(video_id)
        if memory_path is not None and memory_path.exists():
            self._log(f"loading memory cache path={memory_path}")
            memory = self.memory_builder.load_memory(memory_path)
            if not self.memory_builder.memory_matches_builder_config(memory):
                self._log(f"memory cache config mismatch path={memory_path}; rebuilding")
                return None
            self._memory_cache[video_id] = (memory, memory_path)
            return memory, memory_path

        return None

    def _expected_memory_path(self, video_id: str) -> Path | None:
        return self.memory_cache_dir / f"{video_id}.json" if self.memory_cache_dir else None

    def _raise_if_unsupported_video_path(
        self,
        *,
        sample_id: str,
        video_id: str,
        video_path: Path | None,
    ) -> None:
        if video_path is None:
            return
        suffix = video_path.suffix.lower()
        if suffix not in UNSUPPORTED_VIDEO_EXTENSIONS:
            return
        raise LongShOTVideoUnavailableError(
            sample_id=sample_id,
            video_id=video_id,
            reason=(
                f"unsupported video format {suffix}; skipping to avoid ffmpeg frame "
                f"extraction failures for {video_path}"
            ),
        )

    def _resolve_duration_seconds(self, sample: dict[str, Any]) -> float:
        duration = sample.get("duration")
        if duration is None:
            raise ValueError(
                f"LongShOT sample {sample.get('sample_id')} is missing the required duration field"
            )
        return float(duration)

    def _estimate_progress_units(self, sample: dict[str, Any]) -> int:
        units = 1
        if not self._will_prepare_artifacts_for_progress(sample):
            return units
        try:
            duration_seconds = self._resolve_duration_seconds(sample)
        except (TypeError, ValueError):
            return units

        if self.memory_builder.speech_recognizer is not None:
            units += self._estimate_speech_progress_units(duration_seconds)
        if self.memory_builder.visual_summarizer is not None:
            units += self._estimate_visual_progress_units(duration_seconds)
        return max(1, units)

    def _will_prepare_artifacts_for_progress(self, sample: dict[str, Any]) -> bool:
        if self.memory_cache_only:
            return False
        video_id = sample.get("video_id")
        if video_id is None:
            return False
        video_id = str(video_id)
        if video_id in self._memory_cache:
            return False
        memory_path = self._expected_memory_path(video_id)
        if memory_path is not None and memory_path.exists():
            return False
        artifact_dir = self.artifact_cache_dir / video_id if self.artifact_cache_dir else None
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

    def _write_trace(
        self,
        sample_id: str,
        turn_index: int,
        result: VideoRLMResult,
    ) -> Path | None:
        if self.trace_dir is None:
            return None
        output_path = self.trace_dir / f"{sample_id}_turn_{turn_index:03d}.json"
        output_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        return output_path

    def _iter_output_records(self, output_path: Path) -> Iterator[dict[str, Any]]:
        decoder = json.JSONDecoder()
        with output_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                index = 0
                while index < len(text):
                    try:
                        record, end = decoder.raw_decode(text, index)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Malformed JSONL in {output_path} at line {line_number}, "
                            f"column {exc.colno}: {exc.msg}"
                        ) from exc
                    if not isinstance(record, dict):
                        raise ValueError(
                            f"Malformed JSONL in {output_path} at line {line_number}: "
                            f"expected an object, got {type(record).__name__}"
                        )
                    yield record
                    index = end
                    while index < len(text) and text[index].isspace():
                        index += 1

    def _load_completed_ids(self, output_path: Path) -> set[str]:
        if not output_path.exists():
            return set()
        completed = set()
        for record in self._iter_output_records(output_path):
            completed.add(record["sample_id"])
        return completed

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[LongShOT] {message}", flush=True)

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


class _LongShOTProgressReporter:
    def __init__(
        self,
        *,
        runner: LongShOTBenchmarkRunner,
        progress,
        task_id,
        sample_id: str,
        estimated_units: int,
        progress_label: str,
    ):
        self.runner = runner
        self.progress = progress
        self.task_id = task_id
        self.sample_id = sample_id
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
                status=f"{status} {self.sample_id}",
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
            status=f"{status} {self.sample_id}" if status else None,
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


LONGSHOT_CARRIED_EVIDENCE_LIMIT = 12


def _update_longshot_carried_evidence(
    existing: list[Evidence],
    result: VideoRLMResult,
    *,
    turn_index: int,
    question: str,
    answer: str,
    limit: int = LONGSHOT_CARRIED_EVIDENCE_LIMIT,
) -> list[Evidence]:
    candidates = [*existing, *result.state.evidence_ledger]
    best_by_key: dict[str, Evidence] = {}
    for item in candidates:
        if not _longshot_evidence_has_text(item):
            continue
        carried = copy.deepcopy(item)
        key = str(
            carried.metadata.get("longshot_carryover_key")
            or _longshot_evidence_carryover_key(carried)
        )
        fresh_current_turn = not bool(carried.metadata.get("prior_turn_evidence"))
        metadata = dict(carried.metadata)
        metadata["prior_turn_evidence"] = True
        metadata["carried_from_current_turn"] = fresh_current_turn
        metadata["longshot_carryover_key"] = key
        metadata.setdefault("prior_turn_index", turn_index)
        metadata.setdefault("prior_question", question[:500])
        metadata.setdefault("prior_answer", answer[:500])
        metadata.setdefault("role", "support")
        metadata.setdefault("slot", "prior_turn_context")
        carried.metadata = metadata
        previous = best_by_key.get(key)
        carried_score = _longshot_carryover_score(carried)
        previous_score = (
            _longshot_carryover_score(previous) if previous is not None else None
        )
        if previous is None or (
            previous_score is not None and carried_score > previous_score
        ):
            best_by_key[key] = carried
    ranked = sorted(
        best_by_key.values(),
        key=lambda item: (
            -_longshot_carryover_score(item),
            item.time_span.start,
            item.evidence_id,
        ),
    )
    return ranked[:limit]


def _longshot_evidence_has_text(item: Evidence) -> bool:
    return bool(
        str(item.metadata.get("answer_span") or "").strip()
        or item.detail.strip()
        or item.claim.strip()
    )


def _longshot_evidence_carryover_key(item: Evidence) -> str:
    text = str(item.metadata.get("answer_span") or item.detail or item.claim).strip()
    return "|".join(
        [
            item.source_node_id,
            item.modality,
            f"{item.time_span.start:.2f}",
            f"{item.time_span.end:.2f}",
            text[:160],
        ]
    )


def _longshot_carryover_score(item: Evidence) -> float:
    metadata = item.metadata
    role = str(metadata.get("role") or "")
    score = float(item.confidence)
    if item.used_in_final_answer:
        score += 100.0
    if metadata.get("carried_from_current_turn"):
        score += 8.0
    if role == "core":
        score += 30.0
    elif role == "support":
        score += 18.0
    elif role == "background":
        score += 4.0
    if str(metadata.get("answer_span") or "").strip():
        score += 12.0
    if isinstance(metadata.get("evidence_bundle"), dict):
        score += 6.0
    if metadata.get("support_fills_required_slot"):
        score += 5.0
    return score


def _truncate_for_log(text: str, max_length: int = 180) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 3] + "..."


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining_seconds = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{remaining_seconds:02d}s"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{hours}h{remaining_minutes:02d}m"
