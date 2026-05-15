import copy
import json
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from rlm.video.controller import VideoRLM
from rlm.video.memory import VideoMemoryBuilder
from rlm.video.types import VideoMemory, VideoRLMResult

LongShOTHistoryMode = Literal["gold", "candidate"]

LONGSHOT_DATASET_PATH = "MBZUAI/longshot-bench"
LONGSHOT_DATASET_NAME = "postvalid_v1"
LONGSHOT_DATASET_SPLIT = "test"
LONGSHOT_VIDEO_URL_TEMPLATE = "https://www.youtube.com/watch?v={video_id}"
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".mov", ".webm", ".m4v")


def _load_hf_dataset(path: str, name: str | None, split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "LongShOT dataset loading requires the optional 'datasets' package."
        ) from exc

    if name is None:
        return load_dataset(path, split=split)
    return load_dataset(path, name=name, split=split)


def load_longshot_samples(
    dataset_path: str = LONGSHOT_DATASET_PATH,
    dataset_name: str | None = LONGSHOT_DATASET_NAME,
    split: str = LONGSHOT_DATASET_SPLIT,
    *,
    sample_limit: int | None = None,
    sample_ids: Sequence[str] | None = None,
    video_ids: Sequence[str] | None = None,
    task_filters: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
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
    if sample_limit is not None:
        return samples[:sample_limit]
    return samples


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
            return sorted(direct_matches)[0]

        recursive_matches = [
            path
            for path in self.video_dir.rglob(f"{video_id}.*")
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ]
        if recursive_matches:
            return sorted(recursive_matches)[0]
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
            f"LongShOT video unavailable for sample_id={sample_id} "
            f"video_id={video_id}: {reason}"
        )
        self.sample_id = sample_id
        self.video_id = video_id
        self.reason = reason


class LongShOTMemoryUnavailableError(RuntimeError):
    def __init__(self, *, sample_id: str, video_id: str, memory_path: Path | None):
        location = str(memory_path) if memory_path is not None else "no memory cache directory"
        super().__init__(
            f"LongShOT memory unavailable for sample_id={sample_id} "
            f"video_id={video_id}: {location}"
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
        self.history_mode = history_mode
        self.verbose = verbose
        self.show_progress = show_progress
        self.skip_unavailable_videos = skip_unavailable_videos
        self.memory_cache_only = memory_cache_only
        self._memory_cache: dict[str, tuple[VideoMemory, Path | None]] = {}

        for directory in (self.artifact_cache_dir, self.memory_cache_dir, self.trace_dir):
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
            sample.get("sample_id") for sample in samples if sample.get("sample_id") in completed_ids
        }
        progress = self._build_progress()
        progress_task_id = None
        if progress is not None:
            progress.start()
            progress_task_id = progress.add_task(
                "LongShOT",
                total=len(samples),
                completed=len(completed_sample_ids),
                status="starting",
            )
        try:
            for sample_index, sample in enumerate(samples, start=1):
                sample_id = sample.get("sample_id")
                if sample_id in completed_ids:
                    self._progress_update(
                        progress,
                        progress_task_id,
                        description=f"LongShOT {sample_index}/{len(samples)}",
                        status=f"skipped {sample_id}",
                    )
                    self._log(
                        f"sample {sample_index}/{len(samples)} skip completed sample_id={sample_id}"
                    )
                    continue
                self._progress_update(
                    progress,
                    progress_task_id,
                    description=f"LongShOT {sample_index}/{len(samples)}",
                    status=f"running {sample_id}",
                )
                self._log(f"sample {sample_index}/{len(samples)} start sample_id={sample_id}")
                sample_start = time.perf_counter()
                try:
                    result = self.run_sample(sample)
                except LongShOTMemoryUnavailableError as exc:
                    if not self.memory_cache_only:
                        self._progress_update(
                            progress,
                            progress_task_id,
                            status=f"failed {sample_id}",
                        )
                        raise
                    self._progress_advance(
                        progress,
                        progress_task_id,
                        status=f"skipped uncached memory {sample_id}",
                    )
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
                    self._progress_advance(
                        progress,
                        progress_task_id,
                        status=f"skipped unavailable video {sample_id}",
                    )
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
                        json.dump(result, handle, ensure_ascii=False)
                        handle.write("\n")
                    self._log(f"sample {sample_id} appended output={output_file}")
                elapsed = time.perf_counter() - sample_start
                self._progress_advance(
                    progress,
                    progress_task_id,
                    status=f"done {sample_id} in {_format_duration(elapsed)}",
                )
                self._log(f"sample {sample_id} done")
        finally:
            if progress is not None:
                progress.stop()
        self._log(f"run_samples done new_results={len(results)}")
        return results

    def run_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        payload = copy.deepcopy(sample)
        video_id = str(payload["video_id"])
        sample_id = str(payload.get("sample_id", video_id))
        cached = self._load_cached_memory(video_id)
        if cached is not None:
            memory, memory_path = cached
            source_video_path = memory.metadata.get("source_video_path")
            video_path = Path(source_video_path) if source_video_path else None
        else:
            if self.memory_cache_only:
                raise LongShOTMemoryUnavailableError(
                    sample_id=sample_id,
                    video_id=video_id,
                    memory_path=self._expected_memory_path(video_id),
                )
            self._log(f"resolve video start video_id={video_id}")
            try:
                video_path = self.video_resolver.resolve(video_id)
            except (FileNotFoundError, RuntimeError) as exc:
                raise LongShOTVideoUnavailableError(
                    sample_id=sample_id,
                    video_id=video_id,
                    reason=str(exc),
                ) from exc
            self._log(f"resolve video done path={video_path}")
            memory, memory_path = self._load_or_build_memory(payload, video_path)
        self._log(f"memory ready video_id={video_id} memory_path={memory_path}")

        dialogue_context: list[dict[str, str]] = []
        turn_results: list[dict[str, Any]] = []
        pending_question: str | None = None

        for index, turn in enumerate(payload.get("conversations", [])):
            role = turn.get("role")
            content = str(turn.get("content", ""))
            self._log(f"turn index={index} role={role}")

            if role == "user":
                pending_question = content
                dialogue_context.append({"role": "user", "content": content})
                self._log(f"user question queued={_truncate_for_log(content)}")
                continue

            if role != "assistant":
                dialogue_context.append({"role": str(role), "content": content})
                continue

            if pending_question is None:
                raise ValueError(
                    f"Assistant turn at index {index} in LongShOT sample {sample_id} "
                    "does not have a preceding user question"
                )

            self._log(f"VideoRLM run start sample_id={sample_id} turn={index}")
            result = self.video_rlm.run(
                pending_question,
                memory,
                dialogue_context=list(dialogue_context),
                task_type=payload.get("task"),
            )
            self._log(
                f"VideoRLM run done sample_id={sample_id} turn={index} "
                f"steps={result.state.budget.steps_used} "
                f"tool_calls={result.state.budget.tool_calls_used} "
                f"answer={_truncate_for_log(result.answer)}"
            )
            turn["candidate_response"] = result.answer
            trace_path = self._write_trace(sample_id, index, result)
            if trace_path is not None:
                self._log(f"trace written path={trace_path}")
            turn_results.append(
                {
                    "turn_index": index,
                    "question": pending_question,
                    "answer": result.answer,
                    "execution_time": result.execution_time,
                    "steps_used": result.state.budget.steps_used,
                    "tool_calls_used": result.state.budget.tool_calls_used,
                    "trace_path": str(trace_path) if trace_path else None,
                }
            )

            assistant_history = content if self.history_mode == "gold" else result.answer
            dialogue_context.append({"role": "assistant", "content": assistant_history})
            pending_question = None

        payload["video_rlm_metadata"] = {
            "video_path": str(video_path) if video_path is not None else None,
            "memory_path": str(memory_path) if memory_path else None,
            "history_mode": self.history_mode,
            "turn_results": turn_results,
        }
        return payload

    def _load_or_build_memory(
        self,
        sample: dict[str, Any],
        video_path: Path,
    ) -> tuple[VideoMemory, Path | None]:
        video_id = str(sample["video_id"])
        cached = self._load_cached_memory(video_id)
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
            artifacts = self.memory_builder.load_artifacts_dir(artifact_dir)
        if artifacts is None:
            self._log(f"preparing artifacts video_id={video_id}")
            artifacts = self.memory_builder.prepare_artifacts(
                video_path=str(video_path),
                duration_seconds=self._resolve_duration_seconds(sample),
                video_id=video_id,
                metadata={
                    "longshot_sample_id": sample.get("sample_id"),
                    "longshot_task": sample.get("task"),
                },
            )
            if artifact_dir is not None:
                self.memory_builder.save_artifacts_dir(artifacts, artifact_dir)
                self._log(f"artifacts saved dir={artifact_dir}")

        self._log(f"building memory video_id={video_id}")
        memory = self.memory_builder.build_from_artifacts(artifacts)
        if memory_path is not None:
            self.memory_builder.save_memory(memory, memory_path)
            self._log(f"memory saved path={memory_path}")

        self._memory_cache[video_id] = (memory, memory_path)
        return memory, memory_path

    def _load_cached_memory(self, video_id: str) -> tuple[VideoMemory, Path | None] | None:
        if video_id in self._memory_cache:
            self._log(f"memory cache hit video_id={video_id}")
            return self._memory_cache[video_id]

        memory_path = self._expected_memory_path(video_id)
        if memory_path is not None and memory_path.exists():
            self._log(f"loading memory cache path={memory_path}")
            memory = self.memory_builder.load_memory(memory_path)
            self._memory_cache[video_id] = (memory, memory_path)
            return memory, memory_path

        return None

    def _expected_memory_path(self, video_id: str) -> Path | None:
        return self.memory_cache_dir / f"{video_id}.json" if self.memory_cache_dir else None

    def _resolve_duration_seconds(self, sample: dict[str, Any]) -> float:
        duration = sample.get("duration")
        if duration is None:
            raise ValueError(
                f"LongShOT sample {sample.get('sample_id')} is missing the required duration field"
            )
        return float(duration)

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

    def _load_completed_ids(self, output_path: Path) -> set[str]:
        if not output_path.exists():
            return set()
        completed = set()
        with output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                completed.add(json.loads(line)["sample_id"])
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
            TextColumn("{task.completed:.0f}/{task.total:.0f}"),
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
    ) -> None:
        if progress is None or task_id is None:
            return
        kwargs: dict[str, Any] = {}
        if description is not None:
            kwargs["description"] = description
        if status is not None:
            kwargs["status"] = status
        if kwargs:
            progress.update(task_id, **kwargs)

    def _progress_advance(self, progress, task_id, *, status: str) -> None:
        if progress is None or task_id is None:
            return
        progress.update(task_id, advance=1, status=status)


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
