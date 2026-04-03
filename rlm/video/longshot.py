import copy
import json
import subprocess
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

        for sample in samples:
            sample_id = sample.get("sample_id")
            if sample_id in completed_ids:
                continue
            result = self.run_sample(sample)
            results.append(result)
            if output_file is not None:
                with output_file.open("a", encoding="utf-8") as handle:
                    json.dump(result, handle, ensure_ascii=False)
                    handle.write("\n")
        return results

    def run_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        payload = copy.deepcopy(sample)
        video_id = str(payload["video_id"])
        sample_id = str(payload.get("sample_id", video_id))
        video_path = self.video_resolver.resolve(video_id)
        memory, memory_path = self._load_or_build_memory(payload, video_path)

        dialogue_context: list[dict[str, str]] = []
        turn_results: list[dict[str, Any]] = []
        pending_question: str | None = None

        for index, turn in enumerate(payload.get("conversations", [])):
            role = turn.get("role")
            content = str(turn.get("content", ""))

            if role == "user":
                pending_question = content
                dialogue_context.append({"role": "user", "content": content})
                continue

            if role != "assistant":
                dialogue_context.append({"role": str(role), "content": content})
                continue

            if pending_question is None:
                raise ValueError(
                    f"Assistant turn at index {index} in LongShOT sample {sample_id} "
                    "does not have a preceding user question"
                )

            result = self.video_rlm.run(
                pending_question,
                memory,
                dialogue_context=list(dialogue_context),
                task_type=payload.get("task"),
            )
            turn["candidate_response"] = result.answer
            trace_path = self._write_trace(sample_id, index, result)
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
            "video_path": str(video_path),
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
        if video_id in self._memory_cache:
            return self._memory_cache[video_id]

        memory_path = self.memory_cache_dir / f"{video_id}.json" if self.memory_cache_dir else None
        if memory_path is not None and memory_path.exists():
            memory = self.memory_builder.load_memory(memory_path)
            self._memory_cache[video_id] = (memory, memory_path)
            return memory, memory_path

        artifacts = None
        artifact_dir = self.artifact_cache_dir / video_id if self.artifact_cache_dir else None
        if artifact_dir is not None and artifact_dir.exists():
            artifacts = self.memory_builder.load_artifacts_dir(artifact_dir)
        if artifacts is None:
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

        memory = self.memory_builder.build_from_artifacts(artifacts)
        if memory_path is not None:
            self.memory_builder.save_memory(memory, memory_path)

        self._memory_cache[video_id] = (memory, memory_path)
        return memory, memory_path

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
