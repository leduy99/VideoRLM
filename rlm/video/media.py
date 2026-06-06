import math
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from shutil import which
from typing import Literal

from rlm.video.types import TimeSpan

AUDIO_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".wav",
    ".webm",
}
FFMPEG_SHOWINFO_PTS_PATTERN = re.compile(r"pts_time:([0-9]+(?:\.[0-9]+)?)")
FrameExtractionStrategy = Literal["auto", "batch", "seek", "sequence"]
FFMPEG_FRAME_EXTRACTION_BATCH_SIZE = 128
FFMPEG_FRAME_EXTRACTION_MIN_BATCH_FRAMES = 16
FFMPEG_FRAME_EXTRACTION_MAX_SECONDS_PER_FRAME = 1.0
FFMPEG_FRAME_EXTRACTION_MAX_UNIFORM_STEP_DRIFT_SECONDS = 0.05
FFMPEG_FRAME_EXTRACTION_SEEK_MARGIN_SECONDS = 0.25
FFMPEG_FRAME_EXTRACTION_EOF_MARGIN_SECONDS = 0.25
DEFAULT_SCENE_DETECTION_SAMPLE_RATE = None


def get_repo_output_root() -> Path:
    output_root = Path(__file__).resolve().parents[2] / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def get_videorlm_output_root() -> Path:
    output_root = get_repo_output_root() / "videorlm"
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def make_videorlm_temp_dir(prefix: str) -> Path:
    temp_root = get_videorlm_output_root() / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=temp_root))


def is_audio_path(media_path: str | Path) -> bool:
    return Path(media_path).suffix.lower() in AUDIO_EXTENSIONS


def sample_span_timestamps(span: TimeSpan, frame_count: int) -> list[float]:
    if frame_count <= 0:
        raise ValueError(f"frame_count must be positive, got {frame_count}")
    if span.duration == 0:
        return [span.start] * frame_count

    step = span.duration / (frame_count + 1)
    return [span.start + (step * (index + 1)) for index in range(frame_count)]


def sample_span_timestamps_by_rate(
    span: TimeSpan,
    frame_rate: float,
    min_frames: int = 1,
) -> list[float]:
    if frame_rate <= 0:
        raise ValueError(f"frame_rate must be positive, got {frame_rate}")
    if min_frames <= 0:
        raise ValueError(f"min_frames must be positive, got {min_frames}")
    if span.duration == 0:
        return [span.start] * min_frames

    frame_count = max(min_frames, math.ceil(span.duration * frame_rate))
    return sample_span_timestamps(span, frame_count)


def extract_audio_track(
    media_path: str | Path,
    output_path: str | Path,
    ffmpeg_bin: str = "ffmpeg",
    sample_rate: int = 16_000,
) -> Path:
    _require_executable(ffmpeg_bin)
    media = Path(media_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(media),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True)
    return output


def extract_audio_segment(
    media_path: str | Path,
    span: TimeSpan,
    output_path: str | Path,
    ffmpeg_bin: str = "ffmpeg",
    sample_rate: int = 16_000,
) -> Path:
    _require_executable(ffmpeg_bin)
    media = Path(media_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_bin,
        "-y",
        "-ss",
        f"{span.start:.3f}",
        "-t",
        f"{span.duration:.3f}",
        "-i",
        str(media),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True)
    return output


def extract_video_segment(
    media_path: str | Path,
    span: TimeSpan,
    output_path: str | Path,
    ffmpeg_bin: str = "ffmpeg",
    *,
    reencode: bool = True,
) -> Path:
    _require_executable(ffmpeg_bin)
    if span.duration <= 0:
        raise ValueError(f"Video segment duration must be positive, got {span.duration}")
    media = Path(media_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_bin,
        "-y",
        "-ss",
        f"{span.start:.3f}",
        "-t",
        f"{span.duration:.3f}",
        "-i",
        str(media),
        "-map",
        "0:v:0",
        "-an",
    ]
    if reencode:
        command.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"])
    else:
        command.extend(["-c:v", "copy"])
    command.extend(["-movflags", "+faststart", str(output)])
    subprocess.run(command, check=True, capture_output=True)
    return output


def probe_media_duration(media_path: str | Path, ffprobe_bin: str = "ffprobe") -> float:
    _require_executable(ffprobe_bin)
    media = Path(media_path)
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(media),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    duration_text = result.stdout.strip()
    if not duration_text:
        raise ValueError(f"ffprobe returned an empty duration for media_path={media}")
    return float(duration_text)


def _probe_video_stream_duration(media_path: str | Path, ffprobe_bin: str = "ffprobe") -> float | None:
    _require_executable(ffprobe_bin)
    media = Path(media_path)
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(media),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    duration_text = result.stdout.strip()
    if not duration_text:
        return None
    first_value = duration_text.splitlines()[0].strip()
    if not first_value or first_value == "N/A":
        return None
    duration = float(first_value)
    if duration <= 0:
        return None
    return duration


def _probe_frame_extraction_duration(media_path: str | Path, ffprobe_bin: str) -> float:
    stream_duration = _probe_video_stream_duration(media_path, ffprobe_bin=ffprobe_bin)
    if stream_duration is not None:
        return stream_duration
    return probe_media_duration(media_path, ffprobe_bin=ffprobe_bin)


def extract_frame(
    media_path: str | Path,
    timestamp_seconds: float,
    output_path: str | Path,
    ffmpeg_bin: str = "ffmpeg",
    width: int | None = None,
) -> Path:
    _require_executable(ffmpeg_bin)
    media = Path(media_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    for seek_before_input in (True, False):
        command = _extract_frame_command(
            media=media,
            timestamp_seconds=timestamp_seconds,
            output=output,
            ffmpeg_bin=ffmpeg_bin,
            width=width,
            seek_before_input=seek_before_input,
        )
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode == 0 and output.exists():
            return output
        if output.exists() and output.stat().st_size > 0:
            return output
        errors.append(_format_ffmpeg_error(command, result))

    raise RuntimeError(
        f"ffmpeg failed to extract frame at {timestamp_seconds:.3f}s from {media}. "
        + " | ".join(errors)
    )


def _extract_frame_command(
    *,
    media: Path,
    timestamp_seconds: float,
    output: Path,
    ffmpeg_bin: str,
    width: int | None,
    seek_before_input: bool,
) -> list[str]:
    command = [ffmpeg_bin, "-hide_banner", "-nostats", "-y"]
    if seek_before_input:
        command.extend(["-ss", f"{timestamp_seconds:.3f}"])
    command.extend(["-i", str(media)])
    if not seek_before_input:
        command.extend(["-ss", f"{timestamp_seconds:.3f}"])
    command.extend(["-map", "0:v:0", "-an", "-sn", "-dn", "-frames:v", "1"])
    if width is not None:
        command.extend(["-vf", f"scale={width}:-2"])
    command.extend(["-update", "1", "-q:v", "2", str(output)])
    return command


def _format_ffmpeg_error(command: list[str], result: subprocess.CompletedProcess[str]) -> str:
    stderr = (result.stderr or "").strip().splitlines()
    tail = "\n".join(stderr[-8:]) if stderr else "no stderr"
    return f"returncode={result.returncode} command={' '.join(command)} stderr={tail}"


def extract_frames_for_span(
    media_path: str | Path,
    span: TimeSpan,
    frame_count: int = 3,
    ffmpeg_bin: str = "ffmpeg",
    width: int | None = None,
    output_dir: str | Path | None = None,
    extraction_strategy: FrameExtractionStrategy = "auto",
    seek_workers: int = 1,
) -> list[Path]:
    return extract_frames_for_timestamps(
        media_path=media_path,
        timestamps=sample_span_timestamps(span, frame_count),
        ffmpeg_bin=ffmpeg_bin,
        width=width,
        output_dir=output_dir,
        prefix="frame",
        extraction_strategy=extraction_strategy,
        seek_workers=seek_workers,
    )


def extract_frames_for_timestamps(
    media_path: str | Path,
    timestamps: list[float],
    ffmpeg_bin: str = "ffmpeg",
    width: int | None = None,
    output_dir: str | Path | None = None,
    prefix: str = "frame",
    extraction_strategy: FrameExtractionStrategy = "auto",
    seek_workers: int = 1,
) -> list[Path]:
    if output_dir is None:
        temp_dir = make_videorlm_temp_dir("videorlm_frames_")
    else:
        temp_dir = Path(output_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

    paths = [temp_dir / f"{prefix}_{index:03d}.jpg" for index in range(1, len(timestamps) + 1)]
    if not paths:
        return []

    if extraction_strategy not in {"auto", "batch", "seek", "sequence"}:
        raise ValueError(f"Unsupported frame extraction strategy: {extraction_strategy}")
    if seek_workers < 1:
        raise ValueError(f"seek_workers must be at least 1, got {seek_workers}")

    timestamps = _clamp_timestamps_for_media_duration(
        media_path=media_path,
        timestamps=timestamps,
        ffprobe_bin=_ffprobe_bin_for_ffmpeg(ffmpeg_bin),
    )

    should_sequence = extraction_strategy == "sequence"
    if should_sequence:
        try:
            return _extract_frames_for_timestamps_sequence(
                media_path=media_path,
                timestamps=timestamps,
                output_paths=paths,
                ffmpeg_bin=ffmpeg_bin,
                width=width,
            )
        except (OSError, RuntimeError, subprocess.SubprocessError):
            pass

    should_batch = extraction_strategy == "batch" or (
        extraction_strategy == "auto" and _should_batch_frame_extraction(timestamps)
    )
    if should_batch:
        try:
            return _extract_frames_for_timestamps_batched(
                media_path=media_path,
                timestamps=timestamps,
                output_paths=paths,
                ffmpeg_bin=ffmpeg_bin,
                width=width,
            )
        except (OSError, RuntimeError, subprocess.SubprocessError):
            pass

    return _extract_frames_for_timestamps_seeked(
        media_path=media_path,
        timestamps=timestamps,
        output_paths=paths,
        ffmpeg_bin=ffmpeg_bin,
        width=width,
        seek_workers=seek_workers,
    )


def _should_sequence_frame_extraction(timestamps: list[float]) -> bool:
    if len(timestamps) < FFMPEG_FRAME_EXTRACTION_MIN_BATCH_FRAMES:
        return False
    step = _uniform_timestamp_step(timestamps)
    return step is not None and step > 0


def _clamp_timestamps_for_media_duration(
    *,
    media_path: str | Path,
    timestamps: list[float],
    ffprobe_bin: str,
) -> list[float]:
    if not timestamps:
        return []
    try:
        duration = _probe_frame_extraction_duration(media_path, ffprobe_bin=ffprobe_bin)
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError):
        return list(timestamps)
    if duration <= 0:
        return list(timestamps)
    safe_end = max(0.0, duration - FFMPEG_FRAME_EXTRACTION_EOF_MARGIN_SECONDS)
    return [min(max(0.0, timestamp), safe_end) for timestamp in timestamps]


def _ffprobe_bin_for_ffmpeg(ffmpeg_bin: str) -> str:
    if ffmpeg_bin.endswith("ffmpeg"):
        return ffmpeg_bin[: -len("ffmpeg")] + "ffprobe"
    return "ffprobe"


def _uniform_timestamp_step(timestamps: list[float]) -> float | None:
    if len(timestamps) < 2:
        return None
    deltas = [right - left for left, right in zip(timestamps, timestamps[1:], strict=False)]
    if any(delta <= 0 for delta in deltas):
        return None
    step = sum(deltas) / len(deltas)
    tolerance = max(FFMPEG_FRAME_EXTRACTION_MAX_UNIFORM_STEP_DRIFT_SECONDS, step * 0.02)
    if any(abs(delta - step) > tolerance for delta in deltas):
        return None
    return step


def _extract_frames_for_timestamps_sequence(
    *,
    media_path: str | Path,
    timestamps: list[float],
    output_paths: list[Path],
    ffmpeg_bin: str,
    width: int | None,
) -> list[Path]:
    step = _uniform_timestamp_step(timestamps)
    if step is None or step <= 0:
        raise RuntimeError("ffmpeg sequence extraction requires uniformly spaced timestamps")

    _require_executable(ffmpeg_bin)
    output_dir = output_paths[0].parent
    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = _sequence_output_prefix(output_paths)
    output_pattern = str(output_dir / f"{prefix}_%03d.jpg")
    fps = 1.0 / step
    filters = [f"fps=fps={fps:.8f}"]
    if width is not None:
        filters.append(f"scale={width}:-2")
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-nostats",
        "-y",
        "-ss",
        f"{timestamps[0]:.3f}",
        "-i",
        str(Path(media_path)),
        "-frames:v",
        str(len(timestamps)),
        "-vf",
        ",".join(filters),
        "-q:v",
        "2",
        output_pattern,
    ]
    subprocess.run(command, check=True, capture_output=True)
    missing_paths = [path for path in output_paths if not path.exists()]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths[:3])
        if len(missing_paths) > 3:
            missing += ", ..."
        raise RuntimeError(f"ffmpeg sequence frame extraction missed output file(s): {missing}")
    return output_paths


def _sequence_output_prefix(output_paths: list[Path]) -> str:
    first_stem = output_paths[0].stem
    match = re.match(r"(.+)_\d+$", first_stem)
    return match.group(1) if match else first_stem


def _should_batch_frame_extraction(timestamps: list[float]) -> bool:
    if len(timestamps) < FFMPEG_FRAME_EXTRACTION_MIN_BATCH_FRAMES:
        return False
    timestamp_range = max(timestamps) - min(timestamps)
    if timestamp_range <= 0:
        return False
    seconds_per_frame = timestamp_range / max(len(timestamps) - 1, 1)
    return seconds_per_frame <= FFMPEG_FRAME_EXTRACTION_MAX_SECONDS_PER_FRAME


def _extract_frames_for_timestamps_seeked(
    *,
    media_path: str | Path,
    timestamps: list[float],
    output_paths: list[Path],
    ffmpeg_bin: str,
    width: int | None,
    seek_workers: int,
) -> list[Path]:
    def extract_one(item: tuple[float, Path]) -> Path:
        timestamp, frame_path = item
        return extract_frame(
            media_path=media_path,
            timestamp_seconds=timestamp,
            output_path=frame_path,
            ffmpeg_bin=ffmpeg_bin,
            width=width,
        )

    items = list(zip(timestamps, output_paths, strict=True))
    if seek_workers == 1 or len(items) <= 1:
        return [extract_one(item) for item in items]

    worker_count = min(seek_workers, len(items))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(extract_one, items))


def _extract_frames_for_timestamps_batched(
    *,
    media_path: str | Path,
    timestamps: list[float],
    output_paths: list[Path],
    ffmpeg_bin: str,
    width: int | None,
) -> list[Path]:
    if len(timestamps) != len(output_paths):
        raise ValueError(
            "timestamps and output_paths must have the same length, "
            f"got {len(timestamps)} and {len(output_paths)}"
        )
    _require_executable(ffmpeg_bin)
    for start in range(0, len(timestamps), FFMPEG_FRAME_EXTRACTION_BATCH_SIZE):
        end = start + FFMPEG_FRAME_EXTRACTION_BATCH_SIZE
        _run_frame_extraction_batch(
            media_path=media_path,
            timestamps=timestamps[start:end],
            output_paths=output_paths[start:end],
            ffmpeg_bin=ffmpeg_bin,
            width=width,
        )
    return output_paths


def _run_frame_extraction_batch(
    *,
    media_path: str | Path,
    timestamps: list[float],
    output_paths: list[Path],
    ffmpeg_bin: str,
    width: int | None,
) -> None:
    if not timestamps:
        return

    media = Path(media_path)
    seek_start = max(
        0.0,
        min(timestamps) - FFMPEG_FRAME_EXTRACTION_SEEK_MARGIN_SECONDS,
    )
    command = [ffmpeg_bin, "-hide_banner", "-nostats", "-y"]
    if seek_start > 0:
        command.extend(["-ss", f"{seek_start:.3f}"])
    command.extend(["-i", str(media)])

    for timestamp, output_path in zip(timestamps, output_paths, strict=True):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        relative_timestamp = max(0.0, timestamp - seek_start)
        command.extend(["-map", "0:v:0", "-ss", f"{relative_timestamp:.3f}"])
        if width is not None:
            command.extend(["-vf", f"scale={width}:-2"])
        command.extend(["-frames:v", "1", "-update", "1", str(output_path)])

    subprocess.run(command, check=True, capture_output=True)
    missing_paths = [path for path in output_paths if not path.exists()]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths[:3])
        if len(missing_paths) > 3:
            missing += ", ..."
        raise RuntimeError(f"ffmpeg batch frame extraction missed output file(s): {missing}")


def detect_scene_boundary_timestamps(
    media_path: str | Path,
    *,
    span: TimeSpan | None = None,
    ffmpeg_bin: str = "ffmpeg",
    threshold: float = 0.35,
    max_timestamps: int = 8,
    width: int | None = 160,
    sample_rate: float | None = DEFAULT_SCENE_DETECTION_SAMPLE_RATE,
    keyframes_only: bool = False,
) -> list[float]:
    if threshold <= 0 or threshold >= 1:
        raise ValueError(f"threshold must be within (0, 1), got {threshold}")
    if max_timestamps < 0:
        raise ValueError(f"max_timestamps must be non-negative, got {max_timestamps}")
    if max_timestamps == 0:
        return []
    if sample_rate is not None and sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive when provided, got {sample_rate}")
    if span is not None and span.duration <= 0:
        return []

    _require_executable(ffmpeg_bin)
    media = Path(media_path)
    command = [ffmpeg_bin, "-hide_banner", "-nostats"]
    if keyframes_only:
        command.extend(["-skip_frame", "nokey"])
    if span is not None:
        command.extend(["-ss", f"{span.start:.3f}", "-t", f"{span.duration:.3f}"])
    command.extend(["-i", str(media)])
    filters = []
    if sample_rate is not None:
        filters.append(f"fps=fps={sample_rate:.6f}")
    if width is not None:
        filters.append(f"scale={width}:-2")
    filters.append(f"select=gt(scene\\,{threshold:.3f})")
    filters.append("showinfo")
    command.extend(["-vf", ",".join(filters), "-f", "null", "-"])
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    timestamps = [
        _scene_timestamp_from_ffmpeg_value(float(match.group(1)), span)
        for match in FFMPEG_SHOWINFO_PTS_PATTERN.finditer(result.stderr)
    ]
    timestamps = _dedupe_timestamps(timestamps, tolerance=0.1)
    if span is not None:
        timestamps = [timestamp for timestamp in timestamps if span.start <= timestamp <= span.end]
    return _limit_timestamps_by_temporal_coverage(timestamps, max_timestamps)


def _require_executable(name: str) -> None:
    if which(name) is None:
        raise FileNotFoundError(
            f"Required executable '{name}' was not found on PATH. "
            "Install ffmpeg or pass a valid ffmpeg_bin path."
        )


def _scene_timestamp_from_ffmpeg_value(value: float, span: TimeSpan | None) -> float:
    if span is None:
        return value
    if value < span.start - 0.25:
        return span.start + value
    return value


def _dedupe_timestamps(timestamps: list[float], *, tolerance: float) -> list[float]:
    deduped: list[float] = []
    for timestamp in sorted(timestamps):
        if deduped and abs(timestamp - deduped[-1]) <= tolerance:
            continue
        deduped.append(timestamp)
    return deduped


def _limit_timestamps_by_temporal_coverage(
    timestamps: list[float],
    max_count: int,
) -> list[float]:
    if len(timestamps) <= max_count:
        return list(timestamps)
    if max_count == 1:
        return [timestamps[len(timestamps) // 2]]
    indices = sorted(
        {round(position * (len(timestamps) - 1) / (max_count - 1)) for position in range(max_count)}
    )
    return [timestamps[index] for index in indices]
