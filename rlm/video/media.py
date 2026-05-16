import math
import re
import subprocess
import tempfile
from pathlib import Path
from shutil import which

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
    command = [
        ffmpeg_bin,
        "-y",
        "-ss",
        f"{timestamp_seconds:.3f}",
        "-i",
        str(media),
        "-frames:v",
        "1",
    ]
    if width is not None:
        command.extend(["-vf", f"scale={width}:-1"])
    command.append(str(output))
    subprocess.run(command, check=True, capture_output=True)
    return output


def extract_frames_for_span(
    media_path: str | Path,
    span: TimeSpan,
    frame_count: int = 3,
    ffmpeg_bin: str = "ffmpeg",
    width: int | None = None,
    output_dir: str | Path | None = None,
) -> list[Path]:
    timestamps = sample_span_timestamps(span, frame_count)
    if output_dir is None:
        temp_dir = make_videorlm_temp_dir("videorlm_frames_")
    else:
        temp_dir = Path(output_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for index, timestamp in enumerate(timestamps, start=1):
        frame_path = temp_dir / f"frame_{index:03d}.jpg"
        paths.append(
            extract_frame(
                media_path=media_path,
                timestamp_seconds=timestamp,
                output_path=frame_path,
                ffmpeg_bin=ffmpeg_bin,
                width=width,
            )
        )
    return paths


def extract_frames_for_timestamps(
    media_path: str | Path,
    timestamps: list[float],
    ffmpeg_bin: str = "ffmpeg",
    width: int | None = None,
    output_dir: str | Path | None = None,
    prefix: str = "frame",
) -> list[Path]:
    if output_dir is None:
        temp_dir = make_videorlm_temp_dir("videorlm_frames_")
    else:
        temp_dir = Path(output_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for index, timestamp in enumerate(timestamps, start=1):
        frame_path = temp_dir / f"{prefix}_{index:03d}.jpg"
        paths.append(
            extract_frame(
                media_path=media_path,
                timestamp_seconds=timestamp,
                output_path=frame_path,
                ffmpeg_bin=ffmpeg_bin,
                width=width,
            )
        )
    return paths


def detect_scene_boundary_timestamps(
    media_path: str | Path,
    *,
    span: TimeSpan | None = None,
    ffmpeg_bin: str = "ffmpeg",
    threshold: float = 0.35,
    max_timestamps: int = 8,
    width: int | None = 160,
) -> list[float]:
    if threshold <= 0 or threshold >= 1:
        raise ValueError(f"threshold must be within (0, 1), got {threshold}")
    if max_timestamps < 0:
        raise ValueError(f"max_timestamps must be non-negative, got {max_timestamps}")
    if max_timestamps == 0:
        return []
    if span is not None and span.duration <= 0:
        return []

    _require_executable(ffmpeg_bin)
    media = Path(media_path)
    command = [ffmpeg_bin, "-hide_banner", "-nostats"]
    if span is not None:
        command.extend(["-ss", f"{span.start:.3f}", "-t", f"{span.duration:.3f}"])
    command.extend(["-i", str(media)])
    filters = []
    if width is not None:
        filters.append(f"scale={width}:-1")
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
