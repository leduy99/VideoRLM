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


def _require_executable(name: str) -> None:
    if which(name) is None:
        raise FileNotFoundError(
            f"Required executable '{name}' was not found on PATH. "
            "Install ffmpeg or pass a valid ffmpeg_bin path."
        )
