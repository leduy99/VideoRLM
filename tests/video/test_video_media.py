from pathlib import Path

import rlm.video.media as video_media
from rlm.video.media import (
    extract_frames_for_timestamps,
    get_repo_output_root,
    is_audio_path,
    make_videorlm_temp_dir,
    sample_span_timestamps,
    sample_span_timestamps_by_rate,
)
from rlm.video.types import TimeSpan


def test_sample_span_timestamps_evenly_spaced():
    timestamps = sample_span_timestamps(TimeSpan(0.0, 9.0), 2)
    assert timestamps == [3.0, 6.0]


def test_sample_span_timestamps_by_rate_uses_minimum_frame_count():
    timestamps = sample_span_timestamps_by_rate(TimeSpan(0.0, 2.0), 0.5, min_frames=3)

    assert len(timestamps) == 3
    assert timestamps == [0.5, 1.0, 1.5]


def test_sample_span_timestamps_by_rate_uses_rate_when_larger_than_minimum():
    timestamps = sample_span_timestamps_by_rate(TimeSpan(10.0, 14.0), 2.0, min_frames=3)

    assert len(timestamps) == 8
    assert timestamps[0] > 10.0
    assert timestamps[-1] < 14.0


def test_is_audio_path_detects_audio_extensions():
    assert is_audio_path("clip.wav") is True
    assert is_audio_path("clip.mp4") is False


def test_extract_frames_for_timestamps_uses_provided_timestamps(monkeypatch, tmp_path: Path):
    calls = []

    def fake_extract_frame(
        media_path,
        timestamp_seconds,
        output_path,
        ffmpeg_bin="ffmpeg",
        width=None,
    ):
        calls.append((media_path, timestamp_seconds, output_path, ffmpeg_bin, width))
        path = Path(output_path)
        path.write_bytes(b"frame")
        return path

    monkeypatch.setattr(video_media, "extract_frame", fake_extract_frame)

    paths = extract_frames_for_timestamps(
        "video.mp4",
        [1.25, 2.5],
        ffmpeg_bin="ffmpeg-test",
        width=320,
        output_dir=tmp_path,
        prefix="dense",
    )

    assert [path.name for path in paths] == ["dense_001.jpg", "dense_002.jpg"]
    assert [call[1] for call in calls] == [1.25, 2.5]
    assert calls[0][3] == "ffmpeg-test"
    assert calls[0][4] == 320


def test_videorlm_temp_dir_uses_repo_output_root():
    temp_dir = make_videorlm_temp_dir("test_videorlm_")

    try:
        assert temp_dir.exists()
        assert get_repo_output_root() in temp_dir.parents
        assert Path("output") in temp_dir.relative_to(get_repo_output_root().parent).parents
    finally:
        temp_dir.rmdir()
