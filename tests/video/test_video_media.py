from pathlib import Path

from rlm.video.media import (
    get_repo_output_root,
    is_audio_path,
    make_videorlm_temp_dir,
    sample_span_timestamps,
)
from rlm.video.types import TimeSpan


def test_sample_span_timestamps_evenly_spaced():
    timestamps = sample_span_timestamps(TimeSpan(0.0, 9.0), 2)
    assert timestamps == [3.0, 6.0]


def test_is_audio_path_detects_audio_extensions():
    assert is_audio_path("clip.wav") is True
    assert is_audio_path("clip.mp4") is False


def test_videorlm_temp_dir_uses_repo_output_root():
    temp_dir = make_videorlm_temp_dir("test_videorlm_")

    try:
        assert temp_dir.exists()
        assert get_repo_output_root() in temp_dir.parents
        assert Path("output") in temp_dir.relative_to(get_repo_output_root().parent).parents
    finally:
        temp_dir.rmdir()
