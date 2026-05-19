from rlm.video.memory import VideoMemoryBuilder
from rlm.video.types import TimeSpan


class RecordingVisualSummarizer:
    def __init__(self) -> None:
        self.recorded_spans: list[TimeSpan] = []

    def summarize(self, video_path: str, spans: list[TimeSpan]):
        self.recorded_spans = list(spans)
        return []


def test_prepare_artifacts_skips_visual_spans_shorter_than_threshold():
    summarizer = RecordingVisualSummarizer()
    builder = VideoMemoryBuilder(
        visual_summarizer=summarizer,
        scene_duration_seconds=120.0,
        segment_duration_seconds=30.0,
        clip_duration_seconds=10.0,
        min_visual_span_seconds=0.5,
    )

    artifacts = builder.prepare_artifacts(
        video_path="sample.mp4",
        duration_seconds=600.003628,
        video_id="sample",
    )

    assert artifacts.metadata["min_visual_span_seconds"] == 0.5
    assert summarizer.recorded_spans
    assert all(span.duration >= 0.5 for span in summarizer.recorded_spans)
    assert TimeSpan(600.0, 600.003628) not in summarizer.recorded_spans
    assert len(summarizer.recorded_spans) == 65
