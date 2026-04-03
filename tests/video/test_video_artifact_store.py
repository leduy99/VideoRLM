from pathlib import Path

from rlm.video import (
    PreparedArtifactStore,
    PreparedVideoArtifacts,
    SpeechSpan,
    TimeSpan,
    VideoMemoryBuilder,
    VisualSummarySpan,
)


def test_prepared_artifact_store_roundtrip(tmp_path: Path):
    artifacts = PreparedVideoArtifacts(
        video_id="sample",
        duration_seconds=42.0,
        speech_spans=[SpeechSpan(text="hello", time_span=TimeSpan(0.0, 1.0))],
        visual_summaries=[
            VisualSummarySpan(
                summary="person talks on screen",
                time_span=TimeSpan(0.0, 10.0),
                granularity="scene",
            )
        ],
        metadata={"source_video_path": "sample.mp4"},
    )

    store = PreparedArtifactStore()
    output_dir = store.save(artifacts, tmp_path / "artifacts")
    restored = store.load(output_dir)

    assert restored.video_id == "sample"
    assert restored.duration_seconds == 42.0
    assert restored.speech_spans[0].text == "hello"
    assert restored.visual_summaries[0].summary == "person talks on screen"


def test_memory_builder_can_load_artifacts_from_directory(tmp_path: Path):
    artifacts = PreparedVideoArtifacts(
        video_id="sample",
        duration_seconds=20.0,
        speech_spans=[SpeechSpan(text="change the plan", time_span=TimeSpan(2.0, 5.0))],
        metadata={"source_video_path": "sample.mp4"},
    )
    builder = VideoMemoryBuilder(
        scene_duration_seconds=10.0,
        segment_duration_seconds=5.0,
        clip_duration_seconds=2.5,
    )
    builder.save_artifacts_dir(artifacts, tmp_path / "artifact_dir")
    restored = builder.load_artifacts_dir(tmp_path / "artifact_dir")
    memory = builder.build_from_artifacts(restored)

    assert restored.video_id == "sample"
    assert memory.get_node(memory.root_id).children
