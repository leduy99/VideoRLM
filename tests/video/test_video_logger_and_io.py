from pathlib import Path

from rlm.video import (
    PreparedVideoArtifacts,
    SpeechSpan,
    TimeSpan,
    TraceStep,
    VideoMemoryBuilder,
    VideoRLMLogger,
)


def test_memory_save_and_load_roundtrip(tmp_path: Path):
    artifacts = PreparedVideoArtifacts(
        video_id="sample",
        duration_seconds=30.0,
        speech_spans=[
            SpeechSpan(text="hello world", time_span=TimeSpan(0.0, 2.0)),
        ],
        metadata={"source_video_path": "sample.mp4"},
    )
    builder = VideoMemoryBuilder(
        scene_duration_seconds=10.0,
        segment_duration_seconds=5.0,
        clip_duration_seconds=2.5,
    )
    memory = builder.build_from_artifacts(artifacts)

    output_path = tmp_path / "memory.json"
    builder.save_memory(memory, output_path)
    restored = builder.load_memory(output_path)

    assert restored.video_id == "sample"
    assert restored.root_id == memory.root_id
    assert restored.get_node(restored.root_id).children


def test_logger_returns_metadata_and_steps(tmp_path: Path):
    logger = VideoRLMLogger(log_dir=str(tmp_path))
    logger.log_metadata({"controller_model": "mock", "video_id": "sample"})
    logger.log_step(
        TraceStep(
            step_index=1,
            state={},
            action={},
            observation={},
            next_state={},
            raw_model_response="{}",
        )
    )

    trace = logger.get_trace()
    assert trace is not None
    assert trace["metadata"]["video_id"] == "sample"
    assert len(trace["steps"]) == 1
