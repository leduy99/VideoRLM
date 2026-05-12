import json
from pathlib import Path

import rlm.video.cli as video_cli
from rlm.core.types import ModelUsageSummary, UsageSummary
from rlm.video.cli import main
from rlm.video.memory import PreparedVideoArtifacts, VideoMemoryBuilder
from rlm.video.types import ControllerState, SpeechSpan, TimeSpan, VideoRLMResult


def test_cli_build_memory_from_artifact_directory(tmp_path: Path, capsys):
    builder = VideoMemoryBuilder(
        scene_duration_seconds=10.0,
        segment_duration_seconds=5.0,
        clip_duration_seconds=2.5,
    )
    artifacts = PreparedVideoArtifacts(
        video_id="sample",
        duration_seconds=20.0,
        speech_spans=[SpeechSpan(text="change plan", time_span=TimeSpan(1.0, 3.0))],
        metadata={"source_video_path": "sample.mp4"},
    )
    artifact_dir = builder.save_artifacts_dir(artifacts, tmp_path / "artifacts")
    output_path = tmp_path / "memory.json"

    exit_code = main(
        [
            "build-memory",
            "--artifacts",
            str(artifact_dir),
            "--output",
            str(output_path),
            "--scene-duration-seconds",
            "10",
            "--segment-duration-seconds",
            "5",
            "--clip-duration-seconds",
            "2.5",
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    assert "Saved memory JSON" in capsys.readouterr().out


def test_cli_ask_writes_trace(monkeypatch, tmp_path: Path, capsys):
    builder = VideoMemoryBuilder()
    memory = builder.build_from_artifacts(
        PreparedVideoArtifacts(
            video_id="sample",
            duration_seconds=10.0,
            metadata={"source_video_path": "sample.mp4"},
        )
    )
    memory_path = tmp_path / "memory.json"
    builder.save_memory(memory, memory_path)

    class FakeRunner:
        def run(self, question, memory, task_type=None):
            return VideoRLMResult(
                answer="mock answer",
                state=ControllerState(
                    question=question,
                    global_context={"video_id": "sample"},
                ),
                trace=[],
                usage_summary=UsageSummary(
                    model_usage_summaries={
                        "mock": ModelUsageSummary(
                            total_calls=1,
                            total_input_tokens=10,
                            total_output_tokens=10,
                        )
                    }
                ),
                execution_time=0.1,
            )

    monkeypatch.setattr(video_cli, "_build_runner", lambda args, logger=None: FakeRunner())
    trace_path = tmp_path / "trace.json"
    exit_code = main(
        [
            "ask",
            "--memory",
            str(memory_path),
            "--question",
            "What changed?",
            "--controller-base-url",
            "http://127.0.0.1:8000/v1",
            "--trace-out",
            str(trace_path),
        ]
    )

    assert exit_code == 0
    assert trace_path.exists()
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert payload["answer"] == "mock answer"
    assert "mock answer" in capsys.readouterr().out


def test_cli_run_longshot(monkeypatch, tmp_path: Path, capsys):
    class FakeRunner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run_samples(self, samples, output_path=None):
            output = Path(output_path)
            output.write_text(
                json.dumps({"sample_id": samples[0]["sample_id"], "conversations": []}) + "\n",
                encoding="utf-8",
            )
            return [{"sample_id": samples[0]["sample_id"]}]

    class FakeBundle:
        controller = object()
        memory_builder = object()

    monkeypatch.setattr(
        video_cli,
        "_build_qwen_bundle",
        lambda args, logger=None: FakeBundle(),
    )
    monkeypatch.setattr(
        video_cli,
        "LongShOTBenchmarkRunner",
        lambda **kwargs: FakeRunner(**kwargs),
    )
    monkeypatch.setattr(
        video_cli,
        "load_longshot_samples",
        lambda **kwargs: [{"sample_id": "sample_1", "video_id": "video_1"}],
    )
    output_path = tmp_path / "longshot.jsonl"
    video_dir = tmp_path / "videos"
    video_dir.mkdir()

    exit_code = main(
        [
            "run-longshot",
            "--output",
            str(output_path),
            "--video-dir",
            str(video_dir),
            "--base-url",
            "http://127.0.0.1:8000/v1",
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    assert "Saved 1 LongShOT prediction records" in capsys.readouterr().out


def test_cli_build_qwen_bundle_applies_pitome_args(monkeypatch):
    class FakeStack:
        def __init__(self):
            self.ffmpeg_bin = None
            self.frame_count = None
            self.frame_width = None
            self.scene_duration_seconds = None
            self.segment_duration_seconds = None
            self.clip_duration_seconds = None

        def build_bundle(self, **kwargs):
            return self

    fake_stack = FakeStack()
    monkeypatch.setattr(
        video_cli.QwenVideoStackConfig,
        "from_shared_endpoint",
        lambda **kwargs: fake_stack,
    )
    args = video_cli.build_parser().parse_args(
        [
            "run-longshot",
            "--output",
            "out.jsonl",
            "--video-dir",
            "videos",
            "--base-url",
            "http://127.0.0.1:8000/v1",
            "--use-pitome",
            "--frame-count",
            "5",
            "--frame-width",
            "512",
            "--clip-duration-seconds",
            "60",
            "--pitome-dense-frame-rate",
            "2",
            "--pitome-min-frame-count",
            "8",
            "--pitome-embedding-backend",
            "hybrid",
            "--pitome-embedding-size",
            "32",
            "--pitome-max-selected-frames",
            "6",
        ]
    )

    returned = video_cli._build_qwen_bundle(args, logger=None)

    assert returned is fake_stack
    assert fake_stack.use_pitome is True
    assert fake_stack.frame_count == 5
    assert fake_stack.frame_width == 512
    assert fake_stack.clip_duration_seconds == 60
    assert fake_stack.pitome_dense_frame_rate == 2
    assert fake_stack.pitome_min_frame_count == 8
    assert fake_stack.pitome_embedding_backend == "hybrid"
    assert fake_stack.pitome_embedding_size == 32
    assert fake_stack.pitome_max_selected_frames == 6


def test_cli_download_qwen_local_models(monkeypatch, capsys):
    class FakeConfig:
        def download_models(self):
            return {"controller": "/tmp/controller", "visual": "/tmp/visual"}

    monkeypatch.setattr(video_cli, "_build_local_qwen_config", lambda args: FakeConfig())

    exit_code = main(["download-qwen-local-models"])

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "controller:" in stdout
    assert "visual:" in stdout


def test_cli_run_longshot_local(monkeypatch, tmp_path: Path, capsys):
    class FakeRunner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run_samples(self, samples, output_path=None):
            output = Path(output_path)
            output.write_text(
                json.dumps({"sample_id": samples[0]["sample_id"], "conversations": []}) + "\n",
                encoding="utf-8",
            )
            return [{"sample_id": samples[0]["sample_id"]}]

    class FakeBundle:
        controller = object()
        memory_builder = object()

    class FakeConfig:
        def build_bundle(self, **kwargs):
            return FakeBundle()

    monkeypatch.setattr(video_cli, "_build_local_qwen_config", lambda args: FakeConfig())
    monkeypatch.setattr(
        video_cli,
        "LongShOTBenchmarkRunner",
        lambda **kwargs: FakeRunner(**kwargs),
    )
    monkeypatch.setattr(
        video_cli,
        "load_longshot_samples",
        lambda **kwargs: [{"sample_id": "sample_1", "video_id": "video_1"}],
    )
    output_path = tmp_path / "longshot_local.jsonl"
    video_dir = tmp_path / "videos"
    video_dir.mkdir()

    exit_code = main(
        [
            "run-longshot-local",
            "--output",
            str(output_path),
            "--video-dir",
            str(video_dir),
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    assert "Saved 1 LongShOT prediction records" in capsys.readouterr().out
