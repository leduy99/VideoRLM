import json
from pathlib import Path

import rlm.video.longshot as longshot
from rlm.core.types import ModelUsageSummary, UsageSummary
from rlm.video.memory import VideoMemoryBuilder
from rlm.video.types import ControllerState, VideoRLMResult


def test_load_longshot_samples_filters_and_limit(monkeypatch):
    samples = [
        {"sample_id": "sample_b", "video_id": "video_2", "task": "summarization"},
        {"sample_id": "sample_a", "video_id": "video_1", "task": "retrieval"},
        {"sample_id": "sample_c", "video_id": "video_2", "task": "retrieval"},
    ]

    monkeypatch.setattr(longshot, "_load_hf_dataset", lambda path, name, split: samples)
    result = longshot.load_longshot_samples(
        sample_limit=1,
        video_ids=["video_2"],
        task_filters=["retrieval"],
    )

    assert [sample["sample_id"] for sample in result] == ["sample_c"]


def test_longshot_video_resolver_finds_recursive_match(tmp_path: Path):
    nested_dir = tmp_path / "videos" / "nested"
    nested_dir.mkdir(parents=True)
    video_path = nested_dir / "abc123.mp4"
    video_path.write_text("video", encoding="utf-8")

    resolver = longshot.LongShOTVideoResolver(tmp_path / "videos")

    assert resolver.resolve("abc123") == video_path


def test_longshot_video_resolver_downloads_when_missing(monkeypatch, tmp_path: Path):
    resolver = longshot.LongShOTVideoResolver(tmp_path / "videos", download_missing=True)

    def fake_run(command: list[str]) -> None:
        assert command[0] == "yt-dlp"
        output_path = resolver.video_dir / "missing_video.mp4"
        output_path.write_text("video", encoding="utf-8")

    monkeypatch.setattr(resolver, "_run_yt_dlp", fake_run)

    assert resolver.resolve("missing_video") == resolver.video_dir / "missing_video.mp4"


def test_longshot_benchmark_runner_generates_candidate_responses(tmp_path: Path):
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    (video_dir / "sample_video.mp4").write_text("video", encoding="utf-8")

    class FakeVideoRLM:
        def run(self, question, memory, dialogue_context=None, task_type=None):
            return VideoRLMResult(
                answer=f"answer:{question}",
                state=ControllerState(
                    question=question,
                    task_type=task_type,
                    dialogue_context=list(dialogue_context or []),
                ),
                trace=[],
                usage_summary=UsageSummary(
                    model_usage_summaries={
                        "mock": ModelUsageSummary(
                            total_calls=1,
                            total_input_tokens=11,
                            total_output_tokens=7,
                        )
                    }
                ),
                execution_time=0.25,
            )

    runner = longshot.LongShOTBenchmarkRunner(
        video_rlm=FakeVideoRLM(),
        memory_builder=VideoMemoryBuilder(),
        video_resolver=longshot.LongShOTVideoResolver(video_dir),
        memory_cache_dir=tmp_path / "memories",
        trace_dir=tmp_path / "traces",
    )
    sample = {
        "sample_id": "sample_1",
        "video_id": "sample_video",
        "task": "event_understanding",
        "duration": 12.0,
        "conversations": [
            {"role": "user", "content": "What changed?"},
            {"role": "assistant", "content": "gold answer 1"},
            {"role": "user", "content": "Why?"},
            {"role": "assistant", "content": "gold answer 2"},
        ],
    }

    result = runner.run_sample(sample)

    assert result["conversations"][1]["candidate_response"] == "answer:What changed?"
    assert result["conversations"][3]["candidate_response"] == "answer:Why?"
    metadata = result["video_rlm_metadata"]
    assert metadata["history_mode"] == "gold"
    assert Path(metadata["memory_path"]).exists()
    assert Path(metadata["turn_results"][0]["trace_path"]).exists()


def test_longshot_benchmark_runner_appends_jsonl(tmp_path: Path):
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    (video_dir / "sample_video.mp4").write_text("video", encoding="utf-8")

    class FakeVideoRLM:
        def run(self, question, memory, dialogue_context=None, task_type=None):
            return VideoRLMResult(
                answer="mock answer",
                state=ControllerState(question=question),
                trace=[],
                usage_summary=UsageSummary(
                    model_usage_summaries={
                        "mock": ModelUsageSummary(
                            total_calls=1,
                            total_input_tokens=1,
                            total_output_tokens=1,
                        )
                    }
                ),
                execution_time=0.1,
            )

    runner = longshot.LongShOTBenchmarkRunner(
        video_rlm=FakeVideoRLM(),
        memory_builder=VideoMemoryBuilder(),
        video_resolver=longshot.LongShOTVideoResolver(video_dir),
    )
    sample = {
        "sample_id": "sample_1",
        "video_id": "sample_video",
        "duration": 8.0,
        "conversations": [
            {"role": "user", "content": "Question?"},
            {"role": "assistant", "content": "gold answer"},
        ],
    }
    output_path = tmp_path / "results.jsonl"

    first_pass = runner.run_samples([sample], output_path=output_path)
    second_pass = runner.run_samples([sample], output_path=output_path)

    assert len(first_pass) == 1
    assert second_pass == []
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["sample_id"] == "sample_1"
