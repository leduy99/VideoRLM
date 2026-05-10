from pathlib import Path
from types import SimpleNamespace

import rlm.video.adapters as video_adapters
from rlm.video import (
    OpenAICompatibleEmbeddingProvider,
    OpenAICompatibleSpeechRecognizer,
    OpenAICompatibleVisualSummarizer,
    TimeSpan,
)


class FakeAudioTranscriptions:
    def create(self, **kwargs):
        return {
            "text": "hello world",
            "segments": [
                {"start": 0.0, "end": 1.5, "text": "hello"},
                {"start": 1.5, "end": 3.0, "text": "world"},
            ],
        }


class FakeChatCompletions:
    def create(self, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"summary":"A slide shows a launch plan","tags":["slide"],"entities":["launch plan"]}'
                    )
                )
            ]
        )


class FakeEmbeddings:
    def create(self, **kwargs):
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])])


class FakeClient:
    def __init__(self):
        self.audio = SimpleNamespace(transcriptions=FakeAudioTranscriptions())
        self.chat = SimpleNamespace(completions=FakeChatCompletions())
        self.embeddings = FakeEmbeddings()


class RecordingChatCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"summary":"Selected frames","tags":[],"entities":[]}'
                    )
                )
            ]
        )


class RecordingClient:
    def __init__(self):
        self.chat_completions = RecordingChatCompletions()
        self.chat = SimpleNamespace(completions=self.chat_completions)


def test_openai_compatible_speech_recognizer_reads_segments(tmp_path: Path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake-audio")
    recognizer = OpenAICompatibleSpeechRecognizer(
        model_name="qwen-asr",
        client=FakeClient(),
    )

    spans = recognizer.recognize(str(audio_path))

    assert len(spans) == 2
    assert spans[0].text == "hello"
    assert spans[1].time_span.end == 3.0


def test_openai_compatible_visual_summarizer_reads_frames(monkeypatch, tmp_path: Path):
    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"fake-image")

    monkeypatch.setattr(video_adapters, "extract_frames_for_span", lambda **kwargs: [frame_path])
    summarizer = OpenAICompatibleVisualSummarizer(
        model_name="qwen-vl",
        client=FakeClient(),
        frame_count=1,
    )

    summaries = summarizer.summarize("video.mp4", [TimeSpan(0.0, 8.0)])

    assert len(summaries) == 1
    assert summaries[0].summary == "A slide shows a launch plan"
    assert summaries[0].tags == ["slide"]
    assert summaries[0].granularity == "clip"


def test_openai_compatible_visual_summarizer_uses_pitome(monkeypatch, tmp_path: Path):
    selected = [tmp_path / "selected_1.jpg", tmp_path / "selected_2.jpg"]
    for path in selected:
        path.write_bytes(b"fake-image")

    class FakeSelection:
        frame_paths = selected

    calls = []

    def fake_select_visual_frames_for_span(**kwargs):
        calls.append(kwargs)
        return FakeSelection()

    monkeypatch.setattr(video_adapters, "select_visual_frames_for_span", fake_select_visual_frames_for_span)
    client = RecordingClient()
    summarizer = OpenAICompatibleVisualSummarizer(
        model_name="qwen-vl",
        client=client,
        use_pitome=True,
        frame_count=3,
        pitome_dense_frame_rate=2.0,
        pitome_min_frame_count=5,
        pitome_max_selected_frames=1,
        summary_granularity="clip",
    )

    summaries = summarizer.summarize("video.mp4", [TimeSpan(0.0, 60.0)])

    assert len(summaries) == 1
    assert summaries[0].granularity == "clip"
    assert calls[0]["strategy"] == "pitome"
    assert calls[0]["uniform_frame_count"] == 5
    assert calls[0]["dense_frame_rate"] == 2.0
    content = client.chat_completions.calls[0]["messages"][1]["content"]
    assert sum(1 for item in content if item["type"] == "image_url") == 1


def test_openai_compatible_embedding_provider_reads_embeddings():
    provider = OpenAICompatibleEmbeddingProvider(
        model_name="embedding-model",
        client=FakeClient(),
    )

    embedding = provider.embed_text("launch plan")

    assert embedding == [0.1, 0.2, 0.3]
