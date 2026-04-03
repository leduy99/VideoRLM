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


def test_openai_compatible_embedding_provider_reads_embeddings():
    provider = OpenAICompatibleEmbeddingProvider(
        model_name="embedding-model",
        client=FakeClient(),
    )

    embedding = provider.embed_text("launch plan")

    assert embedding == [0.1, 0.2, 0.3]
