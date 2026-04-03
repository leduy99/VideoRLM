from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

import rlm.video.local_adapters as local_adapters
from rlm.video.local_adapters import LocalQwenASRSpeechRecognizer, LocalQwenVisualSummarizer
from rlm.video.types import TimeSpan


class FakeBatch(dict):
    def __init__(self, input_ids):
        super().__init__(input_ids=input_ids)
        self.input_ids = input_ids

    def to(self, device):
        return self


class FakeProcessor:
    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ):
        return FakeBatch(torch.tensor([[1, 2, 3]]))

    def batch_decode(self, generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return ['{"summary":"A person points at a bracelet","tags":["bracelet"],"entities":["Cartier"]}']


class FakeVisionModel:
    def parameters(self):
        yield torch.nn.Parameter(torch.zeros(1))

    def generate(self, **kwargs):
        return torch.tensor([[1, 2, 3, 4, 5]])


class FakeASRModel:
    def transcribe(self, **kwargs):
        return [
            {
                "language": "English",
                "text": "hello world",
                "time_stamps": [
                    {"start": 0.0, "end": 1.5, "text": "hello"},
                    {"start": 1.5, "end": 3.0, "text": "world"},
                ],
            }
        ]


@dataclass
class FakeForcedAlignItem:
    text: str
    start_time: float
    end_time: float


@dataclass
class FakeForcedAlignResult:
    items: list[FakeForcedAlignItem]


@dataclass
class FakeASRTranscription:
    language: str
    text: str
    time_stamps: FakeForcedAlignResult


def test_local_qwen_asr_recognizer_parses_timestamps(tmp_path: Path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake-audio")
    recognizer = LocalQwenASRSpeechRecognizer(
        model_name="Qwen/Qwen3-ASR-0.6B",
        model=FakeASRModel(),
        forced_aligner_name="Qwen/Qwen3-ForcedAligner-0.6B",
    )

    spans = recognizer.recognize(str(audio_path))

    assert [span.text for span in spans] == ["hello", "world"]
    assert spans[1].time_span.end == 3.0


def test_local_qwen_asr_recognizer_groups_object_style_word_timestamps(tmp_path: Path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake-audio")

    class FakeObjectASRModel:
        def transcribe(self, **kwargs):
            return [
                FakeASRTranscription(
                    language="English",
                    text="hello my lovely world",
                    time_stamps=FakeForcedAlignResult(
                        items=[
                            FakeForcedAlignItem("hello", 0.0, 0.4),
                            FakeForcedAlignItem("my", 0.4, 0.6),
                            FakeForcedAlignItem("lovely", 0.6, 1.2),
                            FakeForcedAlignItem("world", 1.2, 1.8),
                            FakeForcedAlignItem("again", 1.8, 2.2),
                            FakeForcedAlignItem("and", 2.2, 2.4),
                            FakeForcedAlignItem("again", 2.4, 2.9),
                            FakeForcedAlignItem("today", 2.9, 3.5),
                        ]
                    ),
                )
            ]

    recognizer = LocalQwenASRSpeechRecognizer(
        model_name="Qwen/Qwen3-ASR-0.6B",
        model=FakeObjectASRModel(),
        forced_aligner_name="Qwen/Qwen3-ForcedAligner-0.6B",
    )

    spans = recognizer.recognize(str(audio_path))

    assert len(spans) == 1
    assert spans[0].text == "hello my lovely world again and again today"
    assert spans[0].time_span.start == 0.0
    assert spans[0].time_span.end == 3.5
    assert spans[0].language == "English"


def test_local_qwen_asr_recognizer_chunks_without_forced_aligner(
    monkeypatch, tmp_path: Path
):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake-audio")

    def fake_probe_media_duration(media_path, ffprobe_bin="ffprobe"):
        return 130.0

    def fake_extract_audio_segment(media_path, span, output_path, ffmpeg_bin="ffmpeg", sample_rate=16000):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"chunk")
        return path

    class FakeChunkASRModel:
        def transcribe(self, **kwargs):
            audio = Path(kwargs["audio"])
            return [{"language": "English", "text": audio.stem.replace("_", " ")}]

    monkeypatch.setattr(local_adapters, "probe_media_duration", fake_probe_media_duration)
    monkeypatch.setattr(local_adapters, "extract_audio_segment", fake_extract_audio_segment)

    recognizer = LocalQwenASRSpeechRecognizer(
        model_name="Qwen/Qwen3-ASR-0.6B",
        model=FakeChunkASRModel(),
        chunk_duration_seconds=60.0,
    )

    spans = recognizer.recognize(str(audio_path))

    assert [span.text for span in spans] == ["chunk 001", "chunk 002", "chunk 003"]
    assert [span.time_span.to_dict() for span in spans] == [
        {"start": 0.0, "end": 60.0},
        {"start": 60.0, "end": 120.0},
        {"start": 120.0, "end": 130.0},
    ]


def test_local_qwen_visual_summarizer_reads_generated_json(monkeypatch, tmp_path: Path):
    frame_path = tmp_path / "frame.jpg"
    Image.new("RGB", (8, 8), color="white").save(frame_path)

    monkeypatch.setattr(local_adapters, "extract_frames_for_span", lambda **kwargs: [frame_path])
    summarizer = LocalQwenVisualSummarizer(
        model_name="Qwen/Qwen3-VL-8B-Instruct",
        model=FakeVisionModel(),
        processor=FakeProcessor(),
        frame_count=1,
    )

    summaries = summarizer.summarize("video.mp4", [TimeSpan(0.0, 8.0)])

    assert len(summaries) == 1
    assert summaries[0].summary == "A person points at a bracelet"
    assert summaries[0].tags == ["bracelet"]
    assert summaries[0].entities == ["Cartier"]
