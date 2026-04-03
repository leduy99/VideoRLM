import base64
import json
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import openai

from rlm.video.media import (
    extract_audio_track,
    extract_frames_for_span,
    get_videorlm_output_root,
    is_audio_path,
)
from rlm.video.types import AudioEvent, OCRSpan, SpeechSpan, TimeSpan, VisualSummarySpan


@runtime_checkable
class SpeechRecognizer(Protocol):
    def recognize(self, video_path: str) -> list[SpeechSpan]:
        ...


@runtime_checkable
class VisualSummarizer(Protocol):
    def summarize(self, video_path: str, spans: list[TimeSpan]) -> list[VisualSummarySpan]:
        ...


@runtime_checkable
class OCRExtractor(Protocol):
    def extract(self, video_path: str) -> list[OCRSpan]:
        ...


@runtime_checkable
class AudioEventExtractor(Protocol):
    def extract(self, video_path: str) -> list[AudioEvent]:
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    def embed_text(self, text: str) -> list[float]:
        ...


@dataclass
class CallableSpeechRecognizer:
    fn: Callable[[str], list[SpeechSpan]]

    def recognize(self, video_path: str) -> list[SpeechSpan]:
        return self.fn(video_path)


@dataclass
class CallableVisualSummarizer:
    fn: Callable[[str, list[TimeSpan]], list[VisualSummarySpan]]

    def summarize(self, video_path: str, spans: list[TimeSpan]) -> list[VisualSummarySpan]:
        return self.fn(video_path, spans)


@dataclass
class CallableOCRExtractor:
    fn: Callable[[str], list[OCRSpan]]

    def extract(self, video_path: str) -> list[OCRSpan]:
        return self.fn(video_path)


@dataclass
class CallableAudioEventExtractor:
    fn: Callable[[str], list[AudioEvent]]

    def extract(self, video_path: str) -> list[AudioEvent]:
        return self.fn(video_path)


@dataclass
class CallableEmbeddingProvider:
    fn: Callable[[str], list[float]]

    def embed_text(self, text: str) -> list[float]:
        return self.fn(text)


@dataclass
class OpenAICompatibleSpeechRecognizer:
    model_name: str
    api_key: str | None = None
    base_url: str | None = None
    prompt: str | None = None
    language: str | None = None
    ffmpeg_bin: str = "ffmpeg"
    timeout: float = 300.0
    client: Any | None = None

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )

    def recognize(self, video_path: str) -> list[SpeechSpan]:
        media_path = Path(video_path)
        temp_root = get_videorlm_output_root() / "tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="videorlm_asr_",
            dir=str(temp_root),
        ) as temp_dir:
            if is_audio_path(media_path):
                audio_path = media_path
            else:
                audio_path = extract_audio_track(
                    media_path=media_path,
                    output_path=Path(temp_dir) / f"{media_path.stem}.wav",
                    ffmpeg_bin=self.ffmpeg_bin,
                )

            with audio_path.open("rb") as audio_file:
                params: dict[str, Any] = {
                    "model": self.model_name,
                    "file": audio_file,
                    "response_format": "verbose_json",
                }
                if self.prompt is not None:
                    params["prompt"] = self.prompt
                if self.language is not None:
                    params["language"] = self.language
                response = self.client.audio.transcriptions.create(**params)

        return self._parse_transcription_response(response)

    def _parse_transcription_response(self, response: Any) -> list[SpeechSpan]:
        payload = _to_dict(response)
        segments = payload.get("segments")
        if segments:
            return [self._segment_to_span(item) for item in segments]

        text = payload.get("text", "").strip()
        if not text:
            return []
        return [SpeechSpan(text=text, time_span=TimeSpan(0.0, 0.0), language=self.language)]

    def _segment_to_span(self, segment: Any) -> SpeechSpan:
        payload = _to_dict(segment)
        start = float(payload.get("start", 0.0))
        end = float(payload.get("end", start))
        if end < start:
            end = start
        return SpeechSpan(
            text=str(payload.get("text", "")).strip(),
            time_span=TimeSpan(start, end),
            language=payload.get("language", self.language),
        )


@dataclass
class OpenAICompatibleVisualSummarizer:
    model_name: str
    api_key: str | None = None
    base_url: str | None = None
    system_prompt: str = (
        "Summarize what is visually present. Return strict JSON with keys "
        "`summary`, `tags`, and `entities`."
    )
    frame_count: int = 3
    ffmpeg_bin: str = "ffmpeg"
    frame_width: int | None = 768
    scene_threshold_seconds: float = 20.0
    timeout: float = 300.0
    client: Any | None = None

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )

    def summarize(self, video_path: str, spans: list[TimeSpan]) -> list[VisualSummarySpan]:
        outputs: list[VisualSummarySpan] = []
        temp_root = get_videorlm_output_root() / "tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="videorlm_vl_",
            dir=str(temp_root),
        ) as temp_dir:
            for index, span in enumerate(spans, start=1):
                frame_dir = Path(temp_dir) / f"span_{index:03d}"
                frame_paths = extract_frames_for_span(
                    media_path=video_path,
                    span=span,
                    frame_count=self.frame_count,
                    ffmpeg_bin=self.ffmpeg_bin,
                    width=self.frame_width,
                    output_dir=frame_dir,
                )
                content = [{"type": "text", "text": self._build_prompt(span)}]
                for frame_path in frame_paths:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": _image_to_data_url(frame_path)},
                        }
                    )

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": content},
                    ],
                )
                response_text = response.choices[0].message.content or ""
                payload = _parse_json_object(response_text)
                outputs.append(
                    VisualSummarySpan(
                        summary=str(payload.get("summary", response_text)).strip(),
                        time_span=span,
                        granularity=self._infer_granularity(span),
                        tags=[str(item) for item in payload.get("tags", [])],
                        entities=[str(item) for item in payload.get("entities", [])],
                    )
                )
        return outputs

    def _build_prompt(self, span: TimeSpan) -> str:
        return (
            "Describe the scene for long-video reasoning. "
            "Mention visible actions, people, objects, slides, or on-screen text. "
            f"Time span: {span.to_display()} seconds."
        )

    def _infer_granularity(self, span: TimeSpan) -> str:
        return "scene" if span.duration >= self.scene_threshold_seconds else "clip"


@dataclass
class OpenAICompatibleEmbeddingProvider:
    model_name: str
    api_key: str | None = None
    base_url: str | None = None
    timeout: float = 300.0
    client: Any | None = None

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )

    def embed_text(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model_name, input=text)
        first_item = response.data[0]
        if isinstance(first_item, dict):
            return [float(item) for item in first_item["embedding"]]
        return [float(item) for item in first_item.embedding]


def _to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return dict(value)


def _parse_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(candidate):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(candidate[index:])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return {"summary": candidate, "tags": [], "entities": []}


def _image_to_data_url(image_path: str | Path) -> str:
    image_bytes = Path(image_path).read_bytes()
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
