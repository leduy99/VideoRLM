from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from rlm.video.adapters import _parse_json_object, _to_dict
from rlm.video.media import (
    extract_audio_segment,
    extract_audio_track,
    extract_frames_for_span,
    get_videorlm_output_root,
    is_audio_path,
    probe_media_duration,
)
from rlm.video.types import SpeechSpan, TimeSpan, VisualSummarySpan


@dataclass
class LocalQwenASRSpeechRecognizer:
    model_name: str
    model_path: str | None = None
    forced_aligner_name: str | None = None
    forced_aligner_path: str | None = None
    device_map: str = "cuda:0"
    torch_dtype: str = "bfloat16"
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    chunk_duration_seconds: float = 60.0
    max_inference_batch_size: int = 8
    max_new_tokens: int = 512
    model: Any | None = None

    def recognize(self, video_path: str) -> list[SpeechSpan]:
        model = self._ensure_loaded()
        media_path = Path(video_path)
        temp_root = get_videorlm_output_root() / "tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        with contextlib.ExitStack() as stack:
            if is_audio_path(media_path):
                audio_path = media_path
            else:
                temp_dir = Path(
                    stack.enter_context(
                        _temporary_directory(prefix="videorlm_local_asr_", dir_path=temp_root)
                    )
                )
                audio_path = extract_audio_track(
                    media_path=media_path,
                    output_path=temp_dir / f"{media_path.stem}.wav",
                    ffmpeg_bin=self.ffmpeg_bin,
                )

            if self.forced_aligner_name or self.forced_aligner_path:
                results = model.transcribe(
                    audio=str(audio_path),
                    language=None,
                    return_time_stamps=True,
                )
                return self._parse_results(results)
            return self._recognize_in_chunks(model=model, audio_path=audio_path, stack=stack)

    def _recognize_in_chunks(self, model, audio_path: Path, stack: contextlib.ExitStack) -> list[SpeechSpan]:
        temp_root = get_videorlm_output_root() / "tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(
            stack.enter_context(_temporary_directory(prefix="videorlm_local_asr_chunks_", dir_path=temp_root))
        )
        duration_seconds = probe_media_duration(audio_path, ffprobe_bin=self.ffprobe_bin)
        spans: list[SpeechSpan] = []

        for index, chunk_span in enumerate(
            _chunk_time_spans(duration_seconds, self.chunk_duration_seconds),
            start=1,
        ):
            chunk_path = extract_audio_segment(
                media_path=audio_path,
                span=chunk_span,
                output_path=temp_dir / f"chunk_{index:03d}.wav",
                ffmpeg_bin=self.ffmpeg_bin,
            )
            chunk_results = model.transcribe(
                audio=str(chunk_path),
                language=None,
                return_time_stamps=False,
            )
            for item in self._parse_results(chunk_results):
                spans.append(_offset_speech_span(item, chunk_span))
        return spans

    def _ensure_loaded(self):
        if self.model is not None:
            return self.model

        import torch
        from qwen_asr import Qwen3ASRModel

        kwargs: dict[str, Any] = {
            "dtype": _resolve_torch_dtype(torch, self.torch_dtype),
            "device_map": self.device_map,
            "max_inference_batch_size": self.max_inference_batch_size,
            "max_new_tokens": self.max_new_tokens,
        }
        aligner = self.forced_aligner_path or self.forced_aligner_name
        if aligner:
            kwargs["forced_aligner"] = aligner
            kwargs["forced_aligner_kwargs"] = {
                "dtype": _resolve_torch_dtype(torch, self.torch_dtype),
                "device_map": self.device_map,
            }
        self.model = Qwen3ASRModel.from_pretrained(self.model_path or self.model_name, **kwargs)
        return self.model

    def _parse_results(self, results: Any) -> list[SpeechSpan]:
        if not results:
            return []
        first = results[0]
        payload = _object_payload(first)

        time_stamps = payload.get("time_stamps")
        language = payload.get("language")
        if time_stamps:
            spans = self._parse_timestamp_items(time_stamps=time_stamps, language=language)
            if spans:
                return spans

        text = str(payload.get("text") or "").strip()
        if not text:
            return []
        return [SpeechSpan(text=text, time_span=TimeSpan(0.0, 0.0), language=language)]

    def _parse_timestamp_items(self, time_stamps: Any, language: str | None) -> list[SpeechSpan]:
        items = _iter_timestamp_items(time_stamps)
        if not items:
            return []

        raw_spans: list[SpeechSpan] = []
        for item in items:
            item_payload = _object_payload(item)
            text = str(
                item_payload.get("text")
                or item_payload.get("content")
                or item_payload.get("sentence")
                or ""
            ).strip()
            start = item_payload.get("start")
            if start is None:
                start = item_payload.get("start_time")
            end = item_payload.get("end")
            if end is None:
                end = item_payload.get("end_time")
            if start is None or end is None:
                span = item_payload.get("time") or item_payload.get("time_span")
                if isinstance(span, (list, tuple)) and len(span) >= 2:
                    start = span[0]
                    end = span[1]
            if start is None or end is None or not text:
                continue
            raw_spans.append(
                SpeechSpan(
                    text=text,
                    time_span=TimeSpan(float(start), float(end)),
                    language=language,
                )
            )

        if not raw_spans:
            return []
        if _looks_like_word_level_alignment(raw_spans):
            return _group_word_level_spans(raw_spans)
        return raw_spans


@dataclass
class LocalQwenVisualSummarizer:
    model_name: str
    model_path: str | None = None
    device: str = "cuda:0"
    device_map: str | dict[str, Any] | None = None
    torch_dtype: str = "bfloat16"
    attn_implementation: str | None = None
    frame_count: int = 3
    ffmpeg_bin: str = "ffmpeg"
    frame_width: int | None = 768
    scene_threshold_seconds: float = 20.0
    max_new_tokens: int = 160
    model: Any | None = None
    processor: Any | None = None

    def summarize(self, video_path: str, spans: list[TimeSpan]) -> list[VisualSummarySpan]:
        model, processor = self._ensure_loaded()
        output_root = get_videorlm_output_root() / "tmp"
        output_root.mkdir(parents=True, exist_ok=True)
        summaries: list[VisualSummarySpan] = []
        with contextlib.ExitStack() as stack:
            temp_dir = Path(
                stack.enter_context(
                    _temporary_directory(prefix="videorlm_local_vl_", dir_path=output_root)
                )
            )
            for index, span in enumerate(spans, start=1):
                frame_dir = temp_dir / f"span_{index:03d}"
                frame_paths = extract_frames_for_span(
                    media_path=video_path,
                    span=span,
                    frame_count=self.frame_count,
                    ffmpeg_bin=self.ffmpeg_bin,
                    width=self.frame_width,
                    output_dir=frame_dir,
                )
                messages = [
                    {
                        "role": "user",
                        "content": [
                            *[
                                {"type": "image", "image": Image.open(frame_path).convert("RGB")}
                                for frame_path in frame_paths
                            ],
                            {"type": "text", "text": self._build_prompt(span)},
                        ],
                    }
                ]
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                if hasattr(inputs, "to"):
                    inputs = inputs.to(self._resolve_input_device(model))
                generated_ids = model.generate(**inputs, max_new_tokens=self.max_new_tokens)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=True)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                payload = _parse_json_object(output_text)
                summaries.append(
                    VisualSummarySpan(
                        summary=str(payload.get("summary", output_text)).strip(),
                        time_span=span,
                        granularity=self._infer_granularity(span),
                        tags=[str(item) for item in payload.get("tags", [])],
                        entities=[str(item) for item in payload.get("entities", [])],
                    )
                )
        return summaries

    def _ensure_loaded(self):
        if self.model is not None and self.processor is not None:
            return self.model, self.processor

        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        model_kwargs: dict[str, Any] = {
            "dtype": _resolve_torch_dtype(torch, self.torch_dtype),
            "device_map": self.device_map or self.device,
        }
        if self.attn_implementation is not None:
            model_kwargs["attn_implementation"] = self.attn_implementation

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path or self.model_name,
            **model_kwargs,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path or self.model_name)
        return self.model, self.processor

    def _resolve_input_device(self, model):
        try:
            return next(model.parameters()).device
        except StopIteration as exc:
            raise ValueError("Vision model has no parameters") from exc

    def _build_prompt(self, span: TimeSpan) -> str:
        return (
            "Summarize this video segment for long-video reasoning. "
            "Return strict JSON with keys `summary`, `tags`, and `entities`. "
            "Mention visible actions, people, objects, slides, or on-screen text. "
            f"Time span: {span.to_display()} seconds."
        )

    def _infer_granularity(self, span: TimeSpan) -> str:
        return "scene" if span.duration >= self.scene_threshold_seconds else "clip"


def _resolve_torch_dtype(torch_module, value: str | Any):
    if not isinstance(value, str):
        return value
    if not hasattr(torch_module, value):
        raise ValueError(f"Unsupported torch dtype: {value}")
    return getattr(torch_module, value)


@contextlib.contextmanager
def _temporary_directory(prefix: str, dir_path: Path):
    import tempfile

    temp_dir = tempfile.TemporaryDirectory(prefix=prefix, dir=str(dir_path))
    try:
        yield temp_dir.name
    finally:
        temp_dir.cleanup()


def _object_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    try:
        return _to_dict(value)
    except TypeError:
        return {}


def _iter_timestamp_items(time_stamps: Any) -> list[Any]:
    if isinstance(time_stamps, (list, tuple)):
        return list(time_stamps)
    if hasattr(time_stamps, "items"):
        items = time_stamps.items
        if isinstance(items, (list, tuple)):
            return list(items)
    payload = _object_payload(time_stamps)
    for key in ("items", "segments", "spans", "timestamps"):
        candidate = payload.get(key)
        if isinstance(candidate, (list, tuple)):
            return list(candidate)
    return []


def _looks_like_word_level_alignment(spans: list[SpeechSpan]) -> bool:
    if len(spans) < 8:
        return False
    short_items = sum(1 for span in spans if len(span.text.split()) <= 3)
    return short_items / len(spans) >= 0.9


def _group_word_level_spans(
    spans: list[SpeechSpan],
    *,
    max_words: int = 18,
    max_duration: float = 12.0,
    max_gap: float = 0.9,
) -> list[SpeechSpan]:
    grouped: list[SpeechSpan] = []
    current: list[SpeechSpan] = []
    current_word_count = 0

    for span in spans:
        if not current:
            current = [span]
            current_word_count = len(span.text.split())
            continue

        last = current[-1]
        gap = span.time_span.start - last.time_span.end
        next_word_count = current_word_count + len(span.text.split())
        next_duration = span.time_span.end - current[0].time_span.start
        sentence_boundary = bool(re.search(r"[.!?]$", last.text.strip()))

        if (
            gap > max_gap
            or next_word_count > max_words
            or next_duration > max_duration
            or sentence_boundary
        ):
            grouped.append(_merge_speech_spans(current))
            current = [span]
            current_word_count = len(span.text.split())
            continue

        current.append(span)
        current_word_count = next_word_count

    if current:
        grouped.append(_merge_speech_spans(current))
    return grouped


def _offset_speech_span(span: SpeechSpan, chunk_span: TimeSpan) -> SpeechSpan:
    if span.time_span.duration == 0:
        time_span = chunk_span
    else:
        time_span = TimeSpan(
            chunk_span.start + span.time_span.start,
            min(chunk_span.end, chunk_span.start + span.time_span.end),
        )
    return SpeechSpan(
        text=span.text,
        time_span=time_span,
        speaker=span.speaker,
        language=span.language,
    )


def _chunk_time_spans(duration_seconds: float, chunk_duration_seconds: float) -> list[TimeSpan]:
    if duration_seconds <= 0:
        return []
    if chunk_duration_seconds <= 0:
        raise ValueError(
            f"chunk_duration_seconds must be positive, got {chunk_duration_seconds}"
        )

    spans: list[TimeSpan] = []
    cursor = 0.0
    while cursor < duration_seconds:
        next_end = min(duration_seconds, cursor + chunk_duration_seconds)
        spans.append(TimeSpan(cursor, next_end))
        cursor = next_end
    return spans


def _merge_speech_spans(spans: list[SpeechSpan]) -> SpeechSpan:
    if not spans:
        raise ValueError("Cannot merge an empty list of speech spans")
    text = _normalize_whitespace(" ".join(span.text.strip() for span in spans if span.text.strip()))
    return SpeechSpan(
        text=text,
        time_span=TimeSpan(spans[0].time_span.start, spans[-1].time_span.end),
        speaker=spans[0].speaker,
        language=spans[0].language,
    )


def _normalize_whitespace(text: str) -> str:
    normalized = " ".join(text.split())
    replacements = {
        " ,": ",",
        " .": ".",
        " !": "!",
        " ?": "?",
        " ;": ";",
        " :": ":",
        " n't": "n't",
        " 'm": "'m",
        " 're": "'re",
        " 've": "'ve",
        " 'll": "'ll",
        " 'd": "'d",
        "( ": "(",
        " )": ")",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return normalized.strip()
