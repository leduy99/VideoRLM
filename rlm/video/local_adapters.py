from __future__ import annotations

import contextlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image

from rlm.video.adapters import ImageTextEmbeddingProvider, _parse_json_object, _to_dict
from rlm.video.media import (
    detect_scene_boundary_timestamps,
    extract_audio_segment,
    extract_audio_track,
    extract_frames_for_span,
    extract_frames_for_timestamps,
    get_videorlm_output_root,
    is_audio_path,
    probe_media_duration,
)
from rlm.video.pitome import (
    FrameSelectionResult,
    compact_frame_embedding,
    fuse_frame_embeddings_with_semantic,
    limit_frame_selection_by_temporal_coverage,
    load_frame_embeddings,
    select_visual_frames_for_span,
)
from rlm.video.types import SpeechSpan, TimeSpan, VideoNodeLevel, VisualSummarySpan


@dataclass
class LocalQwenASRSpeechRecognizer:
    model_name: str
    model_path: str | None = None
    forced_aligner_name: str | None = None
    forced_aligner_path: str | None = None
    device_map: str | dict[str, Any] = "cuda:0"
    torch_dtype: str = "bfloat16"
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    chunk_duration_seconds: float = 60.0
    max_inference_batch_size: int = 8
    max_new_tokens: int = 512
    model: Any | None = None
    verbose: bool = False
    progress_callback: Callable[[dict[str, Any]], None] | None = None

    def recognize(self, video_path: str) -> list[SpeechSpan]:
        self._log(f"recognize start path={video_path}")
        model = self._ensure_loaded()
        media_path = Path(video_path)
        temp_root = get_videorlm_output_root() / "tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        with contextlib.ExitStack() as stack:
            if is_audio_path(media_path):
                audio_path = media_path
                self._log(f"using audio input path={audio_path}")
            else:
                temp_dir = Path(
                    stack.enter_context(
                        _temporary_directory(prefix="videorlm_local_asr_", dir_path=temp_root)
                    )
                )
                self._log("extracting audio track")
                audio_path = extract_audio_track(
                    media_path=media_path,
                    output_path=temp_dir / f"{media_path.stem}.wav",
                    ffmpeg_bin=self.ffmpeg_bin,
                )
                self._log(f"audio track ready path={audio_path}")

            if self.forced_aligner_name or self.forced_aligner_path:
                self._log("forced aligner transcription start")
                self._notify_progress(
                    phase="asr",
                    event="planned",
                    total=1,
                    status="asr forced-aligner 0/1",
                )
                results = model.transcribe(
                    audio=str(audio_path),
                    language=None,
                    return_time_stamps=True,
                )
                spans = self._parse_results(results)
                self._notify_progress(
                    phase="asr",
                    event="advance",
                    advance=1,
                    index=1,
                    total=1,
                    status=f"asr forced-aligner done spans={len(spans)}",
                )
                self._log(f"forced aligner transcription done spans={len(spans)}")
                return spans
            spans = self._recognize_in_chunks(model=model, audio_path=audio_path, stack=stack)
            self._log(f"recognize done spans={len(spans)}")
            return spans

    def _recognize_in_chunks(
        self, model, audio_path: Path, stack: contextlib.ExitStack
    ) -> list[SpeechSpan]:
        temp_root = get_videorlm_output_root() / "tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(
            stack.enter_context(
                _temporary_directory(prefix="videorlm_local_asr_chunks_", dir_path=temp_root)
            )
        )
        duration_seconds = probe_media_duration(audio_path, ffprobe_bin=self.ffprobe_bin)
        chunks = _chunk_time_spans(duration_seconds, self.chunk_duration_seconds)
        self._log(
            f"chunked ASR duration={duration_seconds:.2f}s chunks={len(chunks)} "
            f"chunk_seconds={self.chunk_duration_seconds:.2f}"
        )
        self._notify_progress(
            phase="asr",
            event="planned",
            total=len(chunks),
            status=f"asr 0/{len(chunks)}",
        )
        spans: list[SpeechSpan] = []

        for index, chunk_span in enumerate(chunks, start=1):
            self._log(f"ASR chunk {index}/{len(chunks)} span={chunk_span.to_display()}")
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
            parsed = self._parse_results(chunk_results)
            self._notify_progress(
                phase="asr",
                event="advance",
                advance=1,
                index=index,
                total=len(chunks),
                status=f"asr {index}/{len(chunks)} parsed_spans={len(parsed)}",
            )
            self._log(f"ASR chunk {index}/{len(chunks)} parsed_spans={len(parsed)}")
            for item in parsed:
                spans.append(_offset_speech_span(item, chunk_span))
        return spans

    def _ensure_loaded(self):
        if self.model is not None:
            return self.model

        import torch
        from qwen_asr import Qwen3ASRModel

        self._log(
            f"loading ASR model={self.model_path or self.model_name} "
            f"device_map={self.device_map} dtype={self.torch_dtype}"
        )
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
        self._log("ASR model loaded")
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

    def _notify_progress(self, **payload: Any) -> None:
        if self.progress_callback is not None:
            self.progress_callback(payload)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[LocalQwenASR] {message}", flush=True)


@dataclass
class LazySpeechRecognizer:
    chunk_duration_seconds: float = 120.0
    ffprobe_bin: str = "ffprobe"
    verbose: bool = False
    progress_callback: Callable[[dict[str, Any]], None] | None = None
    progress_unit_weight: int = 1

    def recognize(self, video_path: str) -> list[SpeechSpan]:
        self._log(f"lazy ASR index start path={video_path}")
        duration_seconds = probe_media_duration(video_path, ffprobe_bin=self.ffprobe_bin)
        chunks = _chunk_time_spans(duration_seconds, self.chunk_duration_seconds)
        self._notify_progress(
            phase="asr",
            event="planned",
            total=len(chunks),
            status=f"lazy-asr-index 0/{len(chunks)}",
        )
        spans = [
            SpeechSpan(
                text=(
                    f"Lazy ASR index for {chunk.to_display()}. "
                    "Open this node as speech to run ASR refinement."
                ),
                time_span=chunk,
                language="lazy_asr",
            )
            for chunk in chunks
        ]
        if chunks:
            self._notify_progress(
                phase="asr",
                event="advance",
                advance=len(chunks),
                index=len(chunks),
                total=len(chunks),
                status=f"lazy-asr-index {len(chunks)}/{len(chunks)}",
            )
        self._log(f"lazy ASR index done spans={len(spans)}")
        return spans

    def _notify_progress(self, **payload: Any) -> None:
        if self.progress_callback is not None:
            self.progress_callback(payload)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[LazyASR] {message}", flush=True)


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
    use_pitome: bool = False
    pitome_dense_frame_rate: float = 1.0
    pitome_min_frame_count: int | None = None
    pitome_protect_ratio: float = 0.15
    pitome_similarity_threshold: float = 0.8
    pitome_embedding_size: int = 16
    pitome_embedding_backend: str = "pixel"
    pitome_embedding_device: str | None = None
    pitome_frame_width: int | None = None
    pitome_frame_extraction_strategy: Literal["auto", "batch", "seek", "sequence"] = "auto"
    pitome_frame_extraction_workers: int = 1
    pitome_anchor_frame_count: int = 0
    pitome_max_selected_frames: int | None = None
    pitome_scene_threshold: float = 0.35
    pitome_max_scene_boundary_frames: int = 6
    pitome_scene_sample_rate: float | None = 1.0
    pitome_scene_keyframes_only: bool = True
    frame_embedding_provider: ImageTextEmbeddingProvider | None = None
    summary_granularity: VideoNodeLevel | None = None
    vl_retry_frame_count: int = 4
    verbose: bool = False
    progress_callback: Callable[[dict[str, Any]], None] | None = None

    def summarize(self, video_path: str, spans: list[TimeSpan]) -> list[VisualSummarySpan]:
        self._log(f"summarize start path={video_path} spans={len(spans)}")
        self._notify_progress(
            phase="visual",
            event="planned",
            total=len(spans),
            status=f"visual 0/{len(spans)}",
        )
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
                self._log(f"visual span {index}/{len(spans)} span={span.to_display()}")
                frame_dir = temp_dir / f"span_{index:03d}"
                frame_paths, frame_metadata = self._select_frames(video_path, span, frame_dir)
                self._log(f"visual span {index}/{len(spans)} selected_frames={len(frame_paths)}")
                output_text, frame_metadata = self._generate_with_frame_retry(
                    model=model,
                    processor=processor,
                    frame_paths=frame_paths,
                    span=span,
                    metadata=frame_metadata,
                )
                payload = _parse_json_object(output_text)
                summary_text = str(payload.get("summary", output_text)).strip()
                self._log(
                    f"visual span {index}/{len(spans)} summary={_truncate_for_log(summary_text)}"
                )
                summaries.append(
                    VisualSummarySpan(
                        summary=summary_text,
                        time_span=span,
                        granularity=self._infer_granularity(span),
                        tags=[str(item) for item in payload.get("tags", [])],
                        entities=[str(item) for item in payload.get("entities", [])],
                        metadata=frame_metadata,
                    )
                )
                self._notify_progress(
                    phase="visual",
                    event="advance",
                    advance=1,
                    index=index,
                    total=len(spans),
                    status=f"visual {index}/{len(spans)}",
                )
        self._log(f"summarize done summaries={len(summaries)}")
        return summaries

    def _generate_with_frame_retry(
        self,
        *,
        model,
        processor,
        frame_paths: list[Path],
        span: TimeSpan,
        metadata: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        attempts = [frame_paths]
        retry_paths = self._temporal_frame_subset(frame_paths, self.vl_retry_frame_count)
        if len(retry_paths) < len(frame_paths):
            attempts.append(retry_paths)
        single_frame_paths = self._temporal_frame_subset(frame_paths, 1)
        if len(single_frame_paths) < len(attempts[-1]):
            attempts.append(single_frame_paths)

        last_error: RuntimeError | None = None
        for attempt_index, attempt_paths in enumerate(attempts):
            attempt_metadata = dict(metadata)
            attempt_metadata["vl_input_frame_count"] = len(attempt_paths)
            if attempt_index > 0:
                attempt_metadata["vl_retry_reason"] = "frame_batch_reduced"
                attempt_metadata["vl_original_frame_count"] = len(frame_paths)
                self._log(
                    "VL retry with fewer frames "
                    f"original={len(frame_paths)} retry={len(attempt_paths)} "
                    f"span={span.to_display()}"
                )
            try:
                return (
                    self._generate_summary_text(
                        model=model,
                        processor=processor,
                        frame_paths=attempt_paths,
                        span=span,
                    ),
                    attempt_metadata,
                )
            except RuntimeError as exc:
                last_error = exc
                if attempt_index == len(attempts) - 1 or not _is_retryable_vl_frame_error(exc):
                    raise

        if last_error is not None:
            raise last_error
        raise ValueError("VL frame retry received no frame attempts")

    def _generate_summary_text(
        self,
        *,
        model,
        processor,
        frame_paths: list[Path],
        span: TimeSpan,
    ) -> str:
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
        return processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def _temporal_frame_subset(self, frame_paths: list[Path], max_count: int) -> list[Path]:
        if max_count <= 0:
            raise ValueError(f"max_count must be positive, got {max_count}")
        if len(frame_paths) <= max_count:
            return frame_paths
        if max_count == 1:
            return [frame_paths[len(frame_paths) // 2]]
        step = (len(frame_paths) - 1) / (max_count - 1)
        indexes = [round(index * step) for index in range(max_count)]
        return [frame_paths[index] for index in indexes]

    def _select_frame_paths(self, video_path: str, span: TimeSpan, output_dir: Path) -> list[Path]:
        frame_paths, _ = self._select_frames(video_path, span, output_dir)
        return frame_paths

    def _select_frames(
        self,
        video_path: str,
        span: TimeSpan,
        output_dir: Path,
    ) -> tuple[list[Path], dict[str, Any]]:
        if not self.use_pitome:
            frame_paths = extract_frames_for_span(
                media_path=video_path,
                span=span,
                frame_count=self.frame_count,
                ffmpeg_bin=self.ffmpeg_bin,
                width=self.frame_width,
                output_dir=output_dir,
            )
            return frame_paths, {}

        selection = select_visual_frames_for_span(
            media_path=video_path,
            span=span,
            strategy="pitome",
            uniform_frame_count=self.pitome_min_frame_count or self.frame_count,
            dense_frame_rate=self.pitome_dense_frame_rate,
            ffmpeg_bin=self.ffmpeg_bin,
            width=self.pitome_frame_width if self.pitome_frame_width is not None else self.frame_width,
            output_dir=output_dir,
            protect_ratio=self.pitome_protect_ratio,
            similarity_threshold=self.pitome_similarity_threshold,
            embedding_size=self.pitome_embedding_size,
            embedding_backend=self.pitome_embedding_backend,
            embedding_device=self.pitome_embedding_device,
            anchor_frame_count=self.pitome_anchor_frame_count,
            frame_extraction_strategy=self.pitome_frame_extraction_strategy,
            frame_extraction_seek_workers=self.pitome_frame_extraction_workers,
        )
        selection, boundary_metadata = self._add_boundary_frames(
            video_path=video_path,
            span=span,
            output_dir=output_dir,
            selection=selection,
        )
        if self.pitome_max_selected_frames is not None:
            selection = limit_frame_selection_by_temporal_coverage(
                selection,
                self.pitome_max_selected_frames,
            )
        semantic_metadata = self._semantic_frame_metadata(selection.frame_paths)
        selection = _fuse_selection_with_semantic_embeddings(selection, semantic_metadata)
        metadata = selection.to_metadata()
        metadata.update(_selected_boundary_metadata(selection.timestamps, boundary_metadata))
        metadata.update(semantic_metadata)
        return selection.frame_paths, metadata

    def _add_boundary_frames(
        self,
        *,
        video_path: str,
        span: TimeSpan,
        output_dir: Path,
        selection: FrameSelectionResult,
    ) -> tuple[FrameSelectionResult, dict[str, list[float]]]:
        return _add_boundary_frames_to_selection(
            video_path=video_path,
            span=span,
            output_dir=output_dir,
            selection=selection,
            ffmpeg_bin=self.ffmpeg_bin,
            frame_width=self.pitome_frame_width if self.pitome_frame_width is not None else self.frame_width,
            embedding_size=self.pitome_embedding_size,
            embedding_backend=self.pitome_embedding_backend,
            embedding_device=self.pitome_embedding_device,
            frame_extraction_workers=self.pitome_frame_extraction_workers,
            scene_threshold=self.pitome_scene_threshold,
            max_scene_boundary_frames=self.pitome_max_scene_boundary_frames,
            scene_sample_rate=self.pitome_scene_sample_rate,
            scene_keyframes_only=self.pitome_scene_keyframes_only,
        )

    def _semantic_frame_metadata(self, frame_paths: list[Path]) -> dict[str, Any]:
        if self.frame_embedding_provider is None or not frame_paths:
            return {}
        embeddings = self.frame_embedding_provider.embed_images(frame_paths)
        if not embeddings:
            return {}
        return {
            "semantic_frame_embeddings": embeddings,
            "semantic_frame_embedding_model": getattr(
                self.frame_embedding_provider,
                "model_name",
                None,
            ),
            "semantic_frame_embedding_dim": len(embeddings[0]),
        }

    def _ensure_loaded(self):
        if self.model is not None and self.processor is not None:
            return self.model, self.processor

        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self._log(
            f"loading VL model={self.model_path or self.model_name} "
            f"device_map={self.device_map or self.device} dtype={self.torch_dtype}"
        )
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
        self._log("VL model loaded")
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
        if self.summary_granularity is not None:
            return self.summary_granularity
        return "scene" if span.duration >= self.scene_threshold_seconds else "clip"

    def _notify_progress(self, **payload: Any) -> None:
        if self.progress_callback is not None:
            self.progress_callback(payload)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[LocalQwenVL] {message}", flush=True)


@dataclass
class LazyPiToMeVisualIndexer:
    ffmpeg_bin: str = "ffmpeg"
    frame_width: int | None = 768
    frame_count: int = 3
    pitome_dense_frame_rate: float = 1.0
    pitome_min_frame_count: int | None = None
    pitome_protect_ratio: float = 0.15
    pitome_similarity_threshold: float = 0.8
    pitome_embedding_size: int = 16
    pitome_embedding_backend: str = "pixel"
    pitome_embedding_device: str | None = None
    pitome_frame_width: int | None = None
    pitome_frame_extraction_strategy: Literal["auto", "batch", "seek", "sequence"] = "auto"
    pitome_frame_extraction_workers: int = 1
    pitome_anchor_frame_count: int = 0
    pitome_max_selected_frames: int | None = None
    pitome_scene_threshold: float = 0.35
    pitome_max_scene_boundary_frames: int = 6
    pitome_scene_sample_rate: float | None = 1.0
    pitome_scene_keyframes_only: bool = True
    frame_embedding_provider: ImageTextEmbeddingProvider | None = None
    summary_granularity: VideoNodeLevel | None = "clip"
    verbose: bool = False
    progress_callback: Callable[[dict[str, Any]], None] | None = None
    progress_unit_weight: int = 1

    def summarize(self, video_path: str, spans: list[TimeSpan]) -> list[VisualSummarySpan]:
        self._log(f"lazy visual index start path={video_path} spans={len(spans)}")
        self._notify_progress(
            phase="visual",
            event="planned",
            total=len(spans),
            status=f"visual-index 0/{len(spans)}",
        )
        output_root = get_videorlm_output_root() / "tmp"
        output_root.mkdir(parents=True, exist_ok=True)
        summaries: list[VisualSummarySpan] = []
        with contextlib.ExitStack() as stack:
            temp_dir = Path(
                stack.enter_context(
                    _temporary_directory(prefix="videorlm_lazy_pitome_", dir_path=output_root)
                )
            )
            for index, span in enumerate(spans, start=1):
                frame_dir = temp_dir / f"span_{index:03d}"
                selection = select_visual_frames_for_span(
                    media_path=video_path,
                    span=span,
                    strategy="pitome",
                    uniform_frame_count=self.pitome_min_frame_count or self.frame_count,
                    dense_frame_rate=self.pitome_dense_frame_rate,
                    ffmpeg_bin=self.ffmpeg_bin,
                    width=self.pitome_frame_width if self.pitome_frame_width is not None else self.frame_width,
                    output_dir=frame_dir,
                    protect_ratio=self.pitome_protect_ratio,
                    similarity_threshold=self.pitome_similarity_threshold,
                    embedding_size=self.pitome_embedding_size,
                    embedding_backend=self.pitome_embedding_backend,
                    embedding_device=self.pitome_embedding_device,
                    anchor_frame_count=self.pitome_anchor_frame_count,
                    frame_extraction_strategy=self.pitome_frame_extraction_strategy,
                    frame_extraction_seek_workers=self.pitome_frame_extraction_workers,
                )
                selection, boundary_metadata = self._add_boundary_frames(
                    video_path=video_path,
                    span=span,
                    output_dir=frame_dir,
                    selection=selection,
                )
                if self.pitome_max_selected_frames is not None:
                    selection = limit_frame_selection_by_temporal_coverage(
                        selection,
                        self.pitome_max_selected_frames,
                    )
                semantic_metadata = _semantic_frame_metadata(
                    self.frame_embedding_provider,
                    selection.frame_paths,
                )
                selection = _fuse_selection_with_semantic_embeddings(
                    selection,
                    semantic_metadata,
                )
                metadata = selection.to_metadata()
                metadata.update(
                    _selected_boundary_metadata(selection.timestamps, boundary_metadata)
                )
                metadata.update(semantic_metadata)
                metadata.update(
                    {
                        "visual_summary_mode": "lazy_pitome_index",
                        "on_demand_visual_refinement": True,
                        "lazy_visual_index": True,
                    }
                )
                summary = (
                    f"PiToMe visual index for {span.to_display()} with "
                    f"{len(selection.frame_paths)} representative frames. "
                    "Open this node visually to run QwenVL refinement."
                )
                summaries.append(
                    VisualSummarySpan(
                        summary=summary,
                        time_span=span,
                        granularity=self._infer_granularity(span),
                        tags=["pitome", "visual_index"],
                        entities=[],
                        metadata=metadata,
                    )
                )
                self._notify_progress(
                    phase="visual",
                    event="advance",
                    advance=1,
                    index=index,
                    total=len(spans),
                    status=f"visual-index {index}/{len(spans)}",
                )
        self._log(f"lazy visual index done summaries={len(summaries)}")
        return summaries

    def _add_boundary_frames(
        self,
        *,
        video_path: str,
        span: TimeSpan,
        output_dir: Path,
        selection: FrameSelectionResult,
    ) -> tuple[FrameSelectionResult, dict[str, list[float]]]:
        return _add_boundary_frames_to_selection(
            video_path=video_path,
            span=span,
            output_dir=output_dir,
            selection=selection,
            ffmpeg_bin=self.ffmpeg_bin,
            frame_width=self.pitome_frame_width if self.pitome_frame_width is not None else self.frame_width,
            embedding_size=self.pitome_embedding_size,
            embedding_backend=self.pitome_embedding_backend,
            embedding_device=self.pitome_embedding_device,
            frame_extraction_workers=self.pitome_frame_extraction_workers,
            scene_threshold=self.pitome_scene_threshold,
            max_scene_boundary_frames=self.pitome_max_scene_boundary_frames,
            scene_sample_rate=self.pitome_scene_sample_rate,
            scene_keyframes_only=self.pitome_scene_keyframes_only,
        )

    def _infer_granularity(self, span: TimeSpan) -> str:
        if self.summary_granularity is not None:
            return self.summary_granularity
        return "clip"

    def _notify_progress(self, **payload: Any) -> None:
        if self.progress_callback is not None:
            self.progress_callback(payload)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[LazyPiToMe] {message}", flush=True)


@dataclass
class FasterWhisperSpeechRecognizer:
    model_name: str = "small"
    device: str = "cpu"
    compute_type: str = "default"
    ffmpeg_bin: str = "ffmpeg"
    language: str | None = None
    vad_filter: bool = True
    model: Any | None = None
    verbose: bool = False
    progress_callback: Callable[[dict[str, Any]], None] | None = None

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
                        _temporary_directory(prefix="videorlm_faster_whisper_", dir_path=temp_root)
                    )
                )
                audio_path = extract_audio_track(
                    media_path=media_path,
                    output_path=temp_dir / f"{media_path.stem}.wav",
                    ffmpeg_bin=self.ffmpeg_bin,
                )
            self._notify_progress(
                phase="asr",
                event="planned",
                total=1,
                status="asr faster-whisper 0/1",
            )
            segments, _info = model.transcribe(
                str(audio_path),
                language=self.language,
                vad_filter=self.vad_filter,
            )
            spans = [
                SpeechSpan(
                    text=str(segment.text).strip(),
                    time_span=TimeSpan(float(segment.start), float(segment.end)),
                    language=self.language,
                )
                for segment in segments
                if str(segment.text).strip()
            ]
            self._notify_progress(
                phase="asr",
                event="advance",
                advance=1,
                index=1,
                total=1,
                status=f"asr faster-whisper done spans={len(spans)}",
            )
            return spans

    def _ensure_loaded(self):
        if self.model is not None:
            return self.model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ImportError(
                "faster-whisper speech backend requires `faster-whisper` to be installed."
            ) from exc
        kwargs = {"device": self.device}
        if self.compute_type != "default":
            kwargs["compute_type"] = self.compute_type
        self._log(
            f"loading faster-whisper model={self.model_name} "
            f"device={self.device} compute_type={self.compute_type}"
        )
        self.model = WhisperModel(self.model_name, **kwargs)
        return self.model

    def _notify_progress(self, **payload: Any) -> None:
        if self.progress_callback is not None:
            self.progress_callback(payload)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[FasterWhisper] {message}", flush=True)


def _truncate_for_log(text: str, max_length: int = 180) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 3] + "..."


def _limit_paths_by_temporal_coverage(paths: list[Path], max_count: int) -> list[Path]:
    if max_count <= 0:
        raise ValueError(f"max_count must be positive, got {max_count}")
    if len(paths) <= max_count:
        return paths
    if max_count == 1:
        return [paths[len(paths) // 2]]
    indices = sorted(
        {round(position * (len(paths) - 1) / (max_count - 1)) for position in range(max_count)}
    )
    return [paths[index] for index in indices]


def _semantic_frame_metadata(
    provider: ImageTextEmbeddingProvider | None,
    frame_paths: list[Path],
) -> dict[str, Any]:
    if provider is None or not frame_paths:
        return {}
    embeddings = provider.embed_images(frame_paths)
    if not embeddings:
        return {}
    return {
        "semantic_frame_embeddings": embeddings,
        "semantic_frame_embedding_model": getattr(provider, "model_name", None),
        "semantic_frame_embedding_dim": len(embeddings[0]),
    }


def _fuse_selection_with_semantic_embeddings(
    selection: FrameSelectionResult,
    semantic_metadata: dict[str, Any],
) -> FrameSelectionResult:
    raw_embeddings = semantic_metadata.get("semantic_frame_embeddings", [])
    if not selection.frame_embeddings or not isinstance(raw_embeddings, list):
        return selection
    semantic_embeddings = [
        [float(value) for value in item]
        for item in raw_embeddings
        if isinstance(item, (list, tuple))
    ]
    if len(semantic_embeddings) != len(selection.frame_embeddings):
        return selection
    fused_embeddings = fuse_frame_embeddings_with_semantic(
        selection.frame_embeddings,
        semantic_embeddings,
    )
    backend = selection.embedding_backend or "pitome"
    return FrameSelectionResult(
        strategy=selection.strategy,
        frame_paths=list(selection.frame_paths),
        timestamps=list(selection.timestamps),
        dense_frame_count=selection.dense_frame_count,
        embedding_backend=f"{backend}+semantic",
        embedding_size=selection.embedding_size,
        frame_embeddings=fused_embeddings,
        protected_timestamps=list(selection.protected_timestamps),
        representative_timestamps=list(selection.representative_timestamps),
        energy_scores=list(selection.energy_scores),
        merged_pairs=list(selection.merged_pairs),
    )


def _add_boundary_frames_to_selection(
    *,
    video_path: str,
    span: TimeSpan,
    output_dir: Path,
    selection: FrameSelectionResult,
    ffmpeg_bin: str,
    frame_width: int | None,
    embedding_size: int,
    embedding_backend: str,
    embedding_device: str | None,
    frame_extraction_workers: int,
    scene_threshold: float,
    max_scene_boundary_frames: int,
    scene_sample_rate: float | None,
    scene_keyframes_only: bool,
) -> tuple[FrameSelectionResult, dict[str, list[float]]]:
    boundary_metadata = _boundary_timestamps_for_span(
        video_path=video_path,
        span=span,
        ffmpeg_bin=ffmpeg_bin,
        threshold=scene_threshold,
        max_scene_boundary_frames=max_scene_boundary_frames,
        scene_sample_rate=scene_sample_rate,
        scene_keyframes_only=scene_keyframes_only,
    )
    boundary_timestamps = boundary_metadata["all"]
    if not boundary_timestamps:
        return selection, boundary_metadata

    boundary_paths = extract_frames_for_timestamps(
        media_path=video_path,
        timestamps=boundary_timestamps,
        ffmpeg_bin=ffmpeg_bin,
        width=frame_width,
        output_dir=output_dir / "boundaries",
        prefix="boundary",
        extraction_strategy="seek",
        seek_workers=frame_extraction_workers,
    )
    boundary_embeddings = [
        compact_frame_embedding(embedding)
        for embedding in load_frame_embeddings(
            boundary_paths,
            embedding_size=embedding_size,
            backend=embedding_backend,
            device=embedding_device,
        )
    ]
    return (
        _merge_frame_selection_with_boundaries(
            selection=selection,
            boundary_paths=boundary_paths,
            boundary_timestamps=boundary_timestamps,
            boundary_embeddings=boundary_embeddings,
        ),
        boundary_metadata,
    )


def _boundary_timestamps_for_span(
    *,
    video_path: str,
    span: TimeSpan,
    ffmpeg_bin: str,
    threshold: float,
    max_scene_boundary_frames: int,
    scene_sample_rate: float | None,
    scene_keyframes_only: bool,
) -> dict[str, list[float]]:
    edge_timestamps = _span_boundary_timestamps(span)
    scene_timestamps = detect_scene_boundary_timestamps(
        video_path,
        span=span,
        ffmpeg_bin=ffmpeg_bin,
        threshold=threshold,
        max_timestamps=max_scene_boundary_frames,
        sample_rate=scene_sample_rate,
        keyframes_only=scene_keyframes_only,
    )
    return {
        "all": _merge_timestamps([*edge_timestamps, *scene_timestamps]),
        "edges": edge_timestamps,
        "scene": scene_timestamps,
    }


def _selected_boundary_metadata(
    selected_timestamps: list[float],
    boundary_metadata: dict[str, list[float]],
) -> dict[str, Any]:
    all_boundary_timestamps = _selected_matching_timestamps(
        selected_timestamps,
        boundary_metadata.get("all", []),
    )
    edge_timestamps = _selected_matching_timestamps(
        selected_timestamps,
        boundary_metadata.get("edges", []),
    )
    scene_timestamps = _selected_matching_timestamps(
        selected_timestamps,
        boundary_metadata.get("scene", []),
    )
    return {
        "boundary_frame_timestamps": all_boundary_timestamps,
        "boundary_frame_count": len(all_boundary_timestamps),
        "edge_boundary_frame_timestamps": edge_timestamps,
        "edge_boundary_frame_count": len(edge_timestamps),
        "scene_boundary_frame_timestamps": scene_timestamps,
        "scene_boundary_frame_count": len(scene_timestamps),
    }


def _selected_matching_timestamps(
    selected_timestamps: list[float],
    candidates: list[float],
    *,
    tolerance: float = 0.05,
) -> list[float]:
    return [
        timestamp
        for timestamp in candidates
        if any(abs(timestamp - selected) <= tolerance for selected in selected_timestamps)
    ]


def _merge_timestamps(timestamps: list[float], *, tolerance: float = 0.05) -> list[float]:
    merged: list[float] = []
    for timestamp in sorted(timestamps):
        if merged and abs(timestamp - merged[-1]) <= tolerance:
            continue
        merged.append(timestamp)
    return merged


def _span_boundary_timestamps(span: TimeSpan) -> list[float]:
    if span.duration <= 0:
        return [span.start]
    margin = min(0.5, max(0.05, span.duration * 0.02))
    start = span.start + margin
    end = span.end - margin
    if end <= start:
        return [span.start + (span.duration / 2.0)]
    return [start, end]


def _merge_frame_selection_with_boundaries(
    *,
    selection: FrameSelectionResult,
    boundary_paths: list[Path],
    boundary_timestamps: list[float],
    boundary_embeddings: list[list[float]],
) -> FrameSelectionResult:
    entries: list[tuple[float, Path, list[float] | None, str]] = []
    for index, (timestamp, path) in enumerate(
        zip(selection.timestamps, selection.frame_paths, strict=True)
    ):
        embedding = (
            selection.frame_embeddings[index] if index < len(selection.frame_embeddings) else None
        )
        entries.append((float(timestamp), path, embedding, "pitome"))
    for index, (timestamp, path) in enumerate(
        zip(boundary_timestamps, boundary_paths, strict=True)
    ):
        embedding = boundary_embeddings[index] if index < len(boundary_embeddings) else None
        entries.append((float(timestamp), path, embedding, "boundary"))

    entries.sort(key=lambda item: (item[0], item[1].name))
    deduped: list[tuple[float, Path, list[float] | None, str]] = []
    for entry in entries:
        timestamp = entry[0]
        if deduped and abs(timestamp - deduped[-1][0]) < 0.05:
            if entry[3] == "boundary" and deduped[-1][3] != "boundary":
                deduped[-1] = entry
            continue
        deduped.append(entry)

    merged_embeddings = []
    if all(embedding is not None for _, _, embedding, _ in deduped):
        merged_embeddings = [list(embedding) for _, _, embedding, _ in deduped]
    return FrameSelectionResult(
        strategy=selection.strategy,
        frame_paths=[path for _, path, _, _ in deduped],
        timestamps=[timestamp for timestamp, _, _, _ in deduped],
        dense_frame_count=selection.dense_frame_count,
        embedding_backend=selection.embedding_backend,
        embedding_size=selection.embedding_size,
        frame_embeddings=merged_embeddings,
        protected_timestamps=list(selection.protected_timestamps),
        representative_timestamps=list(selection.representative_timestamps),
        energy_scores=list(selection.energy_scores),
        merged_pairs=list(selection.merged_pairs),
    )


def _is_retryable_vl_frame_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "image features and image tokens do not match" in message or "out of memory" in message


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
        return {key: item for key, item in vars(value).items() if not key.startswith("_")}
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
        raise ValueError(f"chunk_duration_seconds must be positive, got {chunk_duration_seconds}")

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
