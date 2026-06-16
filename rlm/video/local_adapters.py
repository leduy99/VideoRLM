from __future__ import annotations

import contextlib
import difflib
import json
import os
import re
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    compute_memorability_prior,
    filter_cognitive_anchor_metadata,
    fuse_frame_embeddings_with_semantic,
    limit_frame_selection_by_temporal_coverage,
    load_frame_embeddings,
    select_visual_frames_for_span,
)
from rlm.video.types import (
    OCRSpan,
    SpeechSpan,
    TimeSpan,
    VideoNodeLevel,
    VisualSummarySpan,
)


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
    chunk_batch_size: int = 1
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
        if self.chunk_duration_seconds <= 0:
            chunks = [TimeSpan(0.0, duration_seconds)] if duration_seconds > 0 else []
        else:
            chunks = _chunk_time_spans(duration_seconds, self.chunk_duration_seconds)
        if self.chunk_batch_size <= 0:
            raise ValueError(
                f"chunk_batch_size must be positive, got {self.chunk_batch_size}"
            )
        self._log(
            f"chunked ASR duration={duration_seconds:.2f}s chunks={len(chunks)} "
            f"chunk_seconds={self.chunk_duration_seconds:.2f} "
            f"chunk_batch_size={self.chunk_batch_size}"
        )
        self._notify_progress(
            phase="asr",
            event="planned",
            total=len(chunks),
            status=f"asr 0/{len(chunks)}",
        )
        spans: list[SpeechSpan] = []

        for batch_start in range(0, len(chunks), self.chunk_batch_size):
            batch_chunks = chunks[batch_start : batch_start + self.chunk_batch_size]
            batch_paths: list[Path] = []
            for offset, chunk_span in enumerate(batch_chunks):
                index = batch_start + offset + 1
                self._log(f"ASR chunk {index}/{len(chunks)} span={chunk_span.to_display()}")
                batch_paths.append(
                    extract_audio_segment(
                        media_path=audio_path,
                        span=chunk_span,
                        output_path=temp_dir / f"chunk_{index:03d}.wav",
                        ffmpeg_bin=self.ffmpeg_bin,
                    )
                )
            chunk_results_batch = self._transcribe_chunk_batch(model, batch_paths)
            for offset, chunk_results in enumerate(chunk_results_batch):
                index = batch_start + offset + 1
                chunk_span = batch_chunks[offset]
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

    def _transcribe_chunk_batch(self, model, batch_paths: list[Path]) -> list[Any]:
        if len(batch_paths) == 1:
            return [
                model.transcribe(
                    audio=str(batch_paths[0]),
                    language=None,
                    return_time_stamps=False,
                )
            ]
        raw_results = model.transcribe(
            audio=[str(path) for path in batch_paths],
            language=None,
            return_time_stamps=False,
        )
        return _split_asr_batch_results(raw_results, len(batch_paths))

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

    def unload(self) -> None:
        self.model = None
        from rlm.video.gpu_memory import clear_torch_cache

        clear_torch_cache()

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
    pitome_edge_boundary_frames: bool = True
    frame_embedding_provider: ImageTextEmbeddingProvider | None = None
    summary_granularity: VideoNodeLevel | None = None
    prompt_override: str | None = None
    forced_frame_timestamps_override: list[float] | None = None
    vl_max_input_frames: int | None = None
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
        self._notify_progress(
            phase="visual",
            event="status",
            status="visual loading model",
        )
        model, processor = self._ensure_loaded()
        self._notify_progress(
            phase="visual",
            event="status",
            status="visual model ready",
        )
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
                self._notify_progress(
                    phase="visual",
                    event="status",
                    status=f"visual generate {index}/{len(spans)} frames={len(frame_paths)}",
                )
                output_text, frame_metadata = self._generate_with_frame_retry(
                    model=model,
                    processor=processor,
                    frame_paths=frame_paths,
                    span=span,
                    metadata=frame_metadata,
                )
                payload = _parse_json_object(output_text)
                summary_text = _visual_summary_text_from_payload(payload, output_text)
                summary_metadata = dict(frame_metadata)
                summary_metadata.update(_vrrqa_visual_verification_metadata(payload))
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
                        metadata=summary_metadata,
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
        original_frame_count = len(frame_paths)
        frame_paths = self._limit_vl_input_frames(frame_paths)
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
            if len(frame_paths) < original_frame_count:
                attempt_metadata["vl_input_frame_limited"] = True
                attempt_metadata["vl_input_frame_limit"] = self.vl_max_input_frames
                attempt_metadata["vl_original_frame_count"] = original_frame_count
            if attempt_index > 0:
                attempt_metadata["vl_retry_reason"] = "frame_batch_reduced"
                attempt_metadata.setdefault("vl_original_frame_count", len(frame_paths))
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

    def _limit_vl_input_frames(self, frame_paths: list[Path]) -> list[Path]:
        if self.vl_max_input_frames is None:
            return frame_paths
        if self.vl_max_input_frames <= 0:
            raise ValueError(
                f"vl_max_input_frames must be positive when set, got {self.vl_max_input_frames}"
            )
        return self._temporal_frame_subset(frame_paths, self.vl_max_input_frames)

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
        input_device = self._resolve_input_device(model)
        input_dtype = self._resolve_input_dtype(model)
        generation_dtype = _resolve_generation_autocast_dtype(model, input_dtype)
        inputs = _move_inputs_to_device(
            inputs,
            input_device,
            generation_dtype or input_dtype,
        )
        with _generation_autocast_context(input_device, generation_dtype):
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
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
        frame_paths = selection.frame_paths
        if self.forced_frame_timestamps_override:
            frame_paths, relationship_metadata = self._apply_relationship_frame_policy(
                video_path=video_path,
                span=span,
                output_dir=output_dir,
                pitome_frame_paths=selection.frame_paths,
                pitome_timestamps=selection.timestamps,
                forced_timestamps=self.forced_frame_timestamps_override,
            )
            metadata.update(relationship_metadata)
        return frame_paths, metadata

    def _apply_relationship_frame_policy(
        self,
        *,
        video_path: str,
        span: TimeSpan,
        output_dir: Path,
        pitome_frame_paths: list[Path],
        pitome_timestamps: list[float],
        forced_timestamps: list[float],
    ) -> tuple[list[Path], dict[str, Any]]:
        requested_timestamps = _merge_timestamps(
            [
                timestamp
                for timestamp in forced_timestamps
                if span.start <= float(timestamp) <= span.end
            ]
        )
        if not requested_timestamps:
            return pitome_frame_paths, {
                "relationship_frame_policy": "graph_node_start_mid_end",
                "relationship_frame_policy_applied": False,
                "relationship_frame_policy_reason": "no_timestamps_in_span",
            }

        forced_paths = extract_frames_for_timestamps(
            media_path=video_path,
            timestamps=requested_timestamps,
            ffmpeg_bin=self.ffmpeg_bin,
            width=self.pitome_frame_width if self.pitome_frame_width is not None else self.frame_width,
            output_dir=output_dir,
            prefix="relationship",
            extraction_strategy=self.pitome_frame_extraction_strategy,
            seek_workers=self.pitome_frame_extraction_workers,
        )
        max_count = self.pitome_max_selected_frames
        if max_count is not None and len(forced_paths) > max_count:
            forced_paths = _limit_paths_by_temporal_coverage(forced_paths, max_count)
            requested_timestamps = _limit_values_by_temporal_coverage(
                requested_timestamps,
                max_count,
            )
        remaining_capacity = None if max_count is None else max(0, max_count - len(forced_paths))
        pitome_pairs = [
            (path, timestamp)
            for path, timestamp in zip(pitome_frame_paths, pitome_timestamps, strict=False)
            if not _has_nearby_timestamp(timestamp, requested_timestamps)
        ]
        if remaining_capacity is not None:
            pitome_pairs = _limit_pairs_by_temporal_coverage(pitome_pairs, remaining_capacity)
        selected_paths = [*forced_paths, *[path for path, _timestamp in pitome_pairs]]
        return selected_paths, {
            "relationship_frame_policy": "graph_node_start_mid_end",
            "relationship_frame_policy_applied": True,
            "relationship_forced_frame_count": len(forced_paths),
            "relationship_forced_frame_timestamps": requested_timestamps,
            "relationship_pitome_frame_count": len(pitome_pairs),
            "relationship_total_frame_count": len(selected_paths),
        }

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
            include_edge_boundary_frames=self.pitome_edge_boundary_frames,
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
        from transformers import AutoProcessor

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

        model_path = self.model_path or self.model_name
        if _use_image_text_to_text_loader(self.model_name, self.model_path):
            from transformers import AutoModelForImageTextToText

            self._log("using AutoModelForImageTextToText loader")
            self.model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)
        else:
            from transformers import Qwen3VLForConditionalGeneration

            self._log("using Qwen3VLForConditionalGeneration loader")
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                **model_kwargs,
            )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self._log("VL model loaded")
        return self.model, self.processor

    def unload(self) -> None:
        self.model = None
        self.processor = None
        from rlm.video.gpu_memory import clear_torch_cache, unload_component

        unload_component(self.frame_embedding_provider)
        clear_torch_cache()

    def _resolve_input_device(self, model):
        try:
            return next(model.parameters()).device
        except StopIteration as exc:
            raise ValueError("Vision model has no parameters") from exc

    def _resolve_input_dtype(self, model):
        return _model_floating_dtype(model)

    def _build_prompt(self, span: TimeSpan) -> str:
        if self.prompt_override is not None:
            return self.prompt_override
        return (
            "Summarize this LongShotBench video segment for grounded question answering. "
            "Return strict JSON with keys `summary`, `tags`, and `entities`. "
            "Mention exact visible actions, people, objects, slides, UI labels, signs, "
            "code text, math expressions, counts, and shell/output text when visible. "
            "For multi-frame input, preserve frame order, describe important keyframes, "
            "and state what changes between frames. "
            "For code/editor/tutorial content, copy short exact strings and variable names "
            "instead of paraphrasing them. "
            "Also mention spatial relations, viewpoint/visibility, motion direction, temporal "
            "ordering, entity continuity, physical context, and evidence useful for counting "
            "or extracting scene text when visible. "
            "Do not guess audio or speech content from frames alone. "
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
    pitome_edge_boundary_frames: bool = True
    frame_embedding_provider: ImageTextEmbeddingProvider | None = None
    visual_index_batch_size: int = 1
    visual_index_workers: int = 1
    summary_granularity: VideoNodeLevel | None = "clip"
    verbose: bool = False
    progress_callback: Callable[[dict[str, Any]], None] | None = None
    progress_unit_weight: int = 1

    def summarize(self, video_path: str, spans: list[TimeSpan]) -> list[VisualSummarySpan]:
        if self.visual_index_batch_size <= 0:
            raise ValueError(
                f"visual_index_batch_size must be positive, got {self.visual_index_batch_size}"
            )
        if self.visual_index_workers <= 0:
            raise ValueError(
                f"visual_index_workers must be positive, got {self.visual_index_workers}"
            )
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
            for batch_start in range(0, len(spans), self.visual_index_batch_size):
                indexed_batch: list[
                    tuple[int, TimeSpan, FrameSelectionResult, dict[str, list[float]]]
                ] = []
                batch_spans = spans[batch_start : batch_start + self.visual_index_batch_size]
                batch_items = [
                    (batch_start + offset + 1, span)
                    for offset, span in enumerate(batch_spans)
                ]

                def select_one(
                    item: tuple[int, TimeSpan],
                ) -> tuple[int, TimeSpan, FrameSelectionResult, dict[str, list[float]]]:
                    index, span = item
                    selection, boundary_metadata = self._select_visual_index_frames(
                        video_path=video_path,
                        span=span,
                        output_dir=temp_dir / f"span_{index:03d}",
                    )
                    return index, span, selection, boundary_metadata

                if self.visual_index_workers == 1 or len(batch_items) <= 1:
                    indexed_batch = [select_one(item) for item in batch_items]
                else:
                    worker_count = min(self.visual_index_workers, len(batch_items))
                    with ThreadPoolExecutor(max_workers=worker_count) as executor:
                        indexed_batch = list(executor.map(select_one, batch_items))

                semantic_metadatas = _semantic_frame_metadata_batch(
                    self.frame_embedding_provider,
                    [selection.frame_paths for _index, _span, selection, _metadata in indexed_batch],
                )
                for (
                    index,
                    span,
                    selection,
                    boundary_metadata,
                ), semantic_metadata in zip(indexed_batch, semantic_metadatas, strict=True):
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
                            "visual_index_batch_size": self.visual_index_batch_size,
                            "visual_index_workers": self.visual_index_workers,
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

    def unload(self) -> None:
        from rlm.video.gpu_memory import clear_torch_cache, unload_component

        unload_component(self.frame_embedding_provider)
        clear_torch_cache()

    def _select_visual_index_frames(
        self,
        *,
        video_path: str,
        span: TimeSpan,
        output_dir: Path,
    ) -> tuple[FrameSelectionResult, dict[str, list[float]]]:
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
        return selection, boundary_metadata

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
            include_edge_boundary_frames=self.pitome_edge_boundary_frames,
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
class PaddleOCRTextExtractor:
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    window_duration_seconds: float = 45.0
    frame_count: int = 6
    frame_width: int | None = 960
    frame_extraction_strategy: Literal["auto", "batch", "seek", "sequence"] = "seek"
    frame_extraction_workers: int = 1
    lang: str = "en"
    ocr_version: str = "PP-OCRv5"
    text_detection_model_name: str | None = None
    text_recognition_model_name: str | None = None
    text_recognition_batch_size: int | None = None
    device: str | None = None
    min_confidence: float = 0.35
    enable_mkldnn: bool = False
    cache_dir: str | None = None
    model: Any | None = None
    verbose: bool = False
    progress_callback: Callable[[dict[str, Any]], None] | None = None

    def extract(self, video_path: str) -> list[OCRSpan]:
        if self.window_duration_seconds <= 0:
            raise ValueError(
                "window_duration_seconds must be positive, "
                f"got {self.window_duration_seconds}"
            )
        if self.frame_count <= 0:
            raise ValueError(f"frame_count must be positive, got {self.frame_count}")
        if self.frame_extraction_workers <= 0:
            raise ValueError(
                "frame_extraction_workers must be positive, "
                f"got {self.frame_extraction_workers}"
            )
        model = self._ensure_loaded()
        duration_seconds = probe_media_duration(video_path, ffprobe_bin=self.ffprobe_bin)
        windows = _chunk_time_spans(duration_seconds, self.window_duration_seconds)
        self._log(
            f"paddle OCR start path={video_path} windows={len(windows)} "
            f"window_seconds={self.window_duration_seconds:.2f}"
        )
        self._notify_progress(
            phase="ocr",
            event="planned",
            total=len(windows),
            status=f"ocr 0/{len(windows)}",
        )
        output_root = get_videorlm_output_root() / "tmp"
        output_root.mkdir(parents=True, exist_ok=True)
        spans: list[OCRSpan] = []
        with contextlib.ExitStack() as stack:
            temp_dir = Path(
                stack.enter_context(
                    _temporary_directory(prefix="videorlm_paddle_ocr_", dir_path=output_root)
                )
            )
            for index, span in enumerate(windows, start=1):
                frame_paths = extract_frames_for_span(
                    media_path=video_path,
                    span=span,
                    frame_count=self.frame_count,
                    ffmpeg_bin=self.ffmpeg_bin,
                    width=self.frame_width,
                    output_dir=temp_dir / f"window_{index:03d}",
                    extraction_strategy=self.frame_extraction_strategy,
                    seek_workers=self.frame_extraction_workers,
                )
                lines: list[str] = []
                for frame_path in frame_paths:
                    lines.extend(self._predict_frame_lines(model, frame_path))
                text = self._merge_ocr_lines(lines)
                if text:
                    spans.append(OCRSpan(text=text, time_span=span))
                self._notify_progress(
                    phase="ocr",
                    event="advance",
                    advance=1,
                    index=index,
                    total=len(windows),
                    status=f"ocr {index}/{len(windows)}",
                )
        self._log(f"paddle OCR done spans={len(spans)}")
        return spans

    def _ensure_loaded(self) -> Any:
        if self.model is not None:
            return self.model
        cache_dir = Path(self.cache_dir) if self.cache_dir else (
            get_videorlm_output_root() / "cache" / "paddlex"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(cache_dir))
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise ImportError(
                "PaddleOCRTextExtractor requires paddleocr. Install it in the active "
                "environment before using --enable-paddle-ocr."
            ) from exc
        self.model = PaddleOCR(
            lang=self.lang,
            ocr_version=self.ocr_version,
            text_detection_model_name=self.text_detection_model_name,
            text_recognition_model_name=self.text_recognition_model_name,
            text_recognition_batch_size=self.text_recognition_batch_size,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_rec_score_thresh=self.min_confidence,
            enable_mkldnn=self.enable_mkldnn,
            device=self.device,
        )
        return self.model

    def unload(self) -> None:
        self.model = None
        from rlm.video.gpu_memory import clear_torch_cache

        clear_torch_cache()

    def _predict_frame_lines(self, model: Any, frame_path: Path) -> list[str]:
        if hasattr(model, "predict"):
            result = model.predict(
                str(frame_path),
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_rec_score_thresh=self.min_confidence,
            )
        else:
            result = model.ocr(str(frame_path), cls=False)
        return self._extract_lines_from_result(result)

    def _extract_lines_from_result(self, result: object) -> list[str]:
        lines: list[str] = []
        self._collect_ocr_lines(result, lines)
        return [line for line in lines if line]

    def _collect_ocr_lines(self, value: object, lines: list[str]) -> None:
        if value is None:
            return
        payload = self._paddle_result_payload(value)
        if payload is not None:
            rec_texts = payload.get("rec_texts")
            if isinstance(rec_texts, list):
                scores = payload.get("rec_scores", [])
                for index, text in enumerate(rec_texts):
                    score = scores[index] if isinstance(scores, list) and index < len(scores) else None
                    if not self._score_is_usable(score):
                        continue
                    normalized = self._normalize_ocr_line(str(text))
                    if normalized:
                        lines.append(normalized)
            for key in ("res", "ocr_res", "ocr_result", "ocr_results", "result", "results"):
                if key in payload:
                    self._collect_ocr_lines(payload[key], lines)
            return
        if isinstance(value, (list, tuple)):
            if self._looks_like_legacy_text_score(value):
                text = self._normalize_ocr_line(str(value[0]))
                score = value[1]
                if text and self._score_is_usable(score):
                    lines.append(text)
                return
            if len(value) >= 2 and self._looks_like_legacy_text_score(value[1]):
                text_score = value[1]
                text = self._normalize_ocr_line(str(text_score[0]))
                score = text_score[1]
                if text and self._score_is_usable(score):
                    lines.append(text)
                return
            for item in value:
                self._collect_ocr_lines(item, lines)

    def _paddle_result_payload(self, value: object) -> dict[str, object] | None:
        if isinstance(value, dict):
            return value
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            payload = to_dict()
            if isinstance(payload, dict):
                return payload
        json_value = getattr(value, "json", None)
        if callable(json_value):
            payload = json_value()
            if isinstance(payload, dict):
                return payload
        elif isinstance(json_value, dict):
            return json_value
        res_value = getattr(value, "res", None)
        if isinstance(res_value, dict):
            return res_value
        return None

    def _looks_like_legacy_text_score(self, value: object) -> bool:
        return (
            isinstance(value, (list, tuple))
            and len(value) >= 2
            and isinstance(value[0], str)
            and isinstance(value[1], (int, float))
        )

    def _score_is_usable(self, score: object) -> bool:
        if score is None:
            return True
        if not isinstance(score, (int, float)):
            return True
        return float(score) >= self.min_confidence

    def _merge_ocr_lines(self, lines: list[str]) -> str:
        seen: set[str] = set()
        merged: list[str] = []
        for line in lines:
            key = self._ocr_line_key(line)
            if key in seen:
                continue
            if any(self._is_near_duplicate_ocr_line(key, existing) for existing in seen):
                continue
            seen.add(key)
            merged.append(line)
        return "\n".join(merged)

    def _normalize_ocr_line(self, text: str) -> str:
        return " ".join(text.replace("\r", "\n").split()).strip()

    def _ocr_line_key(self, line: str) -> str:
        return re.sub(r"\W+", "", line.casefold())

    def _is_near_duplicate_ocr_line(self, key: str, existing_key: str) -> bool:
        if len(key) < 12 or len(existing_key) < 12:
            return False
        return difflib.SequenceMatcher(None, key, existing_key).ratio() >= 0.94

    def _notify_progress(self, **payload: Any) -> None:
        if self.progress_callback is not None:
            self.progress_callback(payload)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[PaddleOCR] {message}", flush=True)


@dataclass
class FasterWhisperSpeechRecognizer:
    model_name: str = "small"
    device: str = "cpu"
    compute_type: str = "default"
    batch_size: int = 1
    chunk_workers: int = 1
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    chunk_duration_seconds: float = 300.0
    language: str | None = None
    vad_filter: bool = True
    model: Any | None = None
    verbose: bool = False
    progress_callback: Callable[[dict[str, Any]], None] | None = None

    def recognize(self, video_path: str) -> list[SpeechSpan]:
        model = None if self.chunk_workers > 1 else self._ensure_loaded()
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
            return self._recognize_audio_in_chunks(
                model=model,
                audio_path=audio_path,
                stack=stack,
            )

    def _recognize_audio_in_chunks(
        self,
        *,
        model: Any | None,
        audio_path: Path,
        stack: contextlib.ExitStack,
    ) -> list[SpeechSpan]:
        duration_seconds = probe_media_duration(audio_path, ffprobe_bin=self.ffprobe_bin)
        if self.chunk_duration_seconds <= 0:
            chunks = [TimeSpan(0.0, duration_seconds)] if duration_seconds > 0 else []
        else:
            chunks = _chunk_time_spans(duration_seconds, self.chunk_duration_seconds)
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.chunk_workers <= 0:
            raise ValueError(f"chunk_workers must be positive, got {self.chunk_workers}")
        self._notify_progress(
            phase="asr",
            event="planned",
            total=len(chunks),
            status=(
                f"asr faster-whisper 0/{len(chunks)} "
                f"chunk_seconds={self.chunk_duration_seconds:.0f} "
                f"batch_size={self.batch_size} "
                f"chunk_workers={self.chunk_workers}"
            ),
        )
        temp_root = get_videorlm_output_root() / "tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(
            stack.enter_context(
                _temporary_directory(prefix="videorlm_faster_whisper_chunks_", dir_path=temp_root)
            )
        )
        chunk_jobs: list[tuple[int, TimeSpan, Path]] = []
        for index, chunk_span in enumerate(chunks, start=1):
            self._log(
                f"ASR chunk {index}/{len(chunks)} span={chunk_span.to_display()}"
            )
            chunk_path = extract_audio_segment(
                media_path=audio_path,
                span=chunk_span,
                output_path=temp_dir / f"chunk_{index:03d}.wav",
                ffmpeg_bin=self.ffmpeg_bin,
            )
            chunk_jobs.append((index, chunk_span, chunk_path))
        spans: list[SpeechSpan] = []
        if self.chunk_workers == 1 or len(chunk_jobs) <= 1:
            active_model = model if model is not None else self._ensure_loaded()
            transcriber = (
                self._batched_transcriber(active_model) if self.batch_size > 1 else active_model
            )
            for index, chunk_span, chunk_path in chunk_jobs:
                parsed = self._transcribe_audio_path(transcriber, chunk_path)
                for item in parsed:
                    spans.append(_offset_speech_span(item, chunk_span))
                self._notify_progress(
                    phase="asr",
                    event="advance",
                    advance=1,
                    index=index,
                    total=len(chunks),
                    status=(
                        f"asr faster-whisper {index}/{len(chunks)} "
                        f"parsed_spans={len(parsed)}"
                    ),
                )
            return spans

        results_by_index: dict[int, tuple[TimeSpan, list[SpeechSpan]]] = {}
        worker_cache = threading.local()
        worker_count = min(self.chunk_workers, len(chunk_jobs))
        self._log(f"parallel faster-whisper chunk workers={worker_count}")
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    self._transcribe_audio_path_with_worker_model,
                    worker_cache,
                    chunk_path,
                ): (index, chunk_span)
                for index, chunk_span, chunk_path in chunk_jobs
            }
            completed = 0
            for future in as_completed(futures):
                index, chunk_span = futures[future]
                parsed = future.result()
                results_by_index[index] = (chunk_span, parsed)
                completed += 1
                self._notify_progress(
                    phase="asr",
                    event="advance",
                    advance=1,
                    index=completed,
                    total=len(chunks),
                    status=(
                        f"asr faster-whisper {completed}/{len(chunks)} "
                        f"completed_chunk={index} parsed_spans={len(parsed)}"
                    ),
                )
        for index in sorted(results_by_index):
            chunk_span, parsed = results_by_index[index]
            for item in parsed:
                spans.append(_offset_speech_span(item, chunk_span))
        return spans

    def _transcribe_audio_path_with_worker_model(
        self,
        worker_cache: threading.local,
        audio_path: Path,
    ) -> list[SpeechSpan]:
        transcriber = getattr(worker_cache, "transcriber", None)
        if transcriber is None:
            model = self._load_whisper_model()
            transcriber = self._batched_transcriber(model) if self.batch_size > 1 else model
            worker_cache.transcriber = transcriber
        return self._transcribe_audio_path(transcriber, audio_path)

    def _transcribe_audio_path(self, transcriber: Any, audio_path: Path) -> list[SpeechSpan]:
        transcribe_kwargs: dict[str, Any] = {
            "language": self.language,
            "vad_filter": self.vad_filter,
        }
        if self.batch_size > 1:
            transcribe_kwargs["batch_size"] = self.batch_size
        segments, _info = transcriber.transcribe(str(audio_path), **transcribe_kwargs)
        return [
            SpeechSpan(
                text=str(segment.text).strip(),
                time_span=TimeSpan(float(segment.start), float(segment.end)),
                language=self.language,
            )
            for segment in segments
            if str(segment.text).strip()
        ]

    def _ensure_loaded(self):
        if self.model is not None:
            return self.model
        self.model = self._load_whisper_model()
        return self.model

    def _load_whisper_model(self) -> Any:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ImportError(
                "faster-whisper speech backend requires `faster-whisper` to be installed."
            ) from exc
        whisper_device, device_index = _parse_faster_whisper_device(self.device)
        kwargs = {"device": whisper_device}
        if device_index is not None:
            kwargs["device_index"] = device_index
        if self.compute_type != "default":
            kwargs["compute_type"] = self.compute_type
        self._log(
            f"loading faster-whisper model={self.model_name} "
            f"device={self.device} compute_type={self.compute_type} "
            f"batch_size={self.batch_size} "
            f"chunk_seconds={self.chunk_duration_seconds:.0f} "
            f"chunk_workers={self.chunk_workers}"
        )
        return WhisperModel(self.model_name, **kwargs)

    def _batched_transcriber(self, model: Any) -> Any:
        try:
            from faster_whisper import BatchedInferencePipeline
        except ImportError as exc:
            raise ImportError(
                "faster-whisper batch_size > 1 requires a faster-whisper version "
                "with BatchedInferencePipeline."
            ) from exc
        return BatchedInferencePipeline(model=model)

    def unload(self) -> None:
        self.model = None
        from rlm.video.gpu_memory import clear_torch_cache

        clear_torch_cache()

    def _notify_progress(self, **payload: Any) -> None:
        if self.progress_callback is not None:
            self.progress_callback(payload)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[FasterWhisper] {message}", flush=True)


def _parse_faster_whisper_device(device: str) -> tuple[str, int | None]:
    if device.startswith("cuda:"):
        index_text = device.split(":", 1)[1]
        if not index_text.isdigit():
            raise ValueError(f"Invalid faster-whisper CUDA device: {device}")
        return "cuda", int(index_text)
    return device, None


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


def _limit_values_by_temporal_coverage(values: list[float], max_count: int) -> list[float]:
    if max_count <= 0:
        raise ValueError(f"max_count must be positive, got {max_count}")
    if len(values) <= max_count:
        return values
    if max_count == 1:
        return [values[len(values) // 2]]
    indices = sorted(
        {round(position * (len(values) - 1) / (max_count - 1)) for position in range(max_count)}
    )
    return [values[index] for index in indices]


def _limit_pairs_by_temporal_coverage(
    pairs: list[tuple[Path, float]],
    max_count: int,
) -> list[tuple[Path, float]]:
    if max_count <= 0:
        return []
    if len(pairs) <= max_count:
        return pairs
    if max_count == 1:
        return [pairs[len(pairs) // 2]]
    indices = sorted(
        {round(position * (len(pairs) - 1) / (max_count - 1)) for position in range(max_count)}
    )
    return [pairs[index] for index in indices]


def _has_nearby_timestamp(
    timestamp: float,
    candidates: list[float],
    *,
    tolerance: float = 0.05,
) -> bool:
    return any(abs(float(timestamp) - candidate) <= tolerance for candidate in candidates)


def _semantic_frame_metadata(
    provider: ImageTextEmbeddingProvider | None,
    frame_paths: list[Path],
) -> dict[str, Any]:
    return _semantic_frame_metadata_batch(provider, [frame_paths])[0]


def _semantic_frame_metadata_batch(
    provider: ImageTextEmbeddingProvider | None,
    frame_path_groups: list[list[Path]],
) -> list[dict[str, Any]]:
    metadatas: list[dict[str, Any]] = [{} for _group in frame_path_groups]
    if provider is None:
        return metadatas

    flat_paths = [path for group in frame_path_groups for path in group]
    if not flat_paths:
        return metadatas

    embeddings = provider.embed_images(flat_paths)
    if not embeddings:
        return metadatas
    if len(embeddings) != len(flat_paths):
        raise ValueError(
            "Semantic frame embedding provider returned "
            f"{len(embeddings)} embeddings for {len(flat_paths)} frames."
        )

    offset = 0
    model_name = getattr(provider, "model_name", None)
    for index, group in enumerate(frame_path_groups):
        count = len(group)
        group_embeddings = embeddings[offset : offset + count]
        offset += count
        if not group_embeddings:
            continue
        metadatas[index] = {
            "semantic_frame_embeddings": group_embeddings,
            "semantic_frame_embedding_model": model_name,
            "semantic_frame_embedding_dim": len(group_embeddings[0]),
        }
    return metadatas


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
        event_boundary_scores=list(selection.event_boundary_scores),
        visual_novelty_scores=list(selection.visual_novelty_scores),
        cognitive_anchor_metadata=list(selection.cognitive_anchor_metadata),
        memorability_prior=selection.memorability_prior,
    )


def _visual_summary_text_from_payload(payload: dict[str, Any], raw_text: str) -> str:
    summary = payload.get("summary")
    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    evidence = payload.get("evidence")
    best_option = payload.get("best_option")
    if isinstance(evidence, str) and evidence.strip():
        if isinstance(best_option, str) and best_option.strip():
            return f"Best option {best_option.strip().upper()}: {evidence.strip()}"
        return evidence.strip()
    return str(raw_text).strip()


def _vrrqa_visual_verification_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    raw_summary = payload.get("summary")
    if isinstance(raw_summary, str):
        metadata.update(_extract_vrrqa_metadata_from_text(raw_summary))
    if "target_entities" in payload:
        metadata["vrrqa_target_entities"] = payload["target_entities"]
    if "candidate_entities" in payload:
        metadata["vrrqa_candidate_entities"] = payload["candidate_entities"]
    if "entity_grounding" in payload:
        metadata["vrrqa_entity_grounding"] = payload["entity_grounding"]
    if "frame_observations" in payload:
        metadata["vrrqa_frame_observations"] = payload["frame_observations"]
    if "frame_timeline" in payload:
        metadata["vrrqa_frame_timeline"] = payload["frame_timeline"]
    if "visible_relation" in payload:
        metadata["vrrqa_visible_relation"] = payload["visible_relation"]
    if "spatial_relation" in payload:
        metadata["vrrqa_spatial_relation"] = payload["spatial_relation"]
    if "entities_visible" in payload:
        metadata["vrrqa_entities_visible"] = payload["entities_visible"]
    if "co_visible" in payload:
        metadata["vrrqa_co_visible"] = payload["co_visible"]
    if "co_visible_frame_indices" in payload:
        metadata["vrrqa_co_visible_frame_indices"] = payload["co_visible_frame_indices"]
    if "relation_supported" in payload:
        metadata["vrrqa_relation_supported"] = payload["relation_supported"]
    if "relation_votes" in payload:
        metadata["vrrqa_relation_votes"] = payload["relation_votes"]
    if "vote_counts" in payload:
        metadata["vrrqa_vote_counts"] = payload["vote_counts"]
    if "aggregated_relation" in payload:
        metadata["vrrqa_aggregated_relation"] = payload["aggregated_relation"]
    if "motion_trajectory" in payload:
        metadata["vrrqa_motion_trajectory"] = payload["motion_trajectory"]
    if "temporal_order" in payload:
        metadata["vrrqa_temporal_order"] = payload["temporal_order"]
    if "entity_continuity" in payload:
        metadata["vrrqa_entity_continuity"] = payload["entity_continuity"]
    if "physical_context" in payload:
        metadata["vrrqa_physical_context"] = payload["physical_context"]
    if "inferred_relation" in payload:
        metadata["vrrqa_inferred_relation"] = payload["inferred_relation"]
    if "option_comparison" in payload:
        metadata["vrrqa_option_comparison"] = payload["option_comparison"]
    if "verifier_verdict" in payload:
        metadata["vrrqa_verifier_verdict"] = payload["verifier_verdict"]
    if "needs_more_evidence" in payload:
        metadata["vrrqa_needs_more_evidence"] = payload["needs_more_evidence"]
    co_visible_frame_count = _co_visible_frame_count_from_payload(payload)
    if co_visible_frame_count is not None:
        metadata["vrrqa_co_visible_frame_count"] = co_visible_frame_count
    evidence = payload.get("evidence")
    if isinstance(evidence, str) and evidence.strip():
        metadata["vrrqa_evidence"] = evidence.strip()
    option_scores = payload.get("option_scores")
    if isinstance(option_scores, dict):
        metadata["vrrqa_option_scores"] = {
            str(letter).strip().upper(): score for letter, score in option_scores.items()
        }
    best_option = payload.get("best_option")
    if isinstance(best_option, str) and best_option.strip():
        metadata["vrrqa_best_option"] = best_option.strip().upper()[:1]
    if any(key.startswith("vrrqa_") for key in metadata):
        metadata["vrrqa_visual_verification"] = True
    return metadata


def _co_visible_frame_count_from_payload(payload: dict[str, Any]) -> int | None:
    indices = payload.get("co_visible_frame_indices")
    if isinstance(indices, list):
        return len(indices)

    observations = payload.get("frame_observations")
    if not isinstance(observations, list):
        return None
    count = 0
    for observation in observations:
        if not isinstance(observation, dict):
            continue
        if _metadata_bool_value(observation.get("co_visible")) is True:
            count += 1
    return count


def _metadata_bool_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "supported"}:
            return True
        if normalized in {"false", "no", "0", "unsupported"}:
            return False
    return None


def _extract_vrrqa_metadata_from_text(text: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    best_match = re.search(r'"best_option"\s*:\s*"([A-Z])"', text)
    if best_match is not None:
        metadata["vrrqa_best_option"] = best_match.group(1)
    scores_match = re.search(r'"option_scores"\s*:\s*(\{[^{}]*\})', text, flags=re.DOTALL)
    if scores_match is not None:
        try:
            scores = json.loads(scores_match.group(1))
        except json.JSONDecodeError:
            scores = None
        if isinstance(scores, dict):
            metadata["vrrqa_option_scores"] = {
                str(letter).strip().upper(): score for letter, score in scores.items()
            }
    evidence_match = re.search(r'"evidence"\s*:\s*"((?:[^"\\]|\\.)*)"', text, flags=re.DOTALL)
    if evidence_match is not None:
        try:
            evidence = json.loads(f'"{evidence_match.group(1)}"')
        except json.JSONDecodeError:
            evidence = evidence_match.group(1)
        if isinstance(evidence, str) and evidence.strip():
            metadata["vrrqa_evidence"] = evidence.strip()
    return metadata


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
    include_edge_boundary_frames: bool,
) -> tuple[FrameSelectionResult, dict[str, list[float]]]:
    boundary_metadata = _boundary_timestamps_for_span(
        video_path=video_path,
        span=span,
        ffmpeg_bin=ffmpeg_bin,
        threshold=scene_threshold,
        max_scene_boundary_frames=max_scene_boundary_frames,
        scene_sample_rate=scene_sample_rate,
        scene_keyframes_only=scene_keyframes_only,
        include_edge_boundary_frames=include_edge_boundary_frames,
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
    include_edge_boundary_frames: bool,
) -> dict[str, list[float]]:
    edge_timestamps = _span_boundary_timestamps(span) if include_edge_boundary_frames else []
    if not edge_timestamps and max_scene_boundary_frames == 0:
        return {
            "all": [],
            "edges": [],
            "scene": [],
        }
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
    margin = min(0.5, max(0.25, span.duration * 0.02))
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
    selected_timestamps = [timestamp for timestamp, _, _, _ in deduped]
    boundary_anchor_metadata = _boundary_cognitive_anchor_metadata(
        selection=selection,
        selected_timestamps=selected_timestamps,
        boundary_timestamps=boundary_timestamps,
    )
    return FrameSelectionResult(
        strategy=selection.strategy,
        frame_paths=[path for _, path, _, _ in deduped],
        timestamps=selected_timestamps,
        dense_frame_count=selection.dense_frame_count,
        embedding_backend=selection.embedding_backend,
        embedding_size=selection.embedding_size,
        frame_embeddings=merged_embeddings,
        protected_timestamps=list(selection.protected_timestamps),
        representative_timestamps=list(selection.representative_timestamps),
        energy_scores=list(selection.energy_scores),
        merged_pairs=list(selection.merged_pairs),
        event_boundary_scores=list(selection.event_boundary_scores),
        visual_novelty_scores=list(selection.visual_novelty_scores),
        cognitive_anchor_metadata=boundary_anchor_metadata,
        memorability_prior=compute_memorability_prior(
            anchor_metadata=boundary_anchor_metadata,
            event_boundary_scores=[
                float(item.get("score", 0.0)) for item in selection.event_boundary_scores
            ],
            visual_novelty_scores=[
                float(item.get("score", 0.0)) for item in selection.visual_novelty_scores
            ],
            dense_frame_count=selection.dense_frame_count,
        ),
    )


def _boundary_cognitive_anchor_metadata(
    *,
    selection: FrameSelectionResult,
    selected_timestamps: list[float],
    boundary_timestamps: list[float],
) -> list[dict[str, Any]]:
    merged_metadata = filter_cognitive_anchor_metadata(
        selection.cognitive_anchor_metadata,
        selected_timestamps,
    )
    existing_timestamps = {float(item.get("timestamp", -1.0)) for item in merged_metadata}
    for timestamp in boundary_timestamps:
        if not any(abs(timestamp - existing) <= 0.05 for existing in existing_timestamps):
            merged_metadata.append(
                {
                    "timestamp": round(float(timestamp), 3),
                    "dense_index": None,
                    "reasons": ["ffmpeg_scene_or_span_boundary"],
                    "score": 0.9,
                    "event_boundary_score": 0.9,
                    "visual_novelty_score": 0.0,
                }
            )
            existing_timestamps.add(float(timestamp))
            continue
        for item in merged_metadata:
            if abs(float(item.get("timestamp", -1.0)) - timestamp) > 0.05:
                continue
            reasons = [str(reason) for reason in item.get("reasons", [])]
            if "ffmpeg_scene_or_span_boundary" not in reasons:
                reasons.append("ffmpeg_scene_or_span_boundary")
            item["reasons"] = reasons
            item["score"] = max(float(item.get("score", 0.0)), 0.9)
            item["event_boundary_score"] = max(
                float(item.get("event_boundary_score", 0.0)),
                0.9,
            )
            break
    merged_metadata.sort(key=lambda item: float(item.get("timestamp", 0.0)))
    return filter_cognitive_anchor_metadata(merged_metadata, selected_timestamps)


def _is_retryable_vl_frame_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "image features and image tokens do not match" in message or "out of memory" in message


def _resolve_torch_dtype(torch_module, value: str | Any):
    if not isinstance(value, str):
        return value
    if value == "auto":
        return "auto"
    if not hasattr(torch_module, value):
        raise ValueError(f"Unsupported torch dtype: {value}")
    return getattr(torch_module, value)


def _model_floating_dtype(model: Any) -> Any | None:
    if hasattr(model, "parameters"):
        for parameter in model.parameters():
            if hasattr(parameter, "is_floating_point") and parameter.is_floating_point():
                return parameter.dtype
    return getattr(model, "dtype", None)


def _resolve_generation_autocast_dtype(model: Any, input_dtype: Any | None) -> Any | None:
    for dtype in (
        input_dtype,
        _model_config_torch_dtype(model),
        getattr(model, "dtype", None),
    ):
        dtype = _coerce_torch_dtype(dtype)
        if _is_half_precision_torch_dtype(dtype):
            return dtype
    return None


def _model_config_torch_dtype(model: Any) -> Any | None:
    config = getattr(model, "config", None)
    if config is None:
        return None
    return getattr(config, "torch_dtype", None)


def _coerce_torch_dtype(dtype: Any) -> Any | None:
    if not isinstance(dtype, str):
        return dtype
    import torch

    return getattr(torch, dtype, None)


def _is_half_precision_torch_dtype(dtype: Any) -> bool:
    import torch

    return dtype in {torch.float16, torch.bfloat16}


def _generation_autocast_context(device: Any, dtype: Any | None):
    if not _is_half_precision_torch_dtype(dtype):
        return contextlib.nullcontext()
    device_type = getattr(device, "type", str(device).split(":", maxsplit=1)[0])
    if device_type != "cuda":
        return contextlib.nullcontext()

    import torch

    return torch.autocast(device_type=device_type, dtype=dtype)


def _move_inputs_to_device(inputs: Any, device: Any, dtype: Any | None) -> Any:
    if hasattr(inputs, "items"):
        for key, value in list(inputs.items()):
            moved = _move_value_to_device(value, device, dtype)
            inputs[key] = moved
            if hasattr(inputs, key):
                try:
                    setattr(inputs, key, moved)
                except AttributeError:
                    pass
        return inputs
    return _move_value_to_device(inputs, device, dtype)


def _move_value_to_device(value: Any, device: Any, dtype: Any | None) -> Any:
    if hasattr(value, "to"):
        if hasattr(value, "is_floating_point") and value.is_floating_point() and dtype is not None:
            return value.to(device=device, dtype=dtype)
        return value.to(device)
    if isinstance(value, list):
        return [_move_value_to_device(item, device, dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_value_to_device(item, device, dtype) for item in value)
    return value


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
        metadata=dict(span.metadata),
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


def _split_asr_batch_results(results: Any, batch_size: int) -> list[Any]:
    if not isinstance(results, (list, tuple)):
        raise ValueError(
            "Batched Qwen ASR returned a non-list result. "
            "Set SPEECH_ASR_CHUNK_BATCH_SIZE=1 to use sequential ASR."
        )
    if len(results) != batch_size:
        raise ValueError(
            f"Batched Qwen ASR returned {len(results)} results for {batch_size} chunks. "
            "Set SPEECH_ASR_CHUNK_BATCH_SIZE=1 to use sequential ASR."
        )
    grouped: list[Any] = []
    for item in results:
        if isinstance(item, (list, tuple)):
            grouped.append(list(item))
        else:
            grouped.append([item])
    return grouped


def _merge_speech_spans(spans: list[SpeechSpan]) -> SpeechSpan:
    if not spans:
        raise ValueError("Cannot merge an empty list of speech spans")
    text = _normalize_whitespace(" ".join(span.text.strip() for span in spans if span.text.strip()))
    return SpeechSpan(
        text=text,
        time_span=TimeSpan(spans[0].time_span.start, spans[-1].time_span.end),
        speaker=spans[0].speaker,
        language=spans[0].language,
        metadata={
            "merged_span_count": len(spans),
            "source_span_times": [span.time_span.to_dict() for span in spans],
        },
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


def _use_image_text_to_text_loader(model_name: str, model_path: str | None) -> bool:
    model_identifiers = [model_name, model_path or ""]
    normalized_identifiers = [identifier.replace("\\", "/").lower() for identifier in model_identifiers]
    image_text_model_markers = (
        "qwen3.5",
        "qwen3.6",
        "gemma-3",
        "gemma3",
        "paligemma",
        "medgemma",
    )
    return any(
        marker in identifier
        for identifier in normalized_identifiers
        for marker in image_text_model_markers
    )
