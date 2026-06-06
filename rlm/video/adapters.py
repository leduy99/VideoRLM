import base64
import json
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import openai

from rlm.video.media import (
    extract_audio_track,
    extract_frames_for_span,
    get_videorlm_output_root,
    is_audio_path,
)
from rlm.video.pitome import (
    limit_frame_selection_by_temporal_coverage,
    select_visual_frames_for_span,
)
from rlm.video.types import (
    AudioEvent,
    OCRSpan,
    SpeechSpan,
    TimeSpan,
    VideoNodeLevel,
    VisualSummarySpan,
)


@runtime_checkable
class SpeechRecognizer(Protocol):
    def recognize(self, video_path: str) -> list[SpeechSpan]: ...


@runtime_checkable
class VisualSummarizer(Protocol):
    def summarize(self, video_path: str, spans: list[TimeSpan]) -> list[VisualSummarySpan]: ...


@runtime_checkable
class OCRExtractor(Protocol):
    def extract(self, video_path: str) -> list[OCRSpan]: ...


@runtime_checkable
class AudioEventExtractor(Protocol):
    def extract(self, video_path: str) -> list[AudioEvent]: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    def embed_text(self, text: str) -> list[float]: ...


@runtime_checkable
class ImageTextEmbeddingProvider(Protocol):
    def embed_text(self, text: str) -> list[float]: ...

    def embed_images(self, image_paths: list[str | Path]) -> list[list[float]]: ...


@runtime_checkable
class VideoWindowEmbeddingProvider(Protocol):
    def embed_text(self, text: str) -> list[float]: ...

    def embed_video_windows(
        self,
        video_path: str | Path,
        windows: list[TimeSpan],
    ) -> list[list[float]]: ...


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
class CallableVideoWindowEmbeddingProvider:
    text_fn: Callable[[str], list[float]]
    video_fn: Callable[[str | Path, list[TimeSpan]], list[list[float]]]
    model_name: str = "callable-video-window-embedding"

    def embed_text(self, text: str) -> list[float]:
        return self.text_fn(text)

    def embed_video_windows(
        self,
        video_path: str | Path,
        windows: list[TimeSpan],
    ) -> list[list[float]]:
        return self.video_fn(video_path, windows)


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
    image_url_format: Literal["object", "string"] = "object"
    system_prompt: str = (
        "Summarize what is visually present. Return strict JSON with keys "
        "`summary`, `tags`, and `entities`."
    )
    prompt_override: str | None = None
    frame_count: int = 3
    ffmpeg_bin: str = "ffmpeg"
    frame_width: int | None = 768
    scene_threshold_seconds: float = 20.0
    timeout: float = 300.0
    client: Any | None = None
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
    frame_embedding_provider: ImageTextEmbeddingProvider | None = None
    summary_granularity: VideoNodeLevel | None = None
    vl_max_input_frames: int | None = None

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
                frame_paths, frame_metadata = self._select_frames(video_path, span, frame_dir)
                original_frame_count = len(frame_paths)
                frame_paths = self._limit_vl_input_frames(frame_paths)
                if len(frame_paths) < original_frame_count:
                    frame_metadata.update(
                        {
                            "vl_input_frame_limited": True,
                            "vl_input_frame_limit": self.vl_max_input_frames,
                            "vl_original_frame_count": original_frame_count,
                            "vl_input_frame_count": len(frame_paths),
                        }
                    )
                content = [{"type": "text", "text": self._build_prompt(span)}]
                for frame_path in frame_paths:
                    image_url = _image_to_data_url(frame_path)
                    if self.image_url_format == "object":
                        image_url = {"url": image_url}
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": image_url,
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
                summary_metadata = dict(frame_metadata)
                summary_metadata.update(_vrrqa_visual_verification_metadata(payload))
                outputs.append(
                    VisualSummarySpan(
                        summary=str(payload.get("summary", response_text)).strip(),
                        time_span=span,
                        granularity=self._infer_granularity(span),
                        tags=[str(item) for item in payload.get("tags", [])],
                        entities=[str(item) for item in payload.get("entities", [])],
                        metadata=summary_metadata,
                    )
                )
        return outputs

    def _select_frame_paths(self, video_path: str, span: TimeSpan, output_dir: Path) -> list[Path]:
        frame_paths, _ = self._select_frames(video_path, span, output_dir)
        return frame_paths

    def _limit_vl_input_frames(self, frame_paths: list[Path]) -> list[Path]:
        if self.vl_max_input_frames is None:
            return frame_paths
        if self.vl_max_input_frames <= 0:
            raise ValueError(
                f"vl_max_input_frames must be positive when set, got {self.vl_max_input_frames}"
            )
        return _limit_paths_by_temporal_coverage(frame_paths, self.vl_max_input_frames)

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
            width=(
                self.pitome_frame_width
                if self.pitome_frame_width is not None
                else self.frame_width
            ),
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
        if self.pitome_max_selected_frames is not None:
            selection = limit_frame_selection_by_temporal_coverage(
                selection,
                self.pitome_max_selected_frames,
            )
        metadata = selection.to_metadata()
        metadata.update(self._semantic_frame_metadata(selection.frame_paths))
        return selection.frame_paths, metadata

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

    def _build_prompt(self, span: TimeSpan) -> str:
        if self.prompt_override is not None:
            return self.prompt_override
        return (
            "Describe the scene for long-video reasoning. "
            "Mention visible actions, people, objects, slides, or on-screen text. "
            "For multi-frame input, preserve frame order, describe important keyframes, "
            "and state when important entities are visible together. "
            "Also mention spatial relations, viewpoint/visibility, motion direction, temporal "
            "ordering, entity continuity, and physical context when they are visible. "
            f"Time span: {span.to_display()} seconds."
        )

    def _infer_granularity(self, span: TimeSpan) -> str:
        if self.summary_granularity is not None:
            return self.summary_granularity
        return "scene" if span.duration >= self.scene_threshold_seconds else "clip"


@dataclass
class GeminiVisualSummarizer:
    model_name: str
    api_key: str | None = None
    system_prompt: str = (
        "Summarize what is visually present. Return strict JSON with keys "
        "`summary`, `tags`, and `entities`."
    )
    prompt_override: str | None = None
    frame_count: int = 3
    ffmpeg_bin: str = "ffmpeg"
    frame_width: int | None = 768
    scene_threshold_seconds: float = 20.0
    timeout: float = 300.0
    max_new_tokens: int = 512
    client: Any | None = None
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
    frame_embedding_provider: ImageTextEmbeddingProvider | None = None
    summary_granularity: VideoNodeLevel | None = None

    def __post_init__(self) -> None:
        if self.client is not None:
            return
        import os

        from google import genai
        from google.genai import types

        api_key = self.api_key or os.getenv("GEMINI_API_KEY")
        if api_key is None:
            raise ValueError(
                "Gemini API key is required. Set GEMINI_API_KEY env var or pass api_key."
            )
        http_options = types.HttpOptions(timeout=int(self.timeout * 1000))
        self.client = genai.Client(api_key=api_key, http_options=http_options)

    def summarize(self, video_path: str, spans: list[TimeSpan]) -> list[VisualSummarySpan]:
        outputs: list[VisualSummarySpan] = []
        temp_root = get_videorlm_output_root() / "tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="videorlm_gemini_vl_",
            dir=str(temp_root),
        ) as temp_dir:
            for index, span in enumerate(spans, start=1):
                frame_dir = Path(temp_dir) / f"span_{index:03d}"
                frame_paths, frame_metadata = self._select_frames(video_path, span, frame_dir)
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=self._build_contents(frame_paths, span),
                    config=self._build_config(),
                )
                response_text = str(getattr(response, "text", "") or "")
                payload = _parse_json_object(response_text)
                summary_metadata = dict(frame_metadata)
                summary_metadata.update(_vrrqa_visual_verification_metadata(payload))
                outputs.append(
                    VisualSummarySpan(
                        summary=str(payload.get("summary", response_text)).strip(),
                        time_span=span,
                        granularity=self._infer_granularity(span),
                        tags=[str(item) for item in payload.get("tags", [])],
                        entities=[str(item) for item in payload.get("entities", [])],
                        metadata=summary_metadata,
                    )
                )
        return outputs

    def _build_contents(self, frame_paths: list[Path], span: TimeSpan):
        from google.genai import types

        parts = [types.Part(text=self._build_prompt(span))]
        for frame_path in frame_paths:
            parts.append(
                types.Part.from_bytes(
                    data=Path(frame_path).read_bytes(),
                    mime_type="image/jpeg",
                )
            )
        return [types.Content(role="user", parts=parts)]

    def _build_config(self):
        from google.genai import types

        return types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            max_output_tokens=self.max_new_tokens,
        )

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
            width=(
                self.pitome_frame_width
                if self.pitome_frame_width is not None
                else self.frame_width
            ),
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
        if self.pitome_max_selected_frames is not None:
            selection = limit_frame_selection_by_temporal_coverage(
                selection,
                self.pitome_max_selected_frames,
            )
        metadata = selection.to_metadata()
        metadata.update(self._semantic_frame_metadata(selection.frame_paths))
        return selection.frame_paths, metadata

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

    def _build_prompt(self, span: TimeSpan) -> str:
        if self.prompt_override is not None:
            return self.prompt_override
        return (
            "Describe the scene for long-video reasoning. "
            "Mention visible actions, people, objects, slides, or on-screen text. "
            "For multi-frame input, preserve frame order, describe important keyframes, "
            "and state when important entities are visible together. "
            "Also mention spatial relations, viewpoint/visibility, motion direction, temporal "
            "ordering, entity continuity, and physical context when they are visible. "
            f"Time span: {span.to_display()} seconds."
        )

    def _infer_granularity(self, span: TimeSpan) -> str:
        if self.summary_granularity is not None:
            return self.summary_granularity
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


def _vrrqa_visual_verification_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    field_map = {
        "target_entities": "vrrqa_target_entities",
        "candidate_entities": "vrrqa_candidate_entities",
        "entity_grounding": "vrrqa_entity_grounding",
        "frame_observations": "vrrqa_frame_observations",
        "frame_timeline": "vrrqa_frame_timeline",
        "visible_relation": "vrrqa_visible_relation",
        "spatial_relation": "vrrqa_spatial_relation",
        "entities_visible": "vrrqa_entities_visible",
        "co_visible": "vrrqa_co_visible",
        "co_visible_frame_indices": "vrrqa_co_visible_frame_indices",
        "relation_supported": "vrrqa_relation_supported",
        "relation_votes": "vrrqa_relation_votes",
        "vote_counts": "vrrqa_vote_counts",
        "aggregated_relation": "vrrqa_aggregated_relation",
        "motion_trajectory": "vrrqa_motion_trajectory",
        "temporal_order": "vrrqa_temporal_order",
        "entity_continuity": "vrrqa_entity_continuity",
        "physical_context": "vrrqa_physical_context",
        "inferred_relation": "vrrqa_inferred_relation",
        "option_comparison": "vrrqa_option_comparison",
        "verifier_verdict": "vrrqa_verifier_verdict",
        "needs_more_evidence": "vrrqa_needs_more_evidence",
    }
    for source_key, target_key in field_map.items():
        if source_key in payload:
            metadata[target_key] = payload[source_key]
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


def _image_to_data_url(image_path: str | Path) -> str:
    image_bytes = Path(image_path).read_bytes()
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
