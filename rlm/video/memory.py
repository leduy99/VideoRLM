import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rlm.video.adapters import (
    AudioEventExtractor,
    OCRExtractor,
    SpeechRecognizer,
    VisualSummarizer,
)
from rlm.video.types import (
    AudioEvent,
    OCRSpan,
    SpeechSpan,
    TimeSpan,
    VideoMemory,
    VideoNode,
    VideoNodeLevel,
    VisualSummarySpan,
)


@dataclass
class PreparedVideoArtifacts:
    video_id: str
    duration_seconds: float
    speech_spans: list[SpeechSpan] = field(default_factory=list)
    visual_summaries: list[VisualSummarySpan] = field(default_factory=list)
    ocr_spans: list[OCRSpan] = field(default_factory=list)
    audio_events: list[AudioEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "duration_seconds": self.duration_seconds,
            "speech_spans": [item.to_dict() for item in self.speech_spans],
            "visual_summaries": [item.to_dict() for item in self.visual_summaries],
            "ocr_spans": [item.to_dict() for item in self.ocr_spans],
            "audio_events": [item.to_dict() for item in self.audio_events],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreparedVideoArtifacts":
        return cls(
            video_id=data["video_id"],
            duration_seconds=float(data["duration_seconds"]),
            speech_spans=[SpeechSpan.from_dict(item) for item in data.get("speech_spans", [])],
            visual_summaries=[
                VisualSummarySpan.from_dict(item) for item in data.get("visual_summaries", [])
            ],
            ocr_spans=[OCRSpan.from_dict(item) for item in data.get("ocr_spans", [])],
            audio_events=[AudioEvent.from_dict(item) for item in data.get("audio_events", [])],
            metadata=dict(data.get("metadata", {})),
        )


class VideoMemoryBuilder:
    """
    Builds hierarchical VideoMemory from either prepared artifacts or pluggable extractors.

    The builder intentionally avoids hard dependencies on video-processing packages so the
    repository can stay lightweight. In practice, teams can plug in Qwen-VL, Qwen-ASR, or
    other external services behind these adapter protocols.
    """

    def __init__(
        self,
        speech_recognizer: SpeechRecognizer | None = None,
        visual_summarizer: VisualSummarizer | None = None,
        ocr_extractor: OCRExtractor | None = None,
        audio_extractor: AudioEventExtractor | None = None,
        scene_duration_seconds: float = 180.0,
        segment_duration_seconds: float = 45.0,
        clip_duration_seconds: float = 15.0,
    ):
        self.speech_recognizer = speech_recognizer
        self.visual_summarizer = visual_summarizer
        self.ocr_extractor = ocr_extractor
        self.audio_extractor = audio_extractor
        self.scene_duration_seconds = scene_duration_seconds
        self.segment_duration_seconds = segment_duration_seconds
        self.clip_duration_seconds = clip_duration_seconds

    def prepare_artifacts(
        self,
        video_path: str,
        duration_seconds: float,
        video_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PreparedVideoArtifacts:
        if not video_id:
            video_id = Path(video_path).stem

        scene_spans = self._subdivide(TimeSpan(0.0, duration_seconds), self.scene_duration_seconds)
        clip_spans = self._subdivide(TimeSpan(0.0, duration_seconds), self.clip_duration_seconds)
        visual_spans = scene_spans + clip_spans

        speech_spans = (
            self.speech_recognizer.recognize(video_path) if self.speech_recognizer else []
        )
        visual_summaries = (
            self.visual_summarizer.summarize(video_path, visual_spans)
            if self.visual_summarizer
            else []
        )
        ocr_spans = self.ocr_extractor.extract(video_path) if self.ocr_extractor else []
        audio_events = self.audio_extractor.extract(video_path) if self.audio_extractor else []

        payload = dict(metadata or {})
        payload.setdefault("source_video_path", video_path)
        payload.setdefault("duration_seconds", duration_seconds)
        return PreparedVideoArtifacts(
            video_id=video_id,
            duration_seconds=duration_seconds,
            speech_spans=speech_spans,
            visual_summaries=visual_summaries,
            ocr_spans=ocr_spans,
            audio_events=audio_events,
            metadata=payload,
        )

    def build(
        self,
        video_path: str,
        duration_seconds: float,
        video_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VideoMemory:
        artifacts = self.prepare_artifacts(
            video_path=video_path,
            duration_seconds=duration_seconds,
            video_id=video_id,
            metadata=metadata,
        )
        return self.build_from_artifacts(artifacts)

    def build_from_artifacts(self, artifacts: PreparedVideoArtifacts) -> VideoMemory:
        root_span = TimeSpan(0.0, artifacts.duration_seconds)
        root_id = f"{artifacts.video_id}_video"
        nodes: dict[str, VideoNode] = {}

        root_node = self._make_node(
            artifacts=artifacts,
            node_id=root_id,
            level="video",
            time_span=root_span,
            parent_id=None,
        )
        nodes[root_id] = root_node

        for scene_index, scene_span in enumerate(
            self._subdivide(root_span, self.scene_duration_seconds), start=1
        ):
            scene_id = f"{artifacts.video_id}_scene_{scene_index:03d}"
            nodes[scene_id] = self._make_node(
                artifacts=artifacts,
                node_id=scene_id,
                level="scene",
                time_span=scene_span,
                parent_id=root_id,
            )
            nodes[root_id].children.append(scene_id)

            for segment_index, segment_span in enumerate(
                self._subdivide(scene_span, self.segment_duration_seconds), start=1
            ):
                segment_id = f"{scene_id}_seg_{segment_index:03d}"
                nodes[segment_id] = self._make_node(
                    artifacts=artifacts,
                    node_id=segment_id,
                    level="segment",
                    time_span=segment_span,
                    parent_id=scene_id,
                )
                nodes[scene_id].children.append(segment_id)

                for clip_index, clip_span in enumerate(
                    self._subdivide(segment_span, self.clip_duration_seconds), start=1
                ):
                    clip_id = f"{segment_id}_clip_{clip_index:03d}"
                    nodes[clip_id] = self._make_node(
                        artifacts=artifacts,
                        node_id=clip_id,
                        level="clip",
                        time_span=clip_span,
                        parent_id=segment_id,
                    )
                    nodes[segment_id].children.append(clip_id)

        metadata = dict(artifacts.metadata)
        metadata.setdefault("duration_seconds", artifacts.duration_seconds)
        metadata.setdefault("scene_duration_seconds", self.scene_duration_seconds)
        metadata.setdefault("segment_duration_seconds", self.segment_duration_seconds)
        metadata.setdefault("clip_duration_seconds", self.clip_duration_seconds)
        metadata.setdefault("node_count", len(nodes))
        return VideoMemory(
            video_id=artifacts.video_id,
            root_id=root_id,
            nodes=nodes,
            metadata=metadata,
        )

    def save_memory(self, memory: VideoMemory, path: str | Path) -> None:
        output_path = Path(path)
        output_path.write_text(json.dumps(memory.to_dict(), indent=2), encoding="utf-8")

    def load_memory(self, path: str | Path) -> VideoMemory:
        input_path = Path(path)
        return VideoMemory.from_dict(json.loads(input_path.read_text(encoding="utf-8")))

    def save_artifacts(self, artifacts: PreparedVideoArtifacts, path: str | Path) -> None:
        output_path = Path(path)
        output_path.write_text(json.dumps(artifacts.to_dict(), indent=2), encoding="utf-8")

    def load_artifacts(self, path: str | Path) -> PreparedVideoArtifacts:
        input_path = Path(path)
        return PreparedVideoArtifacts.from_dict(json.loads(input_path.read_text(encoding="utf-8")))

    def save_artifacts_dir(self, artifacts: PreparedVideoArtifacts, directory: str | Path) -> Path:
        from rlm.video.artifact_store import PreparedArtifactStore

        return PreparedArtifactStore().save(artifacts, directory)

    def load_artifacts_dir(self, directory: str | Path) -> PreparedVideoArtifacts:
        from rlm.video.artifact_store import PreparedArtifactStore

        return PreparedArtifactStore().load(directory)

    def _make_node(
        self,
        artifacts: PreparedVideoArtifacts,
        node_id: str,
        level: VideoNodeLevel,
        time_span: TimeSpan,
        parent_id: str | None,
    ) -> VideoNode:
        summaries = self._matching_visual_summaries(artifacts.visual_summaries, time_span, level)
        speech_spans = self._overlapping_items(artifacts.speech_spans, time_span)
        ocr_spans = self._overlapping_items(artifacts.ocr_spans, time_span)
        audio_events = self._overlapping_items(artifacts.audio_events, time_span)
        tags = sorted({tag for item in summaries for tag in item.tags})
        entities = sorted({entity for item in summaries for entity in item.entities})
        clip_path = self._build_clip_pointer(artifacts, time_span)

        if summaries:
            visual_summary = " | ".join(item.summary.strip() for item in summaries if item.summary)
        else:
            visual_summary = self._fallback_visual_summary(speech_spans, ocr_spans, level)

        return VideoNode(
            node_id=node_id,
            level=level,
            time_span=time_span,
            visual_summary=visual_summary,
            speech_spans=speech_spans,
            ocr_spans=ocr_spans,
            audio_events=audio_events,
            tags=tags,
            entities=entities,
            clip_path=clip_path,
            parent_id=parent_id,
        )

    def _subdivide(self, span: TimeSpan, window_seconds: float) -> list[TimeSpan]:
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be positive, got {window_seconds}")

        spans: list[TimeSpan] = []
        cursor = span.start
        while cursor < span.end:
            next_end = min(cursor + window_seconds, span.end)
            spans.append(TimeSpan(cursor, next_end))
            cursor = next_end
        return spans

    def _overlapping_items(self, items: list[Any], span: TimeSpan) -> list[Any]:
        return [item for item in items if item.time_span.overlaps(span)]

    def _matching_visual_summaries(
        self,
        summaries: list[VisualSummarySpan],
        span: TimeSpan,
        level: VideoNodeLevel,
    ) -> list[VisualSummarySpan]:
        exact = [
            item
            for item in summaries
            if item.granularity == level and self._same_span(item.time_span, span)
        ]
        if exact:
            return exact

        return [item for item in summaries if item.granularity == level and item.time_span.overlaps(span)]

    def _same_span(self, left: TimeSpan, right: TimeSpan, tol: float = 1e-6) -> bool:
        return abs(left.start - right.start) <= tol and abs(left.end - right.end) <= tol

    def _build_clip_pointer(self, artifacts: PreparedVideoArtifacts, span: TimeSpan) -> str | None:
        source = artifacts.metadata.get("source_video_path")
        if not source:
            return None
        return f"{source}#t={span.start:.2f},{span.end:.2f}"

    def _fallback_visual_summary(
        self,
        speech_spans: list[SpeechSpan],
        ocr_spans: list[OCRSpan],
        level: VideoNodeLevel,
    ) -> str:
        parts = [f"{level} node"]
        if speech_spans:
            parts.append(f"{len(speech_spans)} speech spans")
        if ocr_spans:
            parts.append(f"{len(ocr_spans)} OCR spans")
        return ", ".join(parts)
