import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

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

ParentVisualSummaryMode = Literal["full", "compact"]

ATOM_PATTERN = re.compile(r"\b[\w.+*/<>=!-]+\b")
CODE_ATOM_PATTERN = re.compile(
    r"\b(?:[a-zA-Z_]\w*|\d+(?:\.\d+)?)\s*(?:==|!=|>=|<=|=|\+|-|\*|/|>|<)\s*"
    r"(?:[a-zA-Z_]\w*|\d+(?:\.\d+)?)\b"
)
QUOTED_TEXT_PATTERN = re.compile(r"[\"']([^\"']{2,80})[\"']")
ROLLUP_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "being",
    "code",
    "does",
    "for",
    "from",
    "has",
    "have",
    "into",
    "its",
    "json",
    "keys",
    "mentions",
    "return",
    "scene",
    "segment",
    "shows",
    "summary",
    "that",
    "the",
    "then",
    "this",
    "time",
    "video",
    "with",
}


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
        visual_span_mode: Literal["scene_and_clip", "clip"] = "scene_and_clip",
        aggregate_child_visual_summaries: bool = False,
        parent_visual_summary_mode: ParentVisualSummaryMode = "full",
        verbose: bool = False,
    ):
        if parent_visual_summary_mode not in {"full", "compact"}:
            raise ValueError(
                "parent_visual_summary_mode must be either 'full' or 'compact', "
                f"got {parent_visual_summary_mode!r}"
            )
        self.speech_recognizer = speech_recognizer
        self.visual_summarizer = visual_summarizer
        self.ocr_extractor = ocr_extractor
        self.audio_extractor = audio_extractor
        self.scene_duration_seconds = scene_duration_seconds
        self.segment_duration_seconds = segment_duration_seconds
        self.clip_duration_seconds = clip_duration_seconds
        self.visual_span_mode = visual_span_mode
        self.aggregate_child_visual_summaries = aggregate_child_visual_summaries
        self.parent_visual_summary_mode = parent_visual_summary_mode
        self.verbose = verbose

    def prepare_artifacts(
        self,
        video_path: str,
        duration_seconds: float,
        video_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PreparedVideoArtifacts:
        if not video_id:
            video_id = Path(video_path).stem

        self._log(
            f"prepare_artifacts start video_id={video_id} "
            f"duration={duration_seconds:.2f}s path={video_path}"
        )
        visual_spans = self._visual_spans(TimeSpan(0.0, duration_seconds))
        self._log(f"visual spans planned count={len(visual_spans)} mode={self.visual_span_mode}")

        self._log("speech recognition start")
        speech_spans = (
            self.speech_recognizer.recognize(video_path) if self.speech_recognizer else []
        )
        self._log(f"speech recognition done spans={len(speech_spans)}")
        self._log("visual summarization start")
        visual_summaries = (
            self.visual_summarizer.summarize(video_path, visual_spans)
            if self.visual_summarizer
            else []
        )
        self._log(f"visual summarization done summaries={len(visual_summaries)}")
        self._log("ocr extraction start")
        ocr_spans = self.ocr_extractor.extract(video_path) if self.ocr_extractor else []
        self._log(f"ocr extraction done spans={len(ocr_spans)}")
        self._log("audio event extraction start")
        audio_events = self.audio_extractor.extract(video_path) if self.audio_extractor else []
        self._log(f"audio event extraction done events={len(audio_events)}")

        payload = dict(metadata or {})
        payload.setdefault("source_video_path", video_path)
        payload.setdefault("duration_seconds", duration_seconds)
        payload.setdefault("visual_span_mode", self.visual_span_mode)
        payload.setdefault(
            "aggregate_child_visual_summaries",
            self.aggregate_child_visual_summaries,
        )
        payload.setdefault("parent_visual_summary_mode", self.parent_visual_summary_mode)
        artifacts = PreparedVideoArtifacts(
            video_id=video_id,
            duration_seconds=duration_seconds,
            speech_spans=speech_spans,
            visual_summaries=visual_summaries,
            ocr_spans=ocr_spans,
            audio_events=audio_events,
            metadata=payload,
        )
        self._log(f"prepare_artifacts done video_id={video_id}")
        return artifacts

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
        self._log(
            f"build_memory start video_id={artifacts.video_id} "
            f"speech={len(artifacts.speech_spans)} "
            f"visual={len(artifacts.visual_summaries)} "
            f"ocr={len(artifacts.ocr_spans)} "
            f"audio={len(artifacts.audio_events)}"
        )
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
        metadata.setdefault("visual_span_mode", self.visual_span_mode)
        metadata.setdefault(
            "aggregate_child_visual_summaries",
            self.aggregate_child_visual_summaries,
        )
        metadata.setdefault("parent_visual_summary_mode", self.parent_visual_summary_mode)
        metadata.setdefault("node_count", len(nodes))
        memory = VideoMemory(
            video_id=artifacts.video_id,
            root_id=root_id,
            nodes=nodes,
            metadata=metadata,
        )
        self._attach_compact_visual_detail_pointers(memory)
        self._log(f"build_memory done video_id={artifacts.video_id} nodes={len(nodes)}")
        return memory

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
        summaries = self._matching_visual_summaries(artifacts, time_span, level)
        speech_spans = self._overlapping_items(artifacts.speech_spans, time_span)
        ocr_spans = self._overlapping_items(artifacts.ocr_spans, time_span)
        audio_events = self._overlapping_items(artifacts.audio_events, time_span)
        tags = sorted({tag for item in summaries for tag in item.tags})
        entities = sorted({entity for item in summaries for entity in item.entities})
        clip_path = self._build_clip_pointer(artifacts, time_span)
        metadata: dict[str, Any] = {}

        if summaries:
            if self._should_compact_parent_visual_summary(artifacts, summaries, level):
                visual_keywords = self._visual_keywords(summaries)
                visual_atoms = self._visual_atoms(summaries)
                tags = sorted(set(tags) | set(visual_keywords))
                visual_summary = self._compact_visual_rollup(
                    summaries=summaries,
                    speech_spans=speech_spans,
                    ocr_spans=ocr_spans,
                    audio_events=audio_events,
                    level=level,
                    visual_keywords=visual_keywords,
                    visual_atoms=visual_atoms,
                )
                metadata.update(
                    {
                        "visual_summary_mode": "compact_parent_rollup",
                        "visual_detail_summary_count": len(summaries),
                        "visual_frame_embedding_count": self._visual_frame_embedding_count(
                            summaries
                        ),
                        "visual_keywords": visual_keywords,
                        "visual_atoms": visual_atoms,
                        "visual_source_granularities": sorted(
                            {summary.granularity for summary in summaries}
                        ),
                    }
                )
            else:
                visual_summary = " | ".join(
                    item.summary.strip() for item in summaries if item.summary
                )
                metadata.update(self._visual_summary_node_metadata(summaries))
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
            metadata=metadata,
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

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[VideoMemory] {message}", flush=True)

    def _visual_spans(self, root_span: TimeSpan) -> list[TimeSpan]:
        clip_spans = self._subdivide(root_span, self.clip_duration_seconds)
        if self.visual_span_mode == "clip":
            return clip_spans
        if self.visual_span_mode == "scene_and_clip":
            scene_spans = self._subdivide(root_span, self.scene_duration_seconds)
            return scene_spans + clip_spans
        raise ValueError(f"Unsupported visual_span_mode: {self.visual_span_mode}")

    def _overlapping_items(self, items: list[Any], span: TimeSpan) -> list[Any]:
        return [item for item in items if item.time_span.overlaps(span)]

    def _matching_visual_summaries(
        self,
        artifacts: PreparedVideoArtifacts,
        span: TimeSpan,
        level: VideoNodeLevel,
    ) -> list[VisualSummarySpan]:
        summaries = artifacts.visual_summaries
        exact = [
            item
            for item in summaries
            if item.granularity == level and self._same_span(item.time_span, span)
        ]
        if exact:
            return exact

        matching = [
            item
            for item in summaries
            if item.granularity == level and item.time_span.overlaps(span)
        ]
        if matching:
            return matching
        aggregate_child_summaries = bool(
            artifacts.metadata.get(
                "aggregate_child_visual_summaries",
                self.aggregate_child_visual_summaries,
            )
        )
        if aggregate_child_summaries and level in {"scene", "segment"}:
            return [
                item
                for item in summaries
                if item.granularity == "clip" and item.time_span.overlaps(span)
            ]
        return []

    def _should_compact_parent_visual_summary(
        self,
        artifacts: PreparedVideoArtifacts,
        summaries: list[VisualSummarySpan],
        level: VideoNodeLevel,
    ) -> bool:
        parent_summary_mode = artifacts.metadata.get(
            "parent_visual_summary_mode",
            self.parent_visual_summary_mode,
        )
        aggregate_child_summaries = bool(
            artifacts.metadata.get(
                "aggregate_child_visual_summaries",
                self.aggregate_child_visual_summaries,
            )
        )
        return (
            parent_summary_mode == "compact"
            and aggregate_child_summaries
            and level in {"scene", "segment"}
            and bool(summaries)
            and all(summary.granularity == "clip" for summary in summaries)
        )

    def _compact_visual_rollup(
        self,
        *,
        summaries: list[VisualSummarySpan],
        speech_spans: list[SpeechSpan],
        ocr_spans: list[OCRSpan],
        audio_events: list[AudioEvent],
        level: VideoNodeLevel,
        visual_keywords: list[str],
        visual_atoms: list[str],
    ) -> str:
        parts = [f"{level} visual rollup"]
        if visual_keywords:
            parts.append(f"topics: {', '.join(visual_keywords[:12])}")
        if visual_atoms:
            parts.append(f"visible/code atoms: {'; '.join(visual_atoms[:8])}")
        parts.append(f"{len(summaries)} detailed clip summaries available")
        if speech_spans:
            parts.append(f"{len(speech_spans)} speech spans")
        if ocr_spans:
            parts.append(f"{len(ocr_spans)} OCR spans")
        if audio_events:
            parts.append(f"{len(audio_events)} audio events")
        return "; ".join(parts)

    def _visual_keywords(self, summaries: list[VisualSummarySpan], limit: int = 24) -> list[str]:
        counts: dict[str, int] = {}
        for summary in summaries:
            parts = [summary.summary, " ".join(summary.tags), " ".join(summary.entities)]
            for token in ATOM_PATTERN.findall(" ".join(parts).lower()):
                normalized = token.strip("`.,:;()[]{}")
                if (
                    len(normalized) <= 2
                    or normalized in ROLLUP_STOPWORDS
                    or normalized.replace(".", "", 1).isdigit()
                ):
                    continue
                counts[normalized] = counts.get(normalized, 0) + 1
        return [
            token
            for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
        ]

    def _visual_atoms(self, summaries: list[VisualSummarySpan], limit: int = 16) -> list[str]:
        atoms: list[str] = []
        seen: set[str] = set()
        for summary in summaries:
            candidates = [
                *QUOTED_TEXT_PATTERN.findall(summary.summary),
                *CODE_ATOM_PATTERN.findall(summary.summary),
            ]
            for candidate in candidates:
                normalized = " ".join(candidate.split()).strip("`.,:;()[]{}")
                key = normalized.lower()
                if len(normalized) < 2 or key in seen:
                    continue
                seen.add(key)
                atoms.append(normalized)
                if len(atoms) >= limit:
                    return atoms
        return atoms

    def _visual_summary_node_metadata(
        self,
        summaries: list[VisualSummarySpan],
    ) -> dict[str, Any]:
        if len(summaries) != 1:
            return {}
        return dict(summaries[0].metadata)

    def _visual_frame_embedding_count(self, summaries: list[VisualSummarySpan]) -> int:
        total = 0
        for summary in summaries:
            embeddings = summary.metadata.get("pitome_frame_embeddings", [])
            if isinstance(embeddings, list):
                total += len(embeddings)
        return total

    def _attach_compact_visual_detail_pointers(self, memory: VideoMemory) -> None:
        for node in memory.nodes.values():
            if node.metadata.get("visual_summary_mode") != "compact_parent_rollup":
                continue
            detail_node_ids = [
                descendant.node_id
                for descendant in self._descendant_nodes(memory, node.node_id)
                if descendant.level == "clip" and descendant.visual_summary.strip()
            ]
            if detail_node_ids:
                node.metadata["visual_detail_node_ids"] = detail_node_ids

    def _descendant_nodes(self, memory: VideoMemory, node_id: str) -> list[VideoNode]:
        descendants: list[VideoNode] = []
        stack = list(memory.get_node(node_id).children)
        while stack:
            current_id = stack.pop(0)
            current = memory.get_node(current_id)
            descendants.append(current)
            stack.extend(current.children)
        return descendants

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
