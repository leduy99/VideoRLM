import json
import re
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from rlm.video.adapters import (
    AudioEventExtractor,
    OCRExtractor,
    SpeechRecognizer,
    VisualSummarizer,
)
from rlm.video.gpu_memory import unload_component
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
CODE_ASSIGNMENT_LINE_PATTERN = re.compile(
    r"\b[a-zA-Z_]\w*\s*=\s*(?:[a-zA-Z_]\w*|\d+(?:\.\d+)?|True|False|\"[^\"]*\"|'[^']*')"
    r"(?:\s*(?:==|!=|>=|<=|>|<|\+|\-|\*|/)\s*"
    r"(?:[a-zA-Z_]\w*|\d+(?:\.\d+)?|True|False|\"[^\"]*\"|'[^']*'))?"
)
COMPARISON_OPERATOR_TEXT_PATTERN = re.compile(
    r"==|!=|>=|<=|(?<![A-Za-z0-9])>(?![A-Za-z0-9])|(?<![A-Za-z0-9])<(?![A-Za-z0-9])|"
    r"\bequal(?:s|ity)?\b|\bnot\s+equal\b|\bgreater\s+than\b|\bless\s+than\b",
    flags=re.IGNORECASE,
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
COGNITIVE_EVENT_BOUNDARY_THRESHOLD = 0.55
COGNITIVE_EVENT_BOUNDARY_TOLERANCE_SECONDS = 1.0
COGNITIVE_EVENT_FRAME_EMBEDDING_LIMIT = 32
CAUSAL_OUTCOME_TERMS = {
    "because",
    "caused",
    "effect",
    "result",
    "so",
    "therefore",
    "why",
}
CAUSAL_SETUP_TERMS = {
    "attempt",
    "begin",
    "need",
    "problem",
    "start",
    "try",
    "want",
    "worried",
}
CAUSAL_RESOLUTION_TERMS = {
    "complete",
    "done",
    "fix",
    "fixed",
    "finish",
    "resolve",
    "serve",
    "solve",
}
ACTION_START_TERMS = {
    "enter",
    "grab",
    "hold",
    "lift",
    "open",
    "pick",
    "reach",
    "take",
}
ACTION_COMPLETION_TERMS = {
    "close",
    "drink",
    "drop",
    "leave",
    "place",
    "pour",
    "put",
    "serve",
    "sit",
    "walk",
}


@dataclass
class CognitiveEventGroup:
    clip_node_ids: list[str]
    time_span: TimeSpan
    split_scores: list[float] = field(default_factory=list)


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
        enable_fine_speech_windows: bool = False,
        fine_speech_window_seconds: float = 15.0,
        fine_speech_window_stride_seconds: float = 5.0,
        visual_span_mode: Literal["scene_and_clip", "clip"] = "scene_and_clip",
        aggregate_child_visual_summaries: bool = False,
        parent_visual_summary_mode: ParentVisualSummaryMode = "full",
        enable_cognitive_events: bool = True,
        cognitive_event_boundary_threshold: float = COGNITIVE_EVENT_BOUNDARY_THRESHOLD,
        cognitive_event_max_duration_seconds: float | None = None,
        offload_components_after_phase: bool = False,
        verbose: bool = False,
    ):
        if parent_visual_summary_mode not in {"full", "compact"}:
            raise ValueError(
                "parent_visual_summary_mode must be either 'full' or 'compact', "
                f"got {parent_visual_summary_mode!r}"
            )
        if not 0.0 <= cognitive_event_boundary_threshold <= 1.0:
            raise ValueError(
                "cognitive_event_boundary_threshold must be in [0, 1], "
                f"got {cognitive_event_boundary_threshold!r}"
            )
        if (
            cognitive_event_max_duration_seconds is not None
            and cognitive_event_max_duration_seconds <= 0
        ):
            raise ValueError(
                "cognitive_event_max_duration_seconds must be positive when set, "
                f"got {cognitive_event_max_duration_seconds!r}"
            )
        self.speech_recognizer = speech_recognizer
        self.visual_summarizer = visual_summarizer
        self.ocr_extractor = ocr_extractor
        self.audio_extractor = audio_extractor
        self.scene_duration_seconds = scene_duration_seconds
        self.segment_duration_seconds = segment_duration_seconds
        self.clip_duration_seconds = clip_duration_seconds
        self.enable_fine_speech_windows = enable_fine_speech_windows
        self.fine_speech_window_seconds = fine_speech_window_seconds
        self.fine_speech_window_stride_seconds = fine_speech_window_stride_seconds
        if self.fine_speech_window_seconds <= 0:
            raise ValueError(
                "fine_speech_window_seconds must be positive, "
                f"got {fine_speech_window_seconds}"
            )
        if self.fine_speech_window_stride_seconds <= 0:
            raise ValueError(
                "fine_speech_window_stride_seconds must be positive, "
                f"got {fine_speech_window_stride_seconds}"
            )
        self.visual_span_mode = visual_span_mode
        self.aggregate_child_visual_summaries = aggregate_child_visual_summaries
        self.parent_visual_summary_mode = parent_visual_summary_mode
        self.enable_cognitive_events = enable_cognitive_events
        self.cognitive_event_boundary_threshold = cognitive_event_boundary_threshold
        self.cognitive_event_max_duration_seconds = cognitive_event_max_duration_seconds
        self.offload_components_after_phase = offload_components_after_phase
        self.verbose = verbose

    def prepare_artifacts(
        self,
        video_path: str,
        duration_seconds: float,
        video_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
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
        with _temporary_progress_callback(self.speech_recognizer, progress_callback):
            speech_spans = (
                self.speech_recognizer.recognize(video_path) if self.speech_recognizer else []
            )
        self._offload_component("speech recognizer", self.speech_recognizer)
        self._log(f"speech recognition done spans={len(speech_spans)}")
        self._log("visual summarization start")
        with _temporary_progress_callback(self.visual_summarizer, progress_callback):
            visual_summaries = (
                self.visual_summarizer.summarize(video_path, visual_spans)
                if self.visual_summarizer
                else []
            )
        self._offload_component("visual summarizer", self.visual_summarizer)
        self._log(f"visual summarization done summaries={len(visual_summaries)}")
        self._log("ocr extraction start")
        ocr_spans = self.ocr_extractor.extract(video_path) if self.ocr_extractor else []
        self._offload_component("ocr extractor", self.ocr_extractor)
        self._log(f"ocr extraction done spans={len(ocr_spans)}")
        self._log("audio event extraction start")
        audio_events = self.audio_extractor.extract(video_path) if self.audio_extractor else []
        self._offload_component("audio extractor", self.audio_extractor)
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
        payload.setdefault("enable_cognitive_events", self.enable_cognitive_events)
        payload.setdefault(
            "cognitive_event_boundary_threshold",
            self.cognitive_event_boundary_threshold,
        )
        if self.cognitive_event_max_duration_seconds is not None:
            payload.setdefault(
                "cognitive_event_max_duration_seconds",
                self.cognitive_event_max_duration_seconds,
            )
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

    def _offload_component(self, name: str, component: object | None) -> None:
        if not self.offload_components_after_phase:
            return
        if unload_component(component):
            self._log(f"offloaded {name}")

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
            progress_callback=None,
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
        build_cognitive_events = self._should_build_cognitive_events(artifacts)
        cognitive_event_count = 0

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

                segment_clip_ids: list[str] = []
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
                    segment_clip_ids.append(clip_id)

                self._add_cross_modal_cognitive_anchor_signals(
                    [nodes[clip_id] for clip_id in segment_clip_ids]
                )

                if build_cognitive_events:
                    event_groups = self._build_cognitive_event_groups(
                        segment_span=segment_span,
                        clip_nodes=[nodes[clip_id] for clip_id in segment_clip_ids],
                    )
                    for event_index, event_group in enumerate(event_groups, start=1):
                        event_id = f"{segment_id}_event_{event_index:03d}"
                        event_node = self._make_cognitive_event_node(
                            artifacts=artifacts,
                            node_id=event_id,
                            time_span=event_group.time_span,
                            parent_id=segment_id,
                            child_nodes=[
                                nodes[clip_id] for clip_id in event_group.clip_node_ids
                            ],
                            split_scores=event_group.split_scores,
                        )
                        nodes[event_id] = event_node
                        nodes[segment_id].children.append(event_id)
                        for clip_id in event_group.clip_node_ids:
                            nodes[clip_id].parent_id = event_id
                            nodes[event_id].children.append(clip_id)
                        cognitive_event_count += 1
                else:
                    nodes[segment_id].children.extend(segment_clip_ids)

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
        metadata.setdefault("enable_cognitive_events", self.enable_cognitive_events)
        metadata.setdefault(
            "cognitive_event_boundary_threshold",
            self.cognitive_event_boundary_threshold,
        )
        if self.cognitive_event_max_duration_seconds is not None:
            metadata.setdefault(
                "cognitive_event_max_duration_seconds",
                self.cognitive_event_max_duration_seconds,
            )
        metadata.setdefault("fine_speech_windows_enabled", self.enable_fine_speech_windows)
        metadata.setdefault("fine_speech_window_seconds", self.fine_speech_window_seconds)
        metadata.setdefault(
            "fine_speech_window_stride_seconds",
            self.fine_speech_window_stride_seconds,
        )
        metadata.setdefault("speech_spans_clipped_to_node_intervals", True)
        metadata["cognitive_events_built"] = build_cognitive_events
        metadata["cognitive_event_count"] = cognitive_event_count
        memory = VideoMemory(
            video_id=artifacts.video_id,
            root_id=root_id,
            nodes=nodes,
            metadata=metadata,
        )
        from rlm.video.cross_modal_temporal import build_cross_modal_temporal_index

        memory.cross_modal_index = build_cross_modal_temporal_index(artifacts, nodes)
        memory.metadata["cross_modal_temporal_index"] = {
            "enabled": True,
            "event_count": memory.cross_modal_index.metadata.get("event_count", 0),
            "section_count": memory.cross_modal_index.metadata.get("section_count", 0),
            "code_snapshot_count": memory.cross_modal_index.metadata.get(
                "code_snapshot_count",
                0,
            ),
            "operator_event_count": memory.cross_modal_index.metadata.get(
                "operator_event_count",
                0,
            ),
        }
        self._attach_temporal_occurrence_metadata(memory)
        self._ensure_fine_speech_windows(memory, artifacts.speech_spans)
        self._link_cognitive_events(memory)
        self._attach_compact_visual_detail_pointers(memory)
        memory.metadata["node_count"] = len(memory.nodes)
        self._log(f"build_memory done video_id={artifacts.video_id} nodes={len(nodes)}")
        return memory

    def save_memory(self, memory: VideoMemory, path: str | Path) -> None:
        output_path = Path(path)
        output_path.write_text(json.dumps(memory.to_dict(), indent=2), encoding="utf-8")

    def load_memory(self, path: str | Path) -> VideoMemory:
        input_path = Path(path)
        memory = VideoMemory.from_dict(json.loads(input_path.read_text(encoding="utf-8")))
        self._clip_memory_speech_spans_to_nodes(memory)
        self._ensure_fine_speech_windows(memory)
        memory.metadata["node_count"] = len(memory.nodes)
        return memory

    def memory_matches_builder_config(self, memory: VideoMemory) -> bool:
        expected_float_values = {
            "scene_duration_seconds": self.scene_duration_seconds,
            "segment_duration_seconds": self.segment_duration_seconds,
            "clip_duration_seconds": self.clip_duration_seconds,
        }
        for key, expected in expected_float_values.items():
            actual = memory.metadata.get(key)
            if actual is None or abs(float(actual) - float(expected)) > 1e-6:
                return False
        fine_enabled = bool(memory.metadata.get("fine_speech_windows_enabled", False))
        if fine_enabled != self.enable_fine_speech_windows:
            return False
        if self.enable_fine_speech_windows:
            fine_config = {
                "fine_speech_window_seconds": self.fine_speech_window_seconds,
                "fine_speech_window_stride_seconds": self.fine_speech_window_stride_seconds,
            }
            for key, expected in fine_config.items():
                actual = memory.metadata.get(key)
                if actual is None or abs(float(actual) - float(expected)) > 1e-6:
                    return False
        return True

    def _ensure_fine_speech_windows(
        self,
        memory: VideoMemory,
        source_spans: list[SpeechSpan] | None = None,
    ) -> None:
        if not self.enable_fine_speech_windows:
            return
        duration_seconds = float(memory.metadata.get("duration_seconds") or 0.0)
        if duration_seconds <= 0:
            root = memory.get_node(memory.root_id)
            duration_seconds = root.time_span.end
        if duration_seconds <= 0:
            return

        expected_config = {
            "fine_speech_window_seconds": self.fine_speech_window_seconds,
            "fine_speech_window_stride_seconds": self.fine_speech_window_stride_seconds,
        }
        current_config = {
            key: memory.metadata.get(key)
            for key in (
                "fine_speech_window_seconds",
                "fine_speech_window_stride_seconds",
            )
        }
        existing = [
            node_id
            for node_id, node in memory.nodes.items()
            if node.metadata.get("speech_window_kind") == "fine_asr_window"
        ]
        if existing and current_config == expected_config:
            return
        if existing:
            self._remove_fine_speech_window_nodes(memory, existing)

        spans = source_spans if source_spans is not None else self._base_speech_spans(memory)
        spans = [span for span in spans if span.text.strip()]
        if not spans:
            memory.metadata.update(
                {
                    "fine_speech_windows_enabled": True,
                    **expected_config,
                    "fine_speech_window_count": 0,
                }
            )
            return

        windows = self._overlapping_time_windows(
            TimeSpan(0.0, duration_seconds),
            window_seconds=self.fine_speech_window_seconds,
            stride_seconds=self.fine_speech_window_stride_seconds,
        )
        fine_nodes: list[VideoNode] = []
        for window in windows:
            text, source_refs = self._speech_text_for_window(spans, window)
            if not text:
                continue
            context_node = self._fine_speech_context_node(memory, window)
            node_id = f"{memory.video_id}_asrwin_{len(fine_nodes) + 1:05d}"
            speech_span = SpeechSpan(
                text=text,
                time_span=window,
                metadata={
                    "speech_window_kind": "fine_asr_window",
                    "source_span_refs": source_refs,
                },
            )
            metadata: dict[str, Any] = {
                "speech_window_kind": "fine_asr_window",
                "speech_window_index": len(fine_nodes) + 1,
                "speech_window_duration_seconds": self.fine_speech_window_seconds,
                "speech_window_stride_seconds": self.fine_speech_window_stride_seconds,
                "retrieval_only": True,
                "source_span_refs": source_refs,
            }
            if context_node is not None:
                metadata.update(
                    {
                        "context_node_id": context_node.node_id,
                        "context_node_level": context_node.level,
                        "context_time_span": context_node.time_span.to_dict(),
                        "context_text": self._node_speech_text(context_node, max_chars=2200),
                    }
                )
                metadata.update(
                    {
                        key: value
                        for key, value in context_node.metadata.items()
                        if key
                        in {
                            "temporal_section_tags",
                            "content_section_tags",
                            "section_tags",
                            "speech_occurrences",
                            "temporal_occurrences",
                        }
                    }
                )
            node = VideoNode(
                node_id=node_id,
                level="clip",
                time_span=window,
                visual_summary="",
                speech_spans=[speech_span],
                ocr_spans=[],
                audio_events=[],
                tags=[],
                entities=[],
                clip_path=None,
                parent_id=context_node.node_id if context_node is not None else None,
                metadata=metadata,
            )
            memory.nodes[node_id] = node
            fine_nodes.append(node)
            if context_node is not None:
                context_node.metadata.setdefault("fine_speech_window_node_ids", []).append(
                    node_id
                )

        self._link_fine_speech_windows(fine_nodes)
        memory.metadata.update(
            {
                "fine_speech_windows_enabled": True,
                **expected_config,
                "fine_speech_window_count": len(fine_nodes),
                "fine_speech_window_node_ids": [node.node_id for node in fine_nodes],
            }
        )
        self._log(f"fine speech windows built count={len(fine_nodes)}")

    def _remove_fine_speech_window_nodes(
        self,
        memory: VideoMemory,
        node_ids: list[str],
    ) -> None:
        node_id_set = set(node_ids)
        for node in memory.nodes.values():
            node.children = [child_id for child_id in node.children if child_id not in node_id_set]
            existing = node.metadata.get("fine_speech_window_node_ids")
            if isinstance(existing, list):
                node.metadata["fine_speech_window_node_ids"] = [
                    item for item in existing if item not in node_id_set
                ]
        for node_id in node_ids:
            memory.nodes.pop(node_id, None)

    def _base_speech_spans(self, memory: VideoMemory) -> list[SpeechSpan]:
        root = memory.get_node(memory.root_id)
        if root.speech_spans:
            return list(root.speech_spans)
        seen: set[tuple[float, float, str]] = set()
        spans: list[SpeechSpan] = []
        for node in memory.nodes.values():
            if node.metadata.get("speech_window_kind") == "fine_asr_window":
                continue
            for span in node.speech_spans:
                key = (round(span.time_span.start, 3), round(span.time_span.end, 3), span.text)
                if key in seen:
                    continue
                seen.add(key)
                spans.append(span)
        spans.sort(key=lambda item: (item.time_span.start, item.time_span.end, item.text))
        return spans

    def _overlapping_time_windows(
        self,
        span: TimeSpan,
        *,
        window_seconds: float,
        stride_seconds: float,
    ) -> list[TimeSpan]:
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be positive, got {window_seconds}")
        if stride_seconds <= 0:
            raise ValueError(f"stride_seconds must be positive, got {stride_seconds}")
        windows: list[TimeSpan] = []
        cursor = span.start
        while cursor < span.end:
            window_end = min(span.end, cursor + window_seconds)
            if window_end - cursor >= 0.25:
                windows.append(TimeSpan(cursor, window_end))
            if window_end >= span.end:
                break
            cursor += stride_seconds
        return windows

    def _speech_text_for_window(
        self,
        spans: list[SpeechSpan],
        window: TimeSpan,
    ) -> tuple[str, list[dict[str, Any]]]:
        pieces: list[str] = []
        source_refs: list[dict[str, Any]] = []
        for span in spans:
            if not span.time_span.overlaps(window):
                continue
            overlap_start = max(span.time_span.start, window.start)
            overlap_end = min(span.time_span.end, window.end)
            if overlap_end - overlap_start <= 0.05:
                continue
            text = self._speech_text_slice_for_overlap(span, overlap_start, overlap_end)
            if not text:
                continue
            pieces.append(text)
            source_refs.append(
                {
                    "source_time_span": span.time_span.to_dict(),
                    "overlap_time_span": TimeSpan(overlap_start, overlap_end).to_dict(),
                    "speaker": span.speaker,
                    "language": span.language,
                }
            )
        return self._dedupe_joined_text(pieces), source_refs

    def _speech_text_slice_for_overlap(
        self,
        span: SpeechSpan,
        overlap_start: float,
        overlap_end: float,
    ) -> str:
        text = " ".join(span.text.split()).strip()
        if not text:
            return ""
        duration = max(span.time_span.duration, 1e-6)
        words = text.split()
        if duration <= max(self.fine_speech_window_seconds * 1.5, 1.0) or len(words) <= 24:
            return text
        start_ratio = max(0.0, min(1.0, (overlap_start - span.time_span.start) / duration))
        end_ratio = max(0.0, min(1.0, (overlap_end - span.time_span.start) / duration))
        start_index = max(0, int(len(words) * start_ratio) - 2)
        end_index = min(len(words), int(len(words) * end_ratio + 0.999) + 2)
        if end_index <= start_index:
            end_index = min(len(words), start_index + 1)
        return " ".join(words[start_index:end_index]).strip()

    def _dedupe_joined_text(self, pieces: list[str]) -> str:
        seen: set[str] = set()
        deduped: list[str] = []
        for piece in pieces:
            normalized = " ".join(piece.split()).strip()
            if not normalized:
                continue
            key = re.sub(r"\W+", "", normalized.casefold())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(normalized)
        return " ".join(deduped).strip()

    def _fine_speech_context_node(
        self,
        memory: VideoMemory,
        window: TimeSpan,
    ) -> VideoNode | None:
        candidates = [
            node
            for node in memory.nodes.values()
            if node.level in {"clip", "event", "segment"}
            and node.metadata.get("speech_window_kind") != "fine_asr_window"
            and node.time_span.overlaps(window)
        ]
        if not candidates:
            return None
        level_rank = {"clip": 0, "event": 1, "segment": 2}

        def rank(node: VideoNode) -> tuple[int, float, float, str]:
            overlap = min(node.time_span.end, window.end) - max(node.time_span.start, window.start)
            return (
                level_rank.get(node.level, 3),
                -overlap,
                node.time_span.duration,
                node.node_id,
            )

        return sorted(candidates, key=rank)[0]

    def _node_speech_text(self, node: VideoNode, max_chars: int | None = None) -> str:
        text = " ".join(span.text.strip() for span in node.speech_spans if span.text).strip()
        normalized = " ".join(text.split())
        if max_chars is not None and len(normalized) > max_chars:
            return normalized[:max_chars].rsplit(" ", maxsplit=1)[0]
        return normalized

    def _link_fine_speech_windows(self, fine_nodes: list[VideoNode]) -> None:
        ordered = sorted(fine_nodes, key=lambda node: (node.time_span.start, node.node_id))
        for index, node in enumerate(ordered):
            previous_id = ordered[index - 1].node_id if index > 0 else None
            next_id = ordered[index + 1].node_id if index + 1 < len(ordered) else None
            context_window_ids = [
                item
                for item in (previous_id, node.node_id, next_id)
                if item is not None
            ]
            node.metadata["previous_speech_window_node_id"] = previous_id
            node.metadata["next_speech_window_node_id"] = next_id
            node.metadata["context_window_node_ids"] = context_window_ids

    def _clip_memory_speech_spans_to_nodes(
        self,
        memory: VideoMemory,
        source_spans: list[SpeechSpan] | None = None,
    ) -> None:
        if memory.metadata.get("speech_spans_clipped_to_node_intervals") is True:
            return
        root = memory.get_node(memory.root_id)
        base_spans = source_spans if source_spans is not None else list(root.speech_spans)
        if not base_spans:
            base_spans = self._base_speech_spans(memory)
        if not base_spans:
            return
        for node in memory.nodes.values():
            if node.level == "video" or node.metadata.get("speech_window_kind") == "fine_asr_window":
                continue
            node.speech_spans = self._speech_spans_for_node(base_spans, node.time_span, node.level)
        for node in memory.nodes.values():
            if node.metadata.get("speech_window_kind") != "fine_asr_window":
                continue
            context_node_id = str(node.metadata.get("context_node_id") or "")
            context_node = memory.nodes.get(context_node_id) if context_node_id else None
            if context_node is not None:
                node.metadata["context_text"] = self._node_speech_text(context_node, max_chars=2200)
        memory.metadata["speech_spans_clipped_to_node_intervals"] = True

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
        speech_spans = self._speech_spans_for_node(artifacts.speech_spans, time_span, level)
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

        metadata.update(
            self._section_metadata(
                artifacts=artifacts,
                level=level,
                time_span=time_span,
                visual_summary=visual_summary,
                speech_spans=speech_spans,
                ocr_spans=ocr_spans,
            )
        )

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

    def _section_metadata(
        self,
        *,
        artifacts: PreparedVideoArtifacts,
        level: VideoNodeLevel,
        time_span: TimeSpan,
        visual_summary: str,
        speech_spans: list[SpeechSpan],
        ocr_spans: list[OCRSpan],
    ) -> dict[str, Any]:
        temporal_tags = self._temporal_section_tags(artifacts, level, time_span)
        content_tags = self._content_section_tags(
            visual_summary=visual_summary,
            speech_spans=speech_spans,
            ocr_spans=ocr_spans,
        )
        tags = _dedupe_strings([*temporal_tags, *content_tags])
        if not tags:
            return {}
        return {
            "temporal_section_tags": temporal_tags,
            "content_section_tags": content_tags,
            "section_tags": tags,
        }

    def _attach_temporal_occurrence_metadata(self, memory: VideoMemory) -> None:
        temporal_index = memory.cross_modal_index
        if temporal_index is None:
            return
        for event in temporal_index.all_events():
            if event.source_node_id is None or event.source_node_id not in memory.nodes:
                continue
            source_node = memory.nodes[event.source_node_id]
            compact = self._compact_temporal_event_metadata(event)
            source_node.metadata.setdefault("temporal_occurrences", []).append(compact)
            if event.modality == "asr":
                source_node.metadata.setdefault("speech_occurrences", []).append(compact)
            elif event.modality == "audio":
                source_node.metadata.setdefault("audio_occurrences", []).append(compact)
            elif event.modality == "ocr":
                source_node.metadata.setdefault("ocr_occurrences", []).append(compact)
            elif event.modality == "visual":
                source_node.metadata.setdefault("visual_occurrences", []).append(compact)

    def _compact_temporal_event_metadata(self, event) -> dict[str, Any]:
        metadata = event.metadata
        return {
            "event_id": event.event_id,
            "modality": event.modality,
            "event_type": event.event_type,
            "time_span": event.time_span.to_dict(),
            "section_id": event.section_id,
            "occurrence_index": metadata.get("occurrence_index"),
            "occurrence_count": metadata.get("occurrence_count"),
            "section_local_ordinal": metadata.get("section_local_ordinal"),
            "section_ordinal": metadata.get("section_ordinal"),
            "first_seen": metadata.get("first_seen"),
            "last_seen": metadata.get("last_seen"),
            "previous_same_entity_event": metadata.get("previous_same_entity_event"),
            "next_same_entity_event": metadata.get("next_same_entity_event"),
            "change_type": metadata.get("change_type"),
            "section_tags": list(metadata.get("section_tags", [])),
            "text_preview": (event.text or "")[:180],
        }

    def _temporal_section_tags(
        self,
        artifacts: PreparedVideoArtifacts,
        level: VideoNodeLevel,
        time_span: TimeSpan,
    ) -> list[str]:
        duration = float(artifacts.duration_seconds)
        if duration <= 0:
            return []
        tags: list[str] = []
        midpoint = duration / 2.0
        if time_span.end > midpoint:
            tags.append("second_half")
        if time_span.start < midpoint:
            tags.append("first_half")
        if level not in {"segment", "event", "clip"}:
            return _dedupe_strings(tags)
        for segment_index in self._overlapping_segment_indices(time_span, duration):
            tags.append(f"segment_{segment_index}")
            ordinal = self._segment_ordinal_name(segment_index)
            if ordinal:
                tags.append(ordinal)
        return _dedupe_strings(tags)

    def _overlapping_segment_indices(self, time_span: TimeSpan, duration: float) -> list[int]:
        if self.segment_duration_seconds <= 0:
            return []
        start_index = int(max(0.0, time_span.start) // self.segment_duration_seconds) + 1
        end_time = max(time_span.start, min(duration, time_span.end) - 1e-6)
        end_index = int(end_time // self.segment_duration_seconds) + 1
        return list(range(max(1, start_index), max(start_index, end_index) + 1))

    def _segment_ordinal_name(self, index: int) -> str | None:
        return {
            1: "first_segment",
            2: "second_segment",
            3: "third_segment",
            4: "fourth_segment",
            5: "fifth_segment",
        }.get(index)

    def _content_section_tags(
        self,
        *,
        visual_summary: str,
        speech_spans: list[SpeechSpan],
        ocr_spans: list[OCRSpan],
    ) -> list[str]:
        text = " ".join(
            [
                visual_summary,
                " ".join(span.text for span in speech_spans),
                " ".join(span.text for span in ocr_spans),
            ]
        )
        lowered = text.lower()
        tags: list[str] = []
        assignments = CODE_ASSIGNMENT_LINE_PATTERN.findall(text)
        if assignments or "python script" in lowered or "code editor" in lowered:
            tags.append("code_section")
        if (
            "assignment operator" in lowered
            or "assignment operators" in lowered
            or "variable declaration" in lowered
            or "variables" in lowered
            or re.search(r"\b[x-z]\s*=", text)
        ):
            tags.append("assignment_section")
        if (
            "arithmetic operator" in lowered
            or "arithmetic operators" in lowered
            or any(re.search(r"=\s*[^=<>!]*(?:\+|\-|\*|/)", assignment) for assignment in assignments)
        ):
            tags.append("arithmetic_section")
        if "comparison operator" in lowered or "comparison operators" in lowered or any(
            operator in text for operator in ("==", "!=", ">=", "<=", ">", "<")
        ):
            tags.append("comparison_section")
        if COMPARISON_OPERATOR_TEXT_PATTERN.search(text):
            tags.append("logical_evaluation_section")
        if "shell" in lowered or "output" in lowered or "print(" in lowered:
            tags.append("output_section")
        if (
            ("assignment_section" in tags or "arithmetic_section" in tags)
            and "comparison_section" not in tags
        ):
            tags.append("pre_comparison_section")
        return _dedupe_strings(tags)

    def _should_build_cognitive_events(self, artifacts: PreparedVideoArtifacts) -> bool:
        enabled = bool(
            artifacts.metadata.get("enable_cognitive_events", self.enable_cognitive_events)
        )
        if not enabled:
            return False
        return any(
            self._visual_summary_has_cognitive_signal(summary)
            for summary in artifacts.visual_summaries
        )

    def _visual_summary_has_cognitive_signal(self, summary: VisualSummarySpan) -> bool:
        metadata = summary.metadata
        return (
            metadata.get("visual_summary_mode") == "lazy_pitome_index"
            or bool(metadata.get("lazy_visual_index"))
            or bool(metadata.get("event_boundary_scores"))
            or bool(metadata.get("event_boundary_peak_timestamps"))
            or bool(metadata.get("visual_novelty_scores"))
            or bool(metadata.get("cognitive_anchor_frames"))
            or bool(metadata.get("cognitive_anchor_timestamps"))
            or float(metadata.get("memorability_prior") or 0.0) > 0.0
        )

    def _add_cross_modal_cognitive_anchor_signals(self, clip_nodes: list[VideoNode]) -> None:
        ordered = sorted(clip_nodes, key=lambda node: (node.time_span.start, node.node_id))
        for previous, current in zip(ordered, ordered[1:], strict=False):
            boundary_timestamp = round(
                (previous.time_span.end + current.time_span.start) / 2.0,
                3,
            )
            speech_score = self._speech_topic_shift_score(previous, current)
            ocr_score = self._ocr_change_score(previous, current)
            visual_change_score, visual_change_reasons = self._object_person_action_change_score(
                previous,
                current,
            )
            components = {
                "speech_topic_shift": speech_score,
                "ocr_change": ocr_score,
                "object_person_action_change": visual_change_score,
            }
            boundary_score = round(
                max(
                    speech_score,
                    ocr_score,
                    visual_change_score,
                    (0.35 * speech_score)
                    + (0.25 * ocr_score)
                    + (0.40 * visual_change_score),
                ),
                4,
            )
            reasons = self._cross_modal_anchor_reasons(
                speech_score=speech_score,
                ocr_score=ocr_score,
                visual_change_score=visual_change_score,
                visual_change_reasons=visual_change_reasons,
            )
            if not reasons:
                continue
            anchor = {
                "timestamp": boundary_timestamp,
                "dense_index": None,
                "reasons": reasons,
                "score": boundary_score,
                "event_boundary_score": boundary_score,
                "visual_novelty_score": visual_change_score,
                "cross_modal_boundary_components": components,
            }
            boundary_entry = {
                "timestamp": boundary_timestamp,
                "score": boundary_score,
                "peak": boundary_score >= self.cognitive_event_boundary_threshold,
                "reasons": reasons,
                "components": components,
            }
            for node in (previous, current):
                self._append_boundary_entry(node.metadata, boundary_entry)
                self._append_cognitive_anchor(node.metadata, anchor)
                self._append_timestamp_metadata(
                    node.metadata,
                    "cross_modal_boundary_anchor_timestamps",
                    boundary_timestamp,
                )
                self._append_metadata_record(
                    node.metadata,
                    "cross_modal_boundary_components",
                    boundary_entry,
                )
            if speech_score > 0.0:
                self._append_timestamp_metadata(
                    current.metadata,
                    "speech_topic_shift_anchor_timestamps",
                    boundary_timestamp,
                )
            if ocr_score > 0.0:
                self._append_timestamp_metadata(
                    current.metadata,
                    "ocr_change_anchor_timestamps",
                    boundary_timestamp,
                )
            if visual_change_score > 0.0:
                self._append_timestamp_metadata(
                    current.metadata,
                    "object_person_action_change_anchor_timestamps",
                    boundary_timestamp,
                )

    def _speech_topic_shift_score(self, previous: VideoNode, current: VideoNode) -> float:
        return self._text_shift_score(
            " ".join(span.text for span in previous.speech_spans),
            " ".join(span.text for span in current.speech_spans),
            missing_weight=0.0,
        )

    def _ocr_change_score(self, previous: VideoNode, current: VideoNode) -> float:
        return self._text_shift_score(
            " ".join(span.text for span in previous.ocr_spans),
            " ".join(span.text for span in current.ocr_spans),
            missing_weight=0.0,
        )

    def _text_shift_score(
        self,
        previous_text: str,
        current_text: str,
        *,
        missing_weight: float,
    ) -> float:
        previous_tokens = self._event_tokens(previous_text)
        current_tokens = self._event_tokens(current_text)
        if not previous_tokens and not current_tokens:
            return 0.0
        if not previous_tokens or not current_tokens:
            return missing_weight
        return round(_token_jaccard_distance(previous_tokens, current_tokens), 4)

    def _object_person_action_change_score(
        self,
        previous: VideoNode,
        current: VideoNode,
    ) -> tuple[float, list[str]]:
        previous_roles = self._visual_role_tokens(previous)
        current_roles = self._visual_role_tokens(current)
        component_scores: list[float] = []
        reasons: list[str] = []
        for key, reason in (
            ("actors", "person_change"),
            ("objects", "object_change"),
            ("actions", "action_change"),
        ):
            score = _token_jaccard_distance(previous_roles[key], current_roles[key])
            if previous_roles[key] or current_roles[key]:
                component_scores.append(score)
            if score >= 0.45 and (previous_roles[key] or current_roles[key]):
                reasons.append(reason)
        if not component_scores:
            return 0.0, []
        return round(sum(component_scores) / len(component_scores), 4), reasons

    def _visual_role_tokens(self, node: VideoNode) -> dict[str, set[str]]:
        text = " ".join(
            [
                node.visual_summary,
                " ".join(node.tags),
                " ".join(node.entities),
                " ".join(str(item) for item in node.metadata.get("visual_keywords", [])),
                " ".join(str(item) for item in node.metadata.get("visual_atoms", [])),
            ]
        )
        tokens = self._event_tokens(text)
        actors = tokens & {
            "actor",
            "child",
            "face",
            "man",
            "person",
            "people",
            "presenter",
            "speaker",
            "woman",
        }
        actions = {
            token
            for token in tokens
            if token.endswith("ing")
            or token
            in {
                "carry",
                "close",
                "drink",
                "enter",
                "fix",
                "hold",
                "lift",
                "move",
                "open",
                "pick",
                "place",
                "pour",
                "put",
                "reach",
                "serve",
                "stand",
                "take",
                "walk",
            }
        }
        places = tokens & {
            "bedroom",
            "desk",
            "home",
            "kitchen",
            "office",
            "outdoor",
            "room",
            "screen",
            "shop",
            "stage",
            "street",
            "table",
        }
        objects = tokens - actors - actions - places
        return {"actors": actors, "objects": objects, "actions": actions}

    def _cross_modal_anchor_reasons(
        self,
        *,
        speech_score: float,
        ocr_score: float,
        visual_change_score: float,
        visual_change_reasons: list[str],
    ) -> list[str]:
        reasons: list[str] = []
        if speech_score >= 0.35:
            reasons.append("speech_topic_shift")
        if ocr_score >= 0.35:
            reasons.append("ocr_change")
        if visual_change_score >= 0.35:
            reasons.append("object_person_action_change")
            reasons.extend(visual_change_reasons)
        return _dedupe_strings(reasons)

    def _append_boundary_entry(self, metadata: dict[str, Any], entry: dict[str, Any]) -> None:
        entries = [item for item in metadata.get("event_boundary_scores", []) if isinstance(item, dict)]
        timestamp = float(entry["timestamp"])
        for item in entries:
            if abs(float(item.get("timestamp", -1.0)) - timestamp) > 0.05:
                continue
            item["score"] = round(max(float(item.get("score") or 0.0), float(entry["score"])), 4)
            item["peak"] = bool(item.get("peak")) or bool(entry.get("peak"))
            item["reasons"] = _dedupe_strings(
                [*item.get("reasons", []), *entry.get("reasons", [])]
            )
            item["components"] = {
                **dict(item.get("components") or {}),
                **dict(entry.get("components") or {}),
            }
            break
        else:
            entries.append(dict(entry))
        entries.sort(key=lambda item: float(item.get("timestamp", 0.0)))
        metadata["event_boundary_scores"] = entries
        metadata["event_boundary_peak_timestamps"] = [
            float(item["timestamp"])
            for item in entries
            if item.get("peak") and item.get("timestamp") is not None
        ]

    def _append_cognitive_anchor(self, metadata: dict[str, Any], anchor: dict[str, Any]) -> None:
        anchors = [
            item for item in metadata.get("cognitive_anchor_frames", []) if isinstance(item, dict)
        ]
        timestamp = float(anchor["timestamp"])
        for item in anchors:
            if abs(float(item.get("timestamp", -1.0)) - timestamp) > 0.05:
                continue
            item["score"] = round(max(float(item.get("score") or 0.0), float(anchor["score"])), 4)
            item["event_boundary_score"] = round(
                max(
                    float(item.get("event_boundary_score") or 0.0),
                    float(anchor["event_boundary_score"]),
                ),
                4,
            )
            item["visual_novelty_score"] = round(
                max(
                    float(item.get("visual_novelty_score") or 0.0),
                    float(anchor["visual_novelty_score"]),
                ),
                4,
            )
            item["reasons"] = _dedupe_strings(
                [*item.get("reasons", []), *anchor.get("reasons", [])]
            )
            item["cross_modal_boundary_components"] = {
                **dict(item.get("cross_modal_boundary_components") or {}),
                **dict(anchor.get("cross_modal_boundary_components") or {}),
            }
            break
        else:
            anchors.append(dict(anchor))
        anchors.sort(key=lambda item: float(item.get("timestamp", 0.0)))
        metadata["cognitive_anchor_frames"] = anchors
        metadata["cognitive_anchor_timestamps"] = [
            float(item["timestamp"]) for item in anchors if item.get("timestamp") is not None
        ]
        metadata["cognitive_anchor_frame_count"] = len(anchors)

    def _append_timestamp_metadata(
        self,
        metadata: dict[str, Any],
        key: str,
        timestamp: float,
    ) -> None:
        values = [float(item) for item in metadata.get(key, [])]
        values.append(float(timestamp))
        metadata[key] = _dedupe_sorted_floats(values)

    def _append_metadata_record(
        self,
        metadata: dict[str, Any],
        key: str,
        record: dict[str, Any],
    ) -> None:
        records = [item for item in metadata.get(key, []) if isinstance(item, dict)]
        timestamp = float(record["timestamp"])
        records = [
            item
            for item in records
            if abs(float(item.get("timestamp", -1.0)) - timestamp) > 0.05
        ]
        records.append(dict(record))
        records.sort(key=lambda item: float(item.get("timestamp", 0.0)))
        metadata[key] = records

    def _build_cognitive_event_groups(
        self,
        *,
        segment_span: TimeSpan,
        clip_nodes: list[VideoNode],
    ) -> list[CognitiveEventGroup]:
        if not clip_nodes:
            return []
        ordered = sorted(clip_nodes, key=lambda node: (node.time_span.start, node.node_id))
        groups: list[CognitiveEventGroup] = []
        current_ids = [ordered[0].node_id]
        current_split_scores: list[float] = []

        for previous, current in zip(ordered, ordered[1:], strict=False):
            boundary_score = self._inter_clip_cognitive_boundary_score(previous, current)
            event_too_long = self._would_exceed_cognitive_event_duration(
                start=self._node_by_id(ordered, current_ids[0]).time_span.start,
                end=current.time_span.end,
            )
            should_split = (
                boundary_score >= self.cognitive_event_boundary_threshold
                or event_too_long
            )
            if should_split:
                groups.append(
                    self._cognitive_event_group_from_ids(
                        ordered,
                        current_ids,
                        current_split_scores + [boundary_score],
                    )
                )
                current_ids = [current.node_id]
                current_split_scores = []
            else:
                current_ids.append(current.node_id)
                current_split_scores.append(boundary_score)

        groups.append(
            self._cognitive_event_group_from_ids(ordered, current_ids, current_split_scores)
        )
        return [
            CognitiveEventGroup(
                clip_node_ids=group.clip_node_ids,
                time_span=TimeSpan(
                    max(segment_span.start, group.time_span.start),
                    min(segment_span.end, group.time_span.end),
                ),
                split_scores=list(group.split_scores),
            )
            for group in groups
        ]

    def _node_by_id(self, nodes: list[VideoNode], node_id: str) -> VideoNode:
        for node in nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(f"Unknown node_id while building cognitive events: {node_id}")

    def _cognitive_event_group_from_ids(
        self,
        ordered_nodes: list[VideoNode],
        clip_node_ids: list[str],
        split_scores: list[float],
    ) -> CognitiveEventGroup:
        selected = [self._node_by_id(ordered_nodes, node_id) for node_id in clip_node_ids]
        return CognitiveEventGroup(
            clip_node_ids=list(clip_node_ids),
            time_span=TimeSpan(
                min(node.time_span.start for node in selected),
                max(node.time_span.end for node in selected),
            ),
            split_scores=[round(float(score), 4) for score in split_scores],
        )

    def _would_exceed_cognitive_event_duration(self, *, start: float, end: float) -> bool:
        if self.cognitive_event_max_duration_seconds is None:
            return False
        return (end - start) > self.cognitive_event_max_duration_seconds

    def _inter_clip_cognitive_boundary_score(
        self,
        previous: VideoNode,
        current: VideoNode,
    ) -> float:
        boundary_timestamp = (previous.time_span.end + current.time_span.start) / 2.0
        metadata_boundary = max(
            self._boundary_score_near(previous.metadata, boundary_timestamp),
            self._boundary_score_near(current.metadata, boundary_timestamp),
        )
        visual_shift = _token_jaccard_distance(
            self._event_tokens(previous.visual_summary),
            self._event_tokens(current.visual_summary),
        )
        speech_shift = _token_jaccard_distance(
            self._event_tokens(" ".join(span.text for span in previous.speech_spans)),
            self._event_tokens(" ".join(span.text for span in current.speech_spans)),
        )
        ocr_shift = _token_jaccard_distance(
            self._event_tokens(" ".join(span.text for span in previous.ocr_spans)),
            self._event_tokens(" ".join(span.text for span in current.ocr_spans)),
        )
        memorability = max(
            float(previous.metadata.get("memorability_prior") or 0.0),
            float(current.metadata.get("memorability_prior") or 0.0),
        )
        shift_score = (
            (0.45 * visual_shift)
            + (0.2 * speech_shift)
            + (0.2 * ocr_shift)
            + (0.15 * memorability)
        )
        return round(max(metadata_boundary, shift_score), 4)

    def _boundary_score_near(
        self,
        metadata: dict[str, Any],
        timestamp: float,
        *,
        tolerance_seconds: float = COGNITIVE_EVENT_BOUNDARY_TOLERANCE_SECONDS,
    ) -> float:
        best = 0.0
        for item in metadata.get("event_boundary_scores", []):
            if not isinstance(item, dict):
                continue
            item_timestamp = item.get("timestamp")
            if item_timestamp is None:
                continue
            if abs(float(item_timestamp) - timestamp) <= tolerance_seconds:
                best = max(best, float(item.get("score") or 0.0))
        for item in metadata.get("cognitive_anchor_frames", []):
            if not isinstance(item, dict):
                continue
            item_timestamp = item.get("timestamp")
            if item_timestamp is None:
                continue
            if abs(float(item_timestamp) - timestamp) > tolerance_seconds:
                continue
            reasons = {str(reason) for reason in item.get("reasons", [])}
            if reasons & {"event_boundary_peak", "ffmpeg_scene_or_span_boundary"}:
                best = max(
                    best,
                    float(item.get("event_boundary_score") or item.get("score") or 0.0),
                )
        return round(min(max(best, 0.0), 1.0), 4)

    def _make_cognitive_event_node(
        self,
        *,
        artifacts: PreparedVideoArtifacts,
        node_id: str,
        time_span: TimeSpan,
        parent_id: str,
        child_nodes: list[VideoNode],
        split_scores: list[float],
    ) -> VideoNode:
        speech_spans = self._speech_spans_for_node(artifacts.speech_spans, time_span, "event")
        ocr_spans = self._overlapping_items(artifacts.ocr_spans, time_span)
        audio_events = self._overlapping_items(artifacts.audio_events, time_span)
        tags = sorted({tag for child in child_nodes for tag in child.tags})
        entities = sorted({entity for child in child_nodes for entity in child.entities})
        visual_keywords = self._event_visual_keywords(child_nodes)
        visual_atoms = self._event_visual_atoms(child_nodes)
        spoken_topics = self._event_keywords_from_texts([span.text for span in speech_spans])
        ocr_entities = self._event_keywords_from_texts([span.text for span in ocr_spans])
        audio_labels = sorted({event.label.strip() for event in audio_events if event.label})
        cognitive_anchors = self._event_cognitive_anchor_metadata(child_nodes)
        boundary_scores = self._event_score_timeline(child_nodes, "event_boundary_scores")
        visual_novelty_scores = self._event_score_timeline(child_nodes, "visual_novelty_scores")
        boundary_peaks = self._event_boundary_peak_timestamps(child_nodes, boundary_scores)
        novelty_peaks = self._event_peak_timestamps(child_nodes, "visual_novelty_peak_timestamps")
        memorability_prior = self._event_memorability_prior(child_nodes)
        visual_summary = self._cognitive_event_visual_summary(
            time_span=time_span,
            child_nodes=child_nodes,
            visual_keywords=visual_keywords,
            spoken_topics=spoken_topics,
            ocr_entities=ocr_entities,
            cognitive_anchor_count=len(cognitive_anchors),
        )
        event_schema = self._build_event_schema(
            time_span=time_span,
            visual_summary=visual_summary,
            visual_keywords=visual_keywords,
            visual_atoms=visual_atoms,
            speech_topics=spoken_topics,
            ocr_entities=ocr_entities,
            audio_labels=audio_labels,
        )
        metadata: dict[str, Any] = {
            "node_type": "cognitive_event",
            "cognitive_event": True,
            "event_schema": event_schema,
            "event_source_clip_ids": [child.node_id for child in child_nodes],
            "visual_detail_node_ids": [
                child.node_id for child in child_nodes if child.visual_summary.strip()
            ],
            "visual_keywords": visual_keywords,
            "visual_atoms": visual_atoms,
            "spoken_topics": spoken_topics,
            "ocr_entities": ocr_entities,
            "audio_labels": audio_labels,
            "cognitive_anchor_frames": cognitive_anchors,
            "cognitive_anchor_timestamps": [
                float(item["timestamp"]) for item in cognitive_anchors if "timestamp" in item
            ],
            "event_boundary_scores": boundary_scores,
            "event_boundary_peak_timestamps": boundary_peaks,
            "visual_novelty_scores": visual_novelty_scores,
            "visual_novelty_peak_timestamps": novelty_peaks,
            "memorability_prior": memorability_prior,
            "cognitive_event_boundary_threshold": self.cognitive_event_boundary_threshold,
            "cognitive_event_split_scores": split_scores,
            "on_demand_visual_refinement": any(
                bool(child.metadata.get("on_demand_visual_refinement"))
                for child in child_nodes
            ),
        }
        metadata.update(
            self._section_metadata(
                artifacts=artifacts,
                level="event",
                time_span=time_span,
                visual_summary=visual_summary,
                speech_spans=speech_spans,
                ocr_spans=ocr_spans,
            )
        )
        pitome_embeddings = self._event_child_embeddings(child_nodes, "pitome_frame_embeddings")
        if pitome_embeddings:
            metadata["pitome_frame_embeddings"] = pitome_embeddings
            metadata["pitome_frame_embedding_dim"] = len(pitome_embeddings[0])
            metadata["pitome_frame_embedding_source"] = "cognitive_event_child_anchors"
        semantic_embeddings = self._event_child_embeddings(child_nodes, "semantic_frame_embeddings")
        if semantic_embeddings:
            metadata["semantic_frame_embeddings"] = semantic_embeddings
            metadata["semantic_frame_embedding_dim"] = len(semantic_embeddings[0])
            metadata["semantic_frame_embedding_source"] = "cognitive_event_child_anchors"

        return VideoNode(
            node_id=node_id,
            level="event",
            time_span=time_span,
            visual_summary=visual_summary,
            speech_spans=speech_spans,
            ocr_spans=ocr_spans,
            audio_events=audio_events,
            tags=sorted(set(tags) | set(visual_keywords)),
            entities=entities,
            clip_path=self._build_clip_pointer(artifacts, time_span),
            parent_id=parent_id,
            metadata=metadata,
        )

    def _event_visual_keywords(
        self,
        child_nodes: list[VideoNode],
        *,
        limit: int = 24,
    ) -> list[str]:
        texts = []
        for child in child_nodes:
            texts.append(child.visual_summary)
            texts.extend(str(item) for item in child.tags)
            texts.extend(str(item) for item in child.entities)
            texts.extend(str(item) for item in child.metadata.get("visual_keywords", []))
        return self._event_keywords_from_texts(texts, limit=limit)

    def _event_visual_atoms(self, child_nodes: list[VideoNode], *, limit: int = 16) -> list[str]:
        atoms: list[str] = []
        seen: set[str] = set()
        for child in child_nodes:
            candidates = [
                *[str(item) for item in child.metadata.get("visual_atoms", [])],
                *QUOTED_TEXT_PATTERN.findall(child.visual_summary),
                *CODE_ATOM_PATTERN.findall(child.visual_summary),
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

    def _event_keywords_from_texts(
        self,
        texts: list[str],
        *,
        limit: int = 16,
    ) -> list[str]:
        counts: dict[str, int] = {}
        for text in texts:
            for token in self._event_tokens(text):
                counts[token] = counts.get(token, 0) + 1
        return [
            token
            for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
        ]

    def _event_tokens(self, text: str) -> set[str]:
        tokens: set[str] = set()
        for token in ATOM_PATTERN.findall(text.lower()):
            normalized = token.strip("`.,:;()[]{}")
            if (
                len(normalized) <= 2
                or normalized in ROLLUP_STOPWORDS
                or normalized.replace(".", "", 1).isdigit()
            ):
                continue
            tokens.add(normalized)
        return tokens

    def _event_cognitive_anchor_metadata(
        self,
        child_nodes: list[VideoNode],
    ) -> list[dict[str, Any]]:
        best_by_timestamp: dict[float, dict[str, Any]] = {}
        for child in child_nodes:
            for item in child.metadata.get("cognitive_anchor_frames", []):
                if not isinstance(item, dict) or item.get("timestamp") is None:
                    continue
                timestamp = round(float(item["timestamp"]), 3)
                candidate = dict(item)
                candidate["timestamp"] = timestamp
                candidate["source_node_id"] = child.node_id
                current = best_by_timestamp.get(timestamp)
                if current is None or float(candidate.get("score") or 0.0) > float(
                    current.get("score") or 0.0
                ):
                    best_by_timestamp[timestamp] = candidate
        return [
            best_by_timestamp[timestamp]
            for timestamp in sorted(best_by_timestamp)
        ]

    def _event_score_timeline(
        self,
        child_nodes: list[VideoNode],
        key: str,
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        seen: set[tuple[float, str]] = set()
        for child in child_nodes:
            for item in child.metadata.get(key, []):
                if not isinstance(item, dict) or item.get("timestamp") is None:
                    continue
                timestamp = round(float(item["timestamp"]), 3)
                source_key = (timestamp, child.node_id)
                if source_key in seen:
                    continue
                seen.add(source_key)
                entry = dict(item)
                entry["timestamp"] = timestamp
                entry["source_node_id"] = child.node_id
                entries.append(entry)
        entries.sort(key=lambda item: float(item.get("timestamp", 0.0)))
        return entries

    def _event_boundary_peak_timestamps(
        self,
        child_nodes: list[VideoNode],
        boundary_scores: list[dict[str, Any]],
    ) -> list[float]:
        timestamps = [
            float(item["timestamp"])
            for item in boundary_scores
            if item.get("peak") and item.get("timestamp") is not None
        ]
        for child in child_nodes:
            timestamps.extend(
                float(timestamp)
                for timestamp in child.metadata.get("event_boundary_peak_timestamps", [])
            )
            for item in child.metadata.get("cognitive_anchor_frames", []):
                if not isinstance(item, dict) or item.get("timestamp") is None:
                    continue
                reasons = {str(reason) for reason in item.get("reasons", [])}
                if reasons & {"event_boundary_peak", "ffmpeg_scene_or_span_boundary"}:
                    timestamps.append(float(item["timestamp"]))
        return _dedupe_sorted_floats(timestamps)

    def _event_peak_timestamps(self, child_nodes: list[VideoNode], key: str) -> list[float]:
        timestamps: list[float] = []
        for child in child_nodes:
            timestamps.extend(float(timestamp) for timestamp in child.metadata.get(key, []))
        return _dedupe_sorted_floats(timestamps)

    def _event_memorability_prior(self, child_nodes: list[VideoNode]) -> float:
        scores = [
            float(child.metadata.get("memorability_prior") or 0.0)
            for child in child_nodes
            if child.metadata.get("memorability_prior") is not None
        ]
        if not scores:
            return 0.0
        mean_score = sum(scores) / len(scores)
        return round(max(max(scores), mean_score), 4)

    def _event_child_embeddings(
        self,
        child_nodes: list[VideoNode],
        key: str,
        *,
        limit: int = COGNITIVE_EVENT_FRAME_EMBEDDING_LIMIT,
    ) -> list[list[float]]:
        embeddings: list[list[float]] = []
        dimension: int | None = None
        for child in child_nodes:
            for raw_embedding in child.metadata.get(key, []):
                if not isinstance(raw_embedding, (list, tuple)):
                    continue
                embedding = [float(value) for value in raw_embedding]
                if dimension is None:
                    dimension = len(embedding)
                if len(embedding) != dimension:
                    continue
                embeddings.append(embedding)
                if len(embeddings) >= limit:
                    return embeddings
        return embeddings

    def _cognitive_event_visual_summary(
        self,
        *,
        time_span: TimeSpan,
        child_nodes: list[VideoNode],
        visual_keywords: list[str],
        spoken_topics: list[str],
        ocr_entities: list[str],
        cognitive_anchor_count: int,
    ) -> str:
        parts = [f"cognitive event {time_span.to_display()}"]
        if visual_keywords:
            parts.append(f"visual cues: {', '.join(visual_keywords[:10])}")
        if spoken_topics:
            parts.append(f"speech topics: {', '.join(spoken_topics[:8])}")
        if ocr_entities:
            parts.append(f"OCR entities: {', '.join(ocr_entities[:8])}")
        summaries = [
            " ".join(child.visual_summary.split())
            for child in child_nodes
            if child.visual_summary.strip()
        ]
        if summaries:
            parts.append("child visual states: " + " | ".join(summaries[:3]))
        if cognitive_anchor_count:
            parts.append(f"{cognitive_anchor_count} cognitive anchor frames")
        return "; ".join(parts)

    def _build_event_schema(
        self,
        *,
        time_span: TimeSpan,
        visual_summary: str,
        visual_keywords: list[str],
        visual_atoms: list[str],
        speech_topics: list[str],
        ocr_entities: list[str],
        audio_labels: list[str],
    ) -> dict[str, Any]:
        actor_tokens = [
            token
            for token in visual_keywords
            if token in {"person", "people", "man", "woman", "child", "presenter", "speaker"}
        ]
        place_tokens = [
            token
            for token in visual_keywords
            if token
            in {
                "bedroom",
                "desk",
                "home",
                "kitchen",
                "office",
                "outdoor",
                "room",
                "screen",
                "shop",
                "stage",
                "street",
                "table",
            }
        ]
        action_tokens = [
            token
            for token in visual_keywords + speech_topics
            if token.endswith("ing")
            or token
            in {
                "carry",
                "close",
                "drink",
                "enter",
                "hold",
                "move",
                "open",
                "place",
                "pour",
                "put",
                "reach",
                "stand",
                "take",
                "walk",
            }
        ]
        object_tokens = [
            token
            for token in visual_keywords + visual_atoms
            if token not in set(actor_tokens) | set(place_tokens) | set(action_tokens)
        ]
        event_type = "visual_skim_event"
        if speech_topics and visual_keywords:
            event_type = "multimodal_event"
        elif speech_topics:
            event_type = "spoken_topic_event"
        elif ocr_entities:
            event_type = "text_screen_event"
        return {
            "time_span": [round(time_span.start, 3), round(time_span.end, 3)],
            "place": _dedupe_strings(place_tokens),
            "actors": _dedupe_strings(actor_tokens),
            "objects": _dedupe_strings(object_tokens[:12]),
            "actions": _dedupe_strings(action_tokens[:12]),
            "goals_or_intentions": [],
            "causal_predecessors": [],
            "causal_outcomes": [],
            "spoken_topics": _dedupe_strings(speech_topics),
            "ocr_entities": _dedupe_strings(ocr_entities),
            "visual_state": visual_summary,
            "audio_state": ", ".join(audio_labels),
            "event_type": event_type,
        }

    def _link_cognitive_events(self, memory: VideoMemory) -> None:
        events = sorted(
            [
                node
                for node in memory.nodes.values()
                if node.level == "event" and node.metadata.get("cognitive_event")
            ],
            key=lambda node: (node.time_span.start, node.node_id),
        )
        memory.metadata["cognitive_event_node_ids"] = [node.node_id for node in events]
        for index, node in enumerate(events):
            previous_id = events[index - 1].node_id if index > 0 else None
            next_id = events[index + 1].node_id if index + 1 < len(events) else None
            neighbor_ids = [item for item in [previous_id, next_id] if item is not None]
            node.metadata["previous_cognitive_event_id"] = previous_id
            node.metadata["next_cognitive_event_id"] = next_id
            node.metadata["cognitive_event_neighbor_ids"] = neighbor_ids
            event_schema = node.metadata.get("event_schema")
            if isinstance(event_schema, dict):
                event_schema["temporal_predecessors"] = [previous_id] if previous_id else []
                event_schema["temporal_successors"] = [next_id] if next_id else []
        self._link_cognitive_event_similarity_edges(events)
        self._link_cognitive_event_causal_goal_edges(events)

    def _link_cognitive_event_similarity_edges(self, events: list[VideoNode]) -> None:
        for node in events:
            schema = node.metadata.get("event_schema")
            if not isinstance(schema, dict):
                continue
            same_actor: list[str] = []
            same_object: list[str] = []
            same_place: list[str] = []
            same_topic: list[str] = []
            node_actors = set(_schema_list(schema, "actors"))
            node_objects = set(_schema_list(schema, "objects"))
            node_places = set(_schema_list(schema, "place"))
            node_topics = set(_schema_list(schema, "spoken_topics"))
            for candidate in events:
                if candidate.node_id == node.node_id:
                    continue
                candidate_schema = candidate.metadata.get("event_schema")
                if not isinstance(candidate_schema, dict):
                    continue
                if node_actors & set(_schema_list(candidate_schema, "actors")):
                    same_actor.append(candidate.node_id)
                if node_objects & set(_schema_list(candidate_schema, "objects")):
                    same_object.append(candidate.node_id)
                if node_places & set(_schema_list(candidate_schema, "place")):
                    same_place.append(candidate.node_id)
                if node_topics & set(_schema_list(candidate_schema, "spoken_topics")):
                    same_topic.append(candidate.node_id)
            node.metadata["same_actor_event_ids"] = same_actor[:8]
            node.metadata["same_object_event_ids"] = same_object[:8]
            node.metadata["same_place_event_ids"] = same_place[:8]
            node.metadata["same_topic_event_ids"] = same_topic[:8]

    def _link_cognitive_event_causal_goal_edges(self, events: list[VideoNode]) -> None:
        for node in events:
            node.metadata.setdefault("cause_effect_event_ids", [])
            node.metadata.setdefault("caused_by_event_ids", [])
            node.metadata.setdefault("goal_continuation_event_ids", [])
            node.metadata.setdefault("goal_predecessor_event_ids", [])
        for previous, current in zip(events, events[1:], strict=False):
            previous_schema = previous.metadata.get("event_schema")
            current_schema = current.metadata.get("event_schema")
            if not isinstance(previous_schema, dict) or not isinstance(current_schema, dict):
                continue
            boundary_score = self._event_pair_boundary_score(previous, current)
            shared = self._shared_event_schema_terms(previous_schema, current_schema)
            if self._is_goal_continuation_edge(shared=shared, boundary_score=boundary_score):
                self._append_edge_id(previous.metadata, "goal_continuation_event_ids", current.node_id)
                self._append_edge_id(current.metadata, "goal_predecessor_event_ids", previous.node_id)
                previous_schema["goal_successors"] = _dedupe_strings(
                    [
                        *[str(item) for item in previous_schema.get("goal_successors", [])],
                        current.node_id,
                    ]
                )
                current_schema["goal_predecessors"] = _dedupe_strings(
                    [
                        *[str(item) for item in current_schema.get("goal_predecessors", [])],
                        previous.node_id,
                    ]
                )
            if self._is_cause_effect_edge(
                previous=previous,
                current=current,
                shared=shared,
                boundary_score=boundary_score,
            ):
                self._append_edge_id(previous.metadata, "cause_effect_event_ids", current.node_id)
                self._append_edge_id(current.metadata, "caused_by_event_ids", previous.node_id)
                previous_schema["causal_outcomes"] = _dedupe_strings(
                    [
                        *[str(item) for item in previous_schema.get("causal_outcomes", [])],
                        f"leads_to:{current.node_id}",
                    ]
                )
                current_schema["causal_predecessors"] = _dedupe_strings(
                    [
                        *[str(item) for item in current_schema.get("causal_predecessors", [])],
                        f"caused_by:{previous.node_id}",
                    ]
                )

    def _event_pair_boundary_score(self, previous: VideoNode, current: VideoNode) -> float:
        boundary_timestamp = round((previous.time_span.end + current.time_span.start) / 2.0, 3)
        return max(
            self._boundary_score_near(previous.metadata, boundary_timestamp),
            self._boundary_score_near(current.metadata, boundary_timestamp),
            max(
                [
                    float(score)
                    for score in [
                        *(previous.metadata.get("cognitive_event_split_scores") or []),
                        *(current.metadata.get("cognitive_event_split_scores") or []),
                    ]
                    if isinstance(score, (int, float))
                ],
                default=0.0,
            ),
        )

    def _shared_event_schema_terms(
        self,
        previous_schema: dict[str, Any],
        current_schema: dict[str, Any],
    ) -> dict[str, set[str]]:
        return {
            key: set(_schema_list(previous_schema, key)) & set(_schema_list(current_schema, key))
            for key in ("actors", "objects", "place", "actions", "spoken_topics", "ocr_entities")
        }

    def _is_goal_continuation_edge(
        self,
        *,
        shared: dict[str, set[str]],
        boundary_score: float,
    ) -> bool:
        same_actor = bool(shared["actors"])
        same_context = bool(
            shared["objects"]
            or shared["place"]
            or shared["actions"]
            or shared["spoken_topics"]
            or shared["ocr_entities"]
        )
        return same_actor and same_context and boundary_score <= 0.55

    def _is_cause_effect_edge(
        self,
        *,
        previous: VideoNode,
        current: VideoNode,
        shared: dict[str, set[str]],
        boundary_score: float,
    ) -> bool:
        previous_tokens = self._event_relation_tokens(previous)
        current_tokens = self._event_relation_tokens(current)
        if current_tokens & CAUSAL_OUTCOME_TERMS:
            return True
        if previous_tokens & CAUSAL_SETUP_TERMS and current_tokens & CAUSAL_RESOLUTION_TERMS:
            return True
        if (
            (shared["actors"] or shared["objects"])
            and previous_tokens & ACTION_START_TERMS
            and current_tokens & ACTION_COMPLETION_TERMS
        ):
            return True
        return False

    def _event_relation_tokens(self, node: VideoNode) -> set[str]:
        schema = node.metadata.get("event_schema")
        schema_text = ""
        if isinstance(schema, dict):
            schema_text = " ".join(
                str(item)
                for key in (
                    "actions",
                    "goals_or_intentions",
                    "causal_predecessors",
                    "causal_outcomes",
                    "spoken_topics",
                    "ocr_entities",
                    "visual_state",
                    "audio_state",
                )
                for item in (
                    schema.get(key, [])
                    if isinstance(schema.get(key), list)
                    else [schema.get(key)]
                )
                if item
            )
        return self._relation_token_variants(
            self._event_tokens(
                " ".join(
                    [
                        node.visual_summary,
                        " ".join(node.tags),
                        " ".join(node.entities),
                        " ".join(span.text for span in node.speech_spans),
                        " ".join(span.text for span in node.ocr_spans),
                        schema_text,
                    ]
                )
            )
        )

    def _relation_token_variants(self, tokens: set[str]) -> set[str]:
        variants = set(tokens)
        for token in tokens:
            if len(token) > 4 and token.endswith("ing"):
                variants.add(token[:-3])
            if len(token) > 3 and token.endswith("es"):
                variants.add(token[:-2])
            if len(token) > 3 and token.endswith("s"):
                variants.add(token[:-1])
        return variants

    def _append_edge_id(self, metadata: dict[str, Any], key: str, node_id: str) -> None:
        values = [str(item) for item in metadata.get(key, []) if str(item)]
        values.append(node_id)
        metadata[key] = _dedupe_strings(values)[:8]

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

    def _speech_spans_for_node(
        self,
        speech_spans: list[SpeechSpan],
        node_span: TimeSpan,
        level: VideoNodeLevel,
    ) -> list[SpeechSpan]:
        if level == "video":
            return list(speech_spans)
        clipped_spans: list[SpeechSpan] = []
        for source_index, span in enumerate(speech_spans):
            if not span.time_span.overlaps(node_span):
                continue
            overlap_start = max(span.time_span.start, node_span.start)
            overlap_end = min(span.time_span.end, node_span.end)
            if overlap_end - overlap_start <= 0.05:
                continue
            text = self._speech_text_slice_for_overlap(span, overlap_start, overlap_end)
            if not text:
                continue
            source_text = " ".join(span.text.split()).strip()
            metadata = dict(span.metadata)
            metadata.update(
                {
                    "source_speech_span_index": source_index,
                    "source_speech_time_span": span.time_span.to_dict(),
                    "node_time_span": node_span.to_dict(),
                    "overlap_time_span": TimeSpan(overlap_start, overlap_end).to_dict(),
                    "speech_text_clipped_to_node": True,
                    "speech_text_slice_method": "proportional_word_overlap",
                    "source_text_was_sliced": text != source_text,
                }
            )
            clipped_spans.append(
                SpeechSpan(
                    text=text,
                    time_span=TimeSpan(overlap_start, overlap_end),
                    speaker=span.speaker,
                    language=span.language,
                    metadata=metadata,
                )
            )
        return clipped_spans

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


def _token_jaccard_distance(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    if not left or not right:
        return 1.0
    return 1.0 - (len(left & right) / len(left | right))


def _dedupe_sorted_floats(values: list[float], *, tolerance: float = 0.05) -> list[float]:
    ordered = sorted(float(value) for value in values)
    deduped: list[float] = []
    for value in ordered:
        if deduped and abs(value - deduped[-1]) <= tolerance:
            continue
        deduped.append(round(value, 3))
    return deduped


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = " ".join(str(value).split()).strip()
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _schema_list(schema: dict[str, Any], key: str) -> list[str]:
    value = schema.get(key)
    if not isinstance(value, list):
        return []
    return [
        " ".join(str(item).lower().split()).strip()
        for item in value
        if " ".join(str(item).split()).strip()
    ]


@contextmanager
def _temporary_progress_callback(
    component: Any,
    callback: Callable[[dict[str, Any]], None] | None,
):
    if component is None or callback is None or not hasattr(component, "progress_callback"):
        yield
        return

    original = component.progress_callback
    component.progress_callback = callback
    try:
        yield
    finally:
        component.progress_callback = original
