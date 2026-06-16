import re
from dataclasses import dataclass, field
from typing import Any, Literal

from rlm.core.types import UsageSummary

VideoNodeLevel = Literal["video", "scene", "segment", "event", "clip"]
Modality = Literal["speech", "visual", "ocr", "audio", "cross_modal"]
ActionType = Literal["SEARCH", "OPEN", "SPLIT", "MERGE", "STOP"]
FrontierStatus = Literal["unopened", "opened", "expanded", "exhausted"]
SlotRole = Literal["core", "support", "background", "noise"]
SlotStatus = Literal["missing", "filled", "background_only"]
EventStatus = Literal["missing", "localized"]


@dataclass
class TimeSpan:
    start: float
    end: float

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"Invalid TimeSpan: end={self.end} is before start={self.start}")

    @property
    def duration(self) -> float:
        return self.end - self.start

    def overlaps(self, other: "TimeSpan") -> bool:
        return self.start < other.end and other.start < self.end

    def contains(self, value: float) -> bool:
        return self.start <= value <= self.end

    def to_dict(self) -> dict[str, float]:
        return {"start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TimeSpan":
        return cls(start=float(data["start"]), end=float(data["end"]))

    def to_display(self) -> str:
        return f"{self.start:.2f}-{self.end:.2f}"


@dataclass
class SpeechSpan:
    text: str
    time_span: TimeSpan
    speaker: str | None = None
    language: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "time_span": self.time_span.to_dict(),
            "speaker": self.speaker,
            "language": self.language,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpeechSpan":
        return cls(
            text=data["text"],
            time_span=TimeSpan.from_dict(data["time_span"]),
            speaker=data.get("speaker"),
            language=data.get("language"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class VisualSummarySpan:
    summary: str
    time_span: TimeSpan
    granularity: VideoNodeLevel = "clip"
    tags: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "time_span": self.time_span.to_dict(),
            "granularity": self.granularity,
            "tags": list(self.tags),
            "entities": list(self.entities),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VisualSummarySpan":
        return cls(
            summary=data["summary"],
            time_span=TimeSpan.from_dict(data["time_span"]),
            granularity=data.get("granularity", "clip"),
            tags=list(data.get("tags", [])),
            entities=list(data.get("entities", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class OCRSpan:
    text: str
    time_span: TimeSpan

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "time_span": self.time_span.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OCRSpan":
        return cls(text=data["text"], time_span=TimeSpan.from_dict(data["time_span"]))


@dataclass
class AudioEvent:
    label: str
    time_span: TimeSpan
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "time_span": self.time_span.to_dict(),
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AudioEvent":
        return cls(
            label=data["label"],
            time_span=TimeSpan.from_dict(data["time_span"]),
            confidence=data.get("confidence"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class RawTemporalEvent:
    event_id: str
    time_span: TimeSpan
    modality: str
    event_type: str
    text: str | None = None
    bbox: list[float] | None = None
    screen_region: str | None = None
    confidence: float = 1.0
    section_id: str | None = None
    source_node_id: str | None = None
    linked_events: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def start(self) -> float:
        return self.time_span.start

    @property
    def end(self) -> float:
        return self.time_span.end

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "start": self.time_span.start,
            "end": self.time_span.end,
            "modality": self.modality,
            "event_type": self.event_type,
            "text": self.text,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "screen_region": self.screen_region,
            "confidence": self.confidence,
            "section_id": self.section_id,
            "source_node_id": self.source_node_id,
            "linked_events": list(self.linked_events),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RawTemporalEvent":
        if "time_span" in data:
            time_span = TimeSpan.from_dict(data["time_span"])
        else:
            time_span = TimeSpan(start=float(data["start"]), end=float(data["end"]))
        bbox = data.get("bbox")
        return cls(
            event_id=data["event_id"],
            time_span=time_span,
            modality=str(data["modality"]),
            event_type=str(data["event_type"]),
            text=data.get("text"),
            bbox=[float(item) for item in bbox] if isinstance(bbox, list) else None,
            screen_region=data.get("screen_region"),
            confidence=float(data.get("confidence", 1.0)),
            section_id=data.get("section_id"),
            source_node_id=data.get("source_node_id"),
            linked_events=list(data.get("linked_events", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class SectionNode:
    section_id: str
    ordinal: int
    time_span: TimeSpan
    labels: list[str] = field(default_factory=list)
    evidence_events: list[str] = field(default_factory=list)
    dominant_modalities: list[str] = field(default_factory=list)
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "ordinal": self.ordinal,
            "start": self.time_span.start,
            "end": self.time_span.end,
            "labels": list(self.labels),
            "evidence_events": list(self.evidence_events),
            "dominant_modalities": list(self.dominant_modalities),
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SectionNode":
        if "time_span" in data:
            time_span = TimeSpan.from_dict(data["time_span"])
        else:
            time_span = TimeSpan(start=float(data["start"]), end=float(data["end"]))
        return cls(
            section_id=data["section_id"],
            ordinal=int(data.get("ordinal", 0)),
            time_span=time_span,
            labels=list(data.get("labels", [])),
            evidence_events=list(data.get("evidence_events", [])),
            dominant_modalities=list(data.get("dominant_modalities", [])),
            confidence=float(data.get("confidence", 1.0)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class CodeSnapshot:
    snapshot_id: str
    timestamp: float
    section_id: str | None = None
    active_lines: list[str] = field(default_factory=list)
    variables: dict[str, Any] = field(default_factory=dict)
    derived_from_events: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp,
            "section_id": self.section_id,
            "active_lines": list(self.active_lines),
            "variables": dict(self.variables),
            "derived_from_events": list(self.derived_from_events),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodeSnapshot":
        return cls(
            snapshot_id=data["snapshot_id"],
            timestamp=float(data["timestamp"]),
            section_id=data.get("section_id"),
            active_lines=list(data.get("active_lines", [])),
            variables=dict(data.get("variables", {})),
            derived_from_events=list(data.get("derived_from_events", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class OperatorEvent:
    operator: str
    operator_class: str
    first_seen: float
    section_id: str | None = None
    source: list[str] = field(default_factory=list)
    context: str = ""
    event_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator": self.operator,
            "operator_class": self.operator_class,
            "first_seen": self.first_seen,
            "section_id": self.section_id,
            "source": list(self.source),
            "context": self.context,
            "event_ids": list(self.event_ids),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OperatorEvent":
        return cls(
            operator=data["operator"],
            operator_class=str(data.get("operator_class", "comparison")),
            first_seen=float(data.get("first_seen", 0.0)),
            section_id=data.get("section_id"),
            source=list(data.get("source", [])),
            context=str(data.get("context", "")),
            event_ids=list(data.get("event_ids", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class TemporalLink:
    source: str
    relation: str
    target: str
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "relation": self.relation,
            "target": self.target,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TemporalLink":
        return cls(
            source=data["source"],
            relation=data["relation"],
            target=data["target"],
            confidence=float(data.get("confidence", 1.0)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class CrossModalTemporalIndex:
    sections: list[SectionNode] = field(default_factory=list)
    asr_events: list[RawTemporalEvent] = field(default_factory=list)
    ocr_events: list[RawTemporalEvent] = field(default_factory=list)
    code_line_events: list[RawTemporalEvent] = field(default_factory=list)
    terminal_events: list[RawTemporalEvent] = field(default_factory=list)
    visual_anchor_events: list[RawTemporalEvent] = field(default_factory=list)
    audio_events: list[RawTemporalEvent] = field(default_factory=list)
    code_snapshots: list[CodeSnapshot] = field(default_factory=list)
    operator_events: list[OperatorEvent] = field(default_factory=list)
    temporal_links: list[TemporalLink] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def all_events(self) -> list[RawTemporalEvent]:
        return sorted(
            [
                *self.asr_events,
                *self.ocr_events,
                *self.code_line_events,
                *self.terminal_events,
                *self.visual_anchor_events,
                *self.audio_events,
            ],
            key=lambda item: (item.time_span.start, item.event_id),
        )

    def resolve_section(self, query: str) -> SectionNode | None:
        query_tokens = _temporal_index_tokens(query)
        if not query_tokens:
            return None
        scored: list[tuple[float, float, SectionNode]] = []
        for section in self.sections:
            text = " ".join([section.section_id, *section.labels])
            section_tokens = _temporal_index_tokens(text)
            overlap = len(query_tokens & section_tokens)
            score = float(overlap)
            lowered = query.lower()
            if "comparison" in lowered and "comparison_section" == section.section_id:
                score += 3.0
            if "arithmetic" in lowered and "arithmetic_section" == section.section_id:
                score += 3.0
            if "assignment" in lowered and "assignment_section" == section.section_id:
                score += 3.0
            if "third segment" in lowered and "third_segment" in section.labels:
                score += 3.0
            if "second half" in lowered and "second_half" in section.labels:
                score += 3.0
            if score > 0:
                scored.append((score, -section.time_span.duration, section))
        if not scored:
            return None
        scored.sort(key=lambda item: (-item[0], item[1], item[2].time_span.start))
        return scored[0][2]

    def find_events(
        self,
        *,
        modality: str | None = None,
        event_type: str | None = None,
        text_query: str | None = None,
        section_id: str | None = None,
        temporal_query: str | None = None,
        before: float | None = None,
        after: float | None = None,
        screen_region: str | None = None,
        limit: int | None = None,
    ) -> list[RawTemporalEvent]:
        if section_id is None and temporal_query:
            resolved_section = self.resolve_section(temporal_query)
            if resolved_section is not None:
                section_id = resolved_section.section_id
        query_tokens = _temporal_index_tokens(text_query or "")
        events: list[tuple[float, RawTemporalEvent]] = []
        for event in self.all_events():
            if modality and event.modality != modality:
                continue
            if event_type and event.event_type != event_type:
                continue
            if section_id and event.section_id != section_id:
                continue
            if before is not None and event.time_span.start >= before:
                continue
            if after is not None and event.time_span.end <= after:
                continue
            if screen_region and event.screen_region != screen_region:
                continue
            score = 1.0
            if query_tokens:
                text_tokens = _temporal_index_tokens(event.text or "")
                overlap = len(query_tokens & text_tokens)
                if overlap <= 0:
                    continue
                score += float(overlap)
            if temporal_query:
                score += _temporal_constraint_score(event, temporal_query)
            events.append((score, event))
        events.sort(key=lambda item: (-item[0], item[1].time_span.start, item[1].event_id))
        selected = [event for _, event in events]
        return selected[:limit] if limit is not None else selected

    def resolve_temporal_interval(self, query: str) -> TimeSpan | None:
        section = self.resolve_section(query)
        if section is not None:
            return section.time_span
        duration = float(self.metadata.get("duration_seconds") or 0.0)
        if duration <= 0:
            return None
        lowered = query.lower()
        if any(cue in lowered for cue in ("first half", "beginning", "early part")):
            return TimeSpan(0.0, duration / 2.0)
        if any(cue in lowered for cue in ("second half", "later part", "ending", "final part")):
            return TimeSpan(duration / 2.0, duration)
        return None

    def audio_event_index(self) -> dict[str, Any]:
        events = []
        for event in self.audio_events:
            events.append(
                {
                    "event_id": event.event_id,
                    "label": event.text,
                    "time_span": event.time_span.to_dict(),
                    "confidence": event.confidence,
                    "linked_events": list(event.linked_events),
                    "occurrence_index": event.metadata.get("occurrence_index"),
                    "occurrence_count": event.metadata.get("occurrence_count"),
                    "section_id": event.section_id,
                    "source": event.metadata.get("audio_index_source"),
                    "tags": list(event.metadata.get("audio_tags", [])),
                    "caption": event.metadata.get("audio_caption"),
                }
            )
        return {"events": events, "event_count": len(events)}

    def get_code_snapshot(
        self,
        time_or_section: float | str | None = None,
    ) -> CodeSnapshot | None:
        if not self.code_snapshots:
            return None
        if isinstance(time_or_section, str):
            matching = [
                snapshot
                for snapshot in self.code_snapshots
                if snapshot.section_id == time_or_section
            ]
            if matching:
                return max(matching, key=lambda item: item.timestamp)
        if isinstance(time_or_section, int | float):
            before = [
                snapshot
                for snapshot in self.code_snapshots
                if snapshot.timestamp <= float(time_or_section)
            ]
            if before:
                return max(before, key=lambda item: item.timestamp)
        return max(self.code_snapshots, key=lambda item: item.timestamp)

    def evaluate_code(
        self,
        target_variable: str,
        time_or_section: float | str | None = None,
    ) -> Any | None:
        snapshot = self.get_code_snapshot(time_or_section)
        if snapshot is None:
            return None
        return snapshot.variables.get(target_variable)

    def list_operators(
        self,
        *,
        section_id: str | None = None,
        operator_class: str | None = None,
    ) -> list[str]:
        ordered: list[str] = []
        for event in sorted(self.operator_events, key=lambda item: item.first_seen):
            if section_id and event.section_id != section_id:
                continue
            if operator_class and event.operator_class != operator_class:
                continue
            if event.operator not in ordered:
                ordered.append(event.operator)
        return ordered

    def extract_terminal_output(
        self,
        *,
        section_id: str | None = None,
        time_window: TimeSpan | None = None,
    ) -> list[str]:
        outputs: list[str] = []
        for event in self.terminal_events:
            if section_id and event.section_id != section_id:
                continue
            if time_window and not event.time_span.overlaps(time_window):
                continue
            text = (event.text or "").strip()
            if text and text not in outputs:
                outputs.append(text)
        return outputs

    def to_dict(self) -> dict[str, Any]:
        return {
            "sections": [item.to_dict() for item in self.sections],
            "asr_events": [item.to_dict() for item in self.asr_events],
            "ocr_events": [item.to_dict() for item in self.ocr_events],
            "code_line_events": [item.to_dict() for item in self.code_line_events],
            "terminal_events": [item.to_dict() for item in self.terminal_events],
            "visual_anchor_events": [
                item.to_dict() for item in self.visual_anchor_events
            ],
            "audio_events": [item.to_dict() for item in self.audio_events],
            "code_snapshots": [item.to_dict() for item in self.code_snapshots],
            "operator_events": [item.to_dict() for item in self.operator_events],
            "temporal_links": [item.to_dict() for item in self.temporal_links],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CrossModalTemporalIndex":
        return cls(
            sections=[SectionNode.from_dict(item) for item in data.get("sections", [])],
            asr_events=[
                RawTemporalEvent.from_dict(item) for item in data.get("asr_events", [])
            ],
            ocr_events=[
                RawTemporalEvent.from_dict(item) for item in data.get("ocr_events", [])
            ],
            code_line_events=[
                RawTemporalEvent.from_dict(item)
                for item in data.get("code_line_events", [])
            ],
            terminal_events=[
                RawTemporalEvent.from_dict(item)
                for item in data.get("terminal_events", [])
            ],
            visual_anchor_events=[
                RawTemporalEvent.from_dict(item)
                for item in data.get("visual_anchor_events", [])
            ],
            audio_events=[
                RawTemporalEvent.from_dict(item) for item in data.get("audio_events", [])
            ],
            code_snapshots=[
                CodeSnapshot.from_dict(item) for item in data.get("code_snapshots", [])
            ],
            operator_events=[
                OperatorEvent.from_dict(item) for item in data.get("operator_events", [])
            ],
            temporal_links=[
                TemporalLink.from_dict(item) for item in data.get("temporal_links", [])
            ],
            metadata=dict(data.get("metadata", {})),
        )


def _temporal_index_tokens(text: str) -> set[str]:
    return set(
        token
        for token in re.findall(r"[a-z0-9_]+", text.lower().replace("-", "_"))
        if len(token) > 1
    )


def _temporal_constraint_score(event: RawTemporalEvent, query: str) -> float:
    lowered = query.lower()
    score = 0.0
    occurrence_index = event.metadata.get("occurrence_index")
    occurrence_count = event.metadata.get("occurrence_count")
    section_local_ordinal = event.metadata.get("section_local_ordinal")
    if _is_int_like(occurrence_index):
        index = int(occurrence_index)
        if any(cue in lowered for cue in ("first", "earliest", "initial", "beginning")):
            score += 2.0 if index == 1 else -0.25
        if any(cue in lowered for cue in ("second", "2nd")):
            score += 2.0 if index == 2 else -0.15
        if any(cue in lowered for cue in ("third", "3rd")):
            score += 2.0 if index == 3 else -0.15
    if _is_int_like(occurrence_index) and _is_int_like(occurrence_count):
        if any(cue in lowered for cue in ("last", "latest", "final", "ending")):
            score += 2.0 if int(occurrence_index) == int(occurrence_count) else -0.25
    if _is_int_like(section_local_ordinal):
        local_index = int(section_local_ordinal)
        if "section" in lowered and "first" in lowered:
            score += 1.0 if local_index == 1 else -0.1
        if "section" in lowered and "third" in lowered:
            score += 1.0 if local_index == 3 else -0.1
    if any(cue in lowered for cue in ("before", "previous")) and event.metadata.get(
        "next_same_entity_event"
    ):
        score += 0.35
    if any(cue in lowered for cue in ("after", "next")) and event.metadata.get(
        "previous_same_entity_event"
    ):
        score += 0.35
    return score


def _is_int_like(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, str):
        return value.isdigit()
    return False


@dataclass
class VideoNode:
    node_id: str
    level: VideoNodeLevel
    time_span: TimeSpan
    visual_summary: str = ""
    speech_spans: list[SpeechSpan] = field(default_factory=list)
    ocr_spans: list[OCRSpan] = field(default_factory=list)
    audio_events: list[AudioEvent] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    clip_path: str | None = None
    keyframe_paths: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    parent_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    uncertainty: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "level": self.level,
            "time_span": self.time_span.to_dict(),
            "visual_summary": self.visual_summary,
            "speech_spans": [span.to_dict() for span in self.speech_spans],
            "ocr_spans": [span.to_dict() for span in self.ocr_spans],
            "audio_events": [event.to_dict() for event in self.audio_events],
            "tags": list(self.tags),
            "entities": list(self.entities),
            "clip_path": self.clip_path,
            "keyframe_paths": list(self.keyframe_paths),
            "children": list(self.children),
            "parent_id": self.parent_id,
            "metadata": dict(self.metadata),
            "uncertainty": self.uncertainty,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VideoNode":
        return cls(
            node_id=data["node_id"],
            level=data["level"],
            time_span=TimeSpan.from_dict(data["time_span"]),
            visual_summary=data.get("visual_summary", ""),
            speech_spans=[SpeechSpan.from_dict(item) for item in data.get("speech_spans", [])],
            ocr_spans=[OCRSpan.from_dict(item) for item in data.get("ocr_spans", [])],
            audio_events=[AudioEvent.from_dict(item) for item in data.get("audio_events", [])],
            tags=list(data.get("tags", [])),
            entities=list(data.get("entities", [])),
            clip_path=data.get("clip_path"),
            keyframe_paths=list(data.get("keyframe_paths", [])),
            children=list(data.get("children", [])),
            parent_id=data.get("parent_id"),
            metadata=dict(data.get("metadata", {})),
            uncertainty=data.get("uncertainty"),
        )


@dataclass
class VideoMemory:
    video_id: str
    root_id: str
    nodes: dict[str, VideoNode]
    cross_modal_index: CrossModalTemporalIndex | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_node(self, node_id: str) -> VideoNode:
        if node_id not in self.nodes:
            raise KeyError(f"Unknown node_id: {node_id}")
        return self.nodes[node_id]

    def child_nodes(self, node_id: str) -> list[VideoNode]:
        return [self.get_node(child_id) for child_id in self.get_node(node_id).children]

    def top_level_nodes(self) -> list[VideoNode]:
        return self.child_nodes(self.root_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "root_id": self.root_id,
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "cross_modal_index": (
                self.cross_modal_index.to_dict() if self.cross_modal_index is not None else None
            ),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VideoMemory":
        return cls(
            video_id=data["video_id"],
            root_id=data["root_id"],
            nodes={
                node_id: VideoNode.from_dict(node_data)
                for node_id, node_data in data.get("nodes", {}).items()
            },
            cross_modal_index=(
                CrossModalTemporalIndex.from_dict(data["cross_modal_index"])
                if data.get("cross_modal_index") is not None
                else None
            ),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class FrontierItem:
    node_id: str
    time_span: TimeSpan
    level: VideoNodeLevel
    score: float
    why_candidate: str
    recommended_modalities: list[Modality] = field(default_factory=list)
    status: FrontierStatus = "unopened"

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "time_span": self.time_span.to_dict(),
            "level": self.level,
            "score": self.score,
            "why_candidate": self.why_candidate,
            "recommended_modalities": list(self.recommended_modalities),
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrontierItem":
        return cls(
            node_id=data["node_id"],
            time_span=TimeSpan.from_dict(data["time_span"]),
            level=data["level"],
            score=float(data["score"]),
            why_candidate=data["why_candidate"],
            recommended_modalities=list(data.get("recommended_modalities", [])),
            status=data.get("status", "unopened"),
        )


@dataclass
class EvidenceSlotSpec:
    slot: str
    description: str
    required: bool = True
    preferred_modality: Modality | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "description": self.description,
            "required": self.required,
            "preferred_modality": self.preferred_modality,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceSlotSpec":
        return cls(
            slot=data["slot"],
            description=data["description"],
            required=bool(data.get("required", True)),
            preferred_modality=data.get("preferred_modality"),
        )


@dataclass
class QuestionSpec:
    question_type: str
    required_slots: list[EvidenceSlotSpec]
    preferred_modality: Modality | None = None
    answer_policy: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_type": self.question_type,
            "required_slots": [slot.to_dict() for slot in self.required_slots],
            "preferred_modality": self.preferred_modality,
            "answer_policy": self.answer_policy,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuestionSpec":
        return cls(
            question_type=data["question_type"],
            required_slots=[
                EvidenceSlotSpec.from_dict(item) for item in data.get("required_slots", [])
            ],
            preferred_modality=data.get("preferred_modality"),
            answer_policy=data.get("answer_policy"),
            metadata=dict(data.get("metadata", {})),
        )

    def slot_names(self) -> list[str]:
        return [slot.slot for slot in self.required_slots]

    def get_slot(self, slot_name: str) -> EvidenceSlotSpec | None:
        for slot in self.required_slots:
            if slot.slot == slot_name:
                return slot
        return None


@dataclass
class EvidenceBoardSlot:
    slot: str
    description: str
    required: bool = True
    status: SlotStatus = "missing"
    core_evidence_ids: list[str] = field(default_factory=list)
    support_evidence_ids: list[str] = field(default_factory=list)
    background_evidence_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "description": self.description,
            "required": self.required,
            "status": self.status,
            "core_evidence_ids": list(self.core_evidence_ids),
            "support_evidence_ids": list(self.support_evidence_ids),
            "background_evidence_ids": list(self.background_evidence_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceBoardSlot":
        return cls(
            slot=data["slot"],
            description=data["description"],
            required=bool(data.get("required", True)),
            status=data.get("status", "missing"),
            core_evidence_ids=list(data.get("core_evidence_ids", [])),
            support_evidence_ids=list(data.get("support_evidence_ids", [])),
            background_evidence_ids=list(data.get("background_evidence_ids", [])),
        )


@dataclass
class OpenedTarget:
    node_id: str
    modality: Modality
    target_slot: str | None
    result: str
    step_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "modality": self.modality,
            "target_slot": self.target_slot,
            "result": self.result,
            "step_index": self.step_index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OpenedTarget":
        return cls(
            node_id=data["node_id"],
            modality=data["modality"],
            target_slot=data.get("target_slot"),
            result=data["result"],
            step_index=int(data.get("step_index", 0)),
        )


@dataclass
class EvidenceBoard:
    question_type: str
    slots: dict[str, EvidenceBoardSlot]
    opened_targets: list[OpenedTarget] = field(default_factory=list)
    missing_required_slots: list[str] = field(default_factory=list)
    slot_query_hints: dict[str, list[str]] = field(default_factory=dict)
    slot_refinement_node_ids: dict[str, list[str]] = field(default_factory=dict)
    core_evidence_ids: list[str] = field(default_factory=list)
    support_evidence_ids: list[str] = field(default_factory=list)
    background_evidence_ids: list[str] = field(default_factory=list)
    slot_fill_count: int = 0
    background_only_open_count: int = 0
    duplicate_evidence_count: int = 0
    no_progress_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_type": self.question_type,
            "slots": {name: slot.to_dict() for name, slot in self.slots.items()},
            "opened_targets": [item.to_dict() for item in self.opened_targets],
            "missing_required_slots": list(self.missing_required_slots),
            "slot_query_hints": {
                name: list(queries) for name, queries in self.slot_query_hints.items()
            },
            "slot_refinement_node_ids": {
                name: list(node_ids) for name, node_ids in self.slot_refinement_node_ids.items()
            },
            "core_evidence_ids": list(self.core_evidence_ids),
            "support_evidence_ids": list(self.support_evidence_ids),
            "background_evidence_ids": list(self.background_evidence_ids),
            "slot_fill_count": self.slot_fill_count,
            "background_only_open_count": self.background_only_open_count,
            "duplicate_evidence_count": self.duplicate_evidence_count,
            "no_progress_count": self.no_progress_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceBoard":
        return cls(
            question_type=data["question_type"],
            slots={
                name: EvidenceBoardSlot.from_dict(slot_data)
                for name, slot_data in data.get("slots", {}).items()
            },
            opened_targets=[
                OpenedTarget.from_dict(item) for item in data.get("opened_targets", [])
            ],
            missing_required_slots=list(data.get("missing_required_slots", [])),
            slot_query_hints={
                name: list(queries) for name, queries in data.get("slot_query_hints", {}).items()
            },
            slot_refinement_node_ids={
                name: list(node_ids)
                for name, node_ids in data.get("slot_refinement_node_ids", {}).items()
            },
            core_evidence_ids=list(data.get("core_evidence_ids", [])),
            support_evidence_ids=list(data.get("support_evidence_ids", [])),
            background_evidence_ids=list(data.get("background_evidence_ids", [])),
            slot_fill_count=int(data.get("slot_fill_count", 0)),
            background_only_open_count=int(data.get("background_only_open_count", 0)),
            duplicate_evidence_count=int(data.get("duplicate_evidence_count", 0)),
            no_progress_count=int(data.get("no_progress_count", 0)),
        )

    def is_slot_filled(self, slot_name: str) -> bool:
        slot = self.slots.get(slot_name)
        return slot is not None and slot.status == "filled"


@dataclass
class Evidence:
    evidence_id: str
    claim: str
    modality: Modality
    time_span: TimeSpan
    source_node_id: str
    confidence: float
    detail: str = ""
    used_in_final_answer: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "claim": self.claim,
            "modality": self.modality,
            "time_span": self.time_span.to_dict(),
            "source_node_id": self.source_node_id,
            "confidence": self.confidence,
            "detail": self.detail,
            "used_in_final_answer": self.used_in_final_answer,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Evidence":
        return cls(
            evidence_id=data["evidence_id"],
            claim=data["claim"],
            modality=data["modality"],
            time_span=TimeSpan.from_dict(data["time_span"]),
            source_node_id=data["source_node_id"],
            confidence=float(data["confidence"]),
            detail=data.get("detail", ""),
            used_in_final_answer=bool(data.get("used_in_final_answer", False)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class EventInterval:
    time_span: TimeSpan
    evidence_id: str
    source_node_id: str
    confidence: float
    match_score: float
    detail: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_span": self.time_span.to_dict(),
            "evidence_id": self.evidence_id,
            "source_node_id": self.source_node_id,
            "confidence": self.confidence,
            "match_score": self.match_score,
            "detail": self.detail,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EventInterval":
        return cls(
            time_span=TimeSpan.from_dict(data["time_span"]),
            evidence_id=data["evidence_id"],
            source_node_id=data["source_node_id"],
            confidence=float(data["confidence"]),
            match_score=float(data.get("match_score", data.get("confidence", 0.0))),
            detail=data.get("detail", ""),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class EventMemoryEvent:
    event_id: str
    phrase: str
    source: str = "question"
    option_letter: str | None = None
    status: EventStatus = "missing"
    intervals: list[EventInterval] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "phrase": self.phrase,
            "source": self.source,
            "option_letter": self.option_letter,
            "status": self.status,
            "intervals": [interval.to_dict() for interval in self.intervals],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EventMemoryEvent":
        return cls(
            event_id=data["event_id"],
            phrase=data["phrase"],
            source=data.get("source", "question"),
            option_letter=data.get("option_letter"),
            status=data.get("status", "missing"),
            intervals=[
                EventInterval.from_dict(item) for item in data.get("intervals", [])
            ],
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class EventMemory:
    task_name: str
    question: str
    mode: str | None = None
    events: dict[str, EventMemoryEvent] = field(default_factory=dict)
    relations: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "question": self.question,
            "mode": self.mode,
            "events": {
                event_id: event.to_dict() for event_id, event in self.events.items()
            },
            "relations": [dict(relation) for relation in self.relations],
            "metadata": dict(self.metadata),
            "localized_event_count": self.localized_event_count,
            "missing_event_ids": self.missing_event_ids,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EventMemory":
        return cls(
            task_name=data["task_name"],
            question=data["question"],
            mode=data.get("mode"),
            events={
                event_id: EventMemoryEvent.from_dict(event_data)
                for event_id, event_data in data.get("events", {}).items()
            },
            relations=[dict(item) for item in data.get("relations", [])],
            metadata=dict(data.get("metadata", {})),
        )

    @property
    def localized_event_count(self) -> int:
        return sum(1 for event in self.events.values() if event.status == "localized")

    @property
    def missing_event_ids(self) -> list[str]:
        return [
            event.event_id
            for event in self.events.values()
            if event.status != "localized"
        ]


@dataclass
class BudgetState:
    steps_used: int = 0
    steps_remaining: int = 0
    tool_calls_used: int = 0
    max_depth: int = 0
    current_depth: int = 0
    clips_opened: int = 0
    tokens_spent: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps_used": self.steps_used,
            "steps_remaining": self.steps_remaining,
            "tool_calls_used": self.tool_calls_used,
            "max_depth": self.max_depth,
            "current_depth": self.current_depth,
            "clips_opened": self.clips_opened,
            "tokens_spent": self.tokens_spent,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BudgetState":
        return cls(
            steps_used=int(data.get("steps_used", 0)),
            steps_remaining=int(data.get("steps_remaining", 0)),
            tool_calls_used=int(data.get("tool_calls_used", 0)),
            max_depth=int(data.get("max_depth", 0)),
            current_depth=int(data.get("current_depth", 0)),
            clips_opened=int(data.get("clips_opened", 0)),
            tokens_spent=int(data.get("tokens_spent", 0)),
        )


@dataclass
class ControllerAction:
    action_type: ActionType
    query: str | None = None
    modality: Modality | None = None
    node_id: str | None = None
    target_slot: str | None = None
    evidence_ids: list[str] = field(default_factory=list)
    answer: str | None = None
    rationale: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action_type == "SEARCH" and (not self.query or not self.modality):
            raise ValueError("SEARCH requires query and modality")
        if self.action_type == "OPEN" and (not self.node_id or not self.modality):
            raise ValueError("OPEN requires node_id and modality")
        if self.action_type == "SPLIT" and not self.node_id:
            raise ValueError("SPLIT requires node_id")
        if self.action_type == "MERGE" and not self.evidence_ids:
            raise ValueError("MERGE requires evidence_ids")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "query": self.query,
            "modality": self.modality,
            "node_id": self.node_id,
            "target_slot": self.target_slot,
            "evidence_ids": list(self.evidence_ids),
            "answer": self.answer,
            "rationale": self.rationale,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ControllerAction":
        return cls(
            action_type=data["action_type"],
            query=data.get("query"),
            modality=data.get("modality"),
            node_id=data.get("node_id"),
            target_slot=data.get("target_slot"),
            evidence_ids=list(data.get("evidence_ids", [])),
            answer=data.get("answer"),
            rationale=data.get("rationale"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class Observation:
    kind: str
    summary: str
    frontier: list[FrontierItem] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    node_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "summary": self.summary,
            "frontier": [item.to_dict() for item in self.frontier],
            "evidence": [item.to_dict() for item in self.evidence],
            "node_id": self.node_id,
            "metadata": dict(self.metadata),
        }


@dataclass
class ControllerState:
    question: str
    task_type: str | None = None
    dialogue_context: list[dict[str, str]] = field(default_factory=list)
    question_spec: QuestionSpec | None = None
    subquestion: str | None = None
    frontier: list[FrontierItem] = field(default_factory=list)
    evidence_ledger: list[Evidence] = field(default_factory=list)
    evidence_board: EvidenceBoard | None = None
    event_memory: EventMemory | None = None
    action_history: list[dict[str, Any]] = field(default_factory=list)
    budget: BudgetState = field(default_factory=BudgetState)
    global_context: dict[str, Any] = field(default_factory=dict)
    no_progress_steps: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "task_type": self.task_type,
            "dialogue_context": list(self.dialogue_context),
            "question_spec": self.question_spec.to_dict() if self.question_spec else None,
            "subquestion": self.subquestion,
            "frontier": [item.to_dict() for item in self.frontier],
            "evidence_ledger": [item.to_dict() for item in self.evidence_ledger],
            "evidence_board": self.evidence_board.to_dict() if self.evidence_board else None,
            "event_memory": self.event_memory.to_dict() if self.event_memory else None,
            "action_history": list(self.action_history),
            "budget": self.budget.to_dict(),
            "global_context": dict(self.global_context),
            "no_progress_steps": self.no_progress_steps,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ControllerState":
        return cls(
            question=data["question"],
            task_type=data.get("task_type"),
            dialogue_context=list(data.get("dialogue_context", [])),
            question_spec=(
                QuestionSpec.from_dict(data["question_spec"])
                if data.get("question_spec") is not None
                else None
            ),
            subquestion=data.get("subquestion"),
            frontier=[FrontierItem.from_dict(item) for item in data.get("frontier", [])],
            evidence_ledger=[Evidence.from_dict(item) for item in data.get("evidence_ledger", [])],
            evidence_board=(
                EvidenceBoard.from_dict(data["evidence_board"])
                if data.get("evidence_board") is not None
                else None
            ),
            event_memory=(
                EventMemory.from_dict(data["event_memory"])
                if data.get("event_memory") is not None
                else None
            ),
            action_history=list(data.get("action_history", [])),
            budget=BudgetState.from_dict(data.get("budget", {})),
            global_context=dict(data.get("global_context", {})),
            no_progress_steps=int(data.get("no_progress_steps", 0)),
        )

    def frontier_ids(self) -> set[str]:
        return {item.node_id for item in self.frontier}

    def evidence_by_id(self) -> dict[str, Evidence]:
        return {item.evidence_id: item for item in self.evidence_ledger}


@dataclass
class TraceStep:
    step_index: int
    state: dict[str, Any]
    action: dict[str, Any]
    observation: dict[str, Any]
    next_state: dict[str, Any]
    raw_model_response: str | None = None
    timing: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "state": self.state,
            "action": self.action,
            "observation": self.observation,
            "next_state": self.next_state,
            "raw_model_response": self.raw_model_response,
            "timing": dict(self.timing),
        }


@dataclass
class VideoRLMResult:
    answer: str
    state: ControllerState
    trace: list[dict[str, Any]]
    usage_summary: UsageSummary
    execution_time: float
    timing: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "state": self.state.to_dict(),
            "trace": list(self.trace),
            "usage_summary": self.usage_summary.to_dict(),
            "execution_time": self.execution_time,
            "timing": dict(self.timing),
        }
