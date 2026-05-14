from dataclasses import dataclass, field
from typing import Any, Literal

from rlm.core.types import UsageSummary

VideoNodeLevel = Literal["video", "scene", "segment", "clip"]
Modality = Literal["speech", "visual", "ocr", "audio", "cross_modal"]
ActionType = Literal["SEARCH", "OPEN", "SPLIT", "MERGE", "STOP"]
FrontierStatus = Literal["unopened", "opened", "expanded", "exhausted"]
SlotRole = Literal["core", "support", "background", "noise"]
SlotStatus = Literal["missing", "filled", "background_only"]


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "time_span": self.time_span.to_dict(),
            "speaker": self.speaker,
            "language": self.language,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpeechSpan":
        return cls(
            text=data["text"],
            time_span=TimeSpan.from_dict(data["time_span"]),
            speaker=data.get("speaker"),
            language=data.get("language"),
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "time_span": self.time_span.to_dict(),
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AudioEvent":
        return cls(
            label=data["label"],
            time_span=TimeSpan.from_dict(data["time_span"]),
            confidence=data.get("confidence"),
        )


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "state": self.state,
            "action": self.action,
            "observation": self.observation,
            "next_state": self.next_state,
            "raw_model_response": self.raw_model_response,
        }


@dataclass
class VideoRLMResult:
    answer: str
    state: ControllerState
    trace: list[dict[str, Any]]
    usage_summary: UsageSummary
    execution_time: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "state": self.state.to_dict(),
            "trace": list(self.trace),
            "usage_summary": self.usage_summary.to_dict(),
            "execution_time": self.execution_time,
        }
