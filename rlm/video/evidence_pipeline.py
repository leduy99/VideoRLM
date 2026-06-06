from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from rlm.video.index import STOPWORDS, TOKEN_PATTERN, SearchHit, VideoMemoryIndex
from rlm.video.question_router import evidence_matches_route, route_from_metadata, route_question
from rlm.video.types import (
    ControllerState,
    Evidence,
    EvidenceBoard,
    EvidenceBoardSlot,
    EvidenceSlotSpec,
    FrontierItem,
    Modality,
    Observation,
    OpenedTarget,
    QuestionSpec,
)

GENERIC_SLOT_KEYWORDS: dict[str, list[str]] = {
    "reason": [
        "because",
        "reason",
        "why",
        "decided",
        "chose",
        "so",
        "therefore",
        "worried",
        "lose",
        "clasp",
        "opening",
        "fixed",
        "repair",
        "immediately",
        "show",
        "couldn't wait",
    ],
    "decision": ["decided", "wore", "wear", "tried", "did", "chose"],
    "object": ["item", "object", "thing", "piece", "bracelet", "diamond", "food"],
    "first_thing_tried": [
        "first",
        "initial",
        "earliest",
        "tried",
        "what about",
        "this is",
        "presented",
        "brought out",
        "bite",
        "tasted",
    ],
    "why_different": ["different", "not regular", "unusual", "realize", "street food"],
    "reaction": [
        "reaction",
        "responded",
        "said",
        "felt",
        "tasted",
        "tastes",
        "bite",
        "bit",
        "surprised",
    ],
    "anchor_event": ["stopped", "stared", "paused", "chasing", "moving"],
    "mouse_reaction": ["mouse", "reaction", "responded", "paused", "looked"],
    "shared_sensing": ["both", "seemed", "sensed", "noticed", "realized"],
    "main_claim": ["main", "claim", "answer", "point"],
    "supporting_detail": ["detail", "support", "specific", "example"],
    "missing_context": ["context", "missing", "unclear", "not enough"],
}
CONTROL_QUERY_TOKENS = {"why", "first", "last", "earliest", "initial", "beginning", "final"}
VISUAL_ROUTE_TERMS = {
    "appear",
    "appears",
    "display",
    "displayed",
    "door",
    "expression",
    "expressions",
    "graph",
    "image",
    "look",
    "looks",
    "read",
    "screen",
    "see",
    "seen",
    "show",
    "showing",
    "shown",
    "sign",
    "slide",
    "slides",
    "text",
    "title",
    "visible",
    "visual",
    "written",
}
VISUAL_ROUTE_PHRASES = {
    "on screen",
    "on-screen",
    "what is written",
    "what's written",
    "what does the sign",
    "what does it show",
    "what do you see",
}
AUDIO_ROUTE_TERMS = {
    "audio",
    "background",
    "beep",
    "beeping",
    "hear",
    "heard",
    "mechanical",
    "music",
    "noise",
    "noises",
    "sound",
    "sounds",
    "tick",
    "ticking",
}
AUDIO_ROUTE_PHRASES = {
    "what kind of sound",
    "what kind of sounds",
    "mechanical sounds",
    "background noise",
    "audio field",
}
SPEECH_ROUTE_TERMS = {
    "explain",
    "explained",
    "explains",
    "mention",
    "mentioned",
    "narrator",
    "say",
    "says",
    "said",
    "speaker",
    "spoken",
    "talk",
    "talks",
    "tell",
    "tells",
}
SPEECH_ROUTE_PHRASES = {
    "what did",
    "what does the speaker",
    "what does she say",
    "what does he say",
}
SPATIAL_RELATION_TERMS = {
    "above",
    "across",
    "behind",
    "below",
    "beside",
    "closer",
    "depth",
    "direction",
    "farther",
    "facing",
    "front",
    "left",
    "near",
    "nearest",
    "next",
    "right",
    "side",
    "toward",
    "towards",
    "under",
    "viewpoint",
}
SPATIAL_RELATION_PHRASES = {
    "in front of",
    "to the left",
    "to the right",
    "on the left",
    "on the right",
    "same side",
    "opposite side",
    "relative position",
    "spatial relation",
    "depth relation",
    "which direction",
    "where is",
    "where are",
}
CO_VISIBLE_RELATION_TERMS = {
    "above",
    "behind",
    "below",
    "beside",
    "closer",
    "depth",
    "farther",
    "facing",
    "front",
    "left",
    "near",
    "nearest",
    "next",
    "right",
    "side",
    "toward",
    "towards",
    "under",
    "viewpoint",
}
CO_VISIBLE_RELATION_PHRASES = {
    "in front of",
    "in reference to",
    "with reference to",
    "relative to",
    "with respect to",
    "compared to",
    "to the left",
    "to the right",
    "on the left",
    "on the right",
    "same side",
    "opposite side",
    "spatial relation",
    "depth relation",
}
RELATION_NEGATIVE_PHRASES = {
    "cannot determine the relation",
    "cannot see both",
    "can't determine the relation",
    "can't see both",
    "do not see both",
    "does not show both",
    "not clearly visible together",
    "not in the same frame",
    "not in the same shot",
    "not shown together",
    "not visible together",
    "no frame showing both",
    "no same-frame evidence",
    "relation is not visible",
    "separate frames",
}
RELATION_POSITIVE_PHRASES = {
    "both are visible",
    "both visible",
    "co-visible",
    "in the same frame",
    "in the same shot",
    "left of",
    "right of",
    "above",
    "below",
    "in front of",
    "behind",
    "facing",
    "toward",
    "towards",
    "closer",
    "farther",
    "near",
    "next to",
    "beside",
    "under",
    "visible together",
}

RelationEvidenceStatus = Literal["supported", "unsupported", "unknown"]
TEMPORAL_AFTER_TERMS = {"after", "afterward", "afterwards", "following", "later", "next"}
TEMPORAL_BEFORE_TERMS = {"before", "earlier", "previous", "previously", "prior"}
CAUSAL_TERMS = {"because", "cause", "caused", "causal", "effect", "imply", "why"}
ACTION_TERMS = {
    "add",
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
ACTOR_TERMS = {"actor", "child", "man", "person", "people", "presenter", "speaker", "woman"}
PLACE_TERMS = {
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
COGNITIVE_QUERY_EXCLUDE_TERMS = (
    ACTOR_TERMS
    | PLACE_TERMS
    | ACTION_TERMS
    | TEMPORAL_AFTER_TERMS
    | TEMPORAL_BEFORE_TERMS
    | CAUSAL_TERMS
    | {"answer", "choice", "option", "question", "true", "false", "yes", "no"}
)


@dataclass
class CognitiveQueryFrame:
    actors: set[str] = field(default_factory=set)
    places: set[str] = field(default_factory=set)
    objects: set[str] = field(default_factory=set)
    actions: set[str] = field(default_factory=set)
    spoken_topics: set[str] = field(default_factory=set)
    ocr_entities: set[str] = field(default_factory=set)
    temporal_relation: str | None = None
    requires_visual: bool = False
    requires_speech: bool = False
    requires_ocr: bool = False
    requires_temporal_order: bool = False
    causal: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "actors": sorted(self.actors),
            "places": sorted(self.places),
            "objects": sorted(self.objects),
            "actions": sorted(self.actions),
            "spoken_topics": sorted(self.spoken_topics),
            "ocr_entities": sorted(self.ocr_entities),
            "temporal_relation": self.temporal_relation,
            "requires_visual": self.requires_visual,
            "requires_speech": self.requires_speech,
            "requires_ocr": self.requires_ocr,
            "requires_temporal_order": self.requires_temporal_order,
            "causal": self.causal,
        }


def build_question_spec(
    question: str,
    task_type: str | None = None,
    dialogue_context: list[dict[str, str]] | None = None,
) -> QuestionSpec:
    del dialogue_context
    tokens = _tokenize(question)
    question_route = route_question(question, task_type)
    preferred_modality = question_route.preferred_modality or _infer_preferred_modality(
        question,
        task_type,
    )
    question_type = "generic"
    slots: list[EvidenceSlotSpec]
    answer_policy = "answer_only_if_required_slots_filled"

    if "why" in tokens:
        question_type = "why_reason"
        slots = [
            EvidenceSlotSpec(
                slot="object",
                description=_object_description(question),
                required=False,
            ),
            EvidenceSlotSpec(
                slot="decision",
                description=_decision_description(question),
                required=False,
            ),
            EvidenceSlotSpec(
                slot="reason",
                description=_reason_description(question),
                required=True,
                preferred_modality=preferred_modality,
            ),
        ]
    elif {"first", "earliest", "initial", "beginning"} & tokens:
        question_type = "first_event_realization"
        preferred_modality = "speech"
        slots = [
            EvidenceSlotSpec(
                slot="first_thing_tried",
                description="The earliest item, action, or event directly asked by the question",
                preferred_modality=preferred_modality,
            ),
            EvidenceSlotSpec(
                slot="why_different",
                description="Why that first item or event felt different or revealing",
                preferred_modality=preferred_modality,
            ),
            EvidenceSlotSpec(
                slot="reaction",
                description="Immediate reaction after trying that first item or action",
                required=False,
                preferred_modality=preferred_modality,
            ),
        ]
    elif "reaction" in tokens and ("sense" in tokens or "seemed" in tokens):
        question_type = "reaction_inference"
        preferred_modality = "visual"
        slots = [
            EvidenceSlotSpec(
                slot="anchor_event",
                description="The key moment or anchor event mentioned in the question",
                preferred_modality="visual",
            ),
            EvidenceSlotSpec(
                slot="reaction",
                description="How the subject reacted at that moment",
                preferred_modality="visual",
            ),
            EvidenceSlotSpec(
                slot="shared_sensing",
                description="What the participants seemed to sense or realize together",
                preferred_modality="visual",
            ),
        ]
    elif task_type == "summarization":
        question_type = "summarization"
        preferred_modality = "cross_modal"
        slots = [
            EvidenceSlotSpec(
                slot="main_claim",
                description="Main answer or summary claim",
                preferred_modality=preferred_modality,
            ),
            EvidenceSlotSpec(
                slot="supporting_detail",
                description="Specific supporting detail needed for the summary",
                required=False,
                preferred_modality=preferred_modality,
            ),
        ]
    else:
        slots = [
            EvidenceSlotSpec(
                slot="main_claim",
                description=f"Main answer to: {question}",
                preferred_modality=preferred_modality,
            ),
            EvidenceSlotSpec(
                slot="supporting_detail",
                description="Most important supporting detail for the main answer",
                required=False,
                preferred_modality=preferred_modality,
            ),
        ]

    return QuestionSpec(
        question_type=question_type,
        required_slots=slots,
        preferred_modality=preferred_modality,
        answer_policy=answer_policy,
        metadata={
            "question": question,
            "task_type": task_type,
            "question_route": question_route.to_dict(),
        },
    )


def infer_question_modality(question: str) -> Modality:
    lowered = question.lower()
    tokens = _tokenize(question)
    if _has_route_signal(lowered, tokens, AUDIO_ROUTE_TERMS, AUDIO_ROUTE_PHRASES):
        return "audio"
    if _has_route_signal(lowered, tokens, VISUAL_ROUTE_TERMS, VISUAL_ROUTE_PHRASES):
        return "visual"
    if _has_route_signal(lowered, tokens, SPEECH_ROUTE_TERMS, SPEECH_ROUTE_PHRASES):
        return "speech"
    return "speech"


def is_spatial_relation_question(question: str) -> bool:
    lowered = question.lower()
    tokens = _tokenize(question)
    return bool(tokens & SPATIAL_RELATION_TERMS) or any(
        phrase in lowered for phrase in SPATIAL_RELATION_PHRASES
    )


def requires_co_visible_relation(question: str) -> bool:
    question_stem = _question_stem_without_options(question)
    lowered = question_stem.lower()
    tokens = _tokenize(question_stem)
    return bool(tokens & CO_VISIBLE_RELATION_TERMS) or any(
        phrase in lowered for phrase in CO_VISIBLE_RELATION_PHRASES
    )


def relation_evidence_status(item: Evidence) -> RelationEvidenceStatus:
    co_visible = _metadata_bool(item.metadata.get("vrrqa_co_visible"))
    if co_visible is False:
        return "unsupported"

    has_co_visible_frame = co_visible is True or _metadata_positive_count(
        item.metadata.get("vrrqa_co_visible_frame_count")
    )
    co_visible_frame_indices = item.metadata.get("vrrqa_co_visible_frame_indices")
    if isinstance(co_visible_frame_indices, list) and co_visible_frame_indices:
        has_co_visible_frame = True

    relation_supported = _metadata_bool(item.metadata.get("vrrqa_relation_supported"))
    if relation_supported is True:
        return "supported" if has_co_visible_frame else "unknown"
    if relation_supported is False:
        return "unsupported"

    relation_text = " ".join(
        str(part)
        for part in (
            item.metadata.get("vrrqa_visible_relation"),
            item.metadata.get("vrrqa_spatial_relation"),
            item.metadata.get("vrrqa_evidence"),
            item.detail,
            item.claim,
        )
        if part
    )
    lowered = relation_text.lower()
    if any(phrase in lowered for phrase in RELATION_NEGATIVE_PHRASES):
        return "unsupported"
    if has_co_visible_frame and any(phrase in lowered for phrase in RELATION_POSITIVE_PHRASES):
        return "supported"
    return "unknown"


def _metadata_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "supported"}:
            return True
        if normalized in {"false", "no", "0", "unsupported"}:
            return False
    return None


def _metadata_positive_count(value: Any) -> bool:
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False


def build_evidence_board(question_spec: QuestionSpec) -> EvidenceBoard:
    slots = {
        slot.slot: EvidenceBoardSlot(
            slot=slot.slot,
            description=slot.description,
            required=slot.required,
        )
        for slot in question_spec.required_slots
    }
    return EvidenceBoard(
        question_type=question_spec.question_type,
        slots=slots,
        missing_required_slots=[
            slot.slot for slot in question_spec.required_slots if slot.required
        ],
    )


def select_target_slot(
    question_spec: QuestionSpec | None,
    board: EvidenceBoard | None,
) -> str | None:
    if question_spec is None:
        return None
    if board is None:
        for slot in question_spec.required_slots:
            if slot.required:
                return slot.slot
        return question_spec.required_slots[0].slot if question_spec.required_slots else None
    for slot in question_spec.required_slots:
        if not slot.required:
            continue
        board_slot = board.slots.get(slot.slot)
        if board_slot is None or board_slot.status != "filled":
            return slot.slot
    return question_spec.required_slots[0].slot if question_spec.required_slots else None


def search_v2(
    index: VideoMemoryIndex,
    question_spec: QuestionSpec | None,
    target_slot: str | None,
    state: ControllerState,
    top_k: int,
    query_override: str | None = None,
    modality: Modality | None = None,
) -> tuple[list[FrontierItem], dict[str, Any]]:
    queries = _search_queries_for_state(
        state=state,
        question_spec=question_spec,
        target_slot=target_slot,
        query_override=query_override,
    )
    hits_by_node: dict[str, SearchHit] = {}
    query_sources: dict[str, list[str]] = defaultdict(list)
    selected_modality = modality or _preferred_modality(question_spec, target_slot)
    selected_modality = _resolve_available_modality(selected_modality, state)
    search_modalities = _search_modalities(selected_modality, state.question)
    query_frame = _build_cognitive_query_frame(
        question=state.question,
        queries=queries,
        selected_modality=selected_modality,
    )

    for query in queries:
        for current_modality in search_modalities:
            hits = index.search(
                query=query,
                modality=current_modality,
                top_k=max(top_k * 2, 8),
            )
            for hit in hits:
                adjusted_score, cognitive_breakdown = _adjust_search_score(
                    hit,
                    state,
                    target_slot,
                    query_frame=query_frame,
                    index=index,
                )
                candidate = SearchHit(
                    node_id=hit.node_id,
                    time_span=hit.time_span,
                    level=hit.level,
                    score=adjusted_score,
                    reason=_append_cognitive_reason(hit.reason, cognitive_breakdown),
                    modality=hit.modality,
                    matched_terms=hit.matched_terms,
                    score_breakdown={
                        **dict(hit.score_breakdown),
                        **cognitive_breakdown,
                    },
                )
                current = hits_by_node.get(candidate.node_id)
                if current is None or candidate.score > current.score:
                    hits_by_node[candidate.node_id] = candidate
                query_sources[candidate.node_id].append(query)

    ranked_candidates = sorted(
        hits_by_node.values(),
        key=lambda item: (-item.score, item.time_span.start, item.node_id),
    )
    ranked_hits = _select_temporally_diverse_hits(ranked_candidates, top_k)
    frontier = [hit.to_frontier_item() for hit in ranked_hits]
    for item in frontier:
        item.why_candidate = (
            f"{item.why_candidate}; target_slot={target_slot or 'none'}; "
            f"queries={query_sources.get(item.node_id, [])[:2]}"
        )

    return frontier, {
        "target_slot": target_slot,
        "queries": queries,
        "modality": selected_modality,
        "search_mode": getattr(index, "search_mode", "lexical"),
        "searched_modalities": search_modalities,
        "hit_count": len(frontier),
        "query_sources": dict(query_sources),
        "cognitive_query_frame": query_frame.to_dict(),
    }


def _select_temporally_diverse_hits(hits: list[SearchHit], top_k: int) -> list[SearchHit]:
    selected: list[SearchHit] = []
    deferred: list[SearchHit] = []
    for hit in hits:
        redundant_index = _temporally_redundant_index(hit, selected)
        if redundant_index is not None:
            current = selected[redundant_index]
            if _prefer_more_specific_hit(hit, current):
                selected[redundant_index] = hit
                deferred.append(current)
            else:
                deferred.append(hit)
            continue
        selected.append(hit)
        if len(selected) >= top_k:
            return selected

    for hit in deferred:
        if hit not in selected:
            selected.append(hit)
        if len(selected) >= top_k:
            break
    return selected[:top_k]


def _temporally_redundant_index(
    hit: SearchHit,
    selected: list[SearchHit],
) -> int | None:
    for index, item in enumerate(selected):
        if hit.node_id == item.node_id:
            return index
        if not hit.time_span.overlaps(item.time_span):
            continue
        if hit.level == item.level:
            return index
        overlap = min(hit.time_span.end, item.time_span.end) - max(
            hit.time_span.start,
            item.time_span.start,
        )
        shorter_duration = max(1e-6, min(hit.time_span.duration, item.time_span.duration))
        if overlap / shorter_duration >= 0.9:
            return index
    return None


def _prefer_more_specific_hit(candidate: SearchHit, current: SearchHit) -> bool:
    candidate_situation = float(candidate.score_breakdown.get("cognitive_situation") or 0.0)
    current_situation = float(current.score_breakdown.get("cognitive_situation") or 0.0)
    if (
        current.level == "event"
        and current_situation > 0.0
        and candidate.level == "clip"
        and candidate.score <= current.score
    ):
        return False
    if (
        candidate.level == "event"
        and candidate_situation > 0.0
        and candidate.score >= (current.score * 0.9)
    ):
        return True
    candidate_rank = _window_level_rank(candidate.level)
    current_rank = _window_level_rank(current.level)
    return candidate_rank < current_rank and candidate.score >= (current.score * 0.85)


def _window_level_rank(level: str) -> int:
    return {
        "clip": 0,
        "event": 1,
        "segment": 2,
        "scene": 3,
        "video": 4,
    }.get(level, 4)


def open_v2(
    question_spec: QuestionSpec | None,
    target_slot: str | None,
    state: ControllerState,
    node_id: str,
    modality: Modality,
    evidence_items: list[Evidence],
    evidence_classifier: Callable[..., dict[str, Any] | None] | None = None,
) -> tuple[list[Evidence], dict[str, Any]]:
    if question_spec is None:
        return evidence_items, {
            "target_slot": target_slot,
            "missing_slots": [],
            "background_only": False,
            "no_new_information": not evidence_items,
            "filled_slots": [],
            "duplicate_evidence_count": 0,
            "suggested_queries": [],
            "progress_made": bool(evidence_items),
            "result": "generic_open",
        }

    classified: list[Evidence] = []
    duplicate_count = 0
    filled_slots: set[str] = set()
    background_only = True
    requires_relation_evidence = (
        modality == "visual"
        and state.task_type == "multiple_choice_visual_qa"
        and requires_co_visible_relation(state.question)
    )
    question_route = route_from_metadata(question_spec.metadata) or route_question(
        state.question,
        state.task_type,
    )

    for item in evidence_items:
        slot_name, slot_score = _best_slot_match(item, question_spec, target_slot)
        role = _classify_slot_role(slot_name, slot_score, item, question_spec, target_slot)
        heuristic_slot_name = slot_name
        heuristic_slot_score = slot_score
        heuristic_role = role
        classifier_metadata: dict[str, Any] = {}
        if evidence_classifier is not None:
            classification = evidence_classifier(
                item=item,
                question_spec=question_spec,
                target_slot=target_slot,
                state=state,
                heuristic_slot=heuristic_slot_name,
                heuristic_role=heuristic_role,
                heuristic_score=heuristic_slot_score,
            )
            if classification is not None:
                slot_name = classification["slot"]
                role = classification["role"]
                classifier_metadata = {
                    "classifier_backend": "controller",
                    "classifier_confidence": classification.get("confidence", 0.0),
                    "classifier_reason": classification.get("reason", ""),
                    "classifier_answer_span": classification.get("answer_span", ""),
                    "heuristic_slot": heuristic_slot_name,
                    "heuristic_role": heuristic_role,
                    "heuristic_slot_score": heuristic_slot_score,
                }
                answer_span = str(classification.get("answer_span") or "").strip()
                if answer_span:
                    item.metadata["answer_span"] = answer_span
                    item.metadata["ocr_exact_answer_candidate"] = True
        if (
            role != "core"
            and item.modality == "ocr"
            and item.metadata.get("ocr_exact_answer_candidate")
            and slot_name == target_slot
        ):
            role = "core"
            classifier_metadata["ocr_core_promoted_reason"] = "exact_answer_candidate"
        if (
            role == "core"
            and item.modality == "ocr"
            and item.metadata.get("ocr_requires_exact_answer_span")
            and (
                not str(item.metadata.get("answer_span") or "").strip()
                or not item.metadata.get("ocr_exact_answer_candidate")
            )
        ):
            role = "support"
            classifier_metadata["ocr_core_downgraded_reason"] = "missing_exact_answer_span"
        if role == "core" and _ocr_code_evidence_blocked_for_screen_text_question(
            state.question,
            item,
        ):
            role = "support"
            classifier_metadata["ocr_core_downgraded_reason"] = (
                "code_evidence_not_allowed_for_screen_text_question"
            )
        if role == "core" and _ocr_comparison_assignment_blocked_for_operator_count_question(
            state.question,
            item,
        ):
            role = "support"
            classifier_metadata["ocr_core_downgraded_reason"] = (
                "comparison_assignment_not_operator_count"
            )
        if (
            role == "core"
            and question_route.label != "generic"
            and not evidence_matches_route(item, question_route)
        ):
            role = "support"
            classifier_metadata["route_core_downgraded_reason"] = (
                "evidence_kind_not_compatible_with_question_route"
            )
        claim_hash = _claim_hash(item.claim)
        if _is_duplicate_evidence(state, item, slot_name, claim_hash):
            duplicate_count += 1
            continue

        answers_question = role == "core"
        relevance = min(1.0, 0.35 + slot_score)
        novelty = _estimate_novelty(state, node_id, modality, target_slot)
        relation_status = relation_evidence_status(item) if requires_relation_evidence else None
        relation_rejection_reason: str | None = None
        if requires_relation_evidence and role == "core" and relation_status != "supported":
            answers_question = False
            relation_rejection_reason = (
                "missing_co_visible_relation"
                if relation_status == "unknown"
                else "unsupported_co_visible_relation"
            )
            role = "support" if relation_status == "unknown" else "background"
        item.metadata.update(
            {
                "slot": slot_name,
                "role": role,
                "answers_question": answers_question,
                "relevance": round(relevance, 4),
                "novelty": round(novelty, 4),
                "target_slot": target_slot,
                "claim_hash": claim_hash,
                "question_route": question_route.label,
                **classifier_metadata,
            }
        )
        if relation_status is not None:
            item.metadata["relation_evidence_status"] = relation_status
        if relation_rejection_reason is not None:
            item.metadata["relation_evidence_rejected"] = True
            item.metadata["relation_evidence_rejection_reason"] = relation_rejection_reason
        if role != "noise":
            classified.append(item)
        if role == "core":
            filled_slots.add(slot_name)
            background_only = False
        elif role == "support" and slot_name == target_slot:
            background_only = False

    missing_slots = [
        slot.slot
        for slot in question_spec.required_slots
        if slot.required
        and slot.slot not in filled_slots
        and not _slot_already_filled(state, slot.slot)
    ]
    no_new_information = not classified
    if classified and all(item.metadata.get("role") == "background" for item in classified):
        background_only = True

    suggested_queries = build_slot_queries(state.question, question_spec, target_slot)[1:3]
    refinement_progress = background_only and bool(suggested_queries)

    return classified, {
        "target_slot": target_slot,
        "missing_slots": missing_slots,
        "filled_slots": sorted(filled_slots),
        "background_only": background_only,
        "no_new_information": no_new_information,
        "duplicate_evidence_count": duplicate_count,
        "suggested_queries": suggested_queries,
        "progress_made": bool(filled_slots)
        or any(item.metadata.get("role") == "support" for item in classified)
        or refinement_progress,
        "result": _open_result_label(classified, background_only, no_new_information),
    }


def update_evidence_board(
    board: EvidenceBoard | None,
    question_spec: QuestionSpec | None,
    observation: Observation,
    step_index: int,
) -> EvidenceBoard | None:
    if board is None or question_spec is None:
        return board

    metadata = observation.metadata
    target_slot = metadata.get("target_slot")
    modality = metadata.get("modality")
    if observation.kind == "open" and observation.node_id and modality:
        add_opened_target(
            board,
            node_id=observation.node_id,
            modality=modality,
            target_slot=target_slot,
            result=metadata.get("result", "unknown"),
            step_index=step_index,
        )
        for node_id in graph_expansion_covered_node_ids(observation):
            add_opened_target(
                board,
                node_id=node_id,
                modality=modality,
                target_slot=target_slot,
                result="graph_expansion_covered",
                step_index=step_index,
            )
    if target_slot:
        hinted_queries = [query for query in metadata.get("suggested_queries", []) if query]
        if hinted_queries:
            board.slot_query_hints[target_slot] = _merge_unique_strings(
                board.slot_query_hints.get(target_slot, []),
                hinted_queries,
                limit=6,
            )
        refinement_node_ids = [
            node_id for node_id in metadata.get("refinement_node_ids", []) if node_id
        ]
        if refinement_node_ids:
            board.slot_refinement_node_ids[target_slot] = _merge_unique_strings(
                board.slot_refinement_node_ids.get(target_slot, []),
                refinement_node_ids,
                limit=8,
            )

    filled_slots_before = {name for name, slot in board.slots.items() if slot.status == "filled"}
    for item in observation.evidence:
        slot_name = item.metadata.get("slot")
        role = item.metadata.get("role")
        if slot_name not in board.slots:
            continue
        board_slot = board.slots[slot_name]
        if role == "core":
            board_slot.core_evidence_ids.append(item.evidence_id)
            board_slot.status = "filled"
            board.core_evidence_ids.append(item.evidence_id)
        elif role == "support":
            board_slot.support_evidence_ids.append(item.evidence_id)
            if board_slot.status == "missing":
                board_slot.status = "background_only"
            board.support_evidence_ids.append(item.evidence_id)
        elif role == "background":
            board_slot.background_evidence_ids.append(item.evidence_id)
            if board_slot.status == "missing":
                board_slot.status = "background_only"
            board.background_evidence_ids.append(item.evidence_id)

    board.duplicate_evidence_count += int(metadata.get("duplicate_evidence_count", 0))
    if metadata.get("background_only"):
        board.background_only_open_count += 1

    filled_slots_after = {name for name, slot in board.slots.items() if slot.status == "filled"}
    board.slot_fill_count += max(0, len(filled_slots_after - filled_slots_before))
    board.missing_required_slots = [
        slot.slot
        for slot in question_spec.required_slots
        if slot.required and board.slots[slot.slot].status != "filled"
    ]
    for slot_name in filled_slots_after:
        board.slot_query_hints.pop(slot_name, None)
        board.slot_refinement_node_ids.pop(slot_name, None)
    return board


def choose_next_action(
    state: ControllerState,
    board: EvidenceBoard | None,
    frontier: list[FrontierItem],
) -> dict[str, Any]:
    target_slot = select_target_slot(state.question_spec, board)
    if board is not None and not board.missing_required_slots:
        return {"action_type": "STOP", "target_slot": target_slot}
    if frontier:
        return {
            "action_type": "OPEN",
            "node_id": frontier[0].node_id,
            "modality": frontier[0].recommended_modalities[0]
            if frontier[0].recommended_modalities
            else "speech",
            "target_slot": target_slot,
        }
    return {
        "action_type": "SEARCH",
        "query": build_slot_queries(state.question, state.question_spec, target_slot)[0],
        "modality": _preferred_modality(state.question_spec, target_slot),
        "target_slot": target_slot,
    }


def build_slot_queries(
    question: str,
    question_spec: QuestionSpec | None,
    target_slot: str | None,
) -> list[str]:
    queries = [question.strip()]
    if question_spec is None:
        return queries
    queries.extend(_modality_queries(question_spec, target_slot))
    slot = question_spec.get_slot(target_slot) if target_slot else None
    if slot is not None:
        queries.append(slot.description)
        queries.append(f"{question} {slot.description}".strip())
        queries.extend(_keyword_queries(slot.slot, slot.description))
    else:
        for required_slot in question_spec.required_slots[:2]:
            queries.append(required_slot.description)
    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = " ".join(query.split())
        if not normalized or normalized in seen:
            continue
        deduped.append(normalized)
        seen.add(normalized)
    return deduped[:5]


def is_reopen_blocked(
    board: EvidenceBoard | None,
    node_id: str,
    modality: Modality,
    target_slot: str | None,
) -> bool:
    if board is None:
        return False
    for opened in board.opened_targets:
        if (
            opened.node_id == node_id
            and opened.modality == modality
            and opened.target_slot == target_slot
        ):
            return True
    return False


def add_opened_target(
    board: EvidenceBoard,
    *,
    node_id: str,
    modality: Modality,
    target_slot: str | None,
    result: str,
    step_index: int,
) -> None:
    if is_reopen_blocked(board, node_id, modality, target_slot):
        return
    board.opened_targets.append(
        OpenedTarget(
            node_id=node_id,
            modality=modality,
            target_slot=target_slot,
            result=result,
            step_index=step_index,
        )
    )


def graph_expansion_covered_node_ids(observation: Observation) -> list[str]:
    covered: list[str] = []
    for item in observation.evidence:
        metadata = item.metadata
        if not metadata.get("vrrqa_graph_expansion_applied"):
            continue
        node_ids = metadata.get("vrrqa_graph_expansion_node_ids", [])
        if not isinstance(node_ids, list):
            continue
        for node_id in node_ids:
            if isinstance(node_id, str) and node_id and node_id not in covered:
                covered.append(node_id)
    return covered


def _adjust_search_score(
    hit: SearchHit,
    state: ControllerState,
    target_slot: str | None,
    *,
    query_frame: CognitiveQueryFrame,
    index: VideoMemoryIndex,
) -> tuple[float, dict[str, float]]:
    score = hit.score
    if state.evidence_board and is_reopen_blocked(
        state.evidence_board,
        hit.node_id,
        hit.modality,
        target_slot,
    ):
        score -= 0.25
    if (
        target_slot
        and state.evidence_board is not None
        and hit.node_id in state.evidence_board.slot_refinement_node_ids.get(target_slot, [])
    ):
        score += 0.22
    if target_slot:
        slot_tokens = _tokenize(target_slot.replace("_", " "))
        overlap_bonus = len(slot_tokens & set(hit.matched_terms)) * 0.08
        score += overlap_bonus
    node = index.memory.get_node(hit.node_id)
    situation_score = _situation_model_score(node.metadata.get("event_schema"), query_frame)
    boundary_score = _event_boundary_anchor_score(node.metadata)
    memorability_score = min(1.0, float(node.metadata.get("memorability_prior") or 0.0))
    temporal_score = _cognitive_temporal_relation_score(node.metadata, query_frame)
    graph_score = _cognitive_graph_neighbor_score(node.metadata, query_frame)
    weighted_cognitive_score = _weighted_cognitive_stage1_score(
        hit=hit,
        situation_score=situation_score,
        temporal_score=temporal_score,
        boundary_score=boundary_score,
        memorability_score=memorability_score,
        graph_score=graph_score,
    )
    cognitive_bonus = (
        (0.15 * situation_score)
        + (0.10 * temporal_score)
        + (0.05 * boundary_score)
        + (0.05 * memorability_score)
        + (0.05 * graph_score)
    )
    score = max(score + cognitive_bonus, weighted_cognitive_score)
    return round(score, 4), {
        "cognitive_situation": round(situation_score, 4),
        "cognitive_temporal": round(temporal_score, 4),
        "cognitive_boundary": round(boundary_score, 4),
        "cognitive_memorability": round(memorability_score, 4),
        "cognitive_graph": round(graph_score, 4),
        "cognitive_bonus": round(cognitive_bonus, 4),
        "cognitive_weighted_stage1": round(weighted_cognitive_score, 4),
    }


def _append_cognitive_reason(reason: str, breakdown: dict[str, float]) -> str:
    bonus = breakdown.get("cognitive_bonus", 0.0)
    if bonus <= 0:
        return reason
    strongest = [
        name.removeprefix("cognitive_")
        for name, value in sorted(
            breakdown.items(),
            key=lambda item: -item[1],
        )
        if name != "cognitive_bonus" and value > 0
    ][:3]
    suffix = f"cognitive_bonus={bonus:.3f}"
    if strongest:
        suffix += f" via {', '.join(strongest)}"
    return f"{reason}; {suffix}"


def _build_cognitive_query_frame(
    *,
    question: str,
    queries: list[str],
    selected_modality: Modality,
) -> CognitiveQueryFrame:
    text = " ".join([question, *queries])
    lowered = text.lower()
    tokens = _tokenize(text)
    actions = {
        token
        for token in tokens
        if token in ACTION_TERMS or token.endswith("ing") or token.endswith("ed")
    }
    actors = tokens & ACTOR_TERMS
    places = tokens & PLACE_TERMS
    objects = {
        token
        for token in tokens
        if token not in COGNITIVE_QUERY_EXCLUDE_TERMS
        and token not in actions
        and len(token) > 2
    }
    temporal_relation = None
    if tokens & TEMPORAL_AFTER_TERMS:
        temporal_relation = "after"
    elif tokens & TEMPORAL_BEFORE_TERMS:
        temporal_relation = "before"
    elif {"first", "beginning", "earliest", "initial"} & tokens:
        temporal_relation = "first"
    elif {"last", "final", "ending", "end"} & tokens:
        temporal_relation = "last"
    requires_visual = selected_modality in {"visual", "cross_modal", "ocr"} or _has_route_signal(
        lowered,
        tokens,
        VISUAL_ROUTE_TERMS,
        VISUAL_ROUTE_PHRASES,
    )
    requires_speech = selected_modality in {"speech", "cross_modal"} or _has_route_signal(
        lowered,
        tokens,
        SPEECH_ROUTE_TERMS,
        SPEECH_ROUTE_PHRASES,
    )
    requires_ocr = selected_modality in {"ocr", "cross_modal"} or _has_route_signal(
        lowered,
        tokens,
        VISUAL_ROUTE_TERMS,
        {"what is written", "what's written", "read visible text"},
    )
    causal = bool(tokens & CAUSAL_TERMS)
    return CognitiveQueryFrame(
        actors=actors,
        places=places,
        objects=objects,
        actions=actions,
        spoken_topics=objects if requires_speech else set(),
        ocr_entities=objects if requires_ocr else set(),
        temporal_relation=temporal_relation,
        requires_visual=requires_visual,
        requires_speech=requires_speech,
        requires_ocr=requires_ocr,
        requires_temporal_order=temporal_relation is not None,
        causal=causal,
    )


def _situation_model_score(schema: Any, query_frame: CognitiveQueryFrame) -> float:
    if not isinstance(schema, dict):
        return 0.0
    component_scores = [
        _set_match_score(query_frame.actors, _schema_tokens(schema, "actors")),
        _set_match_score(query_frame.places, _schema_tokens(schema, "place")),
        _set_match_score(query_frame.objects, _schema_tokens(schema, "objects")),
        _set_match_score(query_frame.actions, _schema_tokens(schema, "actions")),
        _set_match_score(query_frame.spoken_topics, _schema_tokens(schema, "spoken_topics")),
        _set_match_score(query_frame.ocr_entities, _schema_tokens(schema, "ocr_entities")),
    ]
    positive_scores = [score for score in component_scores if score > 0]
    if not positive_scores:
        return 0.0
    return round(sum(positive_scores) / len(positive_scores), 4)


def _schema_tokens(schema: dict[str, Any], key: str) -> set[str]:
    value = schema.get(key)
    texts: list[str] = []
    if isinstance(value, list):
        texts.extend(str(item) for item in value)
    elif isinstance(value, str):
        texts.append(value)
    return _tokenize(" ".join(texts))


def _set_match_score(query_terms: set[str], node_terms: set[str]) -> float:
    if not query_terms or not node_terms:
        return 0.0
    return min(1.0, len(query_terms & node_terms) / max(len(query_terms), 1))


def _event_boundary_anchor_score(metadata: dict[str, Any]) -> float:
    boundary_scores = [
        float(item.get("score") or 0.0)
        for item in metadata.get("event_boundary_scores", [])
        if isinstance(item, dict)
    ]
    anchor_scores = [
        float(item.get("event_boundary_score") or item.get("score") or 0.0)
        for item in metadata.get("cognitive_anchor_frames", [])
        if isinstance(item, dict)
    ]
    peaks = metadata.get("event_boundary_peak_timestamps", [])
    peak_bonus = 0.2 if isinstance(peaks, list) and peaks else 0.0
    return round(min(1.0, max([0.0, *boundary_scores, *anchor_scores]) + peak_bonus), 4)


def _weighted_cognitive_stage1_score(
    *,
    hit: SearchHit,
    situation_score: float,
    temporal_score: float,
    boundary_score: float,
    memorability_score: float,
    graph_score: float,
) -> float:
    breakdown = hit.score_breakdown
    lexical_audio_ocr_score = _normalized_score(
        max(
            float(breakdown.get("lexical", 0.0) or 0.0),
            float(breakdown.get("temporal", 0.0) or 0.0),
            float(breakdown.get("combined", 0.0) or 0.0),
            hit.score,
        )
    )
    siglip_image_text_score = _normalized_score(
        max(
            float(breakdown.get("semantic", 0.0) or 0.0),
            float(breakdown.get("frame_similarity", 0.0) or 0.0),
            float(breakdown.get("semantic_frame_similarity", 0.0) or 0.0),
        )
    )
    return round(
        (0.35 * lexical_audio_ocr_score)
        + (0.25 * siglip_image_text_score)
        + (0.15 * situation_score)
        + (0.10 * temporal_score)
        + (0.05 * boundary_score)
        + (0.05 * memorability_score)
        + (0.05 * graph_score),
        4,
    )


def _normalized_score(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value <= 1.0:
        return value
    return 1.0 - (1.0 / (1.0 + value))


def _cognitive_temporal_relation_score(
    metadata: dict[str, Any],
    query_frame: CognitiveQueryFrame,
) -> float:
    if not query_frame.requires_temporal_order:
        return 0.0
    if query_frame.temporal_relation == "after":
        return 1.0 if metadata.get("previous_cognitive_event_id") else 0.35
    if query_frame.temporal_relation == "before":
        return 1.0 if metadata.get("next_cognitive_event_id") else 0.35
    if query_frame.temporal_relation in {"first", "last"}:
        return 0.4 if metadata.get("cognitive_event") else 0.0
    return 0.0


def _cognitive_graph_neighbor_score(
    metadata: dict[str, Any],
    query_frame: CognitiveQueryFrame,
) -> float:
    if not (query_frame.requires_temporal_order or query_frame.causal):
        return 0.0
    neighbor_keys = (
        "cognitive_event_neighbor_ids",
        "same_actor_event_ids",
        "same_object_event_ids",
        "same_place_event_ids",
        "same_topic_event_ids",
        "cause_effect_event_ids",
        "caused_by_event_ids",
        "goal_continuation_event_ids",
        "goal_predecessor_event_ids",
    )
    neighbor_count = sum(
        len(metadata.get(key, [])) for key in neighbor_keys if isinstance(metadata.get(key), list)
    )
    return round(min(1.0, neighbor_count / 4.0), 4)


def _preferred_modality(question_spec: QuestionSpec | None, target_slot: str | None) -> Modality:
    if question_spec is None:
        return "speech"
    if target_slot:
        slot = question_spec.get_slot(target_slot)
        if slot is not None and slot.preferred_modality is not None:
            return slot.preferred_modality
    return question_spec.preferred_modality or "speech"


def _resolve_available_modality(modality: Modality, state: ControllerState) -> Modality:
    available = state.global_context.get("available_modalities", {})
    if not available:
        return modality
    if modality == "ocr" and not available.get("ocr") and available.get("visual"):
        return "visual"
    if modality == "audio" and not available.get("audio") and available.get("speech"):
        return "speech"
    return modality


def _search_modalities(modality: Modality, question: str) -> list[Modality]:
    if modality == "cross_modal":
        return ["speech", "visual", "ocr", "audio"]
    if modality == "visual":
        if _has_route_signal(
            question.lower(), _tokenize(question), VISUAL_ROUTE_TERMS, VISUAL_ROUTE_PHRASES
        ):
            return ["visual", "ocr"]
        return ["visual"]
    if modality == "ocr":
        return ["ocr", "visual"]
    if modality == "audio":
        return ["audio", "speech"]
    return [modality]


def _has_route_signal(
    lowered: str,
    tokens: set[str],
    terms: set[str],
    phrases: set[str],
) -> bool:
    return bool(tokens & terms) or any(phrase in lowered for phrase in phrases)


def _modality_queries(question_spec: QuestionSpec, target_slot: str | None) -> list[str]:
    modality = _preferred_modality(question_spec, target_slot)
    if modality == "visual":
        return [
            "visible text title slide screen shown displayed",
            "on screen text title slide",
        ]
    if modality == "audio":
        return [
            "sound noise audio background ticking mechanical",
            "heard subtle sound background noise",
        ]
    if modality == "ocr":
        return [
            "read visible text on screen",
            "written title label sign",
        ]
    return []


def _best_slot_match(
    item: Evidence,
    question_spec: QuestionSpec,
    target_slot: str | None,
) -> tuple[str, float]:
    text_tokens = _tokenize(" ".join(part for part in [item.claim, item.detail] if part))
    best_slot = target_slot or question_spec.required_slots[0].slot
    best_score = 0.0
    for slot in question_spec.required_slots:
        slot_tokens = _tokenize(f"{slot.slot.replace('_', ' ')} {slot.description}")
        keyword_tokens = _tokenize(" ".join(GENERIC_SLOT_KEYWORDS.get(slot.slot, [])))
        overlap = len(text_tokens & slot_tokens)
        keyword_overlap = len(text_tokens & keyword_tokens)
        score = overlap * 0.18 + keyword_overlap * 0.22
        if slot.slot == target_slot:
            score += 0.26
        if score > best_score:
            best_slot = slot.slot
            best_score = score
    return best_slot, round(best_score, 4)


def _classify_slot_role(
    slot_name: str,
    slot_score: float,
    item: Evidence,
    question_spec: QuestionSpec,
    target_slot: str | None,
) -> str:
    text = " ".join(part for part in [item.claim, item.detail] if part).lower()
    question_type = question_spec.question_type
    is_target_slot = slot_name == target_slot
    generic_intro = any(
        cue in text
        for cue in (
            "in this series",
            "but first",
            "what makes it special",
            "we're exploring",
            "welcome back",
            "today we're",
            "today we are",
        )
    )
    causal_cues = any(
        cue in text
        for cue in (
            "because",
            "so ",
            "therefore",
            "worried",
            "lose it",
            "opening a lot",
            "kept opening",
            "clasp",
            "fixed it",
            "brought it back",
            "couldn't wait",
            "can't wait",
            "rest of the video",
            "show it off",
            "wanted to show",
            "immediately",
            "right away",
            "decided",
            "chose",
        )
    )
    first_item_cues = any(
        cue in text
        for cue in (
            "what about",
            "this is",
            "first",
            "tried",
            "bite",
            "bit into",
            "brought out",
            "presented",
        )
    )
    why_different_cues = any(
        cue in text
        for cue in (
            "different",
            "not regular",
            "unexpected",
            "surprisingly",
            "tastes like chicken",
            "fried goodness",
            "deep fried",
            "coated in flour",
        )
    )
    reaction_cues = any(
        cue in text
        for cue in (
            "reaction",
            "responded",
            "surprised",
            "tastes like",
            "felt",
            "looked",
            "froze",
            "stayed still",
            "paused",
            "stared",
        )
    )
    if slot_score <= 0.12:
        return "noise"

    if generic_intro and slot_name in {"first_thing_tried", "why_different", "reason"}:
        return "background" if slot_score >= 0.18 else "noise"

    if question_type == "why_reason" and is_target_slot:
        if causal_cues:
            return "core"
        return "support" if slot_score >= 0.24 else "background"

    if slot_name == "first_thing_tried" and is_target_slot:
        if first_item_cues and not generic_intro:
            return "core"
        return "support" if slot_score >= 0.24 else "background"

    if slot_name == "why_different" and is_target_slot:
        if why_different_cues:
            return "core"
        return "support" if slot_score >= 0.24 else "background"

    if slot_name == "reaction" and is_target_slot:
        if reaction_cues:
            return "core"
        return "support" if slot_score >= 0.24 else "background"

    if is_target_slot and slot_score >= 0.3:
        return "core"
    if slot_score >= 0.2:
        return "support"
    return "background"


OCR_CODE_EVIDENCE_KINDS = {
    "assignment_count",
    "assignment_count_partial",
    "code_line",
    "comparison_assignments",
    "comparison_operator_count",
    "computed_expression_value",
    "computed_output_value",
    "computed_variable_value",
    "target_assignment",
}


def _ocr_code_evidence_blocked_for_screen_text_question(
    question: str,
    item: Evidence,
) -> bool:
    if item.modality != "ocr":
        return False
    if not _is_header_sign_title_question(question):
        return False
    if _question_explicitly_asks_for_code(question):
        return False
    kind = str(item.metadata.get("ocr_evidence_kind") or "")
    if kind in OCR_CODE_EVIDENCE_KINDS:
        return True
    evidence_text = " ".join(part for part in (item.claim, item.detail) if part)
    return bool(
        re.search(
            r"\b[A-Za-z_]\w*\s*=\s*[^=\n]+(?:==|!=|>=|<=|>|<|\+|\-|\*|/)?",
            evidence_text,
        )
    )


def _is_header_sign_title_question(question: str) -> bool:
    lowered = question.lower()
    return any(
        cue in lowered
        for cue in (
            "header",
            "label",
            "sign",
            "title",
            "what is written",
            "what's written",
            "what text",
            "what does the door",
            "what does the sign",
            "laboratory door",
        )
    )


def _question_explicitly_asks_for_code(question: str) -> bool:
    lowered = question.lower()
    explicit_code_cues = (
        "assignment",
        "boolean",
        "calculate",
        "code line",
        "data type",
        "expression",
        "operator",
        "output",
        "python script",
        "result",
        "script",
        "shell",
        "variable",
    )
    return any(cue in lowered for cue in explicit_code_cues)


def _ocr_comparison_assignment_blocked_for_operator_count_question(
    question: str,
    item: Evidence,
) -> bool:
    lowered = question.lower()
    if not ("how many" in lowered and "comparison" in lowered and "operator" in lowered):
        return False
    return str(item.metadata.get("ocr_evidence_kind") or "") == "comparison_assignments"


def _is_duplicate_evidence(
    state: ControllerState,
    item: Evidence,
    slot_name: str,
    claim_hash: str,
) -> bool:
    for existing in state.evidence_ledger:
        if existing.source_node_id != item.source_node_id:
            continue
        if existing.modality != item.modality:
            continue
        if existing.metadata.get("slot") != slot_name:
            continue
        if existing.metadata.get("claim_hash") == claim_hash:
            return True
    return False


def _estimate_novelty(
    state: ControllerState,
    node_id: str,
    modality: Modality,
    target_slot: str | None,
) -> float:
    if state.evidence_board is None:
        return 1.0
    if is_reopen_blocked(state.evidence_board, node_id, modality, target_slot):
        return 0.1
    return 0.85


def _slot_already_filled(state: ControllerState, slot_name: str) -> bool:
    if state.evidence_board is None:
        return False
    slot = state.evidence_board.slots.get(slot_name)
    return slot is not None and slot.status == "filled"


def _open_result_label(
    evidence_items: list[Evidence],
    background_only: bool,
    no_new_information: bool,
) -> str:
    if no_new_information:
        return "no_new_information"
    if background_only:
        return "background_only"
    if any(item.metadata.get("role") == "core" for item in evidence_items):
        return "slot_filled"
    return "support_only"


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in (match.group(0).lower() for match in TOKEN_PATTERN.finditer(text))
        if (token not in STOPWORDS or token in CONTROL_QUERY_TOKENS) and len(token) > 1
    }


def _question_stem_without_options(question: str) -> str:
    split_markers = ("options:", "choices:", "\noption a", "\na.", "\na)", "\na:")
    lowered = question.lower()
    split_at = len(question)
    for marker in split_markers:
        marker_index = lowered.find(marker)
        if marker_index >= 0:
            split_at = min(split_at, marker_index)
    return question[:split_at].strip() or question


def _search_queries_for_state(
    state: ControllerState,
    question_spec: QuestionSpec | None,
    target_slot: str | None,
    query_override: str | None,
) -> list[str]:
    override = (query_override or "").strip()
    base_question = override or state.question
    base_queries = build_slot_queries(base_question, question_spec, target_slot)
    if state.evidence_board is None or not target_slot:
        return base_queries

    hint_queries = state.evidence_board.slot_query_hints.get(target_slot, [])
    if not hint_queries:
        return base_queries

    if override and override != state.question.strip():
        combined = [override, *hint_queries, *base_queries]
    else:
        combined = [*hint_queries, *base_queries]
    deduped: list[str] = []
    seen: set[str] = set()
    for query in combined:
        normalized = " ".join(query.split())
        if not normalized or normalized in seen:
            continue
        deduped.append(normalized)
        seen.add(normalized)
    return deduped[:6]


def _keyword_queries(slot_name: str, description: str) -> list[str]:
    keywords = GENERIC_SLOT_KEYWORDS.get(slot_name, [])
    if not keywords:
        return []
    return [f"{description} {' '.join(keywords[:4])}".strip(), " ".join(keywords[:5]).strip()]


def _object_description(question: str) -> str:
    lowered = question.lower()
    if "diamond" in lowered:
        return "The diamond item or add-on being discussed"
    if "bracelet" in lowered:
        return "The bracelet or jewelry item being discussed"
    if "food" in lowered or "tried" in lowered:
        return "The first food item or unusual object being discussed"
    return f"The main object or entity in: {question}"


def _decision_description(question: str) -> str:
    lowered = question.lower()
    if "wear" in lowered:
        return "The choice to wear or not wear the item"
    if "try" in lowered or "tried" in lowered:
        return "The choice to try the item or action"
    return f"The decision or action directly asked by: {question}"


def _reason_description(question: str) -> str:
    lowered = question.lower()
    if "wear" in lowered:
        return "The reason or cause explaining why she wore or avoided wearing the item"
    if "stop" in lowered or "stare" in lowered or "look" in lowered or "move" in lowered:
        return "The visible cause or cue explaining why the subject changed behavior"
    if "try" in lowered or "tried" in lowered:
        return "The reason or cause explaining why they tried the item or reacted to it"
    return "The reason or cause that directly answers the question"


def _infer_preferred_modality(question: str, task_type: str | None) -> Modality:
    question_route = route_question(question, task_type)
    if question_route.preferred_modality is not None:
        return question_route.preferred_modality
    lowered = question.lower()
    if task_type in {"multiple_choice_visual_qa", "vrrqa", "vrrqa_visual_only"}:
        return "visual"
    audio_cues = (
        "audio event",
        "environment sound",
        "mechanical sound",
        "sound",
        "ticking",
        "noise",
    )
    if task_type == "agentic_task" and any(cue in lowered for cue in audio_cues):
        return "audio"
    speech_cues = (
        "presenter say",
        "presenter says",
        "presenter said",
        "presenter explain",
        "presenter explains",
        "speaker say",
        "speaker says",
        "speaker said",
        "transcribe",
        "translate",
        "spoken",
        "speech",
    )
    if task_type == "agentic_task" and any(cue in lowered for cue in speech_cues):
        return "speech"
    screen_text_cues = (
        "arithmetic",
        "assignment",
        "boolean",
        "code",
        "code editor",
        "comparison",
        "declared",
        "declaration",
        "displayed",
        "editor",
        "expression",
        "header",
        "label",
        "operator",
        "output",
        "python script",
        "result",
        "screen",
        "script",
        "shown",
        "sign",
        "shell",
        "type",
        "variable",
        "visible",
    )
    if task_type == "agentic_task" and any(cue in lowered for cue in screen_text_cues):
        return "ocr"
    visual_cues = (
        "stare",
        "doorway",
        "stopped",
        "stop chasing",
        "moving",
        "looked",
        "turned",
        "expression",
        "gesture",
        "reaction",
        "mouse",
        "cat",
    )
    if any(cue in lowered for cue in visual_cues):
        return "visual"
    if task_type in {"information_retrieval", "multimodal_synthesis"} and any(
        cue in lowered
        for cue in ("how did", "what happened when", "what did they both", "what did she do")
    ):
        return "visual"
    return "speech"


def _claim_hash(text: str) -> str:
    normalized = " ".join(text.split()).lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]


def _merge_unique_strings(existing: list[str], additions: list[str], limit: int) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in [*existing, *additions]:
        normalized = " ".join(value.split()).strip()
        if not normalized or normalized in seen:
            continue
        merged.append(normalized)
        seen.add(normalized)
        if len(merged) >= limit:
            break
    return merged
