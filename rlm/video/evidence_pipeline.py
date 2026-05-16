from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

from rlm.video.index import STOPWORDS, TOKEN_PATTERN, SearchHit, VideoMemoryIndex
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


def build_question_spec(
    question: str,
    task_type: str | None = None,
    dialogue_context: list[dict[str, str]] | None = None,
) -> QuestionSpec:
    del dialogue_context
    tokens = _tokenize(question)
    preferred_modality = _infer_preferred_modality(question, task_type)
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
        metadata={"question": question, "task_type": task_type},
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

    for query in queries:
        for current_modality in search_modalities:
            hits = index.search(
                query=query,
                modality=current_modality,
                top_k=max(top_k * 2, 8),
            )
            for hit in hits:
                candidate = SearchHit(
                    node_id=hit.node_id,
                    time_span=hit.time_span,
                    level=hit.level,
                    score=_adjust_search_score(hit, state, target_slot),
                    reason=hit.reason,
                    modality=hit.modality,
                    matched_terms=hit.matched_terms,
                    score_breakdown=dict(hit.score_breakdown),
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
    candidate_rank = _window_level_rank(candidate.level)
    current_rank = _window_level_rank(current.level)
    return candidate_rank < current_rank and candidate.score >= (current.score * 0.85)


def _window_level_rank(level: str) -> int:
    return {
        "clip": 0,
        "segment": 1,
        "scene": 2,
        "video": 3,
    }.get(level, 4)


def open_v2(
    question_spec: QuestionSpec | None,
    target_slot: str | None,
    state: ControllerState,
    node_id: str,
    modality: Modality,
    evidence_items: list[Evidence],
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

    for item in evidence_items:
        slot_name, slot_score = _best_slot_match(item, question_spec, target_slot)
        role = _classify_slot_role(slot_name, slot_score, item, question_spec, target_slot)
        claim_hash = _claim_hash(item.claim)
        if _is_duplicate_evidence(state, item, slot_name, claim_hash):
            duplicate_count += 1
            continue

        answers_question = role == "core"
        relevance = min(1.0, 0.35 + slot_score)
        novelty = _estimate_novelty(state, node_id, modality, target_slot)
        item.metadata.update(
            {
                "slot": slot_name,
                "role": role,
                "answers_question": answers_question,
                "relevance": round(relevance, 4),
                "novelty": round(novelty, 4),
                "target_slot": target_slot,
                "claim_hash": claim_hash,
            }
        )
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
        board.opened_targets.append(
            OpenedTarget(
                node_id=observation.node_id,
                modality=modality,
                target_slot=target_slot,
                result=metadata.get("result", "unknown"),
                step_index=step_index,
            )
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


def _adjust_search_score(hit: SearchHit, state: ControllerState, target_slot: str | None) -> float:
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
    return round(score, 4)


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
    lowered = question.lower()
    if task_type in {"multiple_choice_visual_qa", "vrrqa", "vrrqa_visual_only"}:
        return "visual"
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
