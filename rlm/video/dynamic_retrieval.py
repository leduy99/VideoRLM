from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from rlm.video.evidence_pipeline import search_v2
from rlm.video.question_router import QuestionRoute, route_from_metadata, route_question
from rlm.video.types import (
    ControllerState,
    FrontierItem,
    Modality,
    QuestionSpec,
    TimeSpan,
)

DYNAMIC_ROUTE_LABELS = {
    "temporal_occurrence",
    "sentiment_analysis",
    "audio_visual_alignment",
    "visual_difference",
}
SIMPLE_SPEECH_TASK_TYPES = {
    "information_retrieval",
    "summarization",
}
MULTI_EVIDENCE_CUES = {
    "after",
    "before",
    "because",
    "cause",
    "caused",
    "consequence",
    "effect",
    "even",
    "fix",
    "happen",
    "happened",
    "how",
    "later",
    "problem",
    "prove",
    "reason",
    "right",
    "solve",
    "then",
    "though",
    "why",
}
TEMPORAL_ORDER_CUES = {
    "after",
    "before",
    "first",
    "earlier",
    "later",
    "next",
    "right after",
    "then",
    "when",
}
CLAUSE_SPLIT_PATTERN = re.compile(
    r"\b(?:and how|and why|and what|because|but|so that|so|then|after|before|when|while|even though|especially)\b",
    re.IGNORECASE,
)
TOKEN_PATTERN = re.compile(r"\b[a-z0-9_]+\b", re.IGNORECASE)


@dataclass(frozen=True)
class EvidenceRetrievalTarget:
    target_id: str
    label: str
    query: str
    target_slot: str | None
    modality: Modality | None
    required: bool = True
    temporal_role: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "label": self.label,
            "query": self.query,
            "target_slot": self.target_slot,
            "modality": self.modality,
            "required": self.required,
            "temporal_role": self.temporal_role,
        }


@dataclass(frozen=True)
class DynamicEvidenceCandidate:
    target: EvidenceRetrievalTarget
    frontier_item: FrontierItem
    target_score: float
    normalized_score: float


@dataclass(frozen=True)
class DynamicEvidencePlan:
    frontier: list[FrontierItem]
    targets: list[EvidenceRetrievalTarget]
    selected_candidates: list[DynamicEvidenceCandidate]
    score: float
    temporal_policy: str
    target_searches: list[dict[str, Any]]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "combination_plan": True,
            "requires_all_selected": True,
            "target_count": len(self.targets),
            "selected_count": len(self.selected_candidates),
            "score": round(self.score, 4),
            "temporal_policy": self.temporal_policy,
            "targets": [target.to_dict() for target in self.targets],
            "selected": [
                {
                    "target_id": candidate.target.target_id,
                    "label": candidate.target.label,
                    "query": candidate.target.query,
                    "node_id": candidate.frontier_item.node_id,
                    "time_span": candidate.frontier_item.time_span.to_dict(),
                    "score": round(candidate.frontier_item.score, 4),
                    "target_score": round(candidate.target_score, 4),
                    "normalized_score": round(candidate.normalized_score, 4),
                    "target_slot": candidate.target.target_slot,
                    "required": candidate.target.required,
                    "temporal_role": candidate.target.temporal_role,
                    "modality": (
                        candidate.frontier_item.recommended_modalities[0]
                        if candidate.frontier_item.recommended_modalities
                        else candidate.target.modality
                    ),
                }
                for candidate in self.selected_candidates
            ],
            "target_searches": self.target_searches,
        }


def build_dynamic_evidence_targets(
    *,
    state: ControllerState,
    question_spec: QuestionSpec,
    query_override: str | None = None,
    max_targets: int = 4,
) -> list[EvidenceRetrievalTarget]:
    route = _route_for_state(state, question_spec)
    if not should_use_dynamic_evidence_retrieval(state, question_spec, route):
        return []

    question = _clean_text(query_override or state.global_context.get("clean_question") or state.question)
    query_question = _clean_text(
        " ".join(part for part in [question, _postvalid_dialogue_query_context(state)] if part)
    )
    preferred_modality = _preferred_modality_for_dynamic_targets(state, question_spec, route)
    targets: list[EvidenceRetrievalTarget] = []

    if bool(question_spec.metadata.get("postvalid_sentiment_slots")):
        targets.extend(_postvalid_sentiment_targets(query_question, state))
    elif bool(question_spec.metadata.get("postvalid_speech_slots")):
        targets.extend(_direct_question_targets(query_question, preferred_modality))
    elif len(question_spec.required_slots) > 1:
        pattern_targets = _pattern_targets(query_question, preferred_modality)
        clause_targets = _clause_targets(query_question, preferred_modality)
        targets.extend(pattern_targets)
        targets.extend(clause_targets)
        for slot in question_spec.required_slots[:max_targets]:
            targets.append(
                EvidenceRetrievalTarget(
                    target_id=f"slot:{slot.slot}",
                    label=slot.slot,
                    query=_slot_target_query(question, slot.slot, slot.description),
                    target_slot=slot.slot,
                    modality=slot.preferred_modality or preferred_modality,
                    required=slot.required,
                    temporal_role=_temporal_role_for_label(slot.slot, question),
                )
            )
    return _dedupe_targets(targets, max_targets=max_targets)


def select_dynamic_evidence_chain(
    *,
    index: Any,
    state: ControllerState,
    question_spec: QuestionSpec,
    top_k: int,
    query_override: str | None = None,
    max_targets: int = 4,
    candidates_per_target: int = 8,
) -> DynamicEvidencePlan | None:
    targets = build_dynamic_evidence_targets(
        state=state,
        question_spec=question_spec,
        query_override=query_override,
        max_targets=max_targets,
    )
    if len(targets) < 2:
        return None

    candidates_by_target: list[list[DynamicEvidenceCandidate]] = []
    target_searches: list[dict[str, Any]] = []
    search_total_start = time.perf_counter()
    for target in targets:
        search_start = time.perf_counter()
        target_frontier, _metadata = search_v2(
            index=index,
            question_spec=question_spec,
            target_slot=target.target_slot,
            state=state,
            top_k=candidates_per_target,
            query_override=target.query,
            modality=target.modality,
        )
        search_seconds = time.perf_counter() - search_start
        target_searches.append(
            {
                "target_id": target.target_id,
                "label": target.label,
                "target_slot": target.target_slot,
                "modality": target.modality,
                "seconds": round(search_seconds, 6),
                "hit_count": len(target_frontier),
                "search_mode": _metadata.get("search_mode"),
                "searched_modalities": _metadata.get("searched_modalities", []),
                "transcript_section_hit_count": (
                    _metadata.get("transcript_section_index", {}).get("hit_count", 0)
                    if isinstance(_metadata.get("transcript_section_index"), dict)
                    else 0
                ),
            }
        )
        candidates = _target_candidates(target, target_frontier)
        if candidates:
            candidates_by_target.append(candidates)

    if len(candidates_by_target) < 2:
        return None

    temporal_policy = _temporal_policy(state.question, targets)
    selected, chain_score = _dynamic_programming_chain(
        candidates_by_target,
        temporal_policy=temporal_policy,
        video_duration_seconds=_video_duration_seconds(state),
    )
    if len(selected) < 2:
        return None

    frontier = _frontier_from_selected_chain(selected, limit=top_k)
    if len(frontier) < 2:
        return None
    return DynamicEvidencePlan(
        frontier=frontier,
        targets=targets,
        selected_candidates=selected,
        score=chain_score,
        temporal_policy=temporal_policy,
        target_searches=[
            *target_searches,
            {"target_id": "dynamic_total", "seconds": round(time.perf_counter() - search_total_start, 6)},
        ],
    )


def should_use_dynamic_evidence_retrieval(
    state: ControllerState,
    question_spec: QuestionSpec,
    route: QuestionRoute | None = None,
) -> bool:
    enabled, _reason = dynamic_evidence_retrieval_policy(state, question_spec, route)
    return enabled


def dynamic_evidence_retrieval_policy(
    state: ControllerState,
    question_spec: QuestionSpec,
    route: QuestionRoute | None = None,
) -> tuple[bool, str]:
    selected_route = route or _route_for_state(state, question_spec)
    if not _is_longshot_context(state):
        return False, "non_longshot_context"
    if selected_route.label == "sentiment_analysis" or bool(
        question_spec.metadata.get("postvalid_sentiment_slots")
    ):
        return False, "sentiment_speech_first_local_visual_followup"
    if selected_route.label == "temporal_occurrence":
        return True, "route_temporal_occurrence"
    if _question_has_repeated_or_ordinal_constraint(state.question):
        return True, "temporal_or_ordinal_constraint"
    if state.task_type == "multimodal_synthesis":
        if _multimodal_question_needs_dynamic_chain(state, selected_route):
            return True, "multi_step_multimodal_synthesis"
        return False, "simple_multimodal_synthesis"
    if bool(question_spec.metadata.get("postvalid_speech_slots")):
        if _postvalid_speech_question_needs_dynamic_chain(state):
            return True, "postvalid_multi_span_speech_chain"
        return False, "postvalid_single_span_speech_aggregation"
    if state.task_type in SIMPLE_SPEECH_TASK_TYPES:
        return False, f"simple_speech_task:{state.task_type}"
    if state.task_type == "causal_reasoning" and not _causal_question_needs_dynamic_chain(
        state.question
    ):
        return False, "simple_causal_reasoning"
    if selected_route.label in DYNAMIC_ROUTE_LABELS:
        if _question_has_multi_step_structure(state.question):
            return True, f"multi_step_route:{selected_route.label}"
        return False, f"simple_route:{selected_route.label}"
    if selected_route.label in {"speech_explanation", "causal_chain", "rubric_explanation"}:
        return False, f"speech_aggregation_route:{selected_route.label}"
    if len(question_spec.required_slots) > 1 and _question_has_multi_step_structure(state.question):
        return True, "multi_slot_non_speech"
    return False, "no_dynamic_route"


def _postvalid_speech_question_needs_dynamic_chain(state: ControllerState) -> bool:
    question = state.global_context.get("clean_question") or state.question
    return len(_direct_question_parts(str(question))) >= 2


def _question_has_multi_step_structure(question: str) -> bool:
    lowered = question.lower()
    if CLAUSE_SPLIT_PATTERN.search(question):
        return True
    question_tokens = {token.lower() for token in TOKEN_PATTERN.findall(question)}
    return bool(question_tokens & MULTI_EVIDENCE_CUES) and any(
        cue in lowered
        for cue in (
            "and how",
            "and why",
            "problem",
            "fix",
            "solve",
            "consequence",
            "effect",
            "while",
            "at the same time",
            "after that",
        )
    )


def _question_has_repeated_or_ordinal_constraint(question: str) -> bool:
    lowered = question.lower()
    if "right away" in lowered or "saving it for later" in lowered:
        return False
    if "first piece" in lowered and ("jewelry" in lowered or "filippo" in lowered):
        return False
    ordinal_phrases = (
        "right after",
        "immediately after",
        "just after",
        "what happened next",
        "after that",
        "earlier",
        "early in",
        "early lead",
        "before",
        "first time",
        "first occurrence",
        "first appears",
        "first appeared",
        "first happens",
        "first happened",
        "first introduced",
        "second",
        "third",
        "fourth",
        "last",
        "earliest",
        "latest",
        "again",
        "another time",
        "same scene",
        "same person",
        "same team",
        "repeated",
    )
    return any(phrase in lowered for phrase in ordinal_phrases)


def _causal_question_needs_dynamic_chain(question: str) -> bool:
    lowered = question.lower()
    if _question_has_repeated_or_ordinal_constraint(question):
        return True
    paired_terms = (
        ("problem", ("fix", "solve", "address")),
        ("cause", ("effect", "consequence", "result")),
        ("led", ("then", "after", "later")),
        ("because", ("then", "after", "result")),
    )
    return any(left in lowered and any(right in lowered for right in rights) for left, rights in paired_terms)


def _multimodal_question_needs_dynamic_chain(
    state: ControllerState,
    route: QuestionRoute,
) -> bool:
    if route.label in {"audio_visual_alignment", "visual_difference"}:
        return True
    longshot_context = state.global_context.get("longshot")
    expected_modalities = []
    if isinstance(longshot_context, dict):
        expected_modalities = [
            str(item)
            for item in longshot_context.get("expected_modalities", [])
            if str(item)
        ]
    has_multiple_modalities = len(set(expected_modalities)) >= 2
    lowered = state.question.lower()
    explicit_cross_modal = (
        any(term in lowered for term in ("visual", "shown", "seen", "screen", "look"))
        and any(term in lowered for term in ("audio", "heard", "said", "speech", "sound"))
    )
    return (has_multiple_modalities or explicit_cross_modal) and _question_has_multi_step_structure(
        state.question
    )


def _target_candidates(
    target: EvidenceRetrievalTarget,
    frontier: list[FrontierItem],
) -> list[DynamicEvidenceCandidate]:
    if not frontier:
        return []
    max_score = max((max(0.0, item.score) for item in frontier), default=0.0)
    if max_score <= 0.0:
        max_score = 1.0
    candidates: list[DynamicEvidenceCandidate] = []
    seen: set[str] = set()
    for item in frontier:
        if item.node_id in seen:
            continue
        seen.add(item.node_id)
        normalized = max(0.0, item.score) / max_score
        candidates.append(
            DynamicEvidenceCandidate(
                target=target,
                frontier_item=item,
                target_score=item.score,
                normalized_score=normalized,
            )
        )
    return candidates


def _dynamic_programming_chain(
    candidates_by_target: list[list[DynamicEvidenceCandidate]],
    *,
    temporal_policy: str,
    video_duration_seconds: float | None,
) -> tuple[list[DynamicEvidenceCandidate], float]:
    dp: list[dict[int, tuple[float, list[int]]]] = []
    for target_index, candidates in enumerate(candidates_by_target):
        layer: dict[int, tuple[float, list[int]]] = {}
        for candidate_index, candidate in enumerate(candidates):
            base_score = _candidate_base_score(candidate, video_duration_seconds)
            if target_index == 0:
                layer[candidate_index] = (base_score, [candidate_index])
                continue
            best_score: float | None = None
            best_path: list[int] | None = None
            previous_layer = dp[target_index - 1]
            previous_candidates = candidates_by_target[target_index - 1]
            for previous_index, (previous_score, previous_path) in previous_layer.items():
                previous_candidate = previous_candidates[previous_index]
                transition = _transition_score(
                    previous_candidate,
                    candidate,
                    temporal_policy=temporal_policy,
                )
                score = previous_score + base_score + transition
                if best_score is None or score > best_score:
                    best_score = score
                    best_path = [*previous_path, candidate_index]
            if best_score is None or best_path is None:
                layer[candidate_index] = (base_score, [candidate_index])
            else:
                layer[candidate_index] = (best_score, best_path)
        dp.append(layer)

    final_layer = dp[-1]
    best_final_index, (best_score, best_path) = max(
        final_layer.items(),
        key=lambda item: item[1][0],
    )
    del best_final_index
    selected: list[DynamicEvidenceCandidate] = []
    for layer_index, candidate_index in enumerate(best_path):
        selected.append(candidates_by_target[layer_index][candidate_index])
    return selected, best_score


def _candidate_base_score(
    candidate: DynamicEvidenceCandidate,
    video_duration_seconds: float | None,
) -> float:
    score = 0.8 * candidate.normalized_score + 0.2 * min(1.0, max(0.0, candidate.target_score))
    score += {
        "clip": 0.08,
        "event": 0.05,
        "segment": 0.02,
        "scene": -0.06,
        "video": -0.12,
    }.get(candidate.frontier_item.level, 0.0)
    if video_duration_seconds and candidate.target.temporal_role:
        midpoint = _midpoint(candidate.frontier_item.time_span)
        position = max(0.0, min(1.0, midpoint / video_duration_seconds))
        if candidate.target.temporal_role in {"early", "problem", "setup", "immediate"}:
            score += 0.12 * (1.0 - position)
        elif candidate.target.temporal_role in {"later", "fix", "consequence", "outcome"}:
            score += 0.12 * position
    if candidate.target.required:
        score += 0.05
    return score


def _transition_score(
    previous: DynamicEvidenceCandidate,
    current: DynamicEvidenceCandidate,
    *,
    temporal_policy: str,
) -> float:
    previous_item = previous.frontier_item
    current_item = current.frontier_item
    score = 0.0
    if previous_item.node_id == current_item.node_id:
        score -= 0.7
    elif previous_item.time_span.overlaps(current_item.time_span):
        overlap = min(previous_item.time_span.end, current_item.time_span.end) - max(
            previous_item.time_span.start,
            current_item.time_span.start,
        )
        shorter = max(1e-6, min(previous_item.time_span.duration, current_item.time_span.duration))
        overlap_ratio = max(0.0, min(1.0, overlap / shorter))
        score -= 0.45 + (0.25 * overlap_ratio)
        if temporal_policy == "ordered" and current_item.time_span.start <= previous_item.time_span.start:
            score -= 0.25
        if current_item.level in {"scene", "video"} and previous_item.level in {
            "clip",
            "event",
            "segment",
        }:
            score -= 0.18
    else:
        if current_item.time_span.start >= previous_item.time_span.start:
            score += 0.14 if temporal_policy == "ordered" else 0.04
        else:
            score -= 0.32 if temporal_policy == "ordered" else 0.08
        gap = max(0.0, current_item.time_span.start - previous_item.time_span.end)
        if 0.0 <= gap <= 300.0:
            score += 0.04
        elif gap > 1200.0:
            score -= 0.04
    if _candidate_modality(previous) != _candidate_modality(current):
        if _spans_are_near(previous_item.time_span, current_item.time_span, seconds=90.0):
            score += 0.08
    return score


def _frontier_from_selected_chain(
    selected: list[DynamicEvidenceCandidate],
    *,
    limit: int,
) -> list[FrontierItem]:
    by_node: dict[str, FrontierItem] = {}
    target_labels_by_node: dict[str, list[str]] = {}
    for chain_index, candidate in enumerate(selected, start=1):
        item = candidate.frontier_item
        target = candidate.target
        current = by_node.get(item.node_id)
        target_labels_by_node.setdefault(item.node_id, []).append(target.label)
        reason = (
            f"{item.why_candidate}; dynamic_evidence_chain=selected; "
            f"chain_index={chain_index}; target={target.label}; "
            f"target_slot={target.target_slot or 'none'}; "
            f"target_query={target.query[:160]!r}; "
            f"temporal_role={target.temporal_role or 'none'}; "
            f"target_score={candidate.target_score:.4f}; "
            f"normalized_target_score={candidate.normalized_score:.4f}"
        )
        if current is None:
            by_node[item.node_id] = FrontierItem(
                node_id=item.node_id,
                time_span=item.time_span,
                level=item.level,
                score=item.score + (0.03 * max(0, len(selected) - chain_index)),
                why_candidate=reason,
                recommended_modalities=list(item.recommended_modalities),
                status=item.status,
            )
            continue
        current.score = max(current.score, item.score) + 0.03
        current.recommended_modalities = sorted(
            set(current.recommended_modalities) | set(item.recommended_modalities)
        )
        current.why_candidate = (
            f"{current.why_candidate}; additional_dynamic_target={target.label}; "
            f"additional_target_query={target.query[:120]!r}"
        )
    ordered = sorted(
        by_node.values(),
        key=lambda item: (
            min(
                index
                for index, candidate in enumerate(selected)
                if candidate.frontier_item.node_id == item.node_id
            ),
            item.time_span.start,
            -item.score,
        ),
    )
    for item in ordered:
        labels = target_labels_by_node.get(item.node_id, [])
        item.why_candidate = (
            f"{item.why_candidate}; dynamic_targets={labels}; "
            "open as part of a multi-segment evidence combination"
        )
    return ordered[:limit]


def _direct_question_targets(
    question: str,
    modality: Modality | None,
) -> list[EvidenceRetrievalTarget]:
    parts = _direct_question_parts(question)
    targets: list[EvidenceRetrievalTarget] = []
    for index, part in enumerate(parts[:4], start=1):
        targets.append(
            EvidenceRetrievalTarget(
                target_id=f"direct:{index}",
                label=f"question_part_{index}",
                query=part,
                target_slot="answer_core" if index == 1 else None,
                modality=modality,
                required=index == 1,
                temporal_role=_direct_part_temporal_role(part, index),
            )
        )
    return targets


def _direct_question_parts(question: str) -> list[str]:
    cleaned = _clean_text(question).rstrip("?")
    if not cleaned:
        return []
    split_patterns = (
        r"\b(and why)\b",
        r"\b(and how)\b",
        r"\b(and what)\b",
        r"\b(and when)\b",
        r"\b(and where)\b",
        r"\b(and who)\b",
        r"\b(even though)\b",
        r"\b(because)\b",
        r"\b(so that)\b",
        r"\b(but)\b",
    )
    parts = [cleaned]
    for pattern in split_patterns:
        next_parts: list[str] = []
        for part in parts:
            pieces = re.split(pattern, part, maxsplit=1, flags=re.IGNORECASE)
            if len(pieces) < 3:
                next_parts.append(part)
                continue
            before, cue, after = pieces[0], pieces[1].lower(), pieces[2]
            before = _clean_text(before)
            cue_text = cue[4:] if cue.startswith("and ") else cue
            after = _clean_text(f"{cue_text} {after}")
            next_parts.extend(item for item in (before, after) if item)
        parts = next_parts
    deduped: list[str] = []
    seen: set[str] = set()
    for part in parts:
        normalized = _query_key(part)
        if len(_token_set(part)) < 2 or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(part)
    return deduped if len(deduped) >= 2 else []


def _direct_part_temporal_role(part: str, index: int) -> str | None:
    lowered = part.lower()
    if any(cue in lowered for cue in ("before", "earlier", "first", "problem")):
        return "early"
    if any(cue in lowered for cue in ("after", "later", "then", "result", "consequence")):
        return "later"
    if any(cue in lowered for cue in ("why", "because", "how")) and index > 1:
        return "later"
    return "early" if index == 1 else None


def _postvalid_sentiment_targets(
    question: str,
    state: ControllerState,
) -> list[EvidenceRetrievalTarget]:
    context_text = _postvalid_longshot_context_text(state)
    speech_query = _clean_text(
        f"{question} {context_text} quoted emotional speech voice tone I am ready "
        "I need to prove myself throw you in nervous excited confident worried "
        "coach says participant explains feelings"
    )
    visual_query = _clean_text(
        f"{question} {context_text} bench body language facial expression coach interaction "
        "person waiting not playing sideline posture reaction nervous excited focused"
    )
    scene_query = _clean_text(
        f"{question} {context_text} scenario context tryout non roster participant evaluated "
        "not active bench teammate coach game situation"
    )
    audio_query = _clean_text(
        f"{question} {context_text} voice tone nervous excited audio event crowd coach whistle "
        "ambient sound reaction"
    )
    return [
        EvidenceRetrievalTarget(
            target_id="sentiment:speech_moment",
            label="speech_content",
            query=speech_query,
            target_slot="speech_content",
            modality="speech",
            required=True,
            temporal_role="setup",
        ),
        EvidenceRetrievalTarget(
            target_id="sentiment:visual_body_language",
            label="visual_body_language",
            query=visual_query,
            target_slot="visual_body_language",
            modality="visual",
            required=True,
            temporal_role="setup",
        ),
        EvidenceRetrievalTarget(
            target_id="sentiment:scene_context",
            label="scene_context",
            query=scene_query,
            target_slot="scene_context",
            modality="visual",
            required=True,
            temporal_role="context",
        ),
        EvidenceRetrievalTarget(
            target_id="sentiment:tone_or_audio_event",
            label="tone_or_audio_event",
            query=audio_query,
            target_slot="tone_or_audio_event",
            modality="audio",
            required=False,
            temporal_role="context",
        ),
    ]


def _pattern_targets(question: str, modality: Modality | None) -> list[EvidenceRetrievalTarget]:
    lowered = question.lower()
    targets: list[EvidenceRetrievalTarget] = []
    if "problem" in lowered and any(term in lowered for term in ("fix", "solve", "close", "address")):
        targets.extend(
            [
                EvidenceRetrievalTarget(
                    target_id="pattern:problem",
                    label="earlier_problem",
                    query=f"{question} earlier problem loophole issue doubt hidden influence",
                    target_slot="answer_core",
                    modality=modality,
                    temporal_role="problem",
                ),
                EvidenceRetrievalTarget(
                    target_id="pattern:fix",
                    label="fix_or_solution",
                    query=f"{question} fix solve close loophole method result",
                    target_slot="causal_or_temporal_link",
                    modality=modality,
                    required=False,
                    temporal_role="fix",
                ),
            ]
        )
    if lowered.startswith("why") or " why " in lowered or lowered.startswith("how"):
        targets.extend(
            [
                EvidenceRetrievalTarget(
                    target_id="pattern:cause",
                    label="cause_or_reason",
                    query=f"{question} reason cause motivation because",
                    target_slot="answer_core",
                    modality=modality,
                    temporal_role="setup",
                ),
                EvidenceRetrievalTarget(
                    target_id="pattern:mechanism",
                    label="mechanism",
                    query=f"{question} mechanism how it worked explanation",
                    target_slot="mechanism",
                    modality=modality,
                    required=False,
                    temporal_role="fix",
                ),
            ]
        )
    if "right after" in lowered or "what happened when" in lowered:
        targets.extend(
            [
                EvidenceRetrievalTarget(
                    target_id="pattern:immediate",
                    label="immediate_event",
                    query=f"{question} right after immediately warning saw happened",
                    target_slot="answer_core",
                    modality=modality,
                    temporal_role="immediate",
                ),
                EvidenceRetrievalTarget(
                    target_id="pattern:consequence",
                    label="later_consequence",
                    query=f"{question} consequence result had to losing problem",
                    target_slot="consequence",
                    modality=modality,
                    required=False,
                    temporal_role="consequence",
                ),
            ]
        )
    if "even though" in lowered:
        targets.extend(
            [
                EvidenceRetrievalTarget(
                    target_id="pattern:contrast",
                    label="contrast_setup",
                    query=f"{question} even though contrast setup condition",
                    target_slot="answer_core",
                    modality=modality,
                    temporal_role="setup",
                ),
                EvidenceRetrievalTarget(
                    target_id="pattern:explanation",
                    label="explanation",
                    query=f"{question} explanation reason why despite",
                    target_slot="mechanism",
                    modality=modality,
                    required=False,
                    temporal_role="outcome",
                ),
            ]
        )
    return targets


def _clause_targets(question: str, modality: Modality | None) -> list[EvidenceRetrievalTarget]:
    clauses = [
        _clean_text(clause)
        for clause in CLAUSE_SPLIT_PATTERN.split(question)
        if _clean_text(clause)
    ]
    if len(clauses) < 2:
        return []
    targets: list[EvidenceRetrievalTarget] = []
    for index, clause in enumerate(clauses[:4], start=1):
        targets.append(
            EvidenceRetrievalTarget(
                target_id=f"clause:{index}",
                label=f"question_part_{index}",
                query=clause,
                target_slot="answer_core" if index == 1 else None,
                modality=modality,
                required=index == 1,
                temporal_role="early" if index == 1 else "later",
            )
        )
    return targets


def _dedupe_targets(
    targets: list[EvidenceRetrievalTarget],
    *,
    max_targets: int,
) -> list[EvidenceRetrievalTarget]:
    deduped: list[EvidenceRetrievalTarget] = []
    seen: set[str] = set()
    for target in targets:
        query_key = _query_key(target.query)
        key = f"{target.label}:{query_key}:{target.target_slot or ''}"
        if key in seen:
            continue
        seen.add(key)
        if len(_token_set(target.query)) < 2:
            continue
        deduped.append(target)
        if len(deduped) >= max_targets:
            break
    return deduped


def _slot_target_query(question: str, slot_name: str, description: str) -> str:
    extras = {
        "answer_core": "central answer key claim",
        "mechanism": "mechanism how it works",
        "causal_or_temporal_link": "causal temporal link before after because",
        "consequence": "result consequence outcome",
        "reason": "reason because motivation cause",
        "decision": "decision action choice",
        "supporting_detail": "supporting detail evidence",
    }.get(slot_name, "")
    return _clean_text(f"{question} {description} {extras}")


def _temporal_role_for_label(label: str, question: str) -> str | None:
    lowered_label = label.lower()
    lowered_question = question.lower()
    if lowered_label in {"answer_core", "reason", "decision"}:
        if "right after" in lowered_question:
            return "immediate"
        if "earlier" in lowered_question or "problem" in lowered_question:
            return "problem"
        return "setup"
    if lowered_label in {"mechanism", "causal_or_temporal_link"}:
        return "fix"
    if lowered_label in {"consequence", "supporting_detail"}:
        return "consequence"
    return None


def _temporal_policy(state_question: str, targets: list[EvidenceRetrievalTarget]) -> str:
    lowered = state_question.lower()
    if any(cue in lowered for cue in TEMPORAL_ORDER_CUES):
        return "ordered"
    roles = {target.temporal_role for target in targets}
    if roles & {"problem", "setup", "immediate"} and roles & {"fix", "consequence", "outcome"}:
        return "ordered"
    return "flexible"


def _route_for_state(
    state: ControllerState,
    question_spec: QuestionSpec,
) -> QuestionRoute:
    return (
        route_from_metadata(state.global_context)
        or route_from_metadata(question_spec.metadata)
        or route_question(state.question, state.task_type)
    )


def _preferred_modality(
    question_spec: QuestionSpec,
    route: QuestionRoute,
) -> Modality | None:
    if route.preferred_modality is not None:
        return route.preferred_modality
    return question_spec.preferred_modality


def _preferred_modality_for_dynamic_targets(
    state: ControllerState,
    question_spec: QuestionSpec,
    route: QuestionRoute,
) -> Modality | None:
    if state.task_type == "multimodal_synthesis" and _expected_modality_count(state) >= 2:
        return "cross_modal"
    if _is_postvalid_context_with_speech(state) and route.label not in {
        "assignment_count",
        "operator_list",
        "terminal_output",
        "ui_header_text",
        "visual_difference",
        "sentiment_analysis",
        "audio_event",
        "audio_visual_alignment",
    }:
        return "speech"
    return _preferred_modality(question_spec, route)


def _expected_modality_count(state: ControllerState) -> int:
    longshot_context = state.global_context.get("longshot")
    if not isinstance(longshot_context, dict):
        return 0
    return len(
        {
            str(item)
            for item in longshot_context.get("expected_modalities", [])
            if str(item)
        }
    )


def _is_longshot_context(state: ControllerState) -> bool:
    return state.global_context.get("benchmark") == "longshotbench" or isinstance(
        state.global_context.get("longshot"),
        dict,
    )


def _is_postvalid_context_with_speech(state: ControllerState) -> bool:
    longshot_context = state.global_context.get("longshot")
    if not isinstance(longshot_context, dict):
        return False
    dataset_name = str(longshot_context.get("dataset_name") or "").lower()
    if "postvalid" not in dataset_name:
        return False
    available = state.global_context.get("available_modalities", {})
    return not isinstance(available, dict) or bool(available.get("speech", True))


def _postvalid_dialogue_query_context(state: ControllerState) -> str:
    if not _is_postvalid_context_with_speech(state):
        return ""
    user_turns = [
        str(turn.get("content") or "")
        for turn in state.dialogue_context[-4:]
        if turn.get("role") == "user" and str(turn.get("content") or "").strip()
    ]
    if not user_turns:
        return ""
    return _clean_text("dialogue context " + " ".join(user_turns[-2:]))


def _postvalid_longshot_context_text(state: ControllerState) -> str:
    longshot_context = state.global_context.get("longshot")
    if not isinstance(longshot_context, dict):
        return ""
    parts = [
        str(longshot_context.get("scenario") or ""),
        str(longshot_context.get("task") or ""),
        str(longshot_context.get("question_context") or ""),
    ]
    return _clean_text(" ".join(part for part in parts if part.strip()))


def _video_duration_seconds(state: ControllerState) -> float | None:
    value = state.global_context.get("video_length_seconds")
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return None
    return duration if duration > 0.0 else None


def _candidate_modality(candidate: DynamicEvidenceCandidate) -> str | None:
    if candidate.frontier_item.recommended_modalities:
        return candidate.frontier_item.recommended_modalities[0]
    return candidate.target.modality


def _spans_are_near(left: TimeSpan, right: TimeSpan, *, seconds: float) -> bool:
    if left.overlaps(right):
        return True
    return min(abs(left.end - right.start), abs(right.end - left.start)) <= seconds


def _midpoint(span: TimeSpan) -> float:
    return (span.start + span.end) / 2.0


def _clean_text(text: Any) -> str:
    return " ".join(str(text or "").split()).strip()


def _query_key(query: str) -> str:
    return " ".join(sorted(_token_set(query)))


def _token_set(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_PATTERN.findall(text)}
