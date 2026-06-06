from __future__ import annotations

import re
from typing import Any

from rlm.video.types import (
    EventInterval,
    EventMemory,
    EventMemoryEvent,
    Evidence,
    Observation,
    TimeSpan,
)

TOKEN_PATTERN = re.compile(r"\b[a-zA-Z][a-zA-Z0-9_/.-]*\b")
EVENT_STOPWORDS = {
    "a",
    "an",
    "and",
    "before",
    "does",
    "did",
    "for",
    "in",
    "is",
    "it",
    "of",
    "on",
    "option",
    "or",
    "person",
    "someone",
    "something",
    "somewhere",
    "the",
    "then",
    "to",
    "true",
    "which",
    "with",
}
NEGATIVE_VISIBILITY_PHRASES = {
    "absent",
    "cannot be established",
    "cannot confirm",
    "cannot determine",
    "cannot establish",
    "cannot verify",
    "do not see",
    "don't see",
    "is there a shot",
    "not visible",
    "not clearly visible",
    "not explicitly clear",
    "not explicitly visible",
    "not happening",
    "no clear shot",
    "no visible",
    "no evidence",
    "no one is actively",
    "need to look for",
    "does not show",
    "not actively",
    "not pouring",
    "not shown",
    "unsupported",
    "without seeing",
}
PROMPT_ECHO_PHRASES = {
    "answer the user's question",
    "check for visibility",
    "evaluate the actions",
    "evaluate the question",
    "formulate the answer",
    "identify the actions mentioned",
    "listed action phrases",
    "options are",
    "prompt asks",
    "specific sequence",
    "target action phrase",
    "target actions",
    "target entities",
    "target entities/actions",
    "the target actions",
    "the target entities",
    "the options are",
    "the prompt asks",
    "the question asks",
    "the statement",
    "the user wants",
    "therefore, the statement",
    "timeLogic tlqa mode".lower(),
}
POSITIVE_VISUAL_MARKERS = {
    "adding",
    "appears",
    "carrying",
    "carries",
    "cracking",
    "cracks",
    "drinking",
    "fixing",
    "frame",
    "frying",
    "hold",
    "holding",
    "holds",
    "looking",
    "moving",
    "opening",
    "pouring",
    "put",
    "putting",
    "reaching",
    "standing",
    "taking",
    "visible",
    "walking",
}
TOKEN_ALIASES = {
    "blanket": {"blanket", "cloth", "fabric", "garment", "towel"},
    "cup": {"cup", "mug"},
    "dish": {"dish", "plate"},
    "shoe": {"boot", "shoe", "shoes", "sneaker", "sneakers"},
    "teabag": {"bag", "packet", "tea", "teabag"},
}
MATCH_THRESHOLD = 0.67


def build_event_memory_from_global_context(
    global_context: dict[str, Any],
) -> EventMemory | None:
    spec = global_context.get("event_memory_spec")
    if not isinstance(spec, dict):
        return None
    events_payload = spec.get("events")
    if not isinstance(events_payload, list) or not events_payload:
        return None

    events: dict[str, EventMemoryEvent] = {}
    for index, item in enumerate(events_payload, start=1):
        if not isinstance(item, dict):
            continue
        phrase = " ".join(str(item.get("phrase") or "").split()).strip()
        if not phrase:
            continue
        event_id = str(item.get("event_id") or f"event_{index:02d}")
        events[event_id] = EventMemoryEvent(
            event_id=event_id,
            phrase=phrase,
            source=str(item.get("source") or "question"),
            option_letter=(
                str(item["option_letter"]).upper()
                if item.get("option_letter") is not None
                else None
            ),
            metadata={
                key: value
                for key, value in item.items()
                if key not in {"event_id", "phrase", "source", "option_letter"}
            },
        )

    if not events:
        return None
    return EventMemory(
        task_name=str(spec.get("task_name") or spec.get("benchmark") or "video"),
        question=str(spec.get("question") or global_context.get("clean_question") or ""),
        mode=str(spec["mode"]) if spec.get("mode") is not None else None,
        events=events,
        relations=[
            dict(item) for item in (spec.get("relations") or []) if isinstance(item, dict)
        ],
        metadata={
            key: value
            for key, value in spec.items()
            if key not in {"events", "relations", "question", "mode", "task_name"}
        },
    )


def update_event_memory_from_observation(
    event_memory: EventMemory | None,
    observation: Observation,
) -> int:
    if event_memory is None:
        return 0
    updates = 0
    for evidence in observation.evidence:
        updates += update_event_memory_from_evidence(event_memory, evidence)
    if updates:
        event_memory.metadata["last_update_count"] = updates
    return updates


def update_event_memory_from_evidence(
    event_memory: EventMemory,
    evidence: Evidence,
) -> int:
    updates = 0
    if event_memory.mode == "mc":
        updates += _update_option_event_from_best_option(event_memory, evidence)
    passages = _candidate_visual_passages(evidence)
    if not passages:
        return updates

    for event in event_memory.events.values():
        best = _best_positive_event_passage(event.phrase, passages)
        if best is None:
            continue
        score, passage = best
        if _has_interval_for_evidence(event, evidence.evidence_id):
            continue
        interval_span = _passage_time_span(evidence, passage)
        event.intervals.append(
            EventInterval(
                time_span=interval_span,
                evidence_id=evidence.evidence_id,
                source_node_id=evidence.source_node_id,
                confidence=evidence.confidence,
                match_score=score,
                detail=passage[:500],
                metadata={
                    "modality": evidence.modality,
                    "slot": evidence.metadata.get("slot"),
                    "role": evidence.metadata.get("role"),
                    "source_time_span": evidence.time_span.to_dict(),
                },
            )
        )
        event.intervals.sort(
            key=lambda interval: (
                interval.time_span.start,
                interval.time_span.end,
                -interval.confidence,
            )
        )
        event.status = "localized"
        updates += 1
    return updates


def _update_option_event_from_best_option(
    event_memory: EventMemory,
    evidence: Evidence,
) -> int:
    option_letter = _best_option_letter_from_evidence(evidence)
    if option_letter is None:
        return 0
    event = next(
        (
            item
            for item in event_memory.events.values()
            if item.source == "option" and item.option_letter == option_letter
        ),
        None,
    )
    if event is None or _has_interval_for_evidence(event, evidence.evidence_id):
        return 0
    event.intervals.append(
        EventInterval(
            time_span=evidence.time_span,
            evidence_id=evidence.evidence_id,
            source_node_id=evidence.source_node_id,
            confidence=evidence.confidence,
            match_score=1.0,
            detail=_best_option_detail(evidence)[:500],
            metadata={
                "modality": evidence.modality,
                "slot": evidence.metadata.get("slot"),
                "role": evidence.metadata.get("role"),
                "source_time_span": evidence.time_span.to_dict(),
                "source": "best_option",
            },
        )
    )
    event.intervals.sort(
        key=lambda interval: (
            interval.time_span.start,
            interval.time_span.end,
            -interval.confidence,
        )
    )
    event.status = "localized"
    return 1


def _best_option_letter_from_evidence(evidence: Evidence) -> str | None:
    metadata_choice = evidence.metadata.get("vrrqa_best_option")
    if isinstance(metadata_choice, str):
        letter = metadata_choice.strip().upper()[:1]
        if re.fullmatch(r"[A-Z]", letter):
            return letter
    text = _evidence_search_text(evidence)
    match = re.search(
        r"\b(?:best\s+option|selected\s+option|final\s+answer|answer|choice)\s*"
        r"(?:is|=|:)?\s*\(?([A-Z])\)?\b",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    return match.group(1).upper()


def _best_option_detail(evidence: Evidence) -> str:
    for value in (
        evidence.metadata.get("vrrqa_evidence"),
        evidence.metadata.get("vrrqa_summary"),
        evidence.detail,
        evidence.claim,
    ):
        if value:
            return str(value)
    return f"Best option from evidence {evidence.evidence_id}."


def event_match_score(phrase: str, evidence_text: str) -> float:
    token_groups = _action_token_groups(phrase)
    if not token_groups:
        return 0.0
    normalized_phrase = _normalized_action_phrase(phrase)
    normalized_text = _normalized_text(evidence_text)
    if normalized_phrase and normalized_phrase in normalized_text:
        return 1.0

    evidence_variants = _token_variant_set(evidence_text)
    matched = sum(1 for group in token_groups if group & evidence_variants)
    if matched == 0:
        return 0.0
    if matched < min(len(token_groups), 2):
        return 0.0
    if len(token_groups) <= 2 and matched < len(token_groups):
        return 0.0
    return round(matched / len(token_groups), 4)


def event_memory_metrics(event_memory: EventMemory | None) -> dict[str, Any]:
    if event_memory is None:
        return {}
    total = len(event_memory.events)
    localized = event_memory.localized_event_count
    return {
        "event_count": total,
        "localized_event_count": localized,
        "missing_event_count": total - localized,
        "missing_event_ids": event_memory.missing_event_ids,
    }


def _evidence_search_text(evidence: Evidence) -> str:
    parts: list[str] = [evidence.claim, evidence.detail]
    for key in (
        "vrrqa_evidence",
        "vrrqa_summary",
        "vrrqa_temporal_order",
        "vrrqa_frame_timeline",
        "vrrqa_option_comparison",
        "vrrqa_entities_visible",
        "vrrqa_motion_trajectory",
        "vrrqa_physical_context",
        "tags",
        "entities",
    ):
        if key in evidence.metadata:
            parts.append(_flatten_value(evidence.metadata[key]))
    return " ".join(part for part in parts if part)


def _candidate_visual_passages(evidence: Evidence) -> list[str]:
    text = _evidence_search_text(evidence)
    if not text.strip():
        return []
    passages: list[str] = []
    for line in text.splitlines():
        cleaned = _clean_passage(line)
        if not cleaned:
            continue
        if _is_prompt_echo_passage(cleaned):
            continue
        passages.extend(_split_passage(cleaned))
    return [
        passage
        for passage in passages
        if passage and not _is_prompt_echo_passage(passage)
    ]


def _best_positive_event_passage(
    phrase: str,
    passages: list[str],
) -> tuple[float, str] | None:
    scored: list[tuple[float, str]] = []
    for passage in passages:
        if _has_negative_visibility_cue(passage):
            continue
        if _evidence_is_negative_for_event(phrase, passage):
            continue
        if _is_bare_event_label(phrase, passage):
            continue
        if not _looks_like_visual_observation(passage):
            continue
        score = event_match_score(phrase, passage)
        if score >= MATCH_THRESHOLD:
            scored.append((score, passage))
    if not scored:
        return None
    scored.sort(
        key=lambda item: (
            -item[0],
            len(_frame_indices(item[1])) or 99,
            _first_frame_index(item[1]) or 10_000,
        )
    )
    return scored[0]


def _clean_passage(text: str) -> str:
    cleaned = " ".join(str(text).split()).strip()
    cleaned = re.sub(r"^[*\-\d.\s:`]+", "", cleaned).strip()
    return cleaned


def _split_passage(text: str) -> list[str]:
    if len(text) <= 320:
        return [text]
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [_clean_passage(part) for part in parts if _clean_passage(part)]


def _is_prompt_echo_passage(text: str) -> bool:
    lowered = text.lower()
    if lowered.startswith(("a.", "b.", "c.", "d.", "question:", "options:")):
        return True
    if "->" in lowered:
        return True
    if "question" in lowered or "option" in lowered:
        return not lowered.startswith("frame")
    return any(phrase in lowered for phrase in PROMPT_ECHO_PHRASES)


def _has_negative_visibility_cue(text: str) -> bool:
    lowered = " ".join(text.lower().split())
    if any(negative in lowered for negative in NEGATIVE_VISIBILITY_PHRASES):
        return True
    if re.search(r"\bno\b.{0,80}\bvisible\b", lowered):
        return True
    no_action_pattern = (
        r"\bno\b.{0,80}\b("
        r"added|adding|being|shown|used|carrying|fixing|flipping|holding|opening|"
        r"pouring|putting|taking|whisking"
        r")\b"
    )
    if re.search(no_action_pattern, lowered):
        return True
    return bool(
        re.search(
            r"\bnot\b.{0,60}\b("
            r"adding|carrying|fixing|flipping|holding|opening|pouring|putting|"
            r"taking|visible|whisking"
            r")\b",
            lowered,
        )
    )


def _is_bare_event_label(phrase: str, text: str) -> bool:
    if "frame" in text.lower() or "visible" in text.lower():
        return False
    phrase_tokens = set(_tokens(phrase)) - EVENT_STOPWORDS
    text_tokens = set(_tokens(text)) - EVENT_STOPWORDS
    return bool(phrase_tokens) and text_tokens <= phrase_tokens


def _looks_like_visual_observation(text: str) -> bool:
    lowered = text.lower()
    if "frame" in lowered:
        return True
    tokens = set(_tokens(lowered))
    return bool(tokens & POSITIVE_VISUAL_MARKERS)


def _passage_time_span(evidence: Evidence, passage: str) -> TimeSpan:
    frame_indices = _frame_indices(passage)
    timestamps = _metadata_frame_timestamps(evidence.metadata)
    if frame_indices and timestamps:
        matched = [
            timestamps[index - 1]
            for index in frame_indices
            if 1 <= index <= len(timestamps)
        ]
        if matched:
            start = min(matched)
            end = max(matched)
            if start == end:
                end = start
            return TimeSpan(start=start, end=end)
    return evidence.time_span


def _metadata_frame_timestamps(metadata: dict[str, Any]) -> list[float]:
    for key in (
        "selected_frame_timestamps",
        "vrrqa_relationship_frame_timestamps",
        "relationship_forced_frame_timestamps",
    ):
        value = metadata.get(key)
        if isinstance(value, list):
            timestamps = []
            for item in value:
                try:
                    timestamps.append(float(item))
                except (TypeError, ValueError):
                    continue
            if timestamps:
                return timestamps
    return []


def _frame_indices(text: str) -> list[int]:
    indices: list[int] = []
    for match in re.finditer(r"\bframes?\s+([0-9][0-9,\s/\-and]*)", text, re.IGNORECASE):
        raw = match.group(1)
        numbers = [int(item) for item in re.findall(r"\d+", raw)]
        if "-" in raw and len(numbers) >= 2:
            indices.extend(range(numbers[0], numbers[1] + 1))
        else:
            indices.extend(numbers)
    return sorted(set(indices))


def _first_frame_index(text: str) -> int | None:
    indices = _frame_indices(text)
    return indices[0] if indices else None


def _flatten_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_flatten_value(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return " ".join(_flatten_value(item) for item in value)
    return str(value)


def _action_token_groups(text: str) -> list[set[str]]:
    groups: list[set[str]] = []
    seen: set[str] = set()
    for token in _tokens(text):
        if token in EVENT_STOPWORDS or token in {"yes", "no"}:
            continue
        variants = _token_variants(token)
        canonical = sorted(variants)[0]
        if canonical in seen:
            continue
        seen.add(canonical)
        groups.append(variants)
    return groups


def _token_variant_set(text: str) -> set[str]:
    variants: set[str] = set()
    for token in _tokens(text):
        variants.update(_token_variants(token))
    return variants


def _token_variants(token: str) -> set[str]:
    token = token.strip().lower()
    variants = set(TOKEN_ALIASES.get(token, {token}))
    if len(token) > 5 and token.endswith("ing"):
        stem = token[:-3]
        variants.add(stem)
        variants.add(stem + "e")
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            variants.add(stem[:-1])
    if len(token) > 4 and token.endswith("ed"):
        stem = token[:-2]
        variants.add(stem)
        variants.add(stem + "e")
    if len(token) > 4 and token.endswith("ies"):
        variants.add(token[:-3] + "y")
    if len(token) > 3 and token.endswith("s"):
        variants.add(token[:-1])
    return {variant for variant in variants if variant}


def _normalized_action_phrase(text: str) -> str:
    tokens = []
    for token in _tokens(text):
        if token not in EVENT_STOPWORDS and token not in {"yes", "no"}:
            tokens.append(token)
    return " ".join(tokens)


def _normalized_text(text: str) -> str:
    return " ".join(_tokens(text))


def _tokens(text: str) -> list[str]:
    normalized = re.sub(r"[/_-]+", " ", text.lower())
    return TOKEN_PATTERN.findall(normalized)


def _evidence_is_negative_for_event(phrase: str, text: str) -> bool:
    lowered = " ".join(text.lower().split())
    if not any(negative in lowered for negative in NEGATIVE_VISIBILITY_PHRASES):
        return False
    normalized_phrase = _normalized_action_phrase(phrase)
    if not normalized_phrase:
        return False
    index = lowered.find(normalized_phrase)
    if index < 0:
        return False
    window = lowered[max(0, index - 80) : index + len(normalized_phrase) + 80]
    return any(negative in window for negative in NEGATIVE_VISIBILITY_PHRASES)


def _has_interval_for_evidence(event: EventMemoryEvent, evidence_id: str) -> bool:
    return any(interval.evidence_id == evidence_id for interval in event.intervals)
