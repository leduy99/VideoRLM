from __future__ import annotations

import ast
import re
from collections import Counter, defaultdict
from typing import Any

from rlm.video.memory import PreparedVideoArtifacts
from rlm.video.types import (
    CodeSnapshot,
    CrossModalTemporalIndex,
    OperatorEvent,
    RawTemporalEvent,
    SectionNode,
    TemporalLink,
    TimeSpan,
    VideoNode,
)

CODE_ASSIGNMENT_PATTERN = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
    r"((?:[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|True|False|\"[^\"]*\"|'[^']*')"
    r"(?:\s*(?:==|!=|>=|<=|>|<|\+|\-|\*|/)\s*"
    r"(?:[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|True|False|\"[^\"]*\"|'[^']*'))?)"
)
CODE_OPERATOR_PATTERN = re.compile(r"\s*(==|!=|>=|<=|>|<|\+|\-|\*|/|=)\s*")
COMPARISON_OPERATOR_PATTERNS = (
    ("==", (r"==", r"\bequal\s+to\b", r"\bequality\b")),
    ("!=", (r"!=", r"\bnot\s+equal\b", r"\bnot\s+equal\s+to\b")),
    (">", (r"(?<![=<])>(?![=>])", r"\bgreater\s+than\b")),
    ("<", (r"(?<![=<])<(?![=<])", r"\bless\s+than\b")),
)
CONTENT_SECTION_PRIORITY = (
    "output_section",
    "comparison_section",
    "logical_evaluation_section",
    "arithmetic_section",
    "assignment_section",
    "code_section",
)


def build_cross_modal_temporal_index(
    artifacts: PreparedVideoArtifacts,
    nodes: dict[str, VideoNode],
) -> CrossModalTemporalIndex:
    node_lookup = _nodes_by_time(nodes)
    asr_events = _build_asr_events(artifacts, node_lookup)
    ocr_events, code_line_events, terminal_events = _build_ocr_code_terminal_events(
        artifacts,
        node_lookup,
    )
    visual_anchor_events = _build_visual_anchor_events(artifacts, node_lookup)
    audio_events = _build_audio_events(artifacts, node_lookup)
    all_events = [
        *asr_events,
        *ocr_events,
        *code_line_events,
        *terminal_events,
        *visual_anchor_events,
        *audio_events,
    ]
    sections = _build_sections(all_events, artifacts.duration_seconds)
    _assign_section_memberships(all_events, sections)
    _annotate_occurrences(all_events, sections)
    _link_audio_events(audio_events, all_events)
    operator_events = _build_operator_events(all_events)
    code_snapshots = _build_code_snapshots(code_line_events)
    temporal_links = _build_temporal_links(all_events, sections, operator_events, code_snapshots)
    return CrossModalTemporalIndex(
        sections=sections,
        asr_events=asr_events,
        ocr_events=ocr_events,
        code_line_events=code_line_events,
        terminal_events=terminal_events,
        visual_anchor_events=visual_anchor_events,
        audio_events=audio_events,
        code_snapshots=code_snapshots,
        operator_events=operator_events,
        temporal_links=temporal_links,
        metadata={
            "video_id": artifacts.video_id,
            "duration_seconds": artifacts.duration_seconds,
            "event_count": len(all_events),
            "section_count": len(sections),
            "code_snapshot_count": len(code_snapshots),
            "operator_event_count": len(operator_events),
        },
    )


def _nodes_by_time(nodes: dict[str, VideoNode]) -> list[VideoNode]:
    return sorted(
        [node for node in nodes.values() if node.level in {"event", "clip", "segment"}],
        key=lambda item: (item.time_span.duration, item.time_span.start, item.node_id),
    )


def _source_node_for_span(node_lookup: list[VideoNode], span: TimeSpan) -> VideoNode | None:
    for node in node_lookup:
        if node.time_span.overlaps(span):
            return node
    return None


def _event_section_tags(node: VideoNode | None, text: str) -> list[str]:
    tags: list[str] = []
    if node is not None:
        tags.extend(str(item) for item in node.metadata.get("section_tags", []))
    tags.extend(_content_tags_from_text(text))
    return _dedupe(tags)


def _primary_section_id(tags: list[str]) -> str | None:
    for tag in CONTENT_SECTION_PRIORITY:
        if tag in tags:
            return tag
    for tag in tags:
        if tag.endswith("_segment") or tag.startswith("segment_"):
            return tag
    return tags[0] if tags else None


def _build_asr_events(
    artifacts: PreparedVideoArtifacts,
    node_lookup: list[VideoNode],
) -> list[RawTemporalEvent]:
    events: list[RawTemporalEvent] = []
    for index, span in enumerate(artifacts.speech_spans, start=1):
        text = " ".join(span.text.split())
        if not text:
            continue
        node = _source_node_for_span(node_lookup, span.time_span)
        tags = _event_section_tags(node, text)
        events.append(
            RawTemporalEvent(
                event_id=f"asr_{index:05d}",
                time_span=span.time_span,
                modality="asr",
                event_type="speech_explanation",
                text=text,
                confidence=0.86,
                section_id=_primary_section_id(tags),
                source_node_id=node.node_id if node is not None else None,
                metadata={
                    "speaker": span.speaker,
                    "language": span.language,
                    "section_tags": tags,
                },
            )
        )
    return events


def _build_ocr_code_terminal_events(
    artifacts: PreparedVideoArtifacts,
    node_lookup: list[VideoNode],
) -> tuple[list[RawTemporalEvent], list[RawTemporalEvent], list[RawTemporalEvent]]:
    ocr_events: list[RawTemporalEvent] = []
    code_events: list[RawTemporalEvent] = []
    terminal_events: list[RawTemporalEvent] = []
    event_counter = 0
    code_counter = 0
    terminal_counter = 0
    for span_index, span in enumerate(artifacts.ocr_spans, start=1):
        node = _source_node_for_span(node_lookup, span.time_span)
        raw_text = span.text.strip()
        if not raw_text:
            continue
        lines = _unique_text_lines(raw_text)
        leading_screen_lines = _leading_screen_text_lines(lines)
        for line_index, line in enumerate(lines, start=1):
            event_counter += 1
            region = _screen_region_for_line(line, line in leading_screen_lines)
            tags = _event_section_tags(node, line)
            if region == "terminal_panel":
                tags = _dedupe([*tags, "output_section"])
            event = RawTemporalEvent(
                event_id=f"ocr_{event_counter:05d}",
                time_span=span.time_span,
                modality="ocr",
                event_type=_ocr_event_type(line, region),
                text=line,
                screen_region=region,
                confidence=0.9,
                section_id=_primary_section_id(tags),
                source_node_id=node.node_id if node is not None else None,
                metadata={
                    "source_span_index": span_index,
                    "line_index": line_index,
                    "section_tags": tags,
                },
            )
            ocr_events.append(event)

        for assignment in _extract_code_assignments(raw_text):
            code_counter += 1
            tags = _event_section_tags(node, assignment)
            if _assignment_rhs_contains_comparison(assignment):
                tags = _dedupe([*tags, "comparison_section", "logical_evaluation_section"])
            elif _assignment_is_arithmetic(assignment):
                tags = _dedupe([*tags, "arithmetic_section"])
            elif _assignment_is_initial_declaration(assignment):
                tags = _dedupe([*tags, "assignment_section"])
            code_events.append(
                RawTemporalEvent(
                    event_id=f"code_{code_counter:05d}",
                    time_span=span.time_span,
                    modality="code",
                    event_type=_code_event_type(assignment),
                    text=assignment,
                    screen_region="code_editor_body",
                    confidence=0.94,
                    section_id=_primary_section_id(tags),
                    source_node_id=node.node_id if node is not None else None,
                    metadata={"section_tags": tags},
                )
            )

        for output in _extract_terminal_outputs(raw_text):
            terminal_counter += 1
            tags = _event_section_tags(node, output)
            tags = _dedupe([*tags, "output_section"])
            terminal_events.append(
                RawTemporalEvent(
                    event_id=f"terminal_{terminal_counter:05d}",
                    time_span=span.time_span,
                    modality="terminal",
                    event_type="terminal_output",
                    text=output,
                    screen_region="terminal_panel",
                    confidence=0.88,
                    section_id=_primary_section_id(tags),
                    source_node_id=node.node_id if node is not None else None,
                    metadata={"section_tags": tags},
                )
            )
    return ocr_events, _dedupe_events_by_text_and_time(code_events), terminal_events


def _build_visual_anchor_events(
    artifacts: PreparedVideoArtifacts,
    node_lookup: list[VideoNode],
) -> list[RawTemporalEvent]:
    events: list[RawTemporalEvent] = []
    for index, summary in enumerate(artifacts.visual_summaries, start=1):
        text = " ".join(summary.summary.split())
        if not text:
            continue
        node = _source_node_for_span(node_lookup, summary.time_span)
        tags = _event_section_tags(node, " ".join([text, *summary.tags, *summary.entities]))
        events.append(
            RawTemporalEvent(
                event_id=f"visual_{index:05d}",
                time_span=summary.time_span,
                modality="visual",
                event_type="visual_anchor",
                text=text,
                confidence=0.8,
                section_id=_primary_section_id(tags),
                source_node_id=node.node_id if node is not None else None,
                metadata={
                    "granularity": summary.granularity,
                    "tags": list(summary.tags),
                    "entities": list(summary.entities),
                    "section_tags": tags,
                },
            )
        )
    return events


def _build_audio_events(
    artifacts: PreparedVideoArtifacts,
    node_lookup: list[VideoNode],
) -> list[RawTemporalEvent]:
    events: list[RawTemporalEvent] = []
    for index, event in enumerate(artifacts.audio_events, start=1):
        label = event.label.strip()
        if not label:
            continue
        node = _source_node_for_span(node_lookup, event.time_span)
        tags = _event_section_tags(node, label)
        events.append(
            RawTemporalEvent(
                event_id=f"audio_{index:05d}",
                time_span=event.time_span,
                modality="audio",
                event_type="audio_event",
                text=label,
                confidence=float(event.confidence if event.confidence is not None else 0.75),
                section_id=_primary_section_id(tags),
                source_node_id=node.node_id if node is not None else None,
                metadata={
                    "section_tags": tags,
                    "audio_index_source": "prepared_audio_events",
                    "audio_tags": _audio_tags_from_label(label),
                    "audio_caption": event.label.strip(),
                },
            )
        )
    return events


def _annotate_occurrences(
    events: list[RawTemporalEvent],
    sections: list[SectionNode],
) -> None:
    by_signature: dict[str, list[RawTemporalEvent]] = defaultdict(list)
    by_section: dict[str, list[RawTemporalEvent]] = defaultdict(list)
    for event in sorted(events, key=lambda item: (item.time_span.start, item.event_id)):
        signature = _occurrence_signature(event)
        if signature:
            by_signature[signature].append(event)
        if event.section_id:
            by_section[event.section_id].append(event)

    for group in by_signature.values():
        if not group:
            continue
        first_seen = min(item.time_span.start for item in group)
        last_seen = max(item.time_span.end for item in group)
        chain = [item.event_id for item in group]
        for index, event in enumerate(group, start=1):
            event.metadata["first_seen"] = first_seen
            event.metadata["last_seen"] = last_seen
            event.metadata["occurrence_index"] = index
            event.metadata["occurrence_count"] = len(group)
            event.metadata["version_chain"] = list(chain)
            event.metadata["change_type"] = _change_type_for_occurrence(event, index, group)
            if index > 1:
                previous = group[index - 2]
                event.metadata["previous_same_entity_event"] = previous.event_id
                event.linked_events = _dedupe([*event.linked_events, previous.event_id])
            if index < len(group):
                next_event = group[index]
                event.metadata["next_same_entity_event"] = next_event.event_id
                event.linked_events = _dedupe([*event.linked_events, next_event.event_id])

    section_ordinals = {section.section_id: section.ordinal for section in sections}
    for section_id, group in by_section.items():
        for ordinal, event in enumerate(
            sorted(group, key=lambda item: (item.time_span.start, item.event_id)),
            start=1,
        ):
            event.metadata["section_local_ordinal"] = ordinal
            event.metadata["section_ordinal"] = section_ordinals.get(section_id)


def _link_audio_events(
    audio_events: list[RawTemporalEvent],
    all_events: list[RawTemporalEvent],
) -> None:
    for audio_event in audio_events:
        aligned: list[str] = []
        for event in all_events:
            if event.event_id == audio_event.event_id:
                continue
            if event.modality not in {"visual", "ocr", "asr"}:
                continue
            if not audio_event.time_span.overlaps(event.time_span):
                continue
            aligned.append(event.event_id)
            if len(aligned) >= 8:
                break
        audio_event.linked_events = _dedupe([*audio_event.linked_events, *aligned])
        if aligned:
            audio_event.event_type = "audio_visual_alignment"
            audio_event.metadata["aligned_event_ids"] = aligned
            audio_event.metadata["alignment_modalities"] = _dedupe(
                [
                    event.modality
                    for event in all_events
                    if event.event_id in set(aligned)
                ]
            )


def _occurrence_signature(event: RawTemporalEvent) -> str:
    text = event.text or ""
    tokens = [
        token
        for token in re.findall(r"[a-z0-9_]+", text.lower().replace("-", "_"))
        if token not in {"the", "a", "an", "and", "or", "to", "in", "on", "of", "is", "are"}
    ]
    if not tokens:
        return ""
    if event.modality in {"audio", "visual"}:
        key_tokens = tokens[:6]
    elif event.modality == "code":
        key_tokens = tokens[:2]
    else:
        key_tokens = tokens[:5]
    return f"{event.modality}:{event.event_type}:{' '.join(key_tokens)}"


def _change_type_for_occurrence(
    event: RawTemporalEvent,
    index: int,
    group: list[RawTemporalEvent],
) -> str:
    if index == 1:
        return "introduced"
    previous = group[index - 2]
    if (event.text or "").strip().casefold() == (previous.text or "").strip().casefold():
        return "repeated"
    return "changed"


def _audio_tags_from_label(label: str) -> list[str]:
    lowered = label.lower()
    tags = []
    for tag, cues in {
        "speech_like": ("speech", "voice", "talking", "conversation"),
        "music": ("music", "song", "melody", "instrument"),
        "alert": ("beep", "alarm", "alert", "ring"),
        "impact": ("bang", "hit", "crash", "knock"),
        "crowd": ("applause", "cheer", "crowd", "clapping"),
        "environment": ("wind", "rain", "traffic", "engine", "noise"),
    }.items():
        if any(cue in lowered for cue in cues):
            tags.append(tag)
    return tags or ["ambient_audio"]


def _build_sections(
    events: list[RawTemporalEvent],
    duration_seconds: float,
) -> list[SectionNode]:
    grouped: dict[str, list[RawTemporalEvent]] = defaultdict(list)
    for event in events:
        tags = [event.section_id or ""]
        tags.extend(str(item) for item in event.metadata.get("section_tags", []))
        for tag in _dedupe([tag for tag in tags if tag]):
            grouped[tag].append(event)

    if duration_seconds > 0:
        grouped.setdefault("first_half", [])
        grouped.setdefault("second_half", [])

    sections: list[SectionNode] = []
    for ordinal, (section_id, section_events) in enumerate(
        sorted(grouped.items(), key=lambda item: (_section_sort_key(item[0]), item[0])),
        start=1,
    ):
        if section_events:
            start = min(event.time_span.start for event in section_events)
            end = max(event.time_span.end for event in section_events)
            modalities = Counter(event.modality for event in section_events)
            evidence_events = [event.event_id for event in section_events]
        elif section_id == "first_half":
            start, end = 0.0, duration_seconds / 2.0
            modalities = Counter()
            evidence_events = []
        elif section_id == "second_half":
            start, end = duration_seconds / 2.0, duration_seconds
            modalities = Counter()
            evidence_events = []
        else:
            continue
        labels = _section_labels(section_id)
        sections.append(
            SectionNode(
                section_id=section_id,
                ordinal=ordinal,
                time_span=TimeSpan(start, end),
                labels=labels,
                evidence_events=evidence_events,
                dominant_modalities=[name for name, _ in modalities.most_common(4)],
                confidence=0.85 if section_events else 0.5,
            )
        )
    return sections


def _assign_section_memberships(
    events: list[RawTemporalEvent],
    sections: list[SectionNode],
) -> None:
    by_id = {section.section_id: section for section in sections}
    for event in events:
        if event.section_id in by_id:
            continue
        for section in sections:
            if section.time_span.overlaps(event.time_span):
                event.section_id = section.section_id
                break


def _build_operator_events(events: list[RawTemporalEvent]) -> list[OperatorEvent]:
    by_operator: dict[str, OperatorEvent] = {}
    for event in events:
        text = event.text or ""
        for operator in _extract_comparison_operator_list(text):
            existing = by_operator.get(operator)
            if existing is None:
                by_operator[operator] = OperatorEvent(
                    operator=operator,
                    operator_class="comparison",
                    first_seen=event.time_span.start,
                    section_id=event.section_id,
                    source=[event.modality],
                    context=text[:220],
                    event_ids=[event.event_id],
                )
            else:
                existing.first_seen = min(existing.first_seen, event.time_span.start)
                if event.modality not in existing.source:
                    existing.source.append(event.modality)
                existing.event_ids.append(event.event_id)
                if existing.section_id != "comparison_section" and event.section_id:
                    existing.section_id = event.section_id
    return [by_operator[operator] for operator in ("==", "!=", ">", "<") if operator in by_operator]


def _build_code_snapshots(code_events: list[RawTemporalEvent]) -> list[CodeSnapshot]:
    snapshots: list[CodeSnapshot] = []
    active_lines: list[str] = []
    seen_lines: set[str] = set()
    derived_events: list[str] = []
    active_line_event_ids: dict[str, str] = {}
    for index, event in enumerate(
        sorted(code_events, key=lambda item: (item.time_span.start, item.event_id)),
        start=1,
    ):
        line = _normalize_code_assignment(event.text or "")
        if not line:
            continue
        line_key = line.casefold()
        if line_key not in seen_lines:
            seen_lines.add(line_key)
            active_lines.append(line)
            active_line_event_ids[line] = event.event_id
        derived_events.append(event.event_id)
        variables = _evaluate_code_assignments(active_lines)
        snapshots.append(
            CodeSnapshot(
                snapshot_id=f"snapshot_{index:05d}",
                timestamp=event.time_span.end,
                section_id=event.section_id,
                active_lines=list(active_lines),
                variables=variables,
                derived_from_events=list(derived_events),
                metadata={
                    "latest_code_event_id": event.event_id,
                    "active_line_event_ids": dict(active_line_event_ids),
                },
            )
        )
    return snapshots


def _build_temporal_links(
    events: list[RawTemporalEvent],
    sections: list[SectionNode],
    operators: list[OperatorEvent],
    snapshots: list[CodeSnapshot],
) -> list[TemporalLink]:
    links: list[TemporalLink] = []
    section_ids = {section.section_id for section in sections}
    for event in events:
        if event.section_id in section_ids:
            links.append(TemporalLink(event.event_id, "same_section", event.section_id))
    for previous, current in zip(
        sorted(sections, key=lambda item: item.time_span.start),
        sorted(sections, key=lambda item: item.time_span.start)[1:],
        strict=False,
    ):
        if previous.section_id != current.section_id:
            links.append(TemporalLink(previous.section_id, "before", current.section_id))
            links.append(TemporalLink(current.section_id, "after", previous.section_id))
    for operator in operators:
        for event_id in operator.event_ids:
            links.append(TemporalLink(event_id, "mentions_operator", f"operator_{operator.operator}"))
    for snapshot in snapshots:
        for event_id in snapshot.derived_from_events:
            links.append(TemporalLink(snapshot.snapshot_id, "derived_from", event_id))
    return links


def _extract_code_assignments(text: str) -> list[str]:
    assignments: list[str] = []
    seen: set[str] = set()
    sanitized = text.replace("|", " ").replace("\\$", "$")
    for match in CODE_ASSIGNMENT_PATTERN.finditer(sanitized):
        assignment = _normalize_code_assignment(f"{match.group(1)} = {match.group(2)}")
        if not _assignment_is_usable(assignment):
            continue
        key = assignment.casefold()
        if key in seen:
            continue
        seen.add(key)
        assignments.append(assignment)
    return assignments


def _normalize_code_assignment(assignment: str) -> str:
    compact = " ".join(assignment.replace("\\", "").split())
    compact = CODE_OPERATOR_PATTERN.sub(lambda match: f" {match.group(1)} ", compact)
    return " ".join(compact.split())


def _assignment_is_usable(assignment: str) -> bool:
    lhs, _, rhs = assignment.partition("=")
    lhs = lhs.strip()
    rhs = rhs.strip()
    if not lhs or not rhs or lhs.lower() in {"print", "assistant", "shell", "python"}:
        return False
    if not lhs.isidentifier() or lhs != lhs.lower():
        return False
    if any(
        term in rhs.lower()
        for term in ("assistant", "shell", "exception", "python", "variables", "name", "valu")
    ):
        return False
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", rhs) and rhs != rhs.lower():
        return False
    valid_short_variables = {"a", "b", "c", "d", "e", "f", "g", "h", "x", "y", "z", "i"}
    rhs_variables = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", rhs)
    if lhs in valid_short_variables and any(
        token not in valid_short_variables and token not in {"True", "False"}
        for token in rhs_variables
    ):
        return False
    if lhs in {"a", "b"} and re.fullmatch(r"\d+(?:\.\d+)?", rhs):
        try:
            if float(rhs) > 20:
                return False
        except ValueError:
            return False
    if lhs not in {"a", "b", "x", "y", "z", "i"} and re.fullmatch(
        r"\d+(?:\.\d+)?|True|False|\"[^\"]*\"|'[^']*'",
        rhs,
    ):
        return False
    return len(rhs) <= 40


def _assignment_rhs_contains_comparison(assignment: str) -> bool:
    return any(operator in assignment.partition("=")[2] for operator in ("==", "!=", ">=", "<=", ">", "<"))


def _assignment_is_arithmetic(assignment: str) -> bool:
    rhs = assignment.partition("=")[2]
    return any(operator in rhs for operator in ("+", "-", "*", "/")) and not _assignment_rhs_contains_comparison(assignment)


def _assignment_is_initial_declaration(assignment: str) -> bool:
    lhs = assignment.partition("=")[0].strip()
    return lhs in {"x", "y", "z", "i"} or not (
        _assignment_is_arithmetic(assignment) or _assignment_rhs_contains_comparison(assignment)
    )


def _code_event_type(assignment: str) -> str:
    if _assignment_rhs_contains_comparison(assignment):
        return "comparison_line"
    if _assignment_is_arithmetic(assignment):
        return "assignment_line"
    return "assignment_line"


def _evaluate_code_assignments(assignments: list[str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for assignment in assignments:
        lhs, _, rhs = assignment.partition("=")
        variable = lhs.strip()
        if not variable:
            continue
        value = _safe_eval_code_expression(rhs.strip(), values)
        if value is not None:
            values[variable] = value
    return values


def _safe_eval_code_expression(expression: str, values: dict[str, Any]) -> Any | None:
    try:
        tree = ast.parse(expression, mode="eval")
        return _eval_code_ast_node(tree.body, values)
    except (SyntaxError, KeyError, TypeError, ZeroDivisionError):
        return None


def _eval_code_ast_node(node: ast.AST, values: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool | int | float | str):
            return node.value
        raise TypeError("unsupported constant")
    if isinstance(node, ast.Name):
        if node.id not in values:
            raise KeyError(node.id)
        return values[node.id]
    if isinstance(node, ast.BinOp):
        left = _eval_code_ast_node(node.left, values)
        right = _eval_code_ast_node(node.right, values)
        if not _is_numeric_value(left) or not _is_numeric_value(right):
            raise TypeError("unsupported binary operands")
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise TypeError("unsupported comparison chain")
        left = _eval_code_ast_node(node.left, values)
        right = _eval_code_ast_node(node.comparators[0], values)
        operator = node.ops[0]
        if isinstance(operator, ast.Eq):
            return left == right
        if isinstance(operator, ast.NotEq):
            return left != right
        if isinstance(operator, ast.Gt):
            return left > right
        if isinstance(operator, ast.GtE):
            return left >= right
        if isinstance(operator, ast.Lt):
            return left < right
        if isinstance(operator, ast.LtE):
            return left <= right
    raise TypeError("unsupported expression")


def _is_numeric_value(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _unique_text_lines(text: str) -> list[str]:
    candidates = re.split(r"[\n\r]+", text)
    if len(candidates) == 1:
        candidates = re.split(r"\s{2,}|(?<=\))\s+|(?<=True)\s+|(?<=False)\s+", text)
    seen: set[str] = set()
    lines: list[str] = []
    for raw_line in candidates:
        line = " ".join(raw_line.split()).strip()
        if not line:
            continue
        key = re.sub(r"\W+", "", line.casefold())
        if not key or key in seen:
            continue
        seen.add(key)
        lines.append(line)
    return lines


def _leading_screen_text_lines(lines: list[str], limit: int = 8) -> set[str]:
    selected: list[str] = []
    for line in lines:
        if _line_is_code_boundary(line):
            if selected:
                break
            continue
        if _line_is_noise(line):
            continue
        selected.append(line)
        if len(selected) >= limit:
            break
    return set(selected)


def _screen_region_for_line(line: str, is_leading_screen_text: bool) -> str:
    if is_leading_screen_text:
        return "editor_header"
    if CODE_ASSIGNMENT_PATTERN.search(line) or line.strip().startswith("print("):
        return "code_editor_body"
    if _line_looks_terminal_output(line):
        return "terminal_panel"
    return "screen_text"


def _ocr_event_type(line: str, screen_region: str) -> str:
    if screen_region == "editor_header":
        return "ui_header"
    if screen_region == "terminal_panel":
        return "terminal_output"
    if screen_region == "code_editor_body":
        return "code_text"
    return "visible_text"


def _line_is_code_boundary(line: str) -> bool:
    lowered = line.lower().strip()
    if re.fullmatch(r"\d+\.?", lowered):
        return True
    if lowered.startswith(("#", '"""', "print(")):
        return True
    if any(cue in lowered for cue in (".py", "assistant", "exception", "shell", "variables")):
        return True
    return bool(CODE_ASSIGNMENT_PATTERN.search(line))


def _line_is_noise(line: str) -> bool:
    if not re.search(r"[a-zA-Z]", line):
        return True
    return len(line.strip()) <= 1


def _line_looks_terminal_output(line: str) -> bool:
    stripped = line.strip()
    return bool(
        re.fullmatch(r"-?\d+(?:\.\d+)?", stripped)
        or re.fullmatch(r"True|False|bool|<class 'bool'>", stripped)
    )


def _extract_terminal_outputs(text: str) -> list[str]:
    outputs: list[str] = []
    for line in _unique_text_lines(text):
        if _line_looks_terminal_output(line) and line not in outputs:
            outputs.append(line)
    return outputs


def _content_tags_from_text(text: str) -> list[str]:
    lowered = text.lower()
    tags: list[str] = []
    assignments = _extract_code_assignments(text)
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
        or any(_assignment_is_arithmetic(assignment) for assignment in assignments)
    ):
        tags.append("arithmetic_section")
    if (
        "comparison operator" in lowered
        or "comparison operators" in lowered
        or any(_assignment_rhs_contains_comparison(assignment) for assignment in assignments)
        or _extract_comparison_operator_list(text)
    ):
        tags.append("comparison_section")
    if any(
        cue in lowered
        for cue in ("equal to", "not equal", "greater than", "less than", "boolean")
    ):
        tags.append("logical_evaluation_section")
    if "shell" in lowered or "output" in lowered or "print(" in lowered:
        tags.append("output_section")
    if ("assignment_section" in tags or "arithmetic_section" in tags) and (
        "comparison_section" not in tags
    ):
        tags.append("pre_comparison_section")
    return _dedupe(tags)


def _extract_comparison_operator_list(text: str) -> list[str]:
    lowered = text.lower()
    operators: list[str] = []
    for operator, patterns in COMPARISON_OPERATOR_PATTERNS:
        if any(re.search(pattern, lowered) for pattern in patterns):
            operators.append(operator)
    return operators


def _dedupe_events_by_text_and_time(events: list[RawTemporalEvent]) -> list[RawTemporalEvent]:
    deduped: list[RawTemporalEvent] = []
    seen: set[tuple[str, int]] = set()
    for event in events:
        key = (event.text or "", int(event.time_span.start))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(event)
    for index, event in enumerate(deduped, start=1):
        event.event_id = f"code_{index:05d}"
    return deduped


def _section_sort_key(section_id: str) -> tuple[int, str]:
    priority = {
        "first_half": 0,
        "assignment_section": 1,
        "arithmetic_section": 2,
        "pre_comparison_section": 3,
        "comparison_section": 4,
        "logical_evaluation_section": 5,
        "output_section": 6,
        "second_half": 7,
    }
    if section_id.startswith("segment_"):
        try:
            return 20 + int(section_id.split("_", 1)[1]), section_id
        except ValueError:
            pass
    return priority.get(section_id, 50), section_id


def _section_labels(section_id: str) -> list[str]:
    labels = [section_id.replace("_", " ")]
    ordinal_match = re.fullmatch(r"segment_(\d+)", section_id)
    if ordinal_match:
        ordinal = {
            "1": "first segment",
            "2": "second segment",
            "3": "third segment",
            "4": "fourth segment",
            "5": "fifth segment",
        }.get(ordinal_match.group(1))
        if ordinal:
            labels.append(ordinal)
    return _dedupe(labels)


def _dedupe(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
