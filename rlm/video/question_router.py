from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from rlm.video.types import Evidence, Modality

QuestionRouteLabel = Literal[
    "code_value_eval",
    "terminal_output",
    "ui_header_text",
    "assignment_count",
    "operator_list",
    "speech_explanation",
    "sentiment_analysis",
    "audio_event",
    "audio_visual_alignment",
    "visual_difference",
    "causal_chain",
    "temporal_occurrence",
    "rubric_explanation",
    "generic",
]


@dataclass(frozen=True)
class QuestionRoute:
    label: QuestionRouteLabel
    preferred_modality: Modality | None
    required_evidence_kinds: tuple[str, ...] = ()
    allowed_evidence_kinds: tuple[str, ...] = ()
    blocked_evidence_kinds: tuple[str, ...] = ()
    requires_exact_answer_span: bool = True
    requires_computed_value: bool = False
    answer_verifier: str = "route_evidence_compatibility"

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "preferred_modality": self.preferred_modality,
            "required_evidence_kinds": list(self.required_evidence_kinds),
            "allowed_evidence_kinds": list(self.allowed_evidence_kinds),
            "blocked_evidence_kinds": list(self.blocked_evidence_kinds),
            "requires_exact_answer_span": self.requires_exact_answer_span,
            "requires_computed_value": self.requires_computed_value,
            "answer_verifier": self.answer_verifier,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> QuestionRoute:
        label = str(data.get("label") or "generic")
        if label not in ROUTE_LABELS:
            label = "generic"
        modality = data.get("preferred_modality")
        if modality not in {"speech", "visual", "ocr", "audio", "cross_modal", None}:
            modality = None
        return cls(
            label=cast(QuestionRouteLabel, label),
            preferred_modality=cast(Modality | None, modality),
            required_evidence_kinds=tuple(
                str(item) for item in data.get("required_evidence_kinds", [])
            ),
            allowed_evidence_kinds=tuple(
                str(item) for item in data.get("allowed_evidence_kinds", [])
            ),
            blocked_evidence_kinds=tuple(
                str(item) for item in data.get("blocked_evidence_kinds", [])
            ),
            requires_exact_answer_span=bool(data.get("requires_exact_answer_span", True)),
            requires_computed_value=bool(data.get("requires_computed_value", False)),
            answer_verifier=str(data.get("answer_verifier") or "route_evidence_compatibility"),
        )


@dataclass(frozen=True)
class StopVerification:
    accepted: bool
    route_label: QuestionRouteLabel
    reason: str
    compatible_evidence_ids: tuple[str, ...] = ()
    incompatible_evidence_ids: tuple[str, ...] = ()
    missing_requirements: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "route_label": self.route_label,
            "reason": self.reason,
            "compatible_evidence_ids": list(self.compatible_evidence_ids),
            "incompatible_evidence_ids": list(self.incompatible_evidence_ids),
            "missing_requirements": list(self.missing_requirements),
        }


ROUTE_LABELS = {
    "code_value_eval",
    "terminal_output",
    "ui_header_text",
    "assignment_count",
    "operator_list",
    "speech_explanation",
    "sentiment_analysis",
    "audio_event",
    "audio_visual_alignment",
    "visual_difference",
    "causal_chain",
    "temporal_occurrence",
    "rubric_explanation",
    "generic",
}

CODE_VALUE_KINDS = {
    "computed_expression_value",
    "computed_variable_value",
    "target_assignment",
    "code_line",
}
COMPUTED_VALUE_KINDS = {
    "computed_expression_value",
    "computed_output_value",
    "computed_variable_value",
}
CODE_TEXT_KINDS = {
    "assignment_count",
    "assignment_count_partial",
    "code_line",
    "comparison_assignments",
    "comparison_operator_count",
    "comparison_operator_count_partial",
    "comparison_operator_list",
    "computed_expression_value",
    "computed_output_value",
    "computed_variable_value",
    "target_assignment",
}
UI_TEXT_KINDS = {"screen_text_block", "text_line", "raw_compact"}
OPERATOR_LIST_KINDS = {"comparison_operator_count", "comparison_operator_list"}
ASSIGNMENT_COUNT_KINDS = {"assignment_count"}
TERMINAL_OUTPUT_KINDS = {"computed_output_value"}
AUDIO_EVENT_KINDS = {"audio_event"}
AUDIO_VISUAL_ALIGNMENT_KINDS = {"audio_event", "audio_visual_alignment", "visual_anchor"}

REFUSAL_PATTERNS = (
    "could not fill",
    "cannot determine",
    "can't determine",
    "could not determine",
    "not enough evidence",
    "insufficient evidence",
    "missing slots",
    "i don't know",
    "i do not know",
    "unable to answer",
)


def route_question(
    question: str,
    task_type: str | None = None,
    context: Mapping[str, Any] | None = None,
) -> QuestionRoute:
    lowered = question.lower()
    tokens = set(re.findall(r"[a-z0-9_]+", lowered))
    context_tools = _context_terms(context, "required_tools")
    context_modalities = _context_terms(context, "expected_modalities")

    if task_type == "sentiment_analysis":
        return QuestionRoute(
            label="sentiment_analysis",
            preferred_modality="cross_modal",
            requires_exact_answer_span=False,
        )
    if _is_audio_visual_alignment_question(lowered, tokens, context_tools, context_modalities):
        return QuestionRoute(
            label="audio_visual_alignment",
            preferred_modality="cross_modal",
            required_evidence_kinds=("audio_visual_alignment",),
            allowed_evidence_kinds=tuple(sorted(AUDIO_VISUAL_ALIGNMENT_KINDS)),
            requires_exact_answer_span=False,
        )
    if _is_temporal_occurrence_question(lowered, tokens):
        subroute = _temporal_occurrence_subroute(lowered)
        return QuestionRoute(
            label="temporal_occurrence",
            preferred_modality=subroute.preferred_modality,
            required_evidence_kinds=subroute.required_evidence_kinds,
            allowed_evidence_kinds=subroute.allowed_evidence_kinds,
            blocked_evidence_kinds=(
                *subroute.blocked_evidence_kinds,
                "global_operator_count",
                "global_assignment_count",
            ),
            requires_exact_answer_span=subroute.requires_exact_answer_span,
            requires_computed_value=subroute.requires_computed_value,
            answer_verifier=f"temporal_occurrence:{subroute.label}",
        )
    if _is_assignment_count_question(lowered):
        return QuestionRoute(
            label="assignment_count",
            preferred_modality="ocr",
            required_evidence_kinds=("assignment_count",),
            allowed_evidence_kinds=tuple(sorted(ASSIGNMENT_COUNT_KINDS)),
            blocked_evidence_kinds=(
                "assignment_count_partial",
                "code_line",
                "comparison_assignments",
                "target_assignment",
            ),
        )
    if _is_operator_list_question(lowered):
        return QuestionRoute(
            label="operator_list",
            preferred_modality="ocr",
            required_evidence_kinds=tuple(sorted(OPERATOR_LIST_KINDS)),
            allowed_evidence_kinds=tuple(sorted(OPERATOR_LIST_KINDS)),
            blocked_evidence_kinds=(
                "comparison_assignments",
                "code_line",
                "target_assignment",
            ),
        )
    if _is_ui_header_text_question(lowered):
        return QuestionRoute(
            label="ui_header_text",
            preferred_modality="ocr",
            allowed_evidence_kinds=tuple(sorted(UI_TEXT_KINDS)),
            blocked_evidence_kinds=tuple(sorted(CODE_TEXT_KINDS - UI_TEXT_KINDS)),
        )
    if _is_terminal_output_question(lowered):
        return QuestionRoute(
            label="terminal_output",
            preferred_modality="ocr",
            required_evidence_kinds=("computed_output_value",),
            allowed_evidence_kinds=tuple(sorted(TERMINAL_OUTPUT_KINDS)),
            blocked_evidence_kinds=("code_line", "target_assignment", "comparison_assignments"),
            requires_computed_value=True,
        )
    if _is_code_value_question(lowered):
        requires_computed_value = _requires_computed_code_value(lowered)
        allowed = COMPUTED_VALUE_KINDS if requires_computed_value else CODE_VALUE_KINDS
        return QuestionRoute(
            label="code_value_eval",
            preferred_modality="ocr",
            required_evidence_kinds=tuple(sorted(allowed)),
            allowed_evidence_kinds=tuple(sorted(allowed)),
            blocked_evidence_kinds=("comparison_assignments", "assignment_count_partial"),
            requires_computed_value=requires_computed_value,
        )
    if _is_visual_difference_question(lowered, tokens):
        return QuestionRoute(
            label="visual_difference",
            preferred_modality="visual",
            requires_exact_answer_span=False,
        )
    if _is_sentiment_analysis_question(
        lowered,
        tokens,
        task_type,
        context_tools,
        context_modalities,
    ):
        return QuestionRoute(
            label="sentiment_analysis",
            preferred_modality="cross_modal",
            requires_exact_answer_span=False,
        )
    if _is_speech_explanation_question(lowered, tokens, context_tools, context_modalities):
        return QuestionRoute(
            label="speech_explanation",
            preferred_modality="speech",
            requires_exact_answer_span=False,
        )
    if _is_causal_chain_question(lowered, tokens):
        return QuestionRoute(
            label="causal_chain",
            preferred_modality="speech",
            requires_exact_answer_span=False,
        )
    if _is_rubric_explanation_question(lowered, tokens):
        return QuestionRoute(
            label="rubric_explanation",
            preferred_modality="speech",
            requires_exact_answer_span=False,
        )
    if _is_audio_event_question(lowered, tokens, context_tools, context_modalities):
        return QuestionRoute(
            label="audio_event",
            preferred_modality="audio",
            allowed_evidence_kinds=tuple(sorted(AUDIO_EVENT_KINDS)),
            requires_exact_answer_span=False,
        )
    return QuestionRoute(
        label="generic",
        preferred_modality=None,
        requires_exact_answer_span=False,
        answer_verifier="disabled_for_generic_route",
    )


def route_from_metadata(data: Mapping[str, Any] | None) -> QuestionRoute | None:
    if not data:
        return None
    route_data = data.get("question_route")
    if isinstance(route_data, Mapping):
        return QuestionRoute.from_dict(route_data)
    return None


def verify_stop_answer(
    *,
    question: str,
    answer: str,
    evidence_items: Sequence[Evidence],
    route: QuestionRoute | None = None,
) -> StopVerification:
    selected_route = route or route_question(question)
    if selected_route.label == "generic":
        return StopVerification(
            accepted=bool(answer.strip()),
            route_label=selected_route.label,
            reason="generic_route",
        )

    answer_text = answer.strip()
    if not answer_text:
        return StopVerification(
            accepted=False,
            route_label=selected_route.label,
            reason="empty_answer",
            missing_requirements=("answer",),
        )

    compatible: list[Evidence] = []
    incompatible: list[Evidence] = []
    for item in evidence_items:
        if evidence_matches_route(item, selected_route):
            compatible.append(item)
        else:
            incompatible.append(item)

    compatible_ids = tuple(item.evidence_id for item in compatible)
    incompatible_ids = tuple(item.evidence_id for item in incompatible)
    if not compatible:
        return StopVerification(
            accepted=False,
            route_label=selected_route.label,
            reason="no_route_compatible_evidence",
            incompatible_evidence_ids=incompatible_ids,
            missing_requirements=("route_compatible_evidence",),
        )

    if _looks_like_refusal(answer_text):
        return StopVerification(
            accepted=False,
            route_label=selected_route.label,
            reason="refusal_despite_route_compatible_evidence",
            compatible_evidence_ids=compatible_ids,
            incompatible_evidence_ids=incompatible_ids,
        )

    if selected_route.requires_exact_answer_span and not any(
        answer_supported_by_evidence(answer_text, item, selected_route) for item in compatible
    ):
        return StopVerification(
            accepted=False,
            route_label=selected_route.label,
            reason="answer_not_supported_by_exact_evidence_span",
            compatible_evidence_ids=compatible_ids,
            incompatible_evidence_ids=incompatible_ids,
            missing_requirements=("exact_answer_span_match",),
        )

    return StopVerification(
        accepted=True,
        route_label=selected_route.label,
        reason="accepted",
        compatible_evidence_ids=compatible_ids,
        incompatible_evidence_ids=incompatible_ids,
    )


def evidence_matches_route(item: Evidence, route: QuestionRoute) -> bool:
    kind = _evidence_kind(item)
    if kind and kind in set(route.blocked_evidence_kinds):
        return False
    if route.label == "sentiment_analysis":
        return item.modality in {"speech", "visual", "audio", "cross_modal"}
    if route.label in {"speech_explanation", "causal_chain", "rubric_explanation"}:
        return item.modality == route.preferred_modality
    if route.label == "visual_difference":
        return item.modality == "visual"
    if route.label == "audio_event":
        return item.modality == "audio" and (not kind or kind in AUDIO_EVENT_KINDS)
    if route.label == "audio_visual_alignment":
        return item.modality in {"audio", "visual", "cross_modal"} and (
            not kind or kind in AUDIO_VISUAL_ALIGNMENT_KINDS
        )
    if route.label == "temporal_occurrence":
        nested_verifier = route.answer_verifier.partition(":")[2]
        nested_label = nested_verifier or "generic"
        if nested_label in ROUTE_LABELS and nested_label != "temporal_occurrence":
            nested_route = QuestionRoute(
                label=cast(QuestionRouteLabel, nested_label),
                preferred_modality=route.preferred_modality,
                required_evidence_kinds=route.required_evidence_kinds,
                allowed_evidence_kinds=route.allowed_evidence_kinds,
                blocked_evidence_kinds=route.blocked_evidence_kinds,
                requires_exact_answer_span=route.requires_exact_answer_span,
                requires_computed_value=route.requires_computed_value,
            )
            return evidence_matches_route(item, nested_route)
        return item.modality == route.preferred_modality
    if route.label == "ui_header_text":
        return item.modality == "ocr" and kind in UI_TEXT_KINDS and not _text_looks_like_code(item)
    if route.label == "assignment_count":
        return item.modality == "ocr" and kind in ASSIGNMENT_COUNT_KINDS
    if route.label == "operator_list":
        return item.modality == "ocr" and kind in OPERATOR_LIST_KINDS
    if route.label == "terminal_output":
        return item.modality == "ocr" and kind in TERMINAL_OUTPUT_KINDS
    if route.label == "code_value_eval":
        if route.requires_computed_value:
            return item.modality == "ocr" and kind in COMPUTED_VALUE_KINDS
        return item.modality == "ocr" and kind in CODE_VALUE_KINDS
    return True


def answer_supported_by_evidence(answer: str, item: Evidence, route: QuestionRoute) -> bool:
    spans = _answer_spans(item)
    if not spans:
        return False
    normalized_answer = _normalize_answer_text(answer)
    if route.label in {"assignment_count", "terminal_output", "code_value_eval"}:
        answer_numbers = _extract_numbers(answer)
        for span in spans:
            if _normalize_answer_text(span) in normalized_answer:
                return True
            if answer_numbers and set(_extract_numbers(span)) & set(answer_numbers):
                return True
        return False
    if route.label == "operator_list":
        answer_ops = _extract_operators(answer)
        answer_numbers = _extract_numbers(answer)
        for span in spans:
            span_ops = _extract_operators(span)
            if _normalize_answer_text(span) in normalized_answer:
                return True
            if answer_ops and span_ops and bool(answer_ops & span_ops):
                return True
            if answer_numbers and set(_extract_numbers(span)) & set(answer_numbers):
                return True
        return False
    return any(_normalize_answer_text(span) in normalized_answer for span in spans)


def format_answer_for_route(
    answer: str,
    route: QuestionRoute,
    evidence_items: Sequence[Evidence] = (),
) -> str:
    answer_text = answer.strip()
    if not answer_text:
        return answer_text
    if _looks_like_refusal(answer_text):
        return answer_text
    if _looks_like_complete_sentence(answer_text):
        return answer_text

    evidence_kind = ""
    operator_list: list[str] = []
    for item in evidence_items:
        evidence_kind = evidence_kind or _evidence_kind(item)
        operator_list.extend(sorted(_extract_operators(str(item.metadata.get("answer_span") or ""))))
        operator_list.extend(sorted(_extract_operators(item.detail)))
    operator_list = _ordered_unique_operators(operator_list)

    if route.label in {"code_value_eval", "temporal_occurrence"} and _looks_like_number(answer_text):
        if "computed_output_value" in {evidence_kind, _route_nested_kind(route)}:
            return f"The final output value is {answer_text}."
        return f"The expression evaluates to {answer_text}."
    if route.label == "terminal_output" and _looks_like_number(answer_text):
        return f"The final output value is {answer_text}."
    if route.label == "terminal_output":
        return f"The terminal output is {answer_text}."
    if route.label == "assignment_count" and _looks_like_integer(answer_text):
        count = int(float(answer_text))
        noun = "assignment" if count == 1 else "assignments"
        return f"There {'is' if count == 1 else 'are'} {count} {noun}."
    if route.label == "operator_list":
        count = _first_int(answer_text)
        if operator_list:
            return (
                f"There are {count if count is not None else len(operator_list)} comparison "
                f"operators: {_human_join(operator_list)}."
            )
        if count is not None:
            noun = "comparison operator" if count == 1 else "comparison operators"
            return f"There {'is' if count == 1 else 'are'} {count} {noun}."
    if route.label == "ui_header_text":
        return f'The visible text is "{answer_text}".'
    if route.label == "audio_event":
        return f"The audio event is {answer_text}."
    if route.label == "audio_visual_alignment":
        return f"The audio-visual evidence shows {answer_text}."
    if route.label == "visual_difference":
        return f"The visual difference is {answer_text}."
    return answer_text


def _is_audio_event_question(
    lowered: str,
    tokens: set[str],
    context_tools: set[str],
    context_modalities: set[str],
) -> bool:
    if "audio" in context_modalities or "audio" in context_tools:
        return True
    speech_cues = {"say", "says", "said", "speech", "spoken", "transcribe", "translate"}
    if tokens & speech_cues:
        return False
    return any(
        cue in lowered
        for cue in (
            "audio event",
            "background sound",
            "environment sound",
            "mechanical sound",
            "sound is heard",
            "what sound",
            "noise",
            "beep",
            "ticking",
            "heard in the background",
        )
    )


def _route_nested_kind(route: QuestionRoute) -> str:
    return route.answer_verifier.partition(":")[2]


def _looks_like_complete_sentence(text: str) -> bool:
    lowered = text.lower()
    return (
        text.endswith((".", "?", "!"))
        or lowered.startswith(
            (
                "the ",
                "there ",
                "it ",
                "he ",
                "she ",
                "they ",
                "because ",
                "when ",
                "while ",
            )
        )
    )


def _looks_like_number(text: str) -> bool:
    return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", text.strip()))


def _looks_like_integer(text: str) -> bool:
    return bool(re.fullmatch(r"-?\d+(?:\.0+)?", text.strip()))


def _first_int(text: str) -> int | None:
    match = re.search(r"-?\d+", text)
    return int(match.group(0)) if match else None


def _ordered_unique_operators(operators: Sequence[str]) -> list[str]:
    ordered = []
    for operator in ("==", "!=", ">", "<", ">=", "<="):
        if operator in operators and operator not in ordered:
            ordered.append(operator)
    return ordered


def _human_join(items: Sequence[str]) -> str:
    values = list(items)
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])}, and {values[-1]}"


def _is_audio_visual_alignment_question(
    lowered: str,
    tokens: set[str],
    context_tools: set[str],
    context_modalities: set[str],
) -> bool:
    if {"audio", "visual"} <= context_modalities:
        return True
    if "audio_visual_alignment" in context_tools:
        return True
    return (
        bool(tokens & {"sound", "audio", "heard", "noise", "music"})
        and bool(tokens & {"see", "seen", "visible", "screen", "visual", "happen"})
        and any(cue in lowered for cue in ("at the same time", "while", "when", "match", "align"))
    )


def _is_temporal_occurrence_question(lowered: str, tokens: set[str]) -> bool:
    ordinal_tokens = {
        "first",
        "second",
        "third",
        "fourth",
        "last",
        "earliest",
        "latest",
        "before",
        "after",
        "previous",
        "next",
    }
    return bool(tokens & ordinal_tokens) and any(
        cue in lowered
        for cue in (
            "occurrence",
            "time",
            "section",
            "segment",
            "before",
            "after",
            "appears",
            "happens",
            "introduced",
        )
    )


def _temporal_occurrence_subroute(lowered: str) -> QuestionRoute:
    if _is_operator_list_question(lowered):
        return route_question("How many comparison operators are introduced in the tutorial?")
    if _is_assignment_count_question(lowered):
        return route_question("How many variables are declared in the Python script?")
    if _is_terminal_output_question(lowered):
        return route_question("What is the final output value displayed in the shell?")
    if _is_code_value_question(lowered):
        return route_question("What is the result of the expression in the code?")
    return QuestionRoute(
        label="generic",
        preferred_modality=None,
        requires_exact_answer_span=False,
    )


def _is_visual_difference_question(lowered: str, tokens: set[str]) -> bool:
    return bool(tokens & {"changed", "change", "different", "difference", "compare"}) and bool(
        tokens & {"visual", "visually", "scene", "scenes", "screen", "image", "look", "appearance"}
    )


def _is_causal_chain_question(lowered: str, tokens: set[str]) -> bool:
    return bool(tokens & {"why", "because", "cause", "caused", "lead", "led", "consequence"}) or any(
        cue in lowered for cue in ("what happened as a result", "why did", "how did it lead")
    )


def _is_rubric_explanation_question(lowered: str, tokens: set[str]) -> bool:
    return bool(tokens & {"explain", "summarize", "describe"}) and any(
        cue in lowered for cue in ("why", "how", "reason", "evidence", "support")
    )


def _is_assignment_count_question(lowered: str) -> bool:
    return "how many" in lowered and any(
        cue in lowered for cue in ("variable", "assignment", "declaration", "declared")
    )


def _is_operator_list_question(lowered: str) -> bool:
    return "comparison" in lowered and "operator" in lowered and any(
        cue in lowered
        for cue in (
            "introduced",
            "demonstrated",
            "how many",
            "which",
            "what are",
            "what comparison",
        )
    )


def _is_terminal_output_question(lowered: str) -> bool:
    return any(cue in lowered for cue in ("terminal", "shell", "console", "output")) and any(
        cue in lowered for cue in ("displayed", "printed", "final", "value", "result", "executed")
    )


def _is_code_value_question(lowered: str) -> bool:
    code_context_cues = (
        "arithmetic",
        "code",
        "expression",
        "final value",
        "python script",
        "script",
        "value of variable",
        "variable",
    )
    computed_value_cues = ("calculate", "result", "value")
    has_code_context = any(cue in lowered for cue in code_context_cues)
    has_computed_value_cue = any(cue in lowered for cue in computed_value_cues)
    return (
        has_code_context
        and has_computed_value_cue
        and not _is_assignment_count_question(lowered)
    )


def _requires_computed_code_value(lowered: str) -> bool:
    if "code line" in lowered and not any(
        cue in lowered for cue in ("result", "final value", "final output", "output value")
    ):
        return False
    if "mathematical expressions" in lowered and not any(
        cue in lowered for cue in ("result", "final value", "final output", "output value")
    ):
        return False
    return any(
        cue in lowered
        for cue in (
            "result",
            "final value",
            "final output",
            "output value",
            "calculate",
            "evaluates",
            "evaluated",
            "value of variable",
        )
    )


def _is_ui_header_text_question(lowered: str) -> bool:
    if _is_operator_list_question(lowered) or _is_assignment_count_question(lowered):
        return False
    if _is_terminal_output_question(lowered):
        return False
    if any(
        phrase in lowered
        for phrase in (
            "first real sign",
            "first sign that",
            "sign that",
            "sign of",
            "signs that",
        )
    ):
        return False
    explicit_visible_text = any(
        cue in lowered
        for cue in (
            "what is written",
            "what's written",
            "what text",
            "visible text",
            "on screen",
            "on the screen",
            "screen title",
            "screen header",
            "screen label",
            "laboratory door",
        )
    )
    if explicit_visible_text:
        return True
    if any(cue in lowered for cue in ("header", "title")):
        return any(
            context in lowered
            for context in ("screen", "page", "window", "slide", "sign", "door", "interface", "ui")
        )
    if any(cue in lowered for cue in ("label", "sign")):
        return any(
            context in lowered
            for context in ("written", "visible", "screen", "door", "posted", "displayed", "reads")
        )
    return any(
        cue in lowered
        for cue in (
            "what is written",
            "what's written",
            "what text",
            "laboratory door",
            "on screen",
            "visible text",
        )
    )


def _is_speech_explanation_question(
    lowered: str,
    tokens: set[str],
    context_tools: set[str],
    context_modalities: set[str],
) -> bool:
    if "speech" in context_modalities or "speech" in context_tools or "asr" in context_tools:
        return True
    speech_cues = {
        "say",
        "says",
        "said",
        "speaker",
        "presenter",
        "explain",
        "explains",
        "explained",
        "speech",
        "spoken",
        "transcribe",
        "translate",
    }
    return bool(tokens & speech_cues) or any(
        cue in lowered for cue in ("what does the presenter", "what is explained")
    )


def _is_sentiment_analysis_question(
    lowered: str,
    tokens: set[str],
    task_type: str | None,
    context_tools: set[str],
    context_modalities: set[str],
) -> bool:
    if task_type == "sentiment_analysis":
        return True
    sentiment_tokens = {
        "afraid",
        "angry",
        "anxious",
        "confident",
        "emotion",
        "emotional",
        "excited",
        "feel",
        "feeling",
        "feelings",
        "frustrated",
        "happy",
        "hesitant",
        "mood",
        "nervous",
        "reaction",
        "sad",
        "sentiment",
        "tense",
        "tone",
        "upset",
        "worried",
    }
    if tokens & sentiment_tokens:
        return True
    if any(
        phrase in lowered
        for phrase in (
            "body language",
            "facial expression",
            "how did he react",
            "how did she react",
            "how did they react",
            "how does he seem",
            "how does she seem",
            "how do they seem",
            "voice sound",
        )
    ):
        return True
    if ("sentiment" in context_tools or "sentiment" in context_modalities) and bool(
        context_modalities & {"speech", "visual", "audio", "audio_environment"}
    ):
        return True
    return False


def _context_terms(context: Mapping[str, Any] | None, key: str) -> set[str]:
    if not context:
        return set()
    value = context.get(key)
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, Sequence):
        values = [str(item) for item in value]
    else:
        values = []
    return {
        token.lower().replace("-", "_")
        for item in values
        for token in re.findall(r"[a-zA-Z0-9_-]+", item)
    }


def _evidence_kind(item: Evidence) -> str:
    for key in ("ocr_evidence_kind", "derived_evidence_type", "evidence_kind"):
        value = item.metadata.get(key)
        if value:
            return str(value)
    text = " ".join(part for part in (item.claim, item.detail) if part).lower()
    if "computed output" in text:
        return "computed_output_value"
    if "computed expression" in text:
        return "computed_expression_value"
    if "computed" in text and "value" in text:
        return "computed_variable_value"
    if "structured assignment" in text:
        return "target_assignment"
    if "comparison operator count" in text:
        return "comparison_operator_count"
    if "comparison operators" in text:
        return "comparison_operator_list"
    if "structured count" in text:
        return "assignment_count"
    if "ocr code line" in text or re.search(r"\b[A-Za-z_]\w*\s*=\s*[^=\n]+", text):
        return "code_line"
    if "ocr screen text block" in text:
        return "screen_text_block"
    if "ocr text line" in text or item.modality == "ocr":
        return "text_line"
    return ""


def _answer_spans(item: Evidence) -> list[str]:
    spans: list[str] = []
    for key in ("answer_span", "classifier_answer_span"):
        value = item.metadata.get(key)
        if isinstance(value, str) and value.strip():
            spans.append(value.strip())
    return spans


def _looks_like_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(pattern in lowered for pattern in REFUSAL_PATTERNS)


def _text_looks_like_code(item: Evidence) -> bool:
    text = "\n".join(part for part in (item.claim, item.detail) if part)
    return bool(re.search(r"\b[A-Za-z_]\w*\s*=\s*[^=\n]+", text))


def _normalize_answer_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9_.+\-*/=<>!]+", text.lower()))


def _extract_numbers(text: str) -> list[str]:
    return re.findall(r"(?<!\w)-?\d+(?:\.\d+)?(?!\w)", text)


def _extract_operators(text: str) -> set[str]:
    operators = set(re.findall(r"==|!=|>=|<=|(?<![=<])>(?![=>])|(?<![=<])<(?![=<])", text))
    lowered = text.lower()
    phrase_map = {
        "==": ("equal to", "equality"),
        "!=": ("not equal", "not equal to"),
        ">": ("greater than",),
        "<": ("less than",),
    }
    for operator, phrases in phrase_map.items():
        if any(phrase in lowered for phrase in phrases):
            operators.add(operator)
    return operators
