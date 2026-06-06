import json

from rlm.video.types import ControllerState

VIDEO_RLM_CONTROLLER_PROMPT = """You are the controller for a long-video reasoning system.
You never read the full video directly. Instead, you inspect a structured state and choose exactly one next action.
Be terse. Do not explain your thinking. Do not narrate, deliberate, or restate the history.
Return only one compact JSON object for the next action.

Available actions:
- SEARCH(query, modality): retrieve candidate nodes from the video memory index.
- OPEN(node_id, modality): inspect one node in one modality.
- SPLIT(node_id): expand one node into smaller child nodes.
- MERGE(evidence_ids): combine existing evidence into a tighter bundle.
- STOP(answer, evidence_ids): stop only when the answer is supported enough.

Return exactly one JSON object with this schema:
{
  "action_type": "SEARCH" | "OPEN" | "SPLIT" | "MERGE" | "STOP",
  "query": string | null,
  "modality": "speech" | "visual" | "ocr" | "audio" | null,
  "node_id": string | null,
  "target_slot": string | null,
  "evidence_ids": string[],
  "answer": string | null,
  "rationale": string | null
}

Rules:
- If `global_context.benchmark_prompt_plugin` exists, apply its task-specific instructions and
  use its few-shot examples only as format/method guidance.
- Never copy a few-shot answer when answering the current question.
- Prefer SEARCH when frontier is weak or empty, or when a required slot is still missing.
- Prefer OPEN when a frontier node already looks promising for the current target slot.
- For multiple_choice_visual_qa, use `global_context.clean_question` for search wording and
  `global_context.answer_options` as the only final answer choices.
- For multiple_choice_visual_qa, STOP answers must be exactly one valid option letter.
- If `event_memory` exists, treat it as the running timestamped event table. Prefer SEARCH/OPEN
  for missing events, and use localized event intervals plus relation edges before STOP.
- Treat `event` nodes as cognitive episodes: they should usually be opened before broad
  scenes/segments when their situation model matches the question.
- SEARCH should use cue-frame wording: include likely actors, objects, place, action, spoken
  topic, visible text, and temporal relation instead of only copying the question.
- For before/after/why questions, inspect temporal or causal neighbors before STOP when evidence
  does not already cover the event chain.
- For temporal multiple-choice questions, compare every option event against the target event
  intervals in `event_memory`.
- Respect question_spec preferred_modality: use visual for screen/sign/visible text, audio for
  sound/noise, and speech for spoken explanations.
- Respect `global_context.question_route`: for code_value_eval, terminal_output, assignment_count,
  operator_list, and ui_header_text questions, STOP only with evidence whose metadata kind matches
  that route. If `last_stop_verification` says a STOP was rejected, search or open the route's
  preferred modality instead of repeating the same answer.
- Use SPLIT when a cognitive event is still too broad or has internal boundary peaks; this may
  create real subevent nodes.
- Use MERGE when adjacent event evidence or evidence sharing actor/object/place/topic supports one claim.
- Use consolidated_memory/event_schema fields in evidence as durable situation-model facts, not as
  final answers by themselves.
- Do not OPEN the same node with the same modality and the same target_slot twice.
- If a slot has query hints or refinement candidates after a background-only open, use them before STOP.
- If required slots are already filled, prefer STOP.
- If evidence is only background, do not answer from it.
- Use STOP only when you can answer the user's question from core or support evidence and cite
  relevant evidence ids.
- The rationale is optional. Set "rationale": null unless it is truly necessary.
- If you include a rationale, it must be at most 12 words.
- Do not mention previous mistakes, reconsiderations, or internal deliberation.
- Keep STOP answers short and grounded.
- Output JSON only, with no markdown fences or extra commentary.
"""

LONGSHOT_CONTROLLER_PROMPT_SECTION = """LongShotBench instructions:
- Treat this as a LongShotBench post-validation task when `global_context.benchmark` is `longshotbench`.
- Use `global_context.longshot.expected_modalities` and `required_tools` as retrieval hints, not
  as tools to call directly.
- For screen/code/math questions, prefer OCR/visual evidence and open spans that show the exact
  code, label, expression, count, or shell output.
- For speech/translation questions, prefer speech evidence and preserve the exact spoken claim before paraphrasing.
- For audio-environment questions, prefer audio evidence and identify the relevant time window before answering.
- Use dialogue context to resolve follow-up references, but answer only the current user turn.
- STOP with a natural-language answer, not an option letter, unless explicit answer choices are
  present in the current question.
- Keep the final answer concise and directly grounded in evidence ids.
"""

TIMELOGIC_CONTROLLER_PROMPT_SECTION = """TimeLogic/TLQA instructions:
- Use `global_context.timelogic.operator_guide` to map the question template to its temporal
  operator before deciding whether more evidence is needed.
- STOP as soon as the matched operator has all required localized intervals and yields one
  grounded option letter; do not spend extra SEARCH/OPEN steps after symbolic verification is
  possible.
"""


def build_controller_prompt(
    state: ControllerState,
    max_frontier_items: int = 6,
    max_evidence_items: int = 6,
    max_action_history: int = 6,
) -> str:
    frontier = [item.to_dict() for item in state.frontier[:max_frontier_items]]
    recent_actions = [_compact_action_history_item(item) for item in state.action_history[-max_action_history:]]
    evidence_board = _compact_evidence_board(state)
    evidence = _compact_evidence_ledger(state, max_evidence_items)
    payload = {
        "question": state.question,
        "task_type": state.task_type,
        "dialogue_context": state.dialogue_context[-4:],
        "question_spec": state.question_spec.to_dict() if state.question_spec else None,
        "subquestion": state.subquestion,
        "frontier_top": frontier,
        "evidence_board": evidence_board,
        "event_memory": _compact_event_memory(state),
        "compact_evidence_ledger": evidence,
        "recent_action_history": recent_actions,
        "budget": state.budget.to_dict(),
        "global_context": state.global_context,
        "no_progress_steps": state.no_progress_steps,
    }
    prompt = VIDEO_RLM_CONTROLLER_PROMPT
    benchmark = state.global_context.get("benchmark")
    if benchmark == "longshotbench" or state.global_context.get("longshot") is not None:
        prompt += "\n\n" + LONGSHOT_CONTROLLER_PROMPT_SECTION
    if benchmark == "timelogic" or getattr(state.event_memory, "task_name", None) == "timelogic":
        prompt += "\n\n" + TIMELOGIC_CONTROLLER_PROMPT_SECTION
    return prompt + "\n\nCurrent state:\n" + json.dumps(
        payload, indent=2, ensure_ascii=True
    )


def _compact_action_history_item(item: dict) -> dict:
    return {
        "action_type": item.get("action_type"),
        "query": item.get("query"),
        "modality": item.get("modality"),
        "node_id": item.get("node_id"),
        "target_slot": item.get("target_slot"),
        "evidence_ids": list(item.get("evidence_ids", [])),
    }


def _compact_evidence_board(state: ControllerState) -> dict | None:
    if state.evidence_board is None:
        return None
    slots = {}
    for slot_name, slot in state.evidence_board.slots.items():
        slots[slot_name] = {
            "status": slot.status,
            "core_evidence_ids": slot.core_evidence_ids[:2],
            "support_evidence_ids": slot.support_evidence_ids[:2],
            "background_evidence_ids": slot.background_evidence_ids[:2],
        }
    return {
        "question_type": state.evidence_board.question_type,
        "slots": slots,
        "missing_required_slots": list(state.evidence_board.missing_required_slots),
        "query_hints_by_slot": {
            slot_name: hints[:3]
            for slot_name, hints in state.evidence_board.slot_query_hints.items()
            if hints
        },
        "refinement_node_ids_by_slot": {
            slot_name: node_ids[:4]
            for slot_name, node_ids in state.evidence_board.slot_refinement_node_ids.items()
            if node_ids
        },
        "opened_summary": [
            {
                "node_id": item.node_id,
                "modality": item.modality,
                "target_slot": item.target_slot,
                "result": item.result,
            }
            for item in state.evidence_board.opened_targets[-6:]
        ],
        "metrics": state.global_context.get("evidence_metrics", {}),
    }


def _compact_event_memory(state: ControllerState) -> dict | None:
    if state.event_memory is None:
        return None
    events = {}
    for event_id, event in state.event_memory.events.items():
        events[event_id] = {
            "phrase": event.phrase,
            "source": event.source,
            "option_letter": event.option_letter,
            "status": event.status,
            "intervals": [
                {
                    "time_span": interval.time_span.to_dict(),
                    "evidence_id": interval.evidence_id,
                    "confidence": interval.confidence,
                    "match_score": interval.match_score,
                }
                for interval in event.intervals[:3]
            ],
        }
    return {
        "task_name": state.event_memory.task_name,
        "mode": state.event_memory.mode,
        "events": events,
        "relations": state.event_memory.relations[:12],
        "localized_event_count": state.event_memory.localized_event_count,
        "missing_event_ids": state.event_memory.missing_event_ids,
    }


def _compact_evidence_ledger(state: ControllerState, max_evidence_items: int) -> list[dict]:
    items = []
    for item in state.evidence_ledger[:max_evidence_items]:
        cognitive_metadata = _compact_cognitive_metadata(item.metadata)
        items.append(
            {
                "evidence_id": item.evidence_id,
                "slot": item.metadata.get("slot"),
                "role": item.metadata.get("role"),
                "modality": item.modality,
                "claim": item.claim,
                "answer_span": item.metadata.get("answer_span"),
                "ocr_evidence_kind": item.metadata.get("ocr_evidence_kind"),
                "time_span": item.time_span.to_dict(),
                "source_node_id": item.source_node_id,
                "cognitive_metadata": cognitive_metadata,
            }
        )
    return items


def _compact_cognitive_metadata(metadata: dict) -> dict:
    keys = (
        "memory_consolidation",
        "consolidated_node_id",
        "consolidation_count",
        "cognitive_event_merge_applied",
        "merged_event_node_ids",
        "merged_event_adjacent",
        "shared_situation_index_labels",
        "stage2_window_reason",
    )
    compact = {key: metadata[key] for key in keys if key in metadata}
    event_schema = metadata.get("event_schema") or metadata.get("merged_event_schema")
    if isinstance(event_schema, dict):
        compact["event_schema"] = {
            key: event_schema.get(key)
            for key in (
                "actors",
                "objects",
                "actions",
                "place",
                "spoken_topics",
                "ocr_entities",
                "event_type",
            )
            if event_schema.get(key)
        }
    return compact
