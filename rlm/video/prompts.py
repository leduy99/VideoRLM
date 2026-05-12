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
- Prefer SEARCH when frontier is weak or empty, or when a required slot is still missing.
- Prefer OPEN when a frontier node already looks promising for the current target slot.
- Respect question_spec preferred_modality: use visual for screen/sign/visible text, audio for sound/noise, and speech for spoken explanations.
- Use SPLIT when a node is still too broad.
- Use MERGE when multiple evidence items already support one claim.
- Do not OPEN the same node with the same modality and the same target_slot twice.
- If required slots are already filled, prefer STOP.
- If evidence is only background, do not answer from it.
- Use STOP only when you can answer the user's question from core or support evidence and cite relevant evidence ids.
- The rationale is optional. Set "rationale": null unless it is truly necessary.
- If you include a rationale, it must be at most 12 words.
- Do not mention previous mistakes, reconsiderations, or internal deliberation.
- Keep STOP answers short and grounded.
- Output JSON only, with no markdown fences or extra commentary.
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
        "compact_evidence_ledger": evidence,
        "recent_action_history": recent_actions,
        "budget": state.budget.to_dict(),
        "global_context": state.global_context,
        "no_progress_steps": state.no_progress_steps,
    }
    return VIDEO_RLM_CONTROLLER_PROMPT + "\n\nCurrent state:\n" + json.dumps(
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


def _compact_evidence_ledger(state: ControllerState, max_evidence_items: int) -> list[dict]:
    items = []
    for item in state.evidence_ledger[:max_evidence_items]:
        items.append(
            {
                "evidence_id": item.evidence_id,
                "slot": item.metadata.get("slot"),
                "role": item.metadata.get("role"),
                "claim": item.claim,
                "time_span": item.time_span.to_dict(),
            }
        )
    return items
