import json

from rlm.video.types import ControllerState

VIDEO_RLM_CONTROLLER_PROMPT = """You are the controller for a long-video reasoning system.
You never read the full video directly. Instead, you inspect a structured state and choose exactly one next action.

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
  "evidence_ids": string[],
  "answer": string | null,
  "rationale": string | null
}

Rules:
- Prefer SEARCH when frontier is weak or empty.
- Prefer OPEN when a frontier node already looks promising.
- Use SPLIT when a node is still too broad.
- Use MERGE when multiple evidence items already support one claim.
- Use STOP only when you can answer the user's question and cite relevant evidence ids.
- Output JSON only, with no markdown fences or extra commentary.
"""


def build_controller_prompt(
    state: ControllerState,
    max_frontier_items: int = 6,
    max_evidence_items: int = 6,
    max_action_history: int = 6,
) -> str:
    frontier = [item.to_dict() for item in state.frontier[:max_frontier_items]]
    evidence = [item.to_dict() for item in state.evidence_ledger[:max_evidence_items]]
    recent_actions = state.action_history[-max_action_history:]
    payload = {
        "question": state.question,
        "task_type": state.task_type,
        "dialogue_context": state.dialogue_context[-4:],
        "subquestion": state.subquestion,
        "frontier": frontier,
        "evidence_ledger": evidence,
        "recent_action_history": recent_actions,
        "budget": state.budget.to_dict(),
        "global_context": state.global_context,
    }
    return VIDEO_RLM_CONTROLLER_PROMPT + "\n\nCurrent state:\n" + json.dumps(
        payload, indent=2, ensure_ascii=True
    )
