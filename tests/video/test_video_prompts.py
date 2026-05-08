import json

from rlm.video.prompts import VIDEO_RLM_CONTROLLER_PROMPT, build_controller_prompt
from rlm.video.types import (
    BudgetState,
    ControllerState,
    EvidenceBoard,
    EvidenceBoardSlot,
    EvidenceSlotSpec,
    FrontierItem,
    QuestionSpec,
    TimeSpan,
)


def test_controller_prompt_instructs_model_to_keep_actions_short():
    assert "Do not explain your thinking." in VIDEO_RLM_CONTROLLER_PROMPT
    assert 'Set "rationale": null unless it is truly necessary.' in VIDEO_RLM_CONTROLLER_PROMPT
    assert "at most 12 words" in VIDEO_RLM_CONTROLLER_PROMPT
    assert "target_slot" in VIDEO_RLM_CONTROLLER_PROMPT
    assert "Do not OPEN the same node" in VIDEO_RLM_CONTROLLER_PROMPT


def test_build_controller_prompt_strips_verbose_action_history_fields():
    state = ControllerState(
        question="Why did they use quasars?",
        question_spec=QuestionSpec(
            question_type="why_reason",
            required_slots=[
                EvidenceSlotSpec(slot="reason", description="Why they used quasars")
            ],
            preferred_modality="speech",
        ),
        frontier=[
            FrontierItem(
                node_id="scene_001",
                time_span=TimeSpan(0.0, 30.0),
                level="scene",
                score=0.9,
                why_candidate="Matched speech terms quasar and experiment",
                recommended_modalities=["speech"],
            )
        ],
        action_history=[
            {
                "action_type": "OPEN",
                "query": None,
                "modality": "speech",
                "node_id": "scene_001",
                "target_slot": "reason",
                "evidence_ids": [],
                "answer": "A very long answer that should not appear in the prompt history.",
                "rationale": "A very long rationale that should also be removed from prompt history.",
            }
        ],
        budget=BudgetState(steps_used=1, steps_remaining=7, tool_calls_used=1),
        evidence_board=EvidenceBoard(
            question_type="why_reason",
            slots={
                "reason": EvidenceBoardSlot(
                    slot="reason",
                    description="Why they used quasars",
                    status="missing",
                )
            },
            missing_required_slots=["reason"],
        ),
    )

    prompt = build_controller_prompt(state)
    payload = json.loads(prompt.split("Current state:\n", 1)[1])
    recent = payload["recent_action_history"]

    assert len(recent) == 1
    assert recent[0]["action_type"] == "OPEN"
    assert recent[0]["node_id"] == "scene_001"
    assert recent[0]["target_slot"] == "reason"
    assert "rationale" not in recent[0]
    assert "answer" not in recent[0]
    assert "very long rationale" not in prompt
    assert "very long answer" not in prompt
    assert payload["question_spec"]["question_type"] == "why_reason"
    assert payload["evidence_board"]["missing_required_slots"] == ["reason"]
