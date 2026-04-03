from rlm.video.types import (
    BudgetState,
    ControllerAction,
    ControllerState,
    Evidence,
    FrontierItem,
    TimeSpan,
)


def test_timespan_rejects_invalid_range():
    try:
        TimeSpan(5.0, 4.0)
    except ValueError as exc:
        assert "Invalid TimeSpan" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid time span")


def test_controller_action_validation():
    action = ControllerAction(action_type="SEARCH", query="plan", modality="speech")
    assert action.action_type == "SEARCH"


def test_controller_state_roundtrip():
    state = ControllerState(
        question="What changed?",
        frontier=[
            FrontierItem(
                node_id="scene_001",
                time_span=TimeSpan(0.0, 10.0),
                level="scene",
                score=0.9,
                why_candidate="matched speech",
                recommended_modalities=["speech"],
            )
        ],
        evidence_ledger=[
            Evidence(
                evidence_id="e1",
                claim="Speech evidence: the plan changed",
                modality="speech",
                time_span=TimeSpan(1.0, 2.0),
                source_node_id="scene_001",
                confidence=0.8,
            )
        ],
        budget=BudgetState(steps_used=1, steps_remaining=7),
    )

    restored = ControllerState.from_dict(state.to_dict())
    assert restored.question == "What changed?"
    assert restored.frontier[0].node_id == "scene_001"
    assert restored.evidence_ledger[0].evidence_id == "e1"
    assert restored.budget.steps_remaining == 7
