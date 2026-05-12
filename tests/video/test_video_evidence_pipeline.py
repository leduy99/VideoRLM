from rlm.video.evidence_pipeline import (
    build_evidence_board,
    build_question_spec,
    build_slot_queries,
    open_v2,
    search_v2,
    select_target_slot,
    update_evidence_board,
)
from rlm.video.index import SearchHit
from rlm.video.types import ControllerState, Evidence, Observation, TimeSpan


def test_build_question_spec_marks_reason_as_required_for_why_queries():
    spec = build_question_spec(
        "Why did she decide to wear the new diamond add-on right away instead of saving it?",
        task_type="causal_reasoning",
    )

    assert spec.question_type == "why_reason"
    assert spec.required_slots[-1].slot == "reason"
    assert spec.required_slots[-1].required is True
    assert spec.required_slots[0].required is False


def test_evidence_board_selects_first_missing_required_slot():
    spec = build_question_spec(
        "What was the first thing they tried that made them realize this was different?",
        task_type="event_understanding",
    )
    board = build_evidence_board(spec)

    assert select_target_slot(spec, board) == "first_thing_tried"


def test_build_slot_queries_includes_slot_specific_variants():
    spec = build_question_spec(
        "Why did she say she only wears her Cartier Clash bracelet sometimes?",
        task_type="entity_recognition",
    )

    queries = build_slot_queries(spec.metadata["question"], spec, "reason")

    assert queries
    assert any("reason" in query.lower() or "why" in query.lower() for query in queries)


def test_build_question_spec_prefers_visual_for_behavioral_why_question():
    spec = build_question_spec(
        "Why did the blue cat suddenly stop chasing the brown mouse and just stare at the doorway?",
        task_type="information_retrieval",
    )

    assert spec.preferred_modality == "visual"
    assert spec.required_slots[-1].preferred_modality == "visual"


def test_open_v2_demotes_reason_without_causal_signal():
    question = "Why did she say she only wears her Cartier Clash bracelet sometimes?"
    spec = build_question_spec(question, task_type="entity_recognition")
    state = ControllerState(
        question=question,
        task_type="entity_recognition",
        question_spec=spec,
        evidence_board=build_evidence_board(spec),
    )
    evidence = [
        Evidence(
            evidence_id="e1",
            claim="Speech evidence: Cartier love pave bracelet also stacks perfectly with the jewelry that I have.",
            modality="speech",
            time_span=TimeSpan(0.0, 5.0),
            source_node_id="scene_009",
            confidence=0.7,
            detail="Cartier love pave bracelet also stacks perfectly with the jewelry that I have.",
        )
    ]

    classified, metadata = open_v2(
        question_spec=spec,
        target_slot="reason",
        state=state,
        node_id="scene_009",
        modality="speech",
        evidence_items=evidence,
    )

    assert classified
    assert classified[0].metadata["role"] != "core"
    assert "reason" in metadata["missing_slots"]


def test_open_v2_demotes_generic_intro_for_first_thing_tried():
    question = "What was the first thing they tried that made them realize this was different?"
    spec = build_question_spec(question, task_type="event_understanding")
    state = ControllerState(
        question=question,
        task_type="event_understanding",
        question_spec=spec,
        evidence_board=build_evidence_board(spec),
    )
    evidence = [
        Evidence(
            evidence_id="e1",
            claim="Speech evidence: In this series we're exploring street food, but first what makes it special?",
            modality="speech",
            time_span=TimeSpan(0.0, 5.0),
            source_node_id="scene_001",
            confidence=0.9,
            detail="In this series we're exploring street food, but first what makes it special?",
        )
    ]

    classified, metadata = open_v2(
        question_spec=spec,
        target_slot="first_thing_tried",
        state=state,
        node_id="scene_001",
        modality="speech",
        evidence_items=evidence,
    )

    assert classified
    assert classified[0].metadata["role"] != "core"
    assert metadata["background_only"] is True


def test_update_evidence_board_persists_query_hints_and_refinement_nodes():
    question = "Why did she decide to wear the new diamond add-on right away instead of saving it?"
    spec = build_question_spec(question, task_type="causal_reasoning")
    board = build_evidence_board(spec)
    observation = Observation(
        kind="open",
        summary="Background-only open.",
        node_id="scene_006",
        metadata={
            "target_slot": "reason",
            "modality": "speech",
            "result": "background_only",
            "background_only": True,
            "duplicate_evidence_count": 0,
            "suggested_queries": [
                "why wear diamond add-on immediately",
                "save it for later reason diamond add-on",
            ],
            "refinement_node_ids": ["scene_006_seg_001", "scene_006_seg_002"],
        },
    )

    updated = update_evidence_board(board, spec, observation, step_index=1)

    assert updated is not None
    assert updated.slot_query_hints["reason"][0] == "why wear diamond add-on immediately"
    assert updated.slot_refinement_node_ids["reason"] == [
        "scene_006_seg_001",
        "scene_006_seg_002",
    ]


def test_search_v2_uses_slot_query_hints_before_generic_question():
    class RecordingIndex:
        def __init__(self):
            self.queries: list[str] = []

        def search(self, query, modality=None, top_k=5):
            self.queries.append(query)
            return [
                SearchHit(
                    node_id="scene_006_seg_001",
                    time_span=TimeSpan(0.0, 10.0),
                    level="segment",
                    score=0.8,
                    reason="Matched candidate",
                    modality=modality or "speech",
                    matched_terms=["reason"],
                    score_breakdown={"lexical": 0.8},
                )
            ]

    question = "Why did she decide to wear the new diamond add-on right away instead of saving it?"
    spec = build_question_spec(question, task_type="causal_reasoning")
    board = build_evidence_board(spec)
    board.slot_query_hints["reason"] = [
        "why wear diamond add-on immediately",
        "save it for later reason diamond add-on",
    ]
    state = ControllerState(
        question=question,
        task_type="causal_reasoning",
        question_spec=spec,
        evidence_board=board,
    )
    index = RecordingIndex()

    frontier, metadata = search_v2(
        index=index,
        question_spec=spec,
        target_slot="reason",
        state=state,
        top_k=3,
    )

    assert frontier
    assert metadata["queries"][0] == "why wear diamond add-on immediately"
    assert index.queries[0] == "why wear diamond add-on immediately"
