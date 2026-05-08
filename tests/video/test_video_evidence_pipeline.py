from rlm.video.evidence_pipeline import (
    build_evidence_board,
    build_question_spec,
    build_slot_queries,
    select_target_slot,
)


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
