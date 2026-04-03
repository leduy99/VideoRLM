import json

from rlm.video import VideoRLM
from rlm.video.controller import _focus_evidence_detail
from rlm.video.memory import PreparedVideoArtifacts, VideoMemoryBuilder
from rlm.video.types import SpeechSpan, TimeSpan, VisualSummarySpan
from tests.mock_lm import MockLM


def build_memory():
    artifacts = PreparedVideoArtifacts(
        video_id="meeting",
        duration_seconds=90.0,
        speech_spans=[
            SpeechSpan(
                text="We are changing the plan after reviewing the numbers.",
                time_span=TimeSpan(5.0, 12.0),
            ),
            SpeechSpan(
                text="The team approves the updated schedule and everyone agrees.",
                time_span=TimeSpan(40.0, 55.0),
            ),
        ],
        visual_summaries=[
            VisualSummarySpan(
                summary="A slide shows the updated schedule and approval status.",
                time_span=TimeSpan(30.0, 60.0),
                granularity="scene",
                tags=["schedule", "approval"],
            )
        ],
        metadata={"source_video_path": "meeting.mp4"},
    )
    builder = VideoMemoryBuilder(
        scene_duration_seconds=30.0,
        segment_duration_seconds=15.0,
        clip_duration_seconds=5.0,
    )
    return builder.build_from_artifacts(artifacts)


def test_videorlm_controller_runs_end_to_end():
    memory = build_memory()
    responses = [
        json.dumps(
            {
                "action_type": "OPEN",
                "node_id": "meeting_scene_001",
                "modality": "speech",
                "evidence_ids": [],
                "query": None,
                "answer": None,
                "rationale": "Inspect the highest-priority speech evidence first.",
            }
        ),
        json.dumps(
            {
                "action_type": "STOP",
                "node_id": None,
                "modality": None,
                "evidence_ids": ["evidence_00001"],
                "query": None,
                "answer": "The plan changes early in the meeting when the speaker says they are changing it after reviewing the numbers.",
                "rationale": "The answer is directly supported by the opened speech evidence.",
            }
        ),
    ]
    model = MockLM(model_name="mock-controller", responses=responses)
    runner = VideoRLM(controller_client=model, max_steps=4, search_top_k=3, max_frontier_items=4)

    result = runner.run("When does the plan change?", memory, task_type="retrieval")

    assert "plan changes early in the meeting" in result.answer
    assert len(result.trace) == 2
    assert result.state.evidence_ledger
    assert result.state.evidence_ledger[0].used_in_final_answer is True
    assert all(item.node_id != "meeting_scene_001" for item in result.state.frontier)


def test_videorlm_controller_synthesizes_fallback_answer_from_evidence():
    memory = build_memory()
    responses = [
        json.dumps(
            {
                "action_type": "OPEN",
                "node_id": "meeting_scene_001",
                "modality": "speech",
                "evidence_ids": [],
                "query": None,
                "answer": None,
                "rationale": "Inspect the most relevant speech node.",
            }
        ),
        "The plan changes early in the meeting after reviewing the numbers.",
    ]
    model = MockLM(model_name="mock-controller", responses=responses)
    runner = VideoRLM(controller_client=model, max_steps=1, search_top_k=3, max_frontier_items=4)

    result = runner.run("When does the plan change?", memory, task_type="retrieval")

    assert result.answer == "The plan changes early in the meeting after reviewing the numbers."


def test_videorlm_controller_stops_after_repeated_empty_open_steps():
    memory = build_memory()
    responses = [
        json.dumps(
            {
                "action_type": "OPEN",
                "node_id": "meeting_scene_001",
                "modality": "speech",
                "evidence_ids": [],
                "query": None,
                "answer": None,
                "rationale": "Open the main speech node.",
            }
        ),
        json.dumps(
            {
                "action_type": "OPEN",
                "node_id": "meeting_scene_001",
                "modality": "speech",
                "evidence_ids": [],
                "query": None,
                "answer": None,
                "rationale": "Open it again.",
            }
        ),
        json.dumps(
            {
                "action_type": "OPEN",
                "node_id": "meeting_scene_001",
                "modality": "speech",
                "evidence_ids": [],
                "query": None,
                "answer": None,
                "rationale": "Open it again.",
            }
        ),
        "The plan changes early in the meeting after reviewing the numbers.",
    ]
    model = MockLM(model_name="mock-controller", responses=responses)
    runner = VideoRLM(controller_client=model, max_steps=6, search_top_k=3, max_frontier_items=4)

    result = runner.run("When does the plan change?", memory, task_type="retrieval")

    assert result.answer == "The plan changes early in the meeting after reviewing the numbers."
    assert len(result.trace) == 3


def test_focus_evidence_detail_prefers_relevant_sentences():
    detail = (
        "This bracelet looks beautiful and she loves it. "
        "For my Clash bracelet the clasp was opening a lot and she was worried she would lose it. "
        "So she brought it back to Cartier and they fixed it. "
        "After that it worked perfectly."
    )

    focused = _focus_evidence_detail(
        detail,
        "Why did she say she only wears her Cartier Clash bracelet sometimes even though she loves it so much?",
    )

    assert "clasp was opening a lot" in focused
    assert "brought it back to Cartier" in focused


def test_focus_evidence_detail_prefers_late_causal_window_in_long_asr_span():
    detail = (
        "She talks about several bracelets she loves and how they stack beautifully with other pieces. "
        "The Cartier I Love bracelet is the one she wore the most and she mentions it is quite heavy now. "
        "One other bracelet that made her best and worst purchases list is the Cartier Clash bracelet. "
        "For the Clash bracelet the clasp was opening a lot and the bracelet kept opening. "
        "She was really worried that she would lose it because she loved it so much. "
        "So she brought it back to Cartier and they fixed it very quickly. "
        "Now it works perfectly, but she still has not worn it as much as some of her other pieces."
    )

    focused = _focus_evidence_detail(
        detail,
        "Why did she say she only wears her Cartier Clash bracelet sometimes even though she loves it so much?",
        max_chars=260,
    )

    assert "clasp was opening a lot" in focused
    assert "worried that she would lose it" in focused
    assert "quite heavy now" not in focused


def test_focus_evidence_detail_prefers_early_window_for_first_question():
    detail = (
        "They begin with chicken skin and then move to chicken head, saying this is the chicken head and "
        "that YRC Chicken sells over 10 pounds of chicken heads each day. They coat it in flour and deep fry it. "
        "Later in the video they discuss pig head being cut down into manageable portions and cooked again over charcoal."
    )

    focused = _focus_evidence_detail(
        detail,
        "What was the first thing they tried that made them realize this was going to be different from regular street food?",
        max_chars=220,
    )

    assert "chicken head" in focused
    assert "10 pounds" in focused
    if "pig head" in focused:
        assert focused.index("chicken head") < focused.index("pig head")
