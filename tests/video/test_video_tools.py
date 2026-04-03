from rlm.video.memory import PreparedVideoArtifacts, VideoMemoryBuilder
from rlm.video.tools import VideoToolExecutor
from rlm.video.types import ControllerState, SpeechSpan, TimeSpan


def build_memory_for_tools(spans: list[SpeechSpan]):
    artifacts = PreparedVideoArtifacts(
        video_id="tool_sample",
        duration_seconds=60.0,
        speech_spans=spans,
        metadata={"source_video_path": "tool_sample.mp4"},
    )
    return VideoMemoryBuilder(
        scene_duration_seconds=60.0,
        segment_duration_seconds=30.0,
        clip_duration_seconds=15.0,
    ).build_from_artifacts(artifacts)


def test_open_speech_selects_relevant_causal_spans():
    memory = build_memory_for_tools(
        [
            SpeechSpan(
                text="She talks about other rings and says they are beautiful.",
                time_span=TimeSpan(0.0, 10.0),
            ),
            SpeechSpan(
                text=(
                    "For my Cartier Clash bracelet the clasp was opening a lot, "
                    "and she was worried she would lose it."
                ),
                time_span=TimeSpan(10.0, 20.0),
            ),
            SpeechSpan(
                text="So she brought it back to Cartier and they fixed it quickly.",
                time_span=TimeSpan(20.0, 30.0),
            ),
            SpeechSpan(
                text="Later she talks about a necklace and how she styles it.",
                time_span=TimeSpan(30.0, 40.0),
            ),
        ]
    )
    executor = VideoToolExecutor(memory)
    state = ControllerState(
        question="Why did she say she only wears her Cartier Clash bracelet sometimes?",
        action_history=[
            {
                "action_type": "SEARCH",
                "query": "why she only wears Cartier Clash bracelet sometimes",
                "modality": "speech",
            }
        ],
    )

    observation = executor.open("tool_sample_scene_001", "speech", state)

    assert len(observation.evidence) == 2
    assert observation.evidence[0].time_span.start == 10.0
    assert observation.evidence[1].time_span.start == 20.0
    assert "clasp was opening a lot" in observation.evidence[0].detail
    assert "brought it back to Cartier" in observation.evidence[1].detail
    assert "other rings" not in observation.evidence[0].detail


def test_open_speech_first_query_prefers_early_relevant_spans():
    memory = build_memory_for_tools(
        [
            SpeechSpan(
                text="They begin at the market with a quick introduction to the stalls.",
                time_span=TimeSpan(0.0, 10.0),
            ),
            SpeechSpan(
                text=(
                    "The first thing that felt different from regular street food was the chicken head. "
                    "They say YRC sells over 10 pounds of chicken heads each day and coat it in flour and deep fry it."
                ),
                time_span=TimeSpan(10.0, 20.0),
            ),
            SpeechSpan(
                text=(
                    "When they bit into it, they said it tasted like fried goodness "
                    "and surprisingly tasted like chicken."
                ),
                time_span=TimeSpan(20.0, 30.0),
            ),
            SpeechSpan(
                text="Later they discuss pig head being cooked over charcoal.",
                time_span=TimeSpan(30.0, 40.0),
            ),
        ]
    )
    executor = VideoToolExecutor(memory)
    state = ControllerState(
        question=(
            "What was the first thing they tried that made them realize this was going "
            "to be different from regular street food?"
        ),
        action_history=[
            {
                "action_type": "SEARCH",
                "query": (
                    "first thing they tried that made them realize this was going "
                    "to be different from regular street food"
                ),
                "modality": "speech",
            }
        ],
    )

    observation = executor.open("tool_sample_scene_001", "speech", state)

    assert len(observation.evidence) == 2
    assert observation.evidence[0].time_span.start == 10.0
    assert observation.evidence[1].time_span.start == 20.0
    assert "chicken head" in observation.evidence[0].detail
    assert "10 pounds" in observation.evidence[0].detail
    assert "tasted like chicken" in observation.evidence[1].detail
    assert "begin at the market" not in observation.evidence[0].detail
    assert "pig head" not in observation.evidence[1].detail


def test_open_speech_keeps_short_selected_span_intact():
    memory = build_memory_for_tools(
        [
            SpeechSpan(
                text=(
                    "In this series, we're exploring Manila's street food scene. "
                    "Usually we see fish balls and barbecue. "
                    "But this stall brings out a chicken head and says they sell over 10 pounds each day. "
                    "They coat it in flour and deep fry it before serving."
                ),
                time_span=TimeSpan(0.0, 20.0),
            ),
            SpeechSpan(
                text=(
                    "They bite into it and say it mostly tastes like fried goodness. "
                    "Surprisingly, it tastes like chicken."
                ),
                time_span=TimeSpan(20.0, 30.0),
            ),
        ]
    )
    executor = VideoToolExecutor(memory)
    state = ControllerState(
        question=(
            "What was the first thing they tried that made them realize this was going "
            "to be different from regular street food?"
        ),
        action_history=[
            {
                "action_type": "SEARCH",
                "query": (
                    "first thing they tried that made them realize this was going "
                    "to be different from regular street food"
                ),
                "modality": "speech",
            }
        ],
    )

    observation = executor.open("tool_sample_scene_001", "speech", state)

    assert len(observation.evidence) == 2
    assert "chicken head" in observation.evidence[0].detail
    assert "10 pounds" in observation.evidence[0].detail
    assert "deep fry" in observation.evidence[0].detail
    assert "fried goodness" in observation.evidence[1].detail


def test_open_speech_skips_duplicate_existing_evidence():
    shared_span = SpeechSpan(
        text=(
            "The clasp was opening a lot on the Cartier Clash bracelet, so she worried she might lose it."
        ),
        time_span=TimeSpan(10.0, 20.0),
    )
    memory = build_memory_for_tools([shared_span])
    executor = VideoToolExecutor(memory)
    state = ControllerState(
        question="Why did she say she only wears her Cartier Clash bracelet sometimes?",
        evidence_ledger=[
            executor.open("tool_sample_scene_001", "speech", ControllerState(question="seed")).evidence[0]
        ],
        action_history=[
            {
                "action_type": "SEARCH",
                "query": "why she only wears Cartier Clash bracelet sometimes",
                "modality": "speech",
            }
        ],
    )

    observation = executor.open("tool_sample_scene_001", "speech", state)

    assert observation.evidence == []


def test_open_speech_why_query_prefers_causal_late_snippet_over_early_setup():
    memory = build_memory_for_tools(
        [
            SpeechSpan(
                text=(
                    "The one bracelet that I probably wore the most of all is my Cartier I Love "
                    "pave bracelet. This bracelet I've also done an unboxing for. Finally enough, "
                    "also stacks perfectly with the jewelry that I have. Look at this. Actually, wow, "
                    "it's quite heavy now. One other bracelet that actually made to my best and worst "
                    "purchases of two years ago is my Cartier Clash bracelet. Now that I look at it, "
                    "I think it's in pink gold, but Cartier pink gold is not very different from "
                    "Cartier's yellow gold. It's just like tiny little shade pink. Okay, so this is "
                    "yellow gold, this is pink gold, and for me, my Clash bracelet clasp was opening a "
                    "lot. It was opening a lot, so the bracelet kept opening, and I was really worried "
                    "that I will lose it because obviously I really love this bracelet and I don't want "
                    "to lose it. So I brought it back to Cartier. They fixed it in a very speedy time."
                ),
                time_span=TimeSpan(0.0, 60.0),
            )
        ]
    )
    executor = VideoToolExecutor(memory)
    state = ControllerState(
        question=(
            "Why did she say she only wears her Cartier Clash bracelet sometimes "
            "even though she loves it so much?"
        ),
        action_history=[
            {
                "action_type": "SEARCH",
                "query": "why she only wears Cartier Clash bracelet sometimes",
                "modality": "speech",
            }
        ],
    )

    observation = executor.open("tool_sample_scene_001", "speech", state)

    assert len(observation.evidence) == 1
    detail = observation.evidence[0].detail.lower()
    assert "clasp was opening a lot" in detail
    assert "worried" in detail
    assert "brought it back to cartier" in detail
    assert "they fixed it" in detail
    assert "quite heavy now" not in detail


def test_open_speech_why_query_prefers_relevant_continuation_over_topic_shift():
    memory = build_memory_for_tools(
        [
            SpeechSpan(
                text=(
                    "Very very speedy time. Yeah, now it's really perfect. I haven't really worn it "
                    "that much, but I love this bracelet. I think it's such a beautiful piece, and "
                    "it's like a bit more rare of a piece. It's very very special. Now I can open "
                    "the seater. What is wrong with me today? Now I'm stuck in this. Last but not the "
                    "least is the Cartier Pave Jean Sunclou white gold bracelet. It is in white gold "
                    "because this year I got more into sort of like a white metal."
                ),
                time_span=TimeSpan(0.0, 60.0),
            )
        ]
    )
    executor = VideoToolExecutor(memory)
    state = ControllerState(
        question=(
            "Why did she say she only wears her Cartier Clash bracelet sometimes "
            "even though she loves it so much?"
        ),
        action_history=[
            {
                "action_type": "SEARCH",
                "query": "why she only wears Cartier Clash bracelet sometimes",
                "modality": "speech",
            }
        ],
    )

    observation = executor.open("tool_sample_scene_001", "speech", state)

    assert len(observation.evidence) == 1
    detail = observation.evidence[0].detail.lower()
    assert "now it's really perfect" in detail
    assert "haven't really worn it that much" in detail
    assert "love this bracelet" in detail
    assert "last but not the least" not in detail


def test_open_speech_why_query_does_not_pull_unrelated_intro_span():
    memory = build_memory_for_tools(
        [
            SpeechSpan(
                text=(
                    "Check out Leon Diamond. If you are in New York, go see them. "
                    "Ask for Richie to show you the best pieces because that's how it works."
                ),
                time_span=TimeSpan(0.0, 10.0),
            ),
            SpeechSpan(
                text=(
                    "My Clash bracelet clasp was opening a lot, and I was worried that I would lose it. "
                    "So I brought it back to Cartier and they fixed it."
                ),
                time_span=TimeSpan(10.0, 20.0),
            ),
        ]
    )
    executor = VideoToolExecutor(memory)
    state = ControllerState(
        question=(
            "Why did she say she only wears her Cartier Clash bracelet sometimes "
            "even though she loves it so much?"
        ),
        action_history=[
            {
                "action_type": "SEARCH",
                "query": "why she only wears Cartier Clash bracelet sometimes",
                "modality": "speech",
            }
        ],
    )

    observation = executor.open("tool_sample_scene_001", "speech", state)

    assert len(observation.evidence) == 1
    assert observation.evidence[0].time_span.start == 10.0
    assert "leon diamond" not in observation.evidence[0].detail.lower()
