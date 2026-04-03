from rlm.video import CallableEmbeddingProvider
from rlm.video.index import VideoMemoryIndex
from rlm.video.memory import PreparedVideoArtifacts, VideoMemoryBuilder
from rlm.video.types import OCRSpan, SpeechSpan, TimeSpan, VisualSummarySpan


def build_sample_memory():
    artifacts = PreparedVideoArtifacts(
        video_id="sample",
        duration_seconds=120.0,
        speech_spans=[
            SpeechSpan(
                text="We should revise the project plan after the demo.",
                time_span=TimeSpan(10.0, 20.0),
            ),
            SpeechSpan(
                text="The team confirms the new launch timeline.",
                time_span=TimeSpan(70.0, 80.0),
            ),
        ],
        visual_summaries=[
            VisualSummarySpan(
                summary="A presenter stands beside a slide titled project plan.",
                time_span=TimeSpan(0.0, 60.0),
                granularity="scene",
                tags=["slide", "plan"],
            )
        ],
        ocr_spans=[
            OCRSpan(text="PROJECT PLAN", time_span=TimeSpan(12.0, 18.0)),
            OCRSpan(text="Launch timeline", time_span=TimeSpan(71.0, 79.0)),
        ],
        metadata={"source_video_path": "sample.mp4"},
    )
    builder = VideoMemoryBuilder(
        scene_duration_seconds=60.0,
        segment_duration_seconds=30.0,
        clip_duration_seconds=10.0,
    )
    return builder.build_from_artifacts(artifacts)


def test_memory_builder_creates_hierarchy():
    memory = build_sample_memory()
    root = memory.get_node(memory.root_id)
    assert root.level == "video"
    assert len(root.children) == 2
    first_scene = memory.get_node(root.children[0])
    assert first_scene.level == "scene"
    assert first_scene.visual_summary


def test_lexical_index_returns_relevant_nodes():
    memory = build_sample_memory()
    index = VideoMemoryIndex(memory)
    hits = index.search("project plan", modality="speech", top_k=3)
    assert hits
    assert hits[0].modality == "speech"
    assert "Matched speech terms" in hits[0].reason


def test_hybrid_index_can_retrieve_semantic_match_without_lexical_overlap():
    memory = build_sample_memory()

    def embed_text(text: str) -> list[float]:
        lowered = text.lower()
        if any(word in lowered for word in ["roadmap", "plan", "project"]):
            return [1.0, 0.0]
        if any(word in lowered for word in ["schedule", "timeline"]):
            return [0.0, 1.0]
        return [0.0, 0.0]

    index = VideoMemoryIndex(
        memory,
        embedding_provider=CallableEmbeddingProvider(embed_text),
        lexical_weight=0.4,
        semantic_weight=0.6,
    )
    hits = index.search("roadmap", modality="visual", top_k=3)

    assert hits
    assert hits[0].score_breakdown["semantic"] > 0
    assert "semantic similarity" in hits[0].reason


def test_index_tokenizer_ignores_stopwords():
    memory = build_sample_memory()
    index = VideoMemoryIndex(memory)

    tokens = index._tokenize(
        "Why did she say she only wears her Cartier Clash bracelet sometimes even though she loves it so much?"
    )

    assert "cartier" in tokens
    assert "clash" in tokens
    assert "bracelet" in tokens
    assert "did" not in tokens
    assert "she" not in tokens
    assert "it" not in tokens


def test_index_first_query_prefers_earlier_match():
    artifacts = PreparedVideoArtifacts(
        video_id="temporal",
        duration_seconds=300.0,
        speech_spans=[
            SpeechSpan(
                text="They try a fried chicken head early in the market tour.",
                time_span=TimeSpan(10.0, 25.0),
            ),
            SpeechSpan(
                text="Later they try another unusual fried dish.",
                time_span=TimeSpan(210.0, 225.0),
            ),
        ],
        metadata={"source_video_path": "temporal.mp4"},
    )
    memory = VideoMemoryBuilder(
        scene_duration_seconds=150.0,
        segment_duration_seconds=75.0,
        clip_duration_seconds=25.0,
    ).build_from_artifacts(artifacts)
    index = VideoMemoryIndex(memory)

    hits = index.search(
        "What was the first fried thing they tried?",
        modality="speech",
        top_k=3,
    )

    assert hits
    assert hits[0].time_span.start < 100.0
    assert hits[0].score_breakdown["temporal"] > 0
