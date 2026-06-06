from rlm.video import QwenVideoStackConfig


def test_qwen_stack_bundle_builds_expected_components():
    config = QwenVideoStackConfig.from_shared_endpoint(
        base_url="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
        controller_model="Qwen3-8B",
        visual_model="Qwen3-VL-8B",
        speech_model="Qwen3-ASR-0.6B",
        embedding_model="text-embedding-3-small",
    )

    bundle = config.build_bundle(max_steps=6, search_top_k=4, max_frontier_items=5)

    assert bundle.controller.controller_backend_kwargs["model_name"] == "Qwen3-8B"
    assert bundle.speech_recognizer.model_name == "Qwen3-ASR-0.6B"
    assert bundle.visual_summarizer.model_name == "Qwen3-VL-8B"
    assert bundle.embedding_provider is not None


def test_qwen_stack_wires_pitome_visual_preprocessing():
    config = QwenVideoStackConfig.from_shared_endpoint(
        base_url="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
    )
    config.use_pitome = True
    config.clip_duration_seconds = 60.0
    config.pitome_dense_frame_rate = 2.0
    config.pitome_min_frame_count = 8
    config.pitome_max_selected_frames = 6

    bundle = config.build_bundle()

    assert bundle.visual_summarizer.summary_granularity == "clip"
    assert bundle.visual_summarizer.pitome_dense_frame_rate == 2.0
    assert bundle.visual_summarizer.pitome_min_frame_count == 8
    assert bundle.visual_summarizer.pitome_max_selected_frames == 6
    assert bundle.visual_refiner is not None
    assert bundle.visual_refiner.use_pitome is True
    assert bundle.memory_builder.visual_span_mode == "clip"
    assert bundle.memory_builder.aggregate_child_visual_summaries is True
