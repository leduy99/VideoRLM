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
