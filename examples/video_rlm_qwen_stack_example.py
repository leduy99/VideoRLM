"""
Example wiring for the Qwen-based VideoRLM stack.

This shows how to instantiate the controller + memory builder bundle when your
Qwen models are served through an OpenAI-compatible endpoint.
"""

from rlm.video import QwenVideoStackConfig

stack = QwenVideoStackConfig.from_shared_endpoint(
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
    controller_model="Qwen3-8B",
    visual_model="Qwen3-VL-8B",
    speech_model="Qwen3-ASR-0.6B",
    embedding_model=None,
)

bundle = stack.build_bundle(max_steps=8, search_top_k=5)

print("Controller model:", bundle.controller.controller_client.model_name)
print("Speech model:", bundle.speech_recognizer.model_name)
print("Vision model:", bundle.visual_summarizer.model_name)
