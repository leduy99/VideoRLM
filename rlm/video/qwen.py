from dataclasses import dataclass
from typing import Any

from rlm.clients.transformers_local import TransformersClient
from rlm.video.adapters import (
    EmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
    OpenAICompatibleSpeechRecognizer,
    OpenAICompatibleVisualSummarizer,
    SpeechRecognizer,
    VisualSummarizer,
)
from rlm.video.controller import VideoRLM
from rlm.video.huggingface import default_local_model_dir, download_snapshot
from rlm.video.local_adapters import LocalQwenASRSpeechRecognizer, LocalQwenVisualSummarizer
from rlm.video.logger import VideoRLMLogger
from rlm.video.memory import VideoMemoryBuilder


@dataclass
class OpenAICompatibleModelConfig:
    model_name: str
    base_url: str | None = None
    api_key: str | None = None
    timeout: float = 300.0
    extra_client_kwargs: dict[str, Any] | None = None

    def to_client_kwargs(self) -> dict[str, Any]:
        kwargs = {
            "model_name": self.model_name,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "timeout": self.timeout,
        }
        if self.extra_client_kwargs:
            kwargs.update(self.extra_client_kwargs)
        return kwargs


@dataclass
class LocalModelConfig:
    model_name: str
    model_path: str | None = None
    device: str = "cuda:0"
    device_map: str | dict[str, Any] | None = None
    torch_dtype: str = "bfloat16"
    timeout: float = 300.0
    trust_remote_code: bool = False
    attn_implementation: str | None = None
    max_new_tokens: int = 512
    tokenizer_kwargs: dict[str, Any] | None = None
    model_kwargs: dict[str, Any] | None = None

    def resolved_model_path(self) -> str:
        return self.model_path or self.model_name

    def download(self) -> str:
        path = download_snapshot(self.model_name, local_dir=self.model_path)
        self.model_path = str(path)
        return self.model_path


@dataclass
class QwenVideoRuntimeBundle:
    controller: VideoRLM
    memory_builder: VideoMemoryBuilder
    speech_recognizer: SpeechRecognizer
    visual_summarizer: VisualSummarizer
    embedding_provider: EmbeddingProvider | None = None


@dataclass
class QwenVideoStackConfig:
    controller: OpenAICompatibleModelConfig
    visual: OpenAICompatibleModelConfig
    speech: OpenAICompatibleModelConfig
    embedding: OpenAICompatibleModelConfig | None = None
    ffmpeg_bin: str = "ffmpeg"
    frame_count: int = 3
    frame_width: int | None = 768
    scene_threshold_seconds: float = 20.0
    scene_duration_seconds: float = 180.0
    segment_duration_seconds: float = 45.0
    clip_duration_seconds: float = 15.0

    @classmethod
    def from_shared_endpoint(
        cls,
        base_url: str,
        api_key: str | None = None,
        controller_model: str = "Qwen3-8B",
        visual_model: str = "Qwen3-VL-8B",
        speech_model: str = "Qwen3-ASR-0.6B",
        embedding_model: str | None = None,
        timeout: float = 300.0,
    ) -> "QwenVideoStackConfig":
        controller = OpenAICompatibleModelConfig(
            model_name=controller_model,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
        visual = OpenAICompatibleModelConfig(
            model_name=visual_model,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
        speech = OpenAICompatibleModelConfig(
            model_name=speech_model,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
        embedding = None
        if embedding_model is not None:
            embedding = OpenAICompatibleModelConfig(
                model_name=embedding_model,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
            )
        return cls(
            controller=controller,
            visual=visual,
            speech=speech,
            embedding=embedding,
        )

    def build_bundle(
        self,
        *,
        logger: VideoRLMLogger | None = None,
        max_steps: int = 8,
        search_top_k: int = 5,
        max_frontier_items: int = 8,
        enable_hybrid_speech_refinement: bool = False,
        speech_refine_candidate_count: int = 4,
    ) -> QwenVideoRuntimeBundle:
        speech_recognizer = OpenAICompatibleSpeechRecognizer(
            model_name=self.speech.model_name,
            api_key=self.speech.api_key,
            base_url=self.speech.base_url,
            timeout=self.speech.timeout,
            ffmpeg_bin=self.ffmpeg_bin,
        )
        visual_summarizer = OpenAICompatibleVisualSummarizer(
            model_name=self.visual.model_name,
            api_key=self.visual.api_key,
            base_url=self.visual.base_url,
            timeout=self.visual.timeout,
            ffmpeg_bin=self.ffmpeg_bin,
            frame_count=self.frame_count,
            frame_width=self.frame_width,
            scene_threshold_seconds=self.scene_threshold_seconds,
        )
        embedding_provider = None
        if self.embedding is not None:
            embedding_provider = OpenAICompatibleEmbeddingProvider(
                model_name=self.embedding.model_name,
                api_key=self.embedding.api_key,
                base_url=self.embedding.base_url,
                timeout=self.embedding.timeout,
            )

        memory_builder = VideoMemoryBuilder(
            speech_recognizer=speech_recognizer,
            visual_summarizer=visual_summarizer,
            scene_duration_seconds=self.scene_duration_seconds,
            segment_duration_seconds=self.segment_duration_seconds,
            clip_duration_seconds=self.clip_duration_seconds,
        )
        controller = VideoRLM(
            controller_backend="openai",
            controller_backend_kwargs=self.controller.to_client_kwargs(),
            logger=logger,
            max_steps=max_steps,
            search_top_k=search_top_k,
            max_frontier_items=max_frontier_items,
            enable_hybrid_speech_refinement=enable_hybrid_speech_refinement,
            speech_refine_candidate_count=speech_refine_candidate_count,
        )
        return QwenVideoRuntimeBundle(
            controller=controller,
            memory_builder=memory_builder,
            speech_recognizer=speech_recognizer,
            visual_summarizer=visual_summarizer,
            embedding_provider=embedding_provider,
        )


@dataclass
class QwenLocalVideoStackConfig:
    controller: LocalModelConfig
    visual: LocalModelConfig
    speech: LocalModelConfig
    forced_aligner: LocalModelConfig | None = None
    ffmpeg_bin: str = "ffmpeg"
    frame_count: int = 3
    frame_width: int | None = 768
    scene_threshold_seconds: float = 20.0
    scene_duration_seconds: float = 180.0
    segment_duration_seconds: float = 45.0
    clip_duration_seconds: float = 15.0
    controller_enable_thinking: bool = False

    @classmethod
    def default(
        cls,
        *,
        controller_device: str = "cuda:0",
        visual_device: str = "cuda:1",
        speech_device: str = "cuda:2",
        controller_model: str = "Qwen/Qwen3-8B",
        visual_model: str = "Qwen/Qwen3-VL-8B-Instruct",
        speech_model: str = "Qwen/Qwen3-ASR-0.6B",
        forced_aligner_model: str | None = "Qwen/Qwen3-ForcedAligner-0.6B",
        torch_dtype: str = "bfloat16",
        attn_implementation: str | None = None,
    ) -> "QwenLocalVideoStackConfig":
        controller = LocalModelConfig(
            model_name=controller_model,
            model_path=str(default_local_model_dir(controller_model)),
            device=controller_device,
            device_map=controller_device,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            max_new_tokens=256,
        )
        visual = LocalModelConfig(
            model_name=visual_model,
            model_path=str(default_local_model_dir(visual_model)),
            device=visual_device,
            device_map=visual_device,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            max_new_tokens=160,
        )
        speech = LocalModelConfig(
            model_name=speech_model,
            model_path=str(default_local_model_dir(speech_model)),
            device=speech_device,
            device_map=speech_device,
            torch_dtype=torch_dtype,
            max_new_tokens=512,
        )
        forced_aligner = None
        if forced_aligner_model is not None:
            forced_aligner = LocalModelConfig(
                model_name=forced_aligner_model,
                model_path=str(default_local_model_dir(forced_aligner_model)),
                device=speech_device,
                device_map=speech_device,
                torch_dtype=torch_dtype,
            )
        return cls(
            controller=controller,
            visual=visual,
            speech=speech,
            forced_aligner=forced_aligner,
        )

    def download_models(self) -> dict[str, str]:
        downloads = {
            "controller": self.controller.download(),
            "visual": self.visual.download(),
            "speech": self.speech.download(),
        }
        if self.forced_aligner is not None:
            downloads["forced_aligner"] = self.forced_aligner.download()
        return downloads

    def build_bundle(
        self,
        *,
        logger: VideoRLMLogger | None = None,
        max_steps: int = 8,
        search_top_k: int = 5,
        max_frontier_items: int = 8,
        enable_hybrid_speech_refinement: bool = False,
        speech_refine_candidate_count: int = 4,
    ) -> QwenVideoRuntimeBundle:
        controller_client = TransformersClient(
            model_name=self.controller.model_name,
            model_path=self.controller.resolved_model_path(),
            device=self.controller.device,
            device_map=self.controller.device_map,
            torch_dtype=self.controller.torch_dtype,
            trust_remote_code=self.controller.trust_remote_code,
            attn_implementation=self.controller.attn_implementation,
            enable_thinking=self.controller_enable_thinking,
            max_new_tokens=self.controller.max_new_tokens,
            timeout=self.controller.timeout,
            tokenizer_kwargs=self.controller.tokenizer_kwargs,
            model_kwargs=self.controller.model_kwargs,
        )
        speech_recognizer = LocalQwenASRSpeechRecognizer(
            model_name=self.speech.model_name,
            model_path=self.speech.resolved_model_path(),
            forced_aligner_name=self.forced_aligner.model_name if self.forced_aligner else None,
            forced_aligner_path=(
                self.forced_aligner.resolved_model_path() if self.forced_aligner else None
            ),
            device_map=str(self.speech.device_map or self.speech.device),
            torch_dtype=self.speech.torch_dtype,
            ffmpeg_bin=self.ffmpeg_bin,
            max_new_tokens=self.speech.max_new_tokens,
        )
        visual_summarizer = LocalQwenVisualSummarizer(
            model_name=self.visual.model_name,
            model_path=self.visual.resolved_model_path(),
            device=self.visual.device,
            device_map=self.visual.device_map,
            torch_dtype=self.visual.torch_dtype,
            attn_implementation=self.visual.attn_implementation,
            frame_count=self.frame_count,
            ffmpeg_bin=self.ffmpeg_bin,
            frame_width=self.frame_width,
            scene_threshold_seconds=self.scene_threshold_seconds,
            max_new_tokens=self.visual.max_new_tokens,
        )
        memory_builder = VideoMemoryBuilder(
            speech_recognizer=speech_recognizer,
            visual_summarizer=visual_summarizer,
            scene_duration_seconds=self.scene_duration_seconds,
            segment_duration_seconds=self.segment_duration_seconds,
            clip_duration_seconds=self.clip_duration_seconds,
        )
        controller = VideoRLM(
            controller_client=controller_client,
            logger=logger,
            max_steps=max_steps,
            search_top_k=search_top_k,
            max_frontier_items=max_frontier_items,
            enable_hybrid_speech_refinement=enable_hybrid_speech_refinement,
            speech_refine_candidate_count=speech_refine_candidate_count,
        )
        return QwenVideoRuntimeBundle(
            controller=controller,
            memory_builder=memory_builder,
            speech_recognizer=speech_recognizer,
            visual_summarizer=visual_summarizer,
            embedding_provider=None,
        )
