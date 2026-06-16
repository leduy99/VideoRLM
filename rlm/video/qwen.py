from dataclasses import dataclass
from typing import Any, Literal

from rlm.clients.transformers_local import TransformersClient
from rlm.video.adapters import (
    EmbeddingProvider,
    ImageTextEmbeddingProvider,
    OCRExtractor,
    OpenAICompatibleEmbeddingProvider,
    OpenAICompatibleSpeechRecognizer,
    OpenAICompatibleVisualSummarizer,
    SpeechRecognizer,
    VideoWindowEmbeddingProvider,
    VisualSummarizer,
)
from rlm.video.controller import VideoRLM
from rlm.video.huggingface import default_local_model_dir, download_snapshot
from rlm.video.local_adapters import (
    FasterWhisperSpeechRecognizer,
    LazyPiToMeVisualIndexer,
    LazySpeechRecognizer,
    LocalQwenASRSpeechRecognizer,
    LocalQwenVisualSummarizer,
    PaddleOCRTextExtractor,
)
from rlm.video.logger import VideoRLMLogger
from rlm.video.memory import VideoMemoryBuilder
from rlm.video.semantic_embeddings import (
    LocalImageTextEmbeddingProvider,
    LocalInternVideoWindowEmbeddingProvider,
    LocalSentenceTransformerEmbeddingProvider,
)


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
    speech_recognizer: SpeechRecognizer | None
    visual_summarizer: VisualSummarizer
    speech_refiner: SpeechRecognizer | None = None
    visual_refiner: VisualSummarizer | None = None
    ocr_extractor: OCRExtractor | None = None
    embedding_provider: EmbeddingProvider | None = None
    speech_embedding_provider: EmbeddingProvider | None = None
    image_text_embedding_provider: ImageTextEmbeddingProvider | None = None
    video_window_embedding_provider: VideoWindowEmbeddingProvider | None = None


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
    enable_fine_speech_windows: bool = False
    fine_speech_window_seconds: float = 15.0
    fine_speech_window_stride_seconds: float = 5.0
    use_pitome: bool = False
    pitome_dense_frame_rate: float = 1.0
    pitome_min_frame_count: int | None = None
    pitome_protect_ratio: float = 0.15
    pitome_similarity_threshold: float = 0.8
    pitome_embedding_size: int = 16
    pitome_embedding_backend: str = "pixel"
    pitome_embedding_device: str | None = None
    pitome_frame_width: int | None = None
    pitome_frame_extraction_strategy: Literal["auto", "batch", "seek", "sequence"] = "auto"
    pitome_frame_extraction_workers: int = 1
    pitome_anchor_frame_count: int = 0
    pitome_max_selected_frames: int | None = None
    pitome_scene_threshold: float = 0.35
    pitome_max_scene_boundary_frames: int = 6
    pitome_scene_sample_rate: float | None = 1.0
    pitome_scene_keyframes_only: bool = True
    pitome_edge_boundary_frames: bool = True
    visual_index_batch_size: int = 1
    visual_index_workers: int = 1
    vl_max_input_frames: int | None = None
    parent_visual_summary_mode: Literal["full", "compact"] | None = None
    search_mode: Literal["lexical", "graph"] | None = None
    enable_vrrqa_graph_refinement_expansion: bool = True
    vrrqa_graph_refinement_neighbor_count: int = 1
    enable_vrrqa_visual_answer_verifier: bool = True
    vrrqa_visual_verifier_frame_count: int = 8
    enable_refinement_frontier: bool = True
    enable_dynamic_evidence_retrieval: bool = True
    disable_question_routing: bool = True
    visual_image_url_format: Literal["object", "string"] = "object"
    verbose: bool = False

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
        lazy_pitome_mode = self.use_pitome
        full_speech_recognizer = OpenAICompatibleSpeechRecognizer(
            model_name=self.speech.model_name,
            api_key=self.speech.api_key,
            base_url=self.speech.base_url,
            timeout=self.speech.timeout,
            ffmpeg_bin=self.ffmpeg_bin,
        )
        speech_recognizer: SpeechRecognizer = full_speech_recognizer
        speech_refiner: SpeechRecognizer | None = None
        if lazy_pitome_mode:
            speech_recognizer = LazySpeechRecognizer(verbose=self.verbose)
            speech_refiner = full_speech_recognizer

        openai_visual_summarizer = OpenAICompatibleVisualSummarizer(
            model_name=self.visual.model_name,
            api_key=self.visual.api_key,
            base_url=self.visual.base_url,
            image_url_format=self.visual_image_url_format,
            timeout=self.visual.timeout,
            ffmpeg_bin=self.ffmpeg_bin,
            frame_count=self.frame_count,
            frame_width=self.frame_width,
            scene_threshold_seconds=self.scene_threshold_seconds,
            use_pitome=lazy_pitome_mode,
            pitome_dense_frame_rate=self.pitome_dense_frame_rate,
            pitome_min_frame_count=self.pitome_min_frame_count,
            pitome_protect_ratio=self.pitome_protect_ratio,
            pitome_similarity_threshold=self.pitome_similarity_threshold,
            pitome_embedding_size=self.pitome_embedding_size,
            pitome_embedding_backend=self.pitome_embedding_backend,
            pitome_embedding_device=self.pitome_embedding_device,
            pitome_frame_width=self.pitome_frame_width,
            pitome_frame_extraction_strategy=self.pitome_frame_extraction_strategy,
            pitome_frame_extraction_workers=self.pitome_frame_extraction_workers,
            pitome_anchor_frame_count=self.pitome_anchor_frame_count,
            pitome_max_selected_frames=self.pitome_max_selected_frames,
            summary_granularity="clip" if lazy_pitome_mode else None,
            vl_max_input_frames=self.vl_max_input_frames,
        )
        visual_summarizer: VisualSummarizer = openai_visual_summarizer
        visual_refiner: VisualSummarizer | None = None
        if lazy_pitome_mode:
            visual_summarizer = LazyPiToMeVisualIndexer(
                ffmpeg_bin=self.ffmpeg_bin,
                frame_width=self.frame_width,
                frame_count=self.frame_count,
                pitome_dense_frame_rate=self.pitome_dense_frame_rate,
                pitome_min_frame_count=self.pitome_min_frame_count,
                pitome_protect_ratio=self.pitome_protect_ratio,
                pitome_similarity_threshold=self.pitome_similarity_threshold,
                pitome_embedding_size=self.pitome_embedding_size,
                pitome_embedding_backend=self.pitome_embedding_backend,
                pitome_embedding_device=self.pitome_embedding_device,
                pitome_frame_width=self.pitome_frame_width,
                pitome_frame_extraction_strategy=self.pitome_frame_extraction_strategy,
                pitome_frame_extraction_workers=self.pitome_frame_extraction_workers,
                pitome_anchor_frame_count=self.pitome_anchor_frame_count,
                pitome_max_selected_frames=self.pitome_max_selected_frames,
                pitome_scene_threshold=self.pitome_scene_threshold,
                pitome_max_scene_boundary_frames=self.pitome_max_scene_boundary_frames,
                pitome_scene_sample_rate=self.pitome_scene_sample_rate,
                pitome_scene_keyframes_only=self.pitome_scene_keyframes_only,
                pitome_edge_boundary_frames=self.pitome_edge_boundary_frames,
                visual_index_batch_size=self.visual_index_batch_size,
                visual_index_workers=self.visual_index_workers,
                summary_granularity="clip",
                verbose=self.verbose,
            )
            visual_refiner = openai_visual_summarizer
        elif self.enable_vrrqa_visual_answer_verifier:
            visual_refiner = openai_visual_summarizer
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
            enable_fine_speech_windows=self.enable_fine_speech_windows,
            fine_speech_window_seconds=self.fine_speech_window_seconds,
            fine_speech_window_stride_seconds=self.fine_speech_window_stride_seconds,
            visual_span_mode="clip" if lazy_pitome_mode else "scene_and_clip",
            aggregate_child_visual_summaries=lazy_pitome_mode,
            parent_visual_summary_mode=self.parent_visual_summary_mode
            or ("compact" if lazy_pitome_mode else "full"),
            verbose=self.verbose,
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
            search_mode=self.search_mode or ("graph" if lazy_pitome_mode else "lexical"),
            embedding_provider=embedding_provider,
            speech_refiner=speech_refiner,
            visual_refiner=visual_refiner,
            enable_vrrqa_graph_refinement_expansion=(
                self.enable_vrrqa_graph_refinement_expansion
            ),
            vrrqa_graph_refinement_neighbor_count=self.vrrqa_graph_refinement_neighbor_count,
            enable_vrrqa_visual_answer_verifier=self.enable_vrrqa_visual_answer_verifier,
            vrrqa_visual_verifier_frame_count=self.vrrqa_visual_verifier_frame_count,
            enable_refinement_frontier=self.enable_refinement_frontier,
            enable_dynamic_evidence_retrieval=self.enable_dynamic_evidence_retrieval,
            disable_question_routing=self.disable_question_routing,
        )
        return QwenVideoRuntimeBundle(
            controller=controller,
            memory_builder=memory_builder,
            speech_recognizer=speech_recognizer,
            visual_summarizer=visual_summarizer,
            speech_refiner=speech_refiner,
            visual_refiner=visual_refiner,
            embedding_provider=embedding_provider,
            image_text_embedding_provider=None,
            video_window_embedding_provider=None,
        )


@dataclass
class QwenLocalVideoStackConfig:
    controller: LocalModelConfig
    visual: LocalModelConfig
    speech: LocalModelConfig
    api_controller: OpenAICompatibleModelConfig | None = None
    forced_aligner: LocalModelConfig | None = None
    semantic_frame_embedding: LocalModelConfig | None = None
    speech_embedding: LocalModelConfig | None = None
    video_window_embedding: LocalModelConfig | None = None
    enable_speech_recognition: bool = True
    speech_backend: Literal["qwen", "faster-whisper"] = "qwen"
    faster_whisper_model: str = "small"
    faster_whisper_device: str = "cpu"
    faster_whisper_compute_type: str = "default"
    faster_whisper_batch_size: int = 1
    faster_whisper_chunk_workers: int = 1
    lazy_speech_refinement: bool = False
    force_eager_speech_recognition: bool = False
    lazy_visual_refinement: bool = False
    ffmpeg_bin: str = "ffmpeg"
    frame_count: int = 3
    frame_width: int | None = 768
    scene_threshold_seconds: float = 20.0
    scene_duration_seconds: float = 180.0
    segment_duration_seconds: float = 45.0
    clip_duration_seconds: float = 15.0
    speech_chunk_duration_seconds: float = 60.0
    speech_asr_chunk_batch_size: int = 1
    enable_fine_speech_windows: bool = False
    fine_speech_window_seconds: float = 15.0
    fine_speech_window_stride_seconds: float = 5.0
    controller_enable_thinking: bool = False
    use_pitome: bool = False
    pitome_dense_frame_rate: float = 1.0
    pitome_min_frame_count: int | None = None
    pitome_protect_ratio: float = 0.15
    pitome_similarity_threshold: float = 0.8
    pitome_embedding_size: int = 16
    pitome_embedding_backend: str = "pixel"
    pitome_embedding_device: str | None = None
    pitome_frame_width: int | None = None
    pitome_frame_extraction_strategy: Literal["auto", "batch", "seek", "sequence"] = "auto"
    pitome_frame_extraction_workers: int = 1
    pitome_anchor_frame_count: int = 0
    pitome_max_selected_frames: int | None = None
    pitome_scene_threshold: float = 0.35
    pitome_max_scene_boundary_frames: int = 6
    pitome_scene_sample_rate: float | None = 1.0
    pitome_scene_keyframes_only: bool = True
    pitome_edge_boundary_frames: bool = True
    visual_index_batch_size: int = 1
    visual_index_workers: int = 1
    enable_paddle_ocr: bool = False
    paddle_ocr_lang: str = "en"
    paddle_ocr_version: str = "PP-OCRv5"
    paddle_ocr_device: str | None = None
    paddle_ocr_window_seconds: float = 45.0
    paddle_ocr_frame_count: int = 6
    paddle_ocr_frame_width: int | None = 960
    paddle_ocr_min_confidence: float = 0.35
    paddle_ocr_enable_mkldnn: bool = False
    paddle_ocr_cache_dir: str | None = None
    paddle_ocr_frame_extraction_strategy: Literal["auto", "batch", "seek", "sequence"] = "seek"
    paddle_ocr_frame_extraction_workers: int = 1
    paddle_ocr_text_detection_model_name: str | None = None
    paddle_ocr_text_recognition_model_name: str | None = None
    paddle_ocr_text_recognition_batch_size: int | None = None
    vl_max_input_frames: int | None = None
    parent_visual_summary_mode: Literal["full", "compact"] | None = None
    search_mode: Literal["lexical", "graph"] | None = None
    semantic_frame_embedding_batch_size: int = 8
    enable_video_window_reranking: bool = False
    video_window_rerank_candidate_count: int = 24
    video_window_rerank_weight: float = 0.75
    video_window_rerank_window_seconds: float | None = None
    video_window_rerank_min_score: float | None = None
    video_window_embedding_frame_count: int = 8
    video_window_embedding_frame_size: int = 224
    enable_vrrqa_graph_refinement_expansion: bool = True
    vrrqa_graph_refinement_neighbor_count: int = 1
    enable_vrrqa_visual_answer_verifier: bool = True
    vrrqa_visual_verifier_frame_count: int = 8
    enable_controller_evidence_classifier: bool = False
    enable_targeted_asr_refinement: bool = False
    enable_refinement_frontier: bool = True
    offload_components_after_use: bool = False
    enable_dynamic_evidence_retrieval: bool = True
    disable_question_routing: bool = True
    verbose: bool = False

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
        semantic_frame_embedding_model: str | None = None,
        semantic_frame_embedding_device: str = "cpu",
        semantic_frame_embedding_torch_dtype: str = "float32",
        speech_embedding_model: str | None = None,
        speech_embedding_device: str = "cpu",
        video_window_embedding_model: str | None = None,
        video_window_embedding_device: str = "cuda:0",
        video_window_embedding_torch_dtype: str = "float32",
        torch_dtype: str = "bfloat16",
        attn_implementation: str | None = None,
    ) -> "QwenLocalVideoStackConfig":
        controller = LocalModelConfig(
            model_name=controller_model,
            model_path=str(default_local_model_dir(controller_model)),
            device=controller_device,
            device_map={"": controller_device},
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            max_new_tokens=256,
        )
        visual = LocalModelConfig(
            model_name=visual_model,
            model_path=str(default_local_model_dir(visual_model)),
            device=visual_device,
            device_map={"": visual_device},
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            max_new_tokens=512,
        )
        speech = LocalModelConfig(
            model_name=speech_model,
            model_path=str(default_local_model_dir(speech_model)),
            device=speech_device,
            device_map={"": speech_device},
            torch_dtype=torch_dtype,
            max_new_tokens=512,
        )
        forced_aligner = None
        if forced_aligner_model is not None:
            forced_aligner = LocalModelConfig(
                model_name=forced_aligner_model,
                model_path=str(default_local_model_dir(forced_aligner_model)),
                device=speech_device,
                device_map={"": speech_device},
                torch_dtype=torch_dtype,
            )
        semantic_frame_embedding = None
        if semantic_frame_embedding_model is not None:
            semantic_frame_embedding = LocalModelConfig(
                model_name=semantic_frame_embedding_model,
                model_path=None,
                device=semantic_frame_embedding_device,
                torch_dtype=semantic_frame_embedding_torch_dtype,
            )
        speech_embedding = None
        if speech_embedding_model is not None:
            speech_embedding = LocalModelConfig(
                model_name=speech_embedding_model,
                model_path=None,
                device=speech_embedding_device,
                torch_dtype="float32",
            )
        video_window_embedding = None
        if video_window_embedding_model is not None:
            video_window_embedding = LocalModelConfig(
                model_name=video_window_embedding_model,
                model_path=str(default_local_model_dir(video_window_embedding_model)),
                device=video_window_embedding_device,
                torch_dtype=video_window_embedding_torch_dtype,
                trust_remote_code=True,
            )
        return cls(
            controller=controller,
            visual=visual,
            speech=speech,
            forced_aligner=forced_aligner,
            semantic_frame_embedding=semantic_frame_embedding,
            speech_embedding=speech_embedding,
            video_window_embedding=video_window_embedding,
        )

    def download_models(self) -> dict[str, str]:
        downloads = {
            "visual": self.visual.download(),
            "speech": self.speech.download(),
        }
        if self.api_controller is None:
            downloads["controller"] = self.controller.download()
        else:
            downloads["controller"] = f"api:{self.api_controller.model_name}"
        if self.forced_aligner is not None:
            downloads["forced_aligner"] = self.forced_aligner.download()
        if self.semantic_frame_embedding is not None:
            downloads["semantic_frame_embedding"] = self.semantic_frame_embedding.download()
        if self.speech_embedding is not None:
            downloads["speech_embedding"] = self.speech_embedding.download()
        if self.video_window_embedding is not None:
            downloads["video_window_embedding"] = self.video_window_embedding.download()
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
        if self.speech_backend not in {"qwen", "faster-whisper"}:
            raise ValueError(f"Unsupported speech backend: {self.speech_backend}")
        lazy_visual_mode = self.use_pitome or self.lazy_visual_refinement
        lazy_speech_mode = self.lazy_speech_refinement and not self.force_eager_speech_recognition

        if self.api_controller is not None:
            from rlm.clients.openai import OpenAIClient

            controller_client = OpenAIClient(**self.api_controller.to_client_kwargs())
        else:
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
        speech_recognizer = None
        speech_refiner = None
        if self.enable_speech_recognition:
            if self.speech_backend == "faster-whisper":
                full_speech_recognizer = FasterWhisperSpeechRecognizer(
                    model_name=self.faster_whisper_model,
                    device=self.faster_whisper_device,
                    compute_type=self.faster_whisper_compute_type,
                    batch_size=self.faster_whisper_batch_size,
                    chunk_workers=self.faster_whisper_chunk_workers,
                    chunk_duration_seconds=self.speech_chunk_duration_seconds,
                    ffmpeg_bin=self.ffmpeg_bin,
                    verbose=self.verbose,
                )
            else:
                full_speech_recognizer = LocalQwenASRSpeechRecognizer(
                    model_name=self.speech.model_name,
                    model_path=self.speech.resolved_model_path(),
                    forced_aligner_name=self.forced_aligner.model_name
                    if self.forced_aligner
                    else None,
                    forced_aligner_path=(
                        self.forced_aligner.resolved_model_path() if self.forced_aligner else None
                    ),
                    device_map=self.speech.device_map or self.speech.device,
                    torch_dtype=self.speech.torch_dtype,
                    ffmpeg_bin=self.ffmpeg_bin,
                    chunk_duration_seconds=self.speech_chunk_duration_seconds,
                    chunk_batch_size=self.speech_asr_chunk_batch_size,
                    max_new_tokens=self.speech.max_new_tokens,
                    verbose=self.verbose,
                )
            if lazy_speech_mode:
                speech_recognizer = LazySpeechRecognizer(
                    chunk_duration_seconds=self.speech_chunk_duration_seconds,
                    verbose=self.verbose,
                )
                speech_refiner = full_speech_recognizer
            else:
                speech_recognizer = full_speech_recognizer
                if self.enable_targeted_asr_refinement:
                    speech_refiner = full_speech_recognizer
        frame_embedding_provider = self._build_semantic_frame_embedding_provider()
        speech_embedding_provider = self._build_speech_embedding_provider()
        video_window_embedding_provider = self._build_video_window_embedding_provider()
        visual_refiner = None
        visual_summarizer: VisualSummarizer
        qwen_visual_summarizer = LocalQwenVisualSummarizer(
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
            use_pitome=lazy_visual_mode,
            pitome_dense_frame_rate=self.pitome_dense_frame_rate,
            pitome_min_frame_count=self.pitome_min_frame_count,
            pitome_protect_ratio=self.pitome_protect_ratio,
            pitome_similarity_threshold=self.pitome_similarity_threshold,
            pitome_embedding_size=self.pitome_embedding_size,
            pitome_embedding_backend=self.pitome_embedding_backend,
            pitome_embedding_device=self.pitome_embedding_device,
            pitome_frame_width=self.pitome_frame_width,
            pitome_frame_extraction_strategy=self.pitome_frame_extraction_strategy,
            pitome_frame_extraction_workers=self.pitome_frame_extraction_workers,
            pitome_anchor_frame_count=self.pitome_anchor_frame_count,
            pitome_max_selected_frames=self.pitome_max_selected_frames,
            pitome_scene_threshold=self.pitome_scene_threshold,
            pitome_max_scene_boundary_frames=self.pitome_max_scene_boundary_frames,
            pitome_scene_sample_rate=self.pitome_scene_sample_rate,
            pitome_scene_keyframes_only=self.pitome_scene_keyframes_only,
            pitome_edge_boundary_frames=self.pitome_edge_boundary_frames,
            frame_embedding_provider=None if lazy_visual_mode else frame_embedding_provider,
            summary_granularity="clip" if lazy_visual_mode else None,
            vl_max_input_frames=self.vl_max_input_frames,
            verbose=self.verbose,
        )
        if lazy_visual_mode:
            visual_summarizer = LazyPiToMeVisualIndexer(
                ffmpeg_bin=self.ffmpeg_bin,
                frame_width=self.frame_width,
                frame_count=self.frame_count,
                pitome_dense_frame_rate=self.pitome_dense_frame_rate,
                pitome_min_frame_count=self.pitome_min_frame_count,
                pitome_protect_ratio=self.pitome_protect_ratio,
                pitome_similarity_threshold=self.pitome_similarity_threshold,
                pitome_embedding_size=self.pitome_embedding_size,
                pitome_embedding_backend=self.pitome_embedding_backend,
                pitome_embedding_device=self.pitome_embedding_device,
                pitome_frame_width=self.pitome_frame_width,
                pitome_frame_extraction_strategy=self.pitome_frame_extraction_strategy,
                pitome_frame_extraction_workers=self.pitome_frame_extraction_workers,
                pitome_anchor_frame_count=self.pitome_anchor_frame_count,
                pitome_max_selected_frames=self.pitome_max_selected_frames,
                pitome_scene_threshold=self.pitome_scene_threshold,
                pitome_max_scene_boundary_frames=self.pitome_max_scene_boundary_frames,
                pitome_scene_sample_rate=self.pitome_scene_sample_rate,
                pitome_scene_keyframes_only=self.pitome_scene_keyframes_only,
                pitome_edge_boundary_frames=self.pitome_edge_boundary_frames,
                frame_embedding_provider=frame_embedding_provider,
                visual_index_batch_size=self.visual_index_batch_size,
                visual_index_workers=self.visual_index_workers,
                summary_granularity="clip",
                verbose=self.verbose,
            )
            visual_refiner = qwen_visual_summarizer
        else:
            visual_summarizer = qwen_visual_summarizer
            if self.enable_vrrqa_visual_answer_verifier:
                visual_refiner = qwen_visual_summarizer
        image_text_embedding_provider = frame_embedding_provider
        ocr_extractor: OCRExtractor | None = None
        if self.enable_paddle_ocr:
            ocr_extractor = PaddleOCRTextExtractor(
                ffmpeg_bin=self.ffmpeg_bin,
                window_duration_seconds=self.paddle_ocr_window_seconds,
                frame_count=self.paddle_ocr_frame_count,
                frame_width=self.paddle_ocr_frame_width,
                frame_extraction_strategy=self.paddle_ocr_frame_extraction_strategy,
                frame_extraction_workers=self.paddle_ocr_frame_extraction_workers,
                lang=self.paddle_ocr_lang,
                ocr_version=self.paddle_ocr_version,
                text_detection_model_name=self.paddle_ocr_text_detection_model_name,
                text_recognition_model_name=self.paddle_ocr_text_recognition_model_name,
                text_recognition_batch_size=self.paddle_ocr_text_recognition_batch_size,
                device=self.paddle_ocr_device,
                min_confidence=self.paddle_ocr_min_confidence,
                enable_mkldnn=self.paddle_ocr_enable_mkldnn,
                cache_dir=self.paddle_ocr_cache_dir,
                verbose=self.verbose,
            )
        memory_builder = VideoMemoryBuilder(
            speech_recognizer=speech_recognizer,
            visual_summarizer=visual_summarizer,
            ocr_extractor=ocr_extractor,
            scene_duration_seconds=self.scene_duration_seconds,
            segment_duration_seconds=self.segment_duration_seconds,
            clip_duration_seconds=self.clip_duration_seconds,
            enable_fine_speech_windows=self.enable_fine_speech_windows,
            fine_speech_window_seconds=self.fine_speech_window_seconds,
            fine_speech_window_stride_seconds=self.fine_speech_window_stride_seconds,
            visual_span_mode="clip" if lazy_visual_mode else "scene_and_clip",
            aggregate_child_visual_summaries=lazy_visual_mode,
            parent_visual_summary_mode=self.parent_visual_summary_mode
            or ("compact" if lazy_visual_mode else "full"),
            offload_components_after_phase=self.offload_components_after_use,
            verbose=self.verbose,
        )
        controller = VideoRLM(
            controller_client=controller_client,
            logger=logger,
            max_steps=max_steps,
            search_top_k=search_top_k,
            max_frontier_items=max_frontier_items,
            enable_hybrid_speech_refinement=enable_hybrid_speech_refinement,
            speech_refine_candidate_count=speech_refine_candidate_count,
            search_mode=self.search_mode or ("graph" if lazy_visual_mode else "lexical"),
            embedding_provider=None,
            speech_embedding_provider=speech_embedding_provider,
            image_text_embedding_provider=image_text_embedding_provider,
            video_window_embedding_provider=video_window_embedding_provider,
            enable_video_window_reranking=self.enable_video_window_reranking,
            video_window_rerank_candidate_count=self.video_window_rerank_candidate_count,
            video_window_rerank_weight=self.video_window_rerank_weight,
            video_window_rerank_window_seconds=self.video_window_rerank_window_seconds,
            video_window_rerank_min_score=self.video_window_rerank_min_score,
            speech_refiner=speech_refiner,
            enable_targeted_asr_refinement=self.enable_targeted_asr_refinement,
            visual_refiner=visual_refiner,
            enable_vrrqa_graph_refinement_expansion=(
                self.enable_vrrqa_graph_refinement_expansion
            ),
            vrrqa_graph_refinement_neighbor_count=self.vrrqa_graph_refinement_neighbor_count,
            enable_vrrqa_visual_answer_verifier=self.enable_vrrqa_visual_answer_verifier,
            vrrqa_visual_verifier_frame_count=self.vrrqa_visual_verifier_frame_count,
            enable_controller_evidence_classifier=self.enable_controller_evidence_classifier,
            enable_refinement_frontier=self.enable_refinement_frontier,
            offload_components_after_use=self.offload_components_after_use,
            enable_dynamic_evidence_retrieval=self.enable_dynamic_evidence_retrieval,
            disable_question_routing=self.disable_question_routing,
        )
        return QwenVideoRuntimeBundle(
            controller=controller,
            memory_builder=memory_builder,
            speech_recognizer=speech_recognizer,
            visual_summarizer=visual_summarizer,
            speech_refiner=speech_refiner,
            visual_refiner=visual_refiner,
            ocr_extractor=ocr_extractor,
            embedding_provider=None,
            speech_embedding_provider=speech_embedding_provider,
            image_text_embedding_provider=image_text_embedding_provider,
            video_window_embedding_provider=video_window_embedding_provider,
        )

    def _build_semantic_frame_embedding_provider(
        self,
    ) -> ImageTextEmbeddingProvider | None:
        if self.semantic_frame_embedding is None:
            return None
        return LocalImageTextEmbeddingProvider(
            model_name=self.semantic_frame_embedding.model_name,
            model_path=self.semantic_frame_embedding.resolved_model_path(),
            device=self.semantic_frame_embedding.device,
            torch_dtype=self.semantic_frame_embedding.torch_dtype,
            batch_size=self.semantic_frame_embedding_batch_size,
            trust_remote_code=self.semantic_frame_embedding.trust_remote_code,
        )

    def _build_speech_embedding_provider(self) -> EmbeddingProvider | None:
        if self.speech_embedding is None:
            return None
        return LocalSentenceTransformerEmbeddingProvider(
            model_name=self.speech_embedding.model_name,
            model_path=self.speech_embedding.model_path,
            device=self.speech_embedding.device,
        )

    def _build_video_window_embedding_provider(
        self,
    ) -> VideoWindowEmbeddingProvider | None:
        if self.video_window_embedding is None:
            return None
        return LocalInternVideoWindowEmbeddingProvider(
            model_name=self.video_window_embedding.model_name,
            model_path=self.video_window_embedding.resolved_model_path(),
            device=self.video_window_embedding.device,
            torch_dtype=self.video_window_embedding.torch_dtype,
            frame_count=self.video_window_embedding_frame_count,
            frame_size=self.video_window_embedding_frame_size,
            trust_remote_code=self.video_window_embedding.trust_remote_code,
        )
