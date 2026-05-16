from rlm.video.adapters import (
    CallableAudioEventExtractor,
    CallableEmbeddingProvider,
    CallableOCRExtractor,
    CallableSpeechRecognizer,
    CallableVisualSummarizer,
    ImageTextEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
    OpenAICompatibleSpeechRecognizer,
    OpenAICompatibleVisualSummarizer,
)
from rlm.video.artifact_store import PreparedArtifactStore
from rlm.video.controller import VideoRLM
from rlm.video.huggingface import (
    default_local_model_dir,
    download_snapshot,
    get_model_output_root,
    sanitize_repo_id,
)
from rlm.video.index import SearchHit, VideoMemoryIndex
from rlm.video.logger import VideoRLMLogger
from rlm.video.longshot import (
    LONGSHOT_DATASET_NAME,
    LONGSHOT_DATASET_PATH,
    LONGSHOT_DATASET_SPLIT,
    LongShOTBenchmarkRunner,
    LongShOTVideoResolver,
    load_longshot_samples,
)
from rlm.video.memory import PreparedVideoArtifacts, VideoMemoryBuilder
from rlm.video.semantic_embeddings import LocalImageTextEmbeddingProvider
from rlm.video.traces import result_to_training_examples, save_training_examples
from rlm.video.types import (
    ActionType,
    AudioEvent,
    BudgetState,
    ControllerAction,
    ControllerState,
    Evidence,
    EvidenceBoard,
    EvidenceBoardSlot,
    EvidenceSlotSpec,
    FrontierItem,
    Modality,
    Observation,
    OCRSpan,
    OpenedTarget,
    QuestionSpec,
    SlotRole,
    SlotStatus,
    SpeechSpan,
    TimeSpan,
    TraceStep,
    VideoMemory,
    VideoNode,
    VideoNodeLevel,
    VideoRLMResult,
    VisualSummarySpan,
)

_LAZY_EXPORTS = {
    "FasterWhisperSpeechRecognizer": (
        "rlm.video.local_adapters",
        "FasterWhisperSpeechRecognizer",
    ),
    "LazyPiToMeVisualIndexer": ("rlm.video.local_adapters", "LazyPiToMeVisualIndexer"),
    "LazySpeechRecognizer": ("rlm.video.local_adapters", "LazySpeechRecognizer"),
    "LocalModelConfig": ("rlm.video.qwen", "LocalModelConfig"),
    "LocalQwenASRSpeechRecognizer": (
        "rlm.video.local_adapters",
        "LocalQwenASRSpeechRecognizer",
    ),
    "LocalQwenVisualSummarizer": (
        "rlm.video.local_adapters",
        "LocalQwenVisualSummarizer",
    ),
    "OpenAICompatibleModelConfig": ("rlm.video.qwen", "OpenAICompatibleModelConfig"),
    "QwenLocalVideoStackConfig": ("rlm.video.qwen", "QwenLocalVideoStackConfig"),
    "QwenVideoRuntimeBundle": ("rlm.video.qwen", "QwenVideoRuntimeBundle"),
    "QwenVideoStackConfig": ("rlm.video.qwen", "QwenVideoStackConfig"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _LAZY_EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "ActionType",
    "AudioEvent",
    "BudgetState",
    "LONGSHOT_DATASET_NAME",
    "LONGSHOT_DATASET_PATH",
    "LONGSHOT_DATASET_SPLIT",
    "LocalModelConfig",
    "LocalImageTextEmbeddingProvider",
    "LocalQwenASRSpeechRecognizer",
    "LocalQwenVisualSummarizer",
    "CallableAudioEventExtractor",
    "CallableEmbeddingProvider",
    "CallableOCRExtractor",
    "CallableSpeechRecognizer",
    "CallableVisualSummarizer",
    "ControllerAction",
    "ControllerState",
    "Evidence",
    "EvidenceBoard",
    "EvidenceBoardSlot",
    "EvidenceSlotSpec",
    "FrontierItem",
    "LongShOTBenchmarkRunner",
    "LongShOTVideoResolver",
    "Modality",
    "OCRSpan",
    "Observation",
    "OpenedTarget",
    "ImageTextEmbeddingProvider",
    "OpenAICompatibleEmbeddingProvider",
    "OpenAICompatibleModelConfig",
    "OpenAICompatibleSpeechRecognizer",
    "OpenAICompatibleVisualSummarizer",
    "PreparedArtifactStore",
    "PreparedVideoArtifacts",
    "QwenLocalVideoStackConfig",
    "QwenVideoRuntimeBundle",
    "QwenVideoStackConfig",
    "QuestionSpec",
    "SearchHit",
    "SlotRole",
    "SlotStatus",
    "SpeechSpan",
    "TimeSpan",
    "TraceStep",
    "VideoMemory",
    "VideoMemoryBuilder",
    "VideoMemoryIndex",
    "VideoNode",
    "VideoNodeLevel",
    "VideoRLM",
    "VideoRLMLogger",
    "VideoRLMResult",
    "VisualSummarySpan",
    "default_local_model_dir",
    "download_snapshot",
    "get_model_output_root",
    "load_longshot_samples",
    "result_to_training_examples",
    "sanitize_repo_id",
    "save_training_examples",
]
