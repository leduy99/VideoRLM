from __future__ import annotations

import json
import os
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlm.video.types import TimeSpan


@dataclass
class LocalSentenceTransformerEmbeddingProvider:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_path: str | None = None
    device: str = "cpu"
    batch_size: int = 32
    model: Any | None = None

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure_loaded()
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [
            [round(float(value), 6) for value in embedding.tolist()]
            for embedding in embeddings
        ]

    def _ensure_loaded(self):
        if self.model is not None:
            return self.model
        from sentence_transformers import SentenceTransformer

        model_path = self.model_path or self.model_name
        self.model = SentenceTransformer(model_path, device=self.device)
        return self.model

    def unload(self) -> None:
        self.model = None
        from rlm.video.gpu_memory import clear_torch_cache

        clear_torch_cache()


@dataclass
class LocalImageTextEmbeddingProvider:
    model_name: str = "openai/clip-vit-base-patch32"
    model_path: str | None = None
    device: str = "cpu"
    torch_dtype: str = "float32"
    batch_size: int = 8
    trust_remote_code: bool = False
    model: Any | None = None
    processor: Any | None = None

    def embed_text(self, text: str) -> list[float]:
        model, processor = self._ensure_loaded()
        import torch

        inputs = processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = self._to_device(inputs, _model_floating_dtype(model))
        with torch.inference_mode():
            features = model.get_text_features(**inputs)
        return self._normalize_features(features)[0]

    def embed_images(self, image_paths: list[str | Path]) -> list[list[float]]:
        if not image_paths:
            return []

        model, processor = self._ensure_loaded()
        import torch
        from PIL import Image

        embeddings: list[list[float]] = []
        for start in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[start : start + self.batch_size]
            images = []
            for image_path in batch_paths:
                with Image.open(image_path) as image:
                    images.append(image.convert("RGB"))
            inputs = processor(images=images, return_tensors="pt")
            inputs = self._to_device(inputs, _model_floating_dtype(model))
            with torch.inference_mode():
                features = model.get_image_features(**inputs)
            embeddings.extend(self._normalize_features(features))
        return embeddings

    def _ensure_loaded(self):
        if self.model is not None and self.processor is not None:
            return self.model, self.processor

        import torch
        from transformers import AutoModel, AutoProcessor

        model_path = self.model_path or self.model_name
        dtype = _resolve_torch_dtype(torch, self.torch_dtype)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=self.trust_remote_code,
        )
        if not hasattr(self.model, "get_text_features") or not hasattr(
            self.model,
            "get_image_features",
        ):
            raise ValueError(
                "Semantic frame embedding model must expose get_text_features "
                "and get_image_features."
            )
        self.model.to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=self.trust_remote_code,
        )
        return self.model, self.processor

    def unload(self) -> None:
        self.model = None
        self.processor = None
        from rlm.video.gpu_memory import clear_torch_cache

        clear_torch_cache()

    def _to_device(self, inputs, dtype=None):
        return _move_inputs_to_device(inputs, self.device, dtype)

    def _normalize_features(self, features) -> list[list[float]]:
        return _normalize_feature_rows(features)


@dataclass
class LocalInternVideoWindowEmbeddingProvider:
    model_name: str = "OpenGVLab/InternVideo2-Stage2_6B"
    model_path: str | None = None
    device: str = "cuda:0"
    torch_dtype: str = "float32"
    frame_count: int = 8
    frame_size: int = 224
    trust_remote_code: bool = True
    model: Any | None = None

    def embed_text(self, text: str) -> list[float]:
        model = self._ensure_loaded()
        if not hasattr(model, "get_txt_feat"):
            raise ValueError("InternVideo stage-2 model must expose get_txt_feat(text).")
        import torch

        with torch.inference_mode():
            features = model.get_txt_feat(text)
        return _normalize_feature_rows(features)[0]

    def embed_video_windows(
        self,
        video_path: str | Path,
        windows: list[TimeSpan],
    ) -> list[list[float]]:
        if not windows:
            return []
        model = self._ensure_loaded()
        if not hasattr(model, "get_vid_feat"):
            raise ValueError("InternVideo stage-2 model must expose get_vid_feat(frames).")
        import torch

        embeddings: list[list[float]] = []
        frame_count = resolve_internvideo_frame_count(model, self.frame_count)
        for window in windows:
            frames = self._window_to_tensor(Path(video_path), window, frame_count)
            with torch.inference_mode():
                features = model.get_vid_feat(frames)
            embeddings.extend(_normalize_feature_rows(features))
        return embeddings

    def _ensure_loaded(self):
        if self.model is not None:
            return self.model

        import torch
        from transformers import AutoModel

        patch_transformers_internvideo_compat()
        model_path = self.model_path or self.model_name
        local_model_dir = _local_internvideo_model_dir(model_path)
        model_load_path = str(local_model_dir) if local_model_dir is not None else model_path
        ensure_internvideo_text_tokenizer_cached(model_load_path)
        dtype = _resolve_torch_dtype(torch, self.torch_dtype)
        cwd = Path.cwd()
        try:
            if local_model_dir is not None:
                os.chdir(local_model_dir)
            with patch_transformers_eager_init_context():
                self.model = AutoModel.from_pretrained(
                    model_load_path,
                    torch_dtype=dtype,
                    trust_remote_code=self.trust_remote_code,
                )
        finally:
            if local_model_dir is not None:
                os.chdir(cwd)
        self.model.to(self.device)
        self.model.eval()
        self._set_model_device_config(torch)
        return self.model

    def unload(self) -> None:
        self.model = None
        from rlm.video.gpu_memory import clear_torch_cache

        clear_torch_cache()

    def _set_model_device_config(self, torch_module) -> None:
        if self.model is None:
            return
        device = torch_module.device(self.device)
        config = getattr(self.model, "config", None)
        if config is not None:
            config.device = device
        private_config = getattr(self.model, "_config", None)
        if private_config is not None:
            private_config.device = device

    def _window_to_tensor(
        self,
        video_path: Path,
        window: TimeSpan,
        frame_count: int | None = None,
    ):
        effective_frame_count = frame_count if frame_count is not None else self.frame_count
        if effective_frame_count <= 0:
            raise ValueError(f"frame_count must be positive, got {effective_frame_count}")
        if self.frame_size <= 0:
            raise ValueError(f"frame_size must be positive, got {self.frame_size}")

        import cv2
        import numpy as np
        import torch

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video for InternVideo reranking: {video_path}")
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if fps <= 0.0 or frame_total <= 0:
                raise ValueError(
                    "Could not read FPS/frame count for InternVideo reranking video: "
                    f"{video_path}"
                )
            start_frame = max(0, int(round(window.start * fps)))
            end_frame = min(frame_total, max(start_frame + 1, int(round(window.end * fps))))
            indices = _sample_frame_indices(start_frame, end_frame, effective_frame_count)
            frames = [
                self._read_frame(capture, frame_index, video_path)
                for frame_index in indices
            ]
        finally:
            capture.release()

        resized = [
            cv2.resize(frame[:, :, ::-1], (self.frame_size, self.frame_size))
            for frame in frames
        ]
        array = np.asarray(resized, dtype="float32")
        mean = np.array([0.485, 0.456, 0.406], dtype="float32").reshape(1, 1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype="float32").reshape(1, 1, 1, 3)
        array = (array / 255.0 - mean) / std
        array = np.transpose(array, (0, 3, 1, 2))
        return torch.from_numpy(array).unsqueeze(0).to(self.device).float()

    def _read_frame(self, capture, frame_index: int, video_path: Path):
        import cv2

        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()
        if not success:
            raise ValueError(
                f"Could not read frame {frame_index} for InternVideo reranking: {video_path}"
            )
        return frame


def patch_transformers_internvideo_compat() -> None:
    import transformers.modeling_utils as modeling_utils
    import transformers.tokenization_utils as tokenization_utils
    from transformers.modeling_utils import PreTrainedModel
    from transformers.pytorch_utils import apply_chunking_to_forward, prune_linear_layer

    if not hasattr(modeling_utils, "apply_chunking_to_forward"):
        modeling_utils.apply_chunking_to_forward = apply_chunking_to_forward
    if not hasattr(modeling_utils, "prune_linear_layer"):
        modeling_utils.prune_linear_layer = prune_linear_layer
    if not hasattr(modeling_utils, "find_pruneable_heads_and_indices"):
        modeling_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
    if not hasattr(PreTrainedModel, "_convert_head_mask_to_5d"):
        PreTrainedModel._convert_head_mask_to_5d = convert_head_mask_to_5d
    if not hasattr(PreTrainedModel, "get_head_mask"):
        PreTrainedModel.get_head_mask = get_head_mask
    tokenizer_helpers = {
        "_is_control": is_control_character,
        "_is_punctuation": is_punctuation_character,
        "_is_whitespace": is_whitespace_character,
    }
    for helper_name, fallback in tokenizer_helpers.items():
        if not hasattr(tokenization_utils, helper_name):
            setattr(tokenization_utils, helper_name, fallback)


@contextmanager
def patch_transformers_eager_init_context() -> Iterator[None]:
    import inspect

    import transformers.modeling_utils as modeling_utils
    from transformers.modeling_utils import PreTrainedModel

    original_get_init_context = PreTrainedModel.__dict__["get_init_context"]
    original_init_weights = PreTrainedModel.init_weights
    original_mark_tied_weights_as_initialized = getattr(
        PreTrainedModel,
        "mark_tied_weights_as_initialized",
        None,
    )
    original_move_missing_keys_from_meta_to_device = (
        getattr(PreTrainedModel, "_move_missing_keys_from_meta_to_device", None)
    )
    original_adjust_missing_and_unexpected_keys = (
        getattr(PreTrainedModel, "_adjust_missing_and_unexpected_keys", None)
    )
    adjust_load_ignore_container = "list"
    if original_adjust_missing_and_unexpected_keys is not None:
        parameters = inspect.signature(original_adjust_missing_and_unexpected_keys).parameters
        if "loading_info" in parameters:
            adjust_load_ignore_container = "set"

    def get_eager_init_context(cls, *args, **kwargs):
        dtype = kwargs.get("dtype")
        if dtype is None and len(args) >= 4:
            dtype = args[0]
        allow_all_kernels = kwargs.get("allow_all_kernels")
        if allow_all_kernels is None and len(args) >= 4:
            allow_all_kernels = args[3]

        contexts = []
        if dtype is not None and hasattr(modeling_utils, "local_torch_dtype"):
            contexts.append(modeling_utils.local_torch_dtype(dtype, cls.__name__))
        if hasattr(modeling_utils, "init") and hasattr(modeling_utils.init, "no_tie_weights"):
            contexts.append(modeling_utils.init.no_tie_weights())
        elif hasattr(modeling_utils, "no_init_weights"):
            contexts.append(modeling_utils.no_init_weights())
        if hasattr(modeling_utils, "apply_patches"):
            contexts.append(modeling_utils.apply_patches())
        if allow_all_kernels:
            allow_all_hub_kernels = getattr(modeling_utils, "allow_all_hub_kernels", None)
            if allow_all_hub_kernels is not None:
                contexts.append(allow_all_hub_kernels())
        return contexts

    def init_weights_with_runtime_fields(self, *args, **kwargs):
        ensure_transformers_runtime_fields(self)
        return original_init_weights(self, *args, **kwargs)

    def mark_tied_weights_with_runtime_fields(self, *args, **kwargs):
        ensure_transformers_runtime_fields(self)
        if original_mark_tied_weights_as_initialized is None:
            return None
        return original_mark_tied_weights_as_initialized(self, *args, **kwargs)

    def move_missing_keys_with_runtime_fields(self, *args, **kwargs):
        ensure_transformers_runtime_fields(self)
        if original_move_missing_keys_from_meta_to_device is None:
            return None
        return original_move_missing_keys_from_meta_to_device(self, *args, **kwargs)

    def adjust_missing_and_unexpected_keys_with_runtime_fields(self, *args, **kwargs):
        ensure_transformers_runtime_fields(
            self,
            load_ignore_container=adjust_load_ignore_container,
        )
        if original_adjust_missing_and_unexpected_keys is None:
            return args[0] if len(args) == 1 else args
        return original_adjust_missing_and_unexpected_keys(self, *args, **kwargs)

    PreTrainedModel.get_init_context = classmethod(get_eager_init_context)
    PreTrainedModel.init_weights = init_weights_with_runtime_fields
    if original_mark_tied_weights_as_initialized is not None:
        PreTrainedModel.mark_tied_weights_as_initialized = (
            mark_tied_weights_with_runtime_fields
        )
    if original_move_missing_keys_from_meta_to_device is not None:
        PreTrainedModel._move_missing_keys_from_meta_to_device = (
            move_missing_keys_with_runtime_fields
        )
    if original_adjust_missing_and_unexpected_keys is not None:
        PreTrainedModel._adjust_missing_and_unexpected_keys = (
            adjust_missing_and_unexpected_keys_with_runtime_fields
        )
    try:
        yield
    finally:
        PreTrainedModel.get_init_context = original_get_init_context
        PreTrainedModel.init_weights = original_init_weights
        if original_mark_tied_weights_as_initialized is not None:
            PreTrainedModel.mark_tied_weights_as_initialized = (
                original_mark_tied_weights_as_initialized
            )
        if original_move_missing_keys_from_meta_to_device is not None:
            PreTrainedModel._move_missing_keys_from_meta_to_device = (
                original_move_missing_keys_from_meta_to_device
            )
        if original_adjust_missing_and_unexpected_keys is not None:
            PreTrainedModel._adjust_missing_and_unexpected_keys = (
                original_adjust_missing_and_unexpected_keys
            )


def ensure_transformers_runtime_fields(
    model: Any,
    *,
    load_ignore_container: str | None = None,
) -> None:
    if not hasattr(model, "all_tied_weights_keys"):
        get_tied_weights = getattr(model, "get_expanded_tied_weights_keys", None)
        if get_tied_weights is None:
            model.all_tied_weights_keys = {}
        else:
            model.all_tied_weights_keys = model.get_expanded_tied_weights_keys(
                all_submodels=False
            )
    for attr_name in ("_tp_plan", "_ep_plan", "_pp_plan"):
        if not hasattr(model, attr_name) or getattr(model, attr_name) is None:
            setattr(model, attr_name, {})
    for attr_name in (
        "_keys_to_ignore_on_load_unexpected",
        "_keys_to_ignore_on_load_missing",
    ):
        if load_ignore_container is None:
            continue
        container_type = set if load_ignore_container == "set" else list
        if not hasattr(model, attr_name) or getattr(model, attr_name) is None:
            setattr(model, attr_name, container_type())
        elif not isinstance(getattr(model, attr_name), container_type):
            setattr(model, attr_name, container_type(getattr(model, attr_name)))
    for attr_name in (
        "_keep_in_fp32_modules",
        "_keep_in_fp32_modules_strict",
        "_no_split_modules",
        "_skip_keys_device_placement",
        "_keys_to_ignore_on_save",
    ):
        if not hasattr(model, attr_name) or getattr(model, attr_name) is None:
            setattr(model, attr_name, set())


def resolve_internvideo_frame_count(model: Any, default: int) -> int:
    for field_name in ("num_frames_test", "num_frames"):
        for root in (getattr(model, "_config", None), getattr(model, "config", None)):
            value = get_nested_config_value(root, ("model", "vision_encoder", field_name))
            if value is None:
                value = get_nested_config_value(root, ("vision_encoder", field_name))
            if value is not None:
                frame_count = int(value)
                if frame_count <= 0:
                    raise ValueError(
                        "InternVideo model config has non-positive frame count: "
                        f"{frame_count}"
                    )
                return frame_count
    return default


def get_nested_config_value(root: Any, path: tuple[str, ...]) -> Any:
    value = root
    for part in path:
        if value is None:
            return None
        if isinstance(value, dict):
            value = value.get(part)
        else:
            value = getattr(value, part, None)
    return value


def ensure_internvideo_text_tokenizer_cached(model_path: str) -> None:
    text_encoder_repo = _internvideo_text_encoder_repo(model_path)
    from huggingface_hub import hf_hub_download

    try:
        hf_hub_download(repo_id=text_encoder_repo, filename="vocab.txt")
    except Exception as exc:
        raise RuntimeError(
            "InternVideo stage-2 requires the text encoder tokenizer in the Hugging Face "
            f"cache because its remote code loads {text_encoder_repo!r} with "
            "local_files_only=True. Download failed for vocab.txt."
        ) from exc


def _internvideo_text_encoder_repo(model_path: str) -> str:
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        text_encoder = config.get("model", {}).get("text_encoder", {})
        pretrained = text_encoder.get("pretrained")
        if pretrained:
            return str(pretrained)
    return "bert-large-uncased"


def _local_internvideo_model_dir(model_path: str) -> Path | None:
    path = Path(model_path)
    if path.is_dir() and (path / "configs").is_dir():
        return path.resolve()
    return None


def find_pruneable_heads_and_indices(
    heads: list[int] | set[int],
    n_heads: int,
    head_size: int,
    already_pruned_heads: set[int],
) -> tuple[set[int], Any]:
    import torch

    heads = set(heads) - already_pruned_heads
    mask = torch.ones(n_heads, head_size)
    for head in heads:
        adjusted_head = head - sum(1 for pruned_head in already_pruned_heads if pruned_head < head)
        mask[adjusted_head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index = torch.arange(mask.numel())[mask].long()
    return heads, index


def get_head_mask(
    model: Any,
    head_mask: Any,
    num_hidden_layers: int,
    is_attention_chunked: bool = False,
) -> Any:
    if head_mask is None:
        return [None] * num_hidden_layers

    converted_head_mask = model._convert_head_mask_to_5d(head_mask, num_hidden_layers)
    if is_attention_chunked:
        converted_head_mask = converted_head_mask.unsqueeze(-1)
    return converted_head_mask


def convert_head_mask_to_5d(model: Any, head_mask: Any, num_hidden_layers: int) -> Any:
    if head_mask.dim() == 1:
        head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
    elif head_mask.dim() == 2:
        head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    if head_mask.dim() != 5:
        raise ValueError(
            "head_mask must have dimension 1 or 2 after conversion to attention "
            f"heads, got {head_mask.dim()}"
        )
    return head_mask.to(dtype=_model_floating_dtype(model) or head_mask.dtype)


def is_whitespace_character(char: str) -> bool:
    if char in (" ", "\t", "\n", "\r"):
        return True
    return unicodedata.category(char) == "Zs"


def is_control_character(char: str) -> bool:
    if char in ("\t", "\n", "\r"):
        return False
    return unicodedata.category(char).startswith("C")


def is_punctuation_character(char: str) -> bool:
    codepoint = ord(char)
    if (33 <= codepoint <= 47) or (58 <= codepoint <= 64):
        return True
    if (91 <= codepoint <= 96) or (123 <= codepoint <= 126):
        return True
    return unicodedata.category(char).startswith("P")


def _feature_tensor(features: Any) -> Any:
    if hasattr(features, "float"):
        return features

    for field_name in ("pooler_output", "image_embeds", "text_embeds"):
        value = getattr(features, field_name, None)
        if value is not None:
            return value

    if isinstance(features, dict):
        for field_name in ("pooler_output", "image_embeds", "text_embeds"):
            value = features.get(field_name)
            if value is not None:
                return value

    raise ValueError(
        "Semantic embedding model returned unsupported feature output; expected a tensor "
        "or a model output with pooler_output, image_embeds, or text_embeds."
    )


def _normalize_feature_rows(features: Any) -> list[list[float]]:
    tensor = _feature_tensor(features)
    tensor = tensor.float()
    while tensor.dim() > 2 and tensor.shape[1] == 1:
        tensor = tensor.squeeze(1)
    if tensor.dim() > 2:
        tensor = tensor.mean(dim=1)
    norms = tensor.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
    normalized = tensor / norms
    return [
        [round(float(value), 6) for value in row.detach().cpu().tolist()]
        for row in normalized
    ]


def _sample_frame_indices(start_frame: int, end_frame: int, frame_count: int) -> list[int]:
    if frame_count <= 0:
        raise ValueError(f"frame_count must be positive, got {frame_count}")
    last_frame = max(start_frame, end_frame - 1)
    if frame_count == 1:
        return [(start_frame + last_frame) // 2]
    if last_frame == start_frame:
        return [start_frame for _index in range(frame_count)]
    return [
        int(round(start_frame + (position * (last_frame - start_frame) / (frame_count - 1))))
        for position in range(frame_count)
    ]


def _model_floating_dtype(model: Any) -> Any | None:
    if hasattr(model, "parameters"):
        for parameter in model.parameters():
            if hasattr(parameter, "is_floating_point") and parameter.is_floating_point():
                return parameter.dtype
    return getattr(model, "dtype", None)


def _move_inputs_to_device(inputs: Any, device: str, dtype: Any | None) -> Any:
    if hasattr(inputs, "items"):
        for key, value in list(inputs.items()):
            moved = _move_value_to_device(value, device, dtype)
            inputs[key] = moved
            if hasattr(inputs, key):
                try:
                    setattr(inputs, key, moved)
                except AttributeError:
                    pass
        return inputs
    return _move_value_to_device(inputs, device, dtype)


def _move_value_to_device(value: Any, device: str, dtype: Any | None) -> Any:
    if hasattr(value, "to"):
        if hasattr(value, "is_floating_point") and value.is_floating_point() and dtype is not None:
            return value.to(device=device, dtype=dtype)
        return value.to(device)
    if isinstance(value, list):
        return [_move_value_to_device(item, device, dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_value_to_device(item, device, dtype) for item in value)
    return value


def _resolve_torch_dtype(torch_module, value: str | Any):
    if not isinstance(value, str):
        return value
    if not hasattr(torch_module, value):
        raise ValueError(f"Unsupported torch dtype: {value}")
    return getattr(torch_module, value)
