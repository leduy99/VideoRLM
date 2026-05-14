from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
        inputs = self._to_device(inputs)
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
            inputs = self._to_device(inputs)
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

    def _to_device(self, inputs):
        if hasattr(inputs, "to"):
            return inputs.to(self.device)
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    def _normalize_features(self, features) -> list[list[float]]:
        features = features.float()
        norms = features.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
        normalized = features / norms
        return [
            [round(float(value), 6) for value in row.detach().cpu().tolist()] for row in normalized
        ]


def _resolve_torch_dtype(torch_module, value: str | Any):
    if not isinstance(value, str):
        return value
    if not hasattr(torch_module, value):
        raise ValueError(f"Unsupported torch dtype: {value}")
    return getattr(torch_module, value)
