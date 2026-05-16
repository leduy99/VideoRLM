from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from rlm.video.media import (
    FrameExtractionStrategy,
    extract_frames_for_span,
    extract_frames_for_timestamps,
    sample_span_timestamps,
    sample_span_timestamps_by_rate,
)
from rlm.video.types import TimeSpan

FrameSelectionStrategy = Literal["uniform", "pitome"]
FrameEmbeddingBackend = Literal["pixel", "hybrid"]
DEFAULT_STORED_FRAME_EMBEDDING_SIZE = 64
FRAME_EMBEDDING_TORCH_BATCH_SIZE = 128


@dataclass
class FrameSelectionResult:
    strategy: FrameSelectionStrategy
    frame_paths: list[Path]
    timestamps: list[float]
    dense_frame_count: int
    embedding_backend: str | None = None
    embedding_size: int | None = None
    frame_embeddings: list[list[float]] = field(default_factory=list)
    protected_timestamps: list[float] = field(default_factory=list)
    representative_timestamps: list[float] = field(default_factory=list)
    energy_scores: list[float] = field(default_factory=list)
    merged_pairs: list[tuple[float, float]] = field(default_factory=list)

    def to_metadata(self) -> dict[str, Any]:
        metadata = {
            "strategy": self.strategy,
            "dense_frame_count": self.dense_frame_count,
            "selected_frame_count": len(self.timestamps),
            "selected_frame_timestamps": list(self.timestamps),
            "protected_timestamps": list(self.protected_timestamps),
            "representative_timestamps": list(self.representative_timestamps),
            "merged_pairs": [{"left": left, "right": right} for left, right in self.merged_pairs],
        }
        if self.frame_embeddings:
            backend = self.embedding_backend
            metadata.update(
                {
                    "pitome_frame_embeddings": [
                        list(embedding) for embedding in self.frame_embeddings
                    ],
                    "pitome_frame_embedding_backend": backend,
                    "pitome_frame_embedding_source_size": self.embedding_size,
                    "pitome_frame_embedding_dim": len(self.frame_embeddings[0]),
                    "pitome_frame_embedding_semantic_fusion": bool(
                        backend and "+semantic" in backend
                    ),
                }
            )
        return metadata


def limit_frame_selection_by_temporal_coverage(
    selection: FrameSelectionResult,
    max_count: int,
) -> FrameSelectionResult:
    indices = _temporal_coverage_indices(len(selection.frame_paths), max_count)
    if indices == list(range(len(selection.frame_paths))):
        return selection
    return FrameSelectionResult(
        strategy=selection.strategy,
        frame_paths=[selection.frame_paths[index] for index in indices],
        timestamps=[selection.timestamps[index] for index in indices],
        dense_frame_count=selection.dense_frame_count,
        embedding_backend=selection.embedding_backend,
        embedding_size=selection.embedding_size,
        frame_embeddings=[selection.frame_embeddings[index] for index in indices]
        if selection.frame_embeddings
        else [],
        protected_timestamps=list(selection.protected_timestamps),
        representative_timestamps=list(selection.representative_timestamps),
        energy_scores=list(selection.energy_scores),
        merged_pairs=list(selection.merged_pairs),
    )


def select_visual_frames_for_span(
    media_path: str | Path,
    span: TimeSpan,
    *,
    strategy: FrameSelectionStrategy = "pitome",
    uniform_frame_count: int = 3,
    dense_frame_rate: float = 1.0,
    ffmpeg_bin: str = "ffmpeg",
    width: int | None = None,
    output_dir: str | Path | None = None,
    protect_ratio: float = 0.15,
    similarity_threshold: float = 0.8,
    embedding_size: int = 16,
    embedding_backend: FrameEmbeddingBackend = "pixel",
    embedding_device: str | None = None,
    anchor_frame_count: int = 0,
    frame_extraction_strategy: FrameExtractionStrategy = "auto",
    frame_extraction_seek_workers: int = 1,
) -> FrameSelectionResult:
    if strategy == "uniform":
        timestamps = sample_span_timestamps(span, uniform_frame_count)
        frame_paths = extract_frames_for_span(
            media_path=media_path,
            span=span,
            frame_count=uniform_frame_count,
            ffmpeg_bin=ffmpeg_bin,
            width=width,
            output_dir=output_dir,
            extraction_strategy=frame_extraction_strategy,
            seek_workers=frame_extraction_seek_workers,
        )
        return FrameSelectionResult(
            strategy="uniform",
            frame_paths=frame_paths,
            timestamps=timestamps,
            dense_frame_count=len(timestamps),
        )

    dense_timestamps = sample_span_timestamps_by_rate(
        span,
        dense_frame_rate,
        min_frames=max(uniform_frame_count, 1),
    )
    dense_frame_paths = extract_frames_for_timestamps(
        media_path=media_path,
        timestamps=dense_timestamps,
        ffmpeg_bin=ffmpeg_bin,
        width=width,
        output_dir=output_dir,
        prefix="dense",
        extraction_strategy=frame_extraction_strategy,
        seek_workers=frame_extraction_seek_workers,
    )
    embeddings = load_frame_embeddings(
        dense_frame_paths,
        embedding_size=embedding_size,
        backend=embedding_backend,
        device=embedding_device,
    )
    selection = select_frame_indices_from_embeddings(
        embeddings,
        protect_ratio=protect_ratio,
        similarity_threshold=similarity_threshold,
        anchor_count=anchor_frame_count,
        embedding_device=embedding_device,
    )

    selected_indices = selection["selected_indices"]
    selected_timestamps = [dense_timestamps[index] for index in selected_indices]
    selected_frame_paths = [dense_frame_paths[index] for index in selected_indices]
    selected_embeddings = [compact_frame_embedding(embeddings[index]) for index in selected_indices]
    protected_timestamps = [dense_timestamps[index] for index in selection["protected_indices"]]
    representative_timestamps = [
        dense_timestamps[index] for index in selection["representative_indices"]
    ]
    merged_pairs = [
        (dense_timestamps[left], dense_timestamps[right])
        for left, right in selection["merged_pairs"]
    ]

    return FrameSelectionResult(
        strategy="pitome",
        frame_paths=selected_frame_paths,
        timestamps=selected_timestamps,
        dense_frame_count=len(dense_timestamps),
        embedding_backend=embedding_backend,
        embedding_size=embedding_size,
        frame_embeddings=selected_embeddings,
        protected_timestamps=protected_timestamps,
        representative_timestamps=representative_timestamps,
        energy_scores=selection["energy_scores"],
        merged_pairs=merged_pairs,
    )


def select_frame_indices_from_embeddings(
    embeddings: list[list[float]],
    *,
    protect_ratio: float = 0.15,
    similarity_threshold: float = 0.8,
    anchor_count: int = 0,
    embedding_device: str | None = None,
) -> dict[str, Any]:
    if not embeddings:
        return {
            "selected_indices": [],
            "protected_indices": [],
            "representative_indices": [],
            "merged_pairs": [],
            "energy_scores": [],
        }

    _validate_ratio(protect_ratio, name="protect_ratio")
    if anchor_count < 0:
        raise ValueError(f"anchor_count must be non-negative, got {anchor_count}")

    if len(embeddings) == 1:
        return {
            "selected_indices": [0],
            "protected_indices": [0],
            "representative_indices": [],
            "merged_pairs": [],
            "energy_scores": [0.0],
        }

    similarity_matrix = build_similarity_matrix(embeddings, device=embedding_device)
    energy_scores = compute_energy_scores(similarity_matrix, device=embedding_device)
    protected_count = max(1, math.ceil(len(embeddings) * protect_ratio))
    energy_order = sorted(range(len(embeddings)), key=lambda index: energy_scores[index])
    anchor_indices = _uniform_anchor_indices(len(embeddings), anchor_count)
    protected_indices = sorted(set(energy_order[:protected_count]) | set(anchor_indices))

    protected_set = set(protected_indices)
    mergeable_indices = [index for index in range(len(embeddings)) if index not in protected_set]
    set_a = mergeable_indices[::2]
    set_b = mergeable_indices[1::2]
    unmatched_b = set(set_b)
    representatives: set[int] = set()
    leftovers: set[int] = set()
    merged_pairs: list[tuple[int, int]] = []

    for index_a in set_a:
        if not unmatched_b:
            leftovers.add(index_a)
            continue

        best_b = max(
            unmatched_b,
            key=lambda index_b: similarity_matrix[index_a][index_b],
        )
        similarity = similarity_matrix[index_a][best_b]
        if similarity < similarity_threshold:
            leftovers.add(index_a)
            continue

        unmatched_b.remove(best_b)
        representative = index_a if energy_scores[index_a] >= energy_scores[best_b] else best_b
        representatives.add(representative)
        merged_pairs.append((index_a, best_b))

    leftovers.update(unmatched_b)
    selected_indices = sorted(protected_set | representatives | leftovers)
    representative_indices = sorted(representatives)
    return {
        "selected_indices": selected_indices,
        "protected_indices": protected_indices,
        "representative_indices": representative_indices,
        "merged_pairs": merged_pairs,
        "energy_scores": energy_scores,
    }


def build_similarity_matrix(
    embeddings: list[list[float]],
    *,
    device: str | None = None,
) -> list[list[float]]:
    torch_matrix = _build_similarity_matrix_torch(embeddings, device=device)
    if torch_matrix is not None:
        return torch_matrix
    numpy_matrix = _build_similarity_matrix_numpy(embeddings)
    if numpy_matrix is not None:
        return numpy_matrix
    return _build_similarity_matrix_python(embeddings)


def _build_similarity_matrix_torch(
    embeddings: list[list[float]],
    *,
    device: str | None,
) -> list[list[float]] | None:
    torch_module = _load_torch_for_device(device)
    if torch_module is None:
        return None
    if not embeddings:
        return []
    dimension = len(embeddings[0])
    if any(len(item) != dimension for item in embeddings):
        return None

    torch_device = _resolve_torch_device(torch_module, device)
    with torch_module.inference_mode():
        matrix = torch_module.tensor(embeddings, dtype=torch_module.float32, device=torch_device)
        if matrix.ndim != 2:
            return None
        norms = matrix.norm(p=2, dim=1, keepdim=True)
        normalized = matrix / norms.clamp_min(1e-12)
        zero_norm_rows = norms.squeeze(1) == 0
        if bool(zero_norm_rows.any()) and dimension > 0:
            normalized[zero_norm_rows] = 1.0 / math.sqrt(dimension)
        return (normalized @ normalized.T).detach().cpu().tolist()


def _build_similarity_matrix_numpy(embeddings: list[list[float]]) -> list[list[float]] | None:
    try:
        import numpy as np
    except ImportError:
        return None

    if not embeddings:
        return []
    dimension = len(embeddings[0])
    if any(len(item) != dimension for item in embeddings):
        return None

    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        return None

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.maximum(norms, 1e-12)
    zero_norm_rows = norms[:, 0] == 0
    if zero_norm_rows.any() and dimension > 0:
        normalized[zero_norm_rows] = 1.0 / math.sqrt(dimension)
    return (normalized @ normalized.T).tolist()


def _build_similarity_matrix_python(embeddings: list[list[float]]) -> list[list[float]]:
    normalized = [_normalize_embedding(item) for item in embeddings]
    matrix: list[list[float]] = []
    for left in normalized:
        row = []
        for right in normalized:
            row.append(
                sum(
                    left_item * right_item
                    for left_item, right_item in zip(left, right, strict=True)
                )
            )
        matrix.append(row)
    return matrix


def compute_energy_scores(
    similarity_matrix: list[list[float]],
    *,
    device: str | None = None,
) -> list[float]:
    torch_scores = _compute_energy_scores_torch(similarity_matrix, device=device)
    if torch_scores is not None:
        return torch_scores
    return _compute_energy_scores_python(similarity_matrix)


def _compute_energy_scores_torch(
    similarity_matrix: list[list[float]],
    *,
    device: str | None,
) -> list[float] | None:
    torch_module = _load_torch_for_device(device)
    if torch_module is None:
        return None
    if not similarity_matrix:
        return []

    torch_device = _resolve_torch_device(torch_module, device)
    with torch_module.inference_mode():
        matrix = torch_module.tensor(
            similarity_matrix,
            dtype=torch_module.float32,
            device=torch_device,
        )
        item_count = matrix.shape[0]
        indices = torch_module.arange(item_count, device=torch_device)
        distances = (indices[:, None] - indices[None, :]).abs().float()
        weights = 1.0 / (1.0 + distances)
        weights.fill_diagonal_(0.0)
        weighted_totals = (matrix * weights).sum(dim=1)
        weight_sums = weights.sum(dim=1).clamp_min(1e-12)
        scores = weighted_totals / weight_sums
        scores = torch_module.where(weight_sums > 0, scores, torch_module.zeros_like(scores))
        return scores.detach().cpu().tolist()


def _compute_energy_scores_python(similarity_matrix: list[list[float]]) -> list[float]:
    if not similarity_matrix:
        return []

    energy_scores: list[float] = []
    for index, row in enumerate(similarity_matrix):
        total = 0.0
        weight_sum = 0.0
        for other_index, similarity in enumerate(row):
            if index == other_index:
                continue
            weight = 1.0 / (1.0 + abs(index - other_index))
            total += similarity * weight
            weight_sum += weight
        energy_scores.append(total / weight_sum if weight_sum else 0.0)
    return energy_scores


def load_frame_embeddings(
    frame_paths: list[Path],
    *,
    embedding_size: int = 16,
    backend: FrameEmbeddingBackend = "pixel",
    device: str | None = None,
) -> list[list[float]]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "PiToMe frame selection requires Pillow to load extracted frame images."
        ) from exc

    if embedding_size <= 0:
        raise ValueError(f"embedding_size must be positive, got {embedding_size}")
    if backend not in {"pixel", "hybrid"}:
        raise ValueError(f"Unsupported PiToMe embedding backend: {backend}")

    torch_embeddings = _load_frame_embeddings_torch(
        frame_paths,
        embedding_size=embedding_size,
        backend=backend,
        device=device,
    )
    if torch_embeddings is not None:
        return torch_embeddings

    embeddings: list[list[float]] = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as image:
            if backend == "pixel":
                embeddings.append(_pixel_embedding(image, embedding_size))
            else:
                embeddings.append(_hybrid_embedding(image, embedding_size))
    return embeddings


def _load_frame_embeddings_torch(
    frame_paths: list[Path],
    *,
    embedding_size: int,
    backend: FrameEmbeddingBackend,
    device: str | None,
) -> list[list[float]] | None:
    torch_module = _load_torch_for_device(device)
    if torch_module is None:
        return None

    try:
        import numpy as np
        from PIL import Image
        from torch.nn import functional as torch_functional
    except ImportError:
        if device:
            raise
        return None

    torch_device = _resolve_torch_device(torch_module, device)
    embeddings: list[list[float]] = []
    for start in range(0, len(frame_paths), FRAME_EMBEDDING_TORCH_BATCH_SIZE):
        batch_paths = frame_paths[start : start + FRAME_EMBEDDING_TORCH_BATCH_SIZE]
        if not batch_paths:
            continue

        pixel_arrays = []
        histogram_arrays = []
        edge_arrays = []
        edge_size = max(4, min(16, embedding_size))
        for frame_path in batch_paths:
            with Image.open(frame_path) as image:
                rgb = image.convert("RGB")
                pixel_arrays.append(
                    np.asarray(
                        rgb.resize((embedding_size, embedding_size)),
                        dtype=np.float32,
                    )
                )
                if backend == "hybrid":
                    histogram_arrays.append(
                        np.asarray(rgb.resize((128, 128)), dtype=np.int64)
                    )
                    edge_arrays.append(
                        np.asarray(
                            rgb.convert("L").resize((edge_size + 1, edge_size + 1)),
                            dtype=np.float32,
                        )
                    )

        with torch_module.inference_mode():
            pixels = torch_module.from_numpy(np.stack(pixel_arrays)).to(torch_device) / 255.0
            features = pixels.reshape(len(batch_paths), -1)
            if backend == "hybrid":
                features = features * 0.75
                histograms = torch_module.from_numpy(np.stack(histogram_arrays)).to(torch_device)
                histogram_features = []
                for channel_index in range(3):
                    buckets = ((histograms[..., channel_index] * 16) // 256).clamp(0, 15)
                    one_hot = torch_functional.one_hot(buckets, num_classes=16)
                    counts = one_hot.sum(dim=(1, 2)).float()
                    histogram_features.append(counts / float(128 * 128))
                histogram_feature = torch_module.cat(histogram_features, dim=1) * 2.0

                edges = torch_module.from_numpy(np.stack(edge_arrays)).to(torch_device)
                current = edges[:, :edge_size, :edge_size]
                right = edges[:, :edge_size, 1 : edge_size + 1]
                down = edges[:, 1 : edge_size + 1, :edge_size]
                edge_feature = ((current - right).abs() + (current - down).abs()) / 510.0
                edge_feature = edge_feature.reshape(len(batch_paths), -1)
                features = torch_module.cat([features, histogram_feature, edge_feature], dim=1)
            embeddings.extend(features.detach().cpu().tolist())
    return embeddings


def compact_frame_embedding(
    embedding: list[float],
    *,
    output_size: int = DEFAULT_STORED_FRAME_EMBEDDING_SIZE,
) -> list[float]:
    if output_size <= 0:
        raise ValueError(f"output_size must be positive, got {output_size}")
    if not embedding:
        return []
    if len(embedding) <= output_size:
        compact = list(embedding)
    else:
        compact = []
        length = len(embedding)
        for index in range(output_size):
            start = round(index * length / output_size)
            end = round((index + 1) * length / output_size)
            chunk = embedding[start : max(end, start + 1)]
            compact.append(sum(chunk) / len(chunk))
    return [round(value, 6) for value in _normalize_embedding(compact)]


def fuse_frame_embeddings_with_semantic(
    frame_embeddings: list[list[float]],
    semantic_embeddings: list[list[float]],
    *,
    frame_weight: float = 0.75,
    semantic_weight: float = 1.0,
    output_size: int = DEFAULT_STORED_FRAME_EMBEDDING_SIZE,
) -> list[list[float]]:
    if len(frame_embeddings) != len(semantic_embeddings):
        raise ValueError(
            "frame_embeddings and semantic_embeddings must have the same length, "
            f"got {len(frame_embeddings)} and {len(semantic_embeddings)}"
        )
    fused: list[list[float]] = []
    for frame_embedding, semantic_embedding in zip(
        frame_embeddings,
        semantic_embeddings,
        strict=True,
    ):
        if not frame_embedding or not semantic_embedding:
            fused.append(compact_frame_embedding(frame_embedding, output_size=output_size))
            continue
        frame_part = [value * frame_weight for value in compact_frame_embedding(frame_embedding)]
        semantic_part = [
            value * semantic_weight for value in compact_frame_embedding(semantic_embedding)
        ]
        fused.append(
            compact_frame_embedding(
                [*frame_part, *semantic_part],
                output_size=output_size,
            )
        )
    return fused


def _load_torch_for_device(device: str | None):
    if device is None or not str(device).strip():
        return None
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PiToMe GPU acceleration requires PyTorch. Install torch or omit "
            "--pitome-embedding-device."
        ) from exc
    return torch


def _resolve_torch_device(torch_module: Any, device: str | None):
    if device is None or not str(device).strip():
        raise ValueError("device must be provided for Torch PiToMe acceleration")
    value = str(device).strip()
    if value == "auto":
        if torch_module.cuda.is_available():
            return torch_module.device("cuda:0")
        if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
            return torch_module.device("mps")
        return torch_module.device("cpu")

    torch_device = torch_module.device(value)
    if torch_device.type == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError(
            f"PiToMe embedding device {value!r} requested but CUDA is not available."
        )
    if (
        torch_device.type == "mps"
        and hasattr(torch_module.backends, "mps")
        and not torch_module.backends.mps.is_available()
    ):
        raise RuntimeError(f"PiToMe embedding device {value!r} requested but MPS is not available.")
    return torch_device


def _pixel_embedding(image: Any, embedding_size: int) -> list[float]:
    resized = image.convert("RGB").resize((embedding_size, embedding_size))
    values: list[float] = []
    for pixel in resized.getdata():
        values.extend(channel / 255.0 for channel in pixel)
    return values


def _hybrid_embedding(image: Any, embedding_size: int) -> list[float]:
    rgb = image.convert("RGB")
    values = [value * 0.75 for value in _pixel_embedding(rgb, embedding_size)]
    values.extend(value * 2.0 for value in _color_histogram_embedding(rgb))
    values.extend(_edge_embedding(rgb, embedding_size))
    return values


def _color_histogram_embedding(image: Any, bins: int = 16) -> list[float]:
    small = image.resize((128, 128))
    channel_counts = [[0.0 for _ in range(bins)] for _ in range(3)]
    pixel_count = 0
    for pixel in small.getdata():
        pixel_count += 1
        for channel_index, channel in enumerate(pixel):
            bucket = min(bins - 1, int(channel * bins / 256))
            channel_counts[channel_index][bucket] += 1.0
    scale = 1.0 / max(pixel_count, 1)
    return [count * scale for channel in channel_counts for count in channel]


def _edge_embedding(image: Any, embedding_size: int) -> list[float]:
    edge_size = max(4, min(16, embedding_size))
    grayscale = image.convert("L").resize((edge_size + 1, edge_size + 1))
    pixels = list(grayscale.getdata())

    def value_at(x: int, y: int) -> int:
        return pixels[y * (edge_size + 1) + x]

    values: list[float] = []
    for y in range(edge_size):
        for x in range(edge_size):
            current = value_at(x, y)
            right = value_at(x + 1, y)
            down = value_at(x, y + 1)
            values.append((abs(current - right) + abs(current - down)) / 510.0)
    return values


def _normalize_embedding(embedding: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in embedding))
    if norm == 0:
        if not embedding:
            return []
        scale = 1.0 / math.sqrt(len(embedding))
        return [scale] * len(embedding)
    return [value / norm for value in embedding]


def _uniform_anchor_indices(item_count: int, anchor_count: int) -> list[int]:
    if item_count <= 0 or anchor_count <= 0:
        return []
    if anchor_count >= item_count:
        return list(range(item_count))
    if anchor_count == 1:
        return [item_count // 2]
    return sorted(
        {
            round(position * (item_count - 1) / (anchor_count - 1))
            for position in range(anchor_count)
        }
    )


def _temporal_coverage_indices(item_count: int, max_count: int) -> list[int]:
    if max_count <= 0:
        raise ValueError(f"max_count must be positive, got {max_count}")
    if item_count <= max_count:
        return list(range(item_count))
    if max_count == 1:
        return [item_count // 2]
    return sorted(
        {round(position * (item_count - 1) / (max_count - 1)) for position in range(max_count)}
    )


def _validate_ratio(value: float, *, name: str) -> None:
    if value < 0 or value > 1:
        raise ValueError(f"{name} must be within [0, 1], got {value}")
