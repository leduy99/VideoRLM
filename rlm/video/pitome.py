from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from rlm.video.media import (
    extract_frames_for_span,
    extract_frames_for_timestamps,
    sample_span_timestamps,
    sample_span_timestamps_by_rate,
)
from rlm.video.types import TimeSpan

FrameSelectionStrategy = Literal["uniform", "pitome"]
FrameEmbeddingBackend = Literal["pixel", "hybrid"]


@dataclass
class FrameSelectionResult:
    strategy: FrameSelectionStrategy
    frame_paths: list[Path]
    timestamps: list[float]
    dense_frame_count: int
    protected_timestamps: list[float] = field(default_factory=list)
    representative_timestamps: list[float] = field(default_factory=list)
    energy_scores: list[float] = field(default_factory=list)
    merged_pairs: list[tuple[float, float]] = field(default_factory=list)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "dense_frame_count": self.dense_frame_count,
            "selected_frame_count": len(self.timestamps),
            "protected_timestamps": list(self.protected_timestamps),
            "representative_timestamps": list(self.representative_timestamps),
            "merged_pairs": [
                {"left": left, "right": right}
                for left, right in self.merged_pairs
            ],
        }


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
    )
    embeddings = load_frame_embeddings(
        dense_frame_paths,
        embedding_size=embedding_size,
        backend=embedding_backend,
    )
    selection = select_frame_indices_from_embeddings(
        embeddings,
        protect_ratio=protect_ratio,
        similarity_threshold=similarity_threshold,
    )

    selected_indices = selection["selected_indices"]
    selected_timestamps = [dense_timestamps[index] for index in selected_indices]
    selected_frame_paths = [dense_frame_paths[index] for index in selected_indices]
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

    if len(embeddings) == 1:
        return {
            "selected_indices": [0],
            "protected_indices": [0],
            "representative_indices": [],
            "merged_pairs": [],
            "energy_scores": [0.0],
        }

    similarity_matrix = build_similarity_matrix(embeddings)
    energy_scores = compute_energy_scores(similarity_matrix)
    protected_count = max(1, math.ceil(len(embeddings) * protect_ratio))
    energy_order = sorted(range(len(embeddings)), key=lambda index: energy_scores[index])
    protected_indices = sorted(energy_order[:protected_count])

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
        representative = (
            index_a if energy_scores[index_a] >= energy_scores[best_b] else best_b
        )
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


def build_similarity_matrix(embeddings: list[list[float]]) -> list[list[float]]:
    normalized = [_normalize_embedding(item) for item in embeddings]
    matrix: list[list[float]] = []
    for left in normalized:
        row = []
        for right in normalized:
            row.append(sum(left_item * right_item for left_item, right_item in zip(left, right, strict=True)))
        matrix.append(row)
    return matrix


def compute_energy_scores(similarity_matrix: list[list[float]]) -> list[float]:
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

    embeddings: list[list[float]] = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as image:
            if backend == "pixel":
                embeddings.append(_pixel_embedding(image, embedding_size))
            else:
                embeddings.append(_hybrid_embedding(image, embedding_size))
    return embeddings


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


def _validate_ratio(value: float, *, name: str) -> None:
    if value < 0 or value > 1:
        raise ValueError(f"{name} must be within [0, 1], got {value}")
