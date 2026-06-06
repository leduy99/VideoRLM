import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from rlm.video.adapters import VideoWindowEmbeddingProvider
from rlm.video.types import FrontierItem, TimeSpan, VideoMemory


@dataclass
class VideoWindowReranker:
    embedding_provider: VideoWindowEmbeddingProvider
    candidate_count: int = 24
    stage2_weight: float = 0.75
    window_seconds: float | None = None
    min_stage2_score: float | None = None
    adaptive_stage2_weight: bool = True

    def __post_init__(self) -> None:
        if self.candidate_count <= 0:
            raise ValueError(f"candidate_count must be positive, got {self.candidate_count}")
        if not 0.0 <= self.stage2_weight <= 1.0:
            raise ValueError(f"stage2_weight must be in [0, 1], got {self.stage2_weight}")
        if self.window_seconds is not None and self.window_seconds <= 0:
            raise ValueError(f"window_seconds must be positive, got {self.window_seconds}")

    def rerank(
        self,
        *,
        query: str,
        candidates: list[FrontierItem],
        memory: VideoMemory,
        top_k: int,
    ) -> tuple[list[FrontierItem], dict[str, Any]]:
        if top_k <= 0:
            return [], {"stage2_rerank_applied": False, "stage2_rerank_reason": "top_k<=0"}
        if not candidates:
            return [], {"stage2_rerank_applied": False, "stage2_rerank_reason": "no_candidates"}

        effective_weight = self._effective_stage2_weight(query)
        visual_candidates = [
            item for item in candidates if self._is_visual_candidate(item, query=query)
        ]
        passthrough = [item for item in candidates if item not in visual_candidates]
        if not visual_candidates:
            return candidates[:top_k], {
                "stage2_rerank_applied": False,
                "stage2_rerank_reason": "no_visual_candidates",
            }

        source_video_path = self._source_video_path(memory, visual_candidates)
        query_embedding = self.embedding_provider.embed_text(query)
        if not query_embedding:
            raise ValueError("Video window reranker text embedding provider returned no embedding.")

        rerank_candidates = visual_candidates[: self.candidate_count]
        window_records = [
            self._candidate_window(item, memory, query=query) for item in rerank_candidates
        ]
        windows = [record[0] for record in window_records]
        video_embeddings = self.embedding_provider.embed_video_windows(source_video_path, windows)
        if len(video_embeddings) != len(windows):
            raise ValueError(
                "Video window embedding provider returned "
                f"{len(video_embeddings)} embeddings for {len(windows)} windows."
            )

        max_stage1 = max((max(0.0, item.score) for item in rerank_candidates), default=0.0)
        max_stage1 = max(max_stage1, 1e-12)
        reranked: list[tuple[FrontierItem, float]] = []
        for item, (window, window_reason), video_embedding in zip(
            rerank_candidates,
            window_records,
            video_embeddings,
            strict=True,
        ):
            stage2_score = _cosine_similarity(query_embedding, video_embedding)
            stage1_score = max(0.0, item.score) / max_stage1
            combined_score = (
                ((1.0 - effective_weight) * stage1_score)
                + (effective_weight * stage2_score)
            )
            reranked.append(
                (
                    self._reranked_item(
                        item=item,
                        combined_score=combined_score,
                        stage2_score=stage2_score,
                        window=window,
                        window_reason=window_reason,
                    ),
                    stage2_score,
                )
            )

        filtered, floor_relaxed = self._apply_min_stage2_score(reranked)
        filtered.sort(
            key=lambda item: (
                -item[0].score,
                item[0].time_span.start,
                item[0].node_id,
            )
        )
        reranked_node_ids = {ranked_item.node_id for ranked_item, _score in filtered}
        remainder = [
            replace(item)
            for item in visual_candidates[self.candidate_count :] + passthrough
            if item.node_id not in reranked_node_ids
        ]
        output = [item for item, _score in filtered] + remainder
        return output[:top_k], {
            "stage2_rerank_applied": True,
            "stage2_rerank_model": getattr(
                self.embedding_provider,
                "model_name",
                self.embedding_provider.__class__.__name__,
            ),
            "stage2_rerank_candidate_count": len(rerank_candidates),
            "stage2_rerank_window_count": len(windows),
            "stage2_rerank_query": query,
            "stage2_rerank_weight": effective_weight,
            "stage2_rerank_base_weight": self.stage2_weight,
            "stage2_rerank_adaptive_weight": self.adaptive_stage2_weight,
            "stage2_rerank_window_seconds": self.window_seconds,
            "stage2_rerank_min_score": self.min_stage2_score,
            "stage2_rerank_floor_relaxed": floor_relaxed,
        }

    def _apply_min_stage2_score(
        self,
        reranked: list[tuple[FrontierItem, float]],
    ) -> tuple[list[tuple[FrontierItem, float]], bool]:
        if self.min_stage2_score is None:
            return reranked, False
        filtered = [item for item in reranked if item[1] >= self.min_stage2_score]
        if filtered:
            return filtered, False
        return reranked, True

    def _reranked_item(
        self,
        *,
        item: FrontierItem,
        combined_score: float,
        stage2_score: float,
        window: TimeSpan,
        window_reason: str,
    ) -> FrontierItem:
        reason = (
            f"{item.why_candidate}; stage2_video_window_rerank="
            f"{stage2_score:.4f}; stage2_window={window.to_display()}"
        )
        if window_reason:
            reason += f"; stage2_window_reason={window_reason}"
        return replace(item, score=round(max(0.0, combined_score), 4), why_candidate=reason)

    def _candidate_window(
        self,
        item: FrontierItem,
        memory: VideoMemory,
        *,
        query: str,
    ) -> tuple[TimeSpan, str]:
        event_chain_window = self._event_chain_window(item, memory, query)
        if event_chain_window is not None:
            return event_chain_window, "event-chain"
        if self.window_seconds is None or item.time_span.duration >= self.window_seconds:
            return item.time_span, ""
        duration = float(memory.metadata.get("duration_seconds") or item.time_span.end)
        half = self.window_seconds / 2.0
        center = (item.time_span.start + item.time_span.end) / 2.0
        start = max(0.0, center - half)
        end = min(duration, center + half)
        if end - start < self.window_seconds and start <= 0.0:
            end = min(duration, self.window_seconds)
        if end - start < self.window_seconds and end >= duration:
            start = max(0.0, duration - self.window_seconds)
        return TimeSpan(start, max(start, end)), "fixed-window"

    def _event_chain_window(
        self,
        item: FrontierItem,
        memory: VideoMemory,
        query: str,
    ) -> TimeSpan | None:
        if item.node_id not in memory.nodes:
            return None
        node = memory.get_node(item.node_id)
        if node.level != "event" and not _query_needs_event_chain(query):
            return None
        node_ids = [node.node_id]
        for key in ("previous_cognitive_event_id", "next_cognitive_event_id"):
            neighbor_id = node.metadata.get(key)
            if isinstance(neighbor_id, str) and neighbor_id in memory.nodes:
                node_ids.append(neighbor_id)
        if len(node_ids) <= 1:
            return None
        nodes = [memory.get_node(node_id) for node_id in dict.fromkeys(node_ids)]
        return TimeSpan(
            min(candidate.time_span.start for candidate in nodes),
            max(candidate.time_span.end for candidate in nodes),
        )

    def _source_video_path(
        self,
        memory: VideoMemory,
        candidates: list[FrontierItem],
    ) -> Path:
        source = memory.metadata.get("source_video_path")
        if source:
            return Path(str(source))
        for item in candidates:
            node = memory.get_node(item.node_id)
            if not node.clip_path:
                continue
            return Path(str(node.clip_path).split("#t=", maxsplit=1)[0])
        raise ValueError(
            "Video window reranking requires memory.metadata['source_video_path'] "
            "or candidate node clip_path values."
        )

    def _is_visual_candidate(self, item: FrontierItem, *, query: str) -> bool:
        query_tokens = _tokenize(query)
        if (
            "ocr" in item.recommended_modalities
            and "visual" not in item.recommended_modalities
            and _query_is_ocr_or_speech(query_tokens)
        ):
            return False
        if "visual" in item.recommended_modalities:
            return True
        if "cross_modal" in item.recommended_modalities:
            return True
        return "ocr" in item.recommended_modalities

    def _effective_stage2_weight(self, query: str) -> float:
        if not self.adaptive_stage2_weight:
            return self.stage2_weight
        query_tokens = _tokenize(query)
        visualness = 1.0 if _has_any(query_tokens, VISUAL_TERMS) else 0.0
        motionness = 1.0 if _has_any(query_tokens, MOTION_TERMS) else 0.0
        speechness = 1.0 if _has_any(query_tokens, SPEECH_TERMS) else 0.0
        ocrness = 1.0 if _has_any(query_tokens, OCR_TERMS) else 0.0
        causalness = 1.0 if _has_any(query_tokens, CAUSAL_TERMS) else 0.0
        temporalness = 1.0 if _has_any(query_tokens, TEMPORAL_TERMS) else 0.0
        adaptive = _clamp(
            0.55
            + (0.2 * visualness)
            + (0.1 * motionness)
            + (0.05 * causalness)
            + (0.05 * temporalness)
            - (0.25 * speechness)
            - (0.2 * ocrness),
            0.2,
            0.85,
        )
        if ocrness:
            adaptive = _clamp(adaptive, 0.25, 0.45)
        elif speechness:
            adaptive = _clamp(adaptive, 0.15, 0.35)
        elif causalness:
            adaptive = _clamp(adaptive, 0.50, 0.65)
        elif temporalness:
            adaptive = _clamp(adaptive, 0.55, 0.70)
        elif visualness and motionness:
            adaptive = _clamp(adaptive, 0.75, 0.85)
        elif visualness:
            adaptive = _clamp(adaptive, 0.65, 0.80)
        return round(_clamp(adaptive, 0.0, 1.0), 4)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


VISUAL_TERMS = {
    "action",
    "appear",
    "display",
    "frame",
    "image",
    "look",
    "object",
    "person",
    "see",
    "show",
    "visible",
    "visual",
}
MOTION_TERMS = {
    "after",
    "before",
    "carry",
    "enter",
    "follow",
    "move",
    "moving",
    "open",
    "pour",
    "put",
    "reach",
    "walk",
}
SPEECH_TERMS = {"explain", "mention", "narrator", "say", "said", "speaker", "talk", "tell"}
OCR_TERMS = {"label", "read", "screen", "sign", "subtitle", "text", "title", "written"}
CAUSAL_TERMS = {"because", "cause", "effect", "imply", "why"}
TEMPORAL_TERMS = {"after", "before", "earlier", "first", "last", "later", "next", "previous"}


def _tokenize(text: str) -> set[str]:
    return {match.group(0).lower() for match in re.finditer(r"\b\w+\b", text)}


def _has_any(tokens: set[str], terms: set[str]) -> bool:
    return bool(tokens & terms)


def _query_is_ocr_or_speech(tokens: set[str]) -> bool:
    return _has_any(tokens, OCR_TERMS) or _has_any(tokens, SPEECH_TERMS)


def _query_needs_event_chain(query: str) -> bool:
    tokens = _tokenize(query)
    return _has_any(tokens, TEMPORAL_TERMS | CAUSAL_TERMS)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
