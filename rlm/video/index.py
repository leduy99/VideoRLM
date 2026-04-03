import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from rlm.video.adapters import EmbeddingProvider
from rlm.video.types import FrontierItem, Modality, TimeSpan, VideoMemory, VideoNodeLevel

TOKEN_PATTERN = re.compile(r"\b\w+\b")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "but",
    "by",
    "did",
    "do",
    "does",
    "even",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "much",
    "my",
    "of",
    "on",
    "only",
    "or",
    "our",
    "out",
    "she",
    "so",
    "some",
    "sometimes",
    "that",
    "the",
    "their",
    "them",
    "there",
    "they",
    "this",
    "to",
    "too",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}


@dataclass
class SearchHit:
    node_id: str
    time_span: TimeSpan
    level: VideoNodeLevel
    score: float
    reason: str
    modality: Modality
    matched_terms: list[str]
    score_breakdown: dict[str, float] = field(default_factory=dict)

    def to_frontier_item(self) -> FrontierItem:
        return FrontierItem(
            node_id=self.node_id,
            time_span=self.time_span,
            level=self.level,
            score=self.score,
            why_candidate=self.reason,
            recommended_modalities=[self.modality],
        )


class VideoMemoryIndex:
    def __init__(
        self,
        memory: VideoMemory,
        embedding_provider: EmbeddingProvider | None = None,
        lexical_weight: float = 0.7,
        semantic_weight: float = 0.3,
    ):
        self.memory = memory
        self.embedding_provider = embedding_provider
        self.lexical_weight = lexical_weight
        self.semantic_weight = semantic_weight
        self._embedding_cache: dict[tuple[str, str], list[float]] = {}

    def search(
        self,
        query: str,
        modality: Modality | None = None,
        top_k: int = 5,
        levels: Iterable[VideoNodeLevel] | None = None,
    ) -> list[SearchHit]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        allowed_levels = set(levels) if levels else None
        hits: list[SearchHit] = []

        for node in self.memory.nodes.values():
            if node.level == "video":
                continue
            if allowed_levels and node.level not in allowed_levels:
                continue

            hit = self._score_node(node_id=node.node_id, query=query, modality=modality)
            if hit is None:
                continue
            hits.append(hit)

        hits.sort(key=lambda item: (-item.score, item.time_span.start, item.node_id))
        return hits[:top_k]

    def _score_node(
        self,
        node_id: str,
        query: str,
        modality: Modality | None,
    ) -> SearchHit | None:
        node = self.memory.get_node(node_id)
        query_lower = query.lower()
        query_tokens = self._tokenize(query)

        modalities: list[Modality]
        if modality is None:
            modalities = ["speech", "visual", "ocr", "audio"]
        else:
            modalities = [modality]

        best_hit: SearchHit | None = None
        for current_modality in modalities:
            text = self._node_text(node, current_modality)
            if not text:
                continue

            lexical_score, overlap = self._lexical_score(query_lower, query_tokens, text)
            semantic_score = self._semantic_score(query, text)
            temporal_score = self._temporal_score(query_tokens, node.time_span)
            if lexical_score <= 0 and semantic_score <= 0:
                continue

            score = self._combine_scores(lexical_score, semantic_score, temporal_score)
            reason = self._build_reason(
                modality=current_modality,
                node_id=node.node_id,
                overlap=overlap,
                lexical_score=lexical_score,
                semantic_score=semantic_score,
                temporal_score=temporal_score,
            )
            hit = SearchHit(
                node_id=node.node_id,
                time_span=node.time_span,
                level=node.level,
                score=score,
                reason=reason,
                modality=current_modality,
                matched_terms=overlap,
                score_breakdown={
                    "lexical": lexical_score,
                    "semantic": semantic_score,
                    "temporal": temporal_score,
                    "combined": score,
                },
            )
            if best_hit is None or hit.score > best_hit.score:
                best_hit = hit

        return best_hit

    def _node_text(self, node, modality: Modality) -> str:
        if modality == "speech":
            return " ".join(item.text for item in node.speech_spans)
        if modality == "visual":
            parts = [node.visual_summary, " ".join(node.tags), " ".join(node.entities)]
            return " ".join(part for part in parts if part)
        if modality == "ocr":
            return " ".join(item.text for item in node.ocr_spans)
        if modality == "audio":
            return " ".join(item.label for item in node.audio_events)
        return ""

    def _lexical_score(
        self,
        query_lower: str,
        query_tokens: set[str],
        text: str,
    ) -> tuple[float, list[str]]:
        doc_tokens = self._tokenize(text)
        overlap = sorted(query_tokens & doc_tokens)
        if not overlap:
            return 0.0, []

        overlap_ratio = len(overlap) / len(query_tokens)
        density_bonus = sum(text.lower().count(term) for term in overlap) / max(len(doc_tokens), 1)
        phrase_bonus = 0.25 if query_lower in text.lower() else 0.0
        score = min(1.0, overlap_ratio + density_bonus + phrase_bonus)
        return round(score, 4), overlap

    def _semantic_score(self, query: str, text: str) -> float:
        if self.embedding_provider is None:
            return 0.0
        query_vector = self._embed_cached(("query", query), query)
        text_vector = self._embed_cached(("text", text), text)
        if not query_vector or not text_vector:
            return 0.0
        return round(self._cosine_similarity(query_vector, text_vector), 4)

    def _combine_scores(
        self,
        lexical_score: float,
        semantic_score: float,
        temporal_score: float,
    ) -> float:
        if self.embedding_provider is None:
            return round(lexical_score + temporal_score, 4)
        combined = (self.lexical_weight * lexical_score) + (self.semantic_weight * semantic_score)
        return round(combined + temporal_score, 4)

    def _build_reason(
        self,
        modality: Modality,
        node_id: str,
        overlap: list[str],
        lexical_score: float,
        semantic_score: float,
        temporal_score: float,
    ) -> str:
        parts = []
        if overlap:
            parts.append(f"Matched {modality} terms {', '.join(overlap[:4])}")
        if semantic_score > 0:
            parts.append(f"semantic similarity {semantic_score:.2f}")
        if temporal_score > 0:
            parts.append(f"temporal prior {temporal_score:.2f}")
        if not parts:
            parts.append(f"Matched {modality} content")
        return f"{'; '.join(parts)} in node {node_id}"

    def _temporal_score(self, query_tokens: set[str], span: TimeSpan) -> float:
        duration = float(self.memory.metadata.get("duration_seconds") or 0.0)
        if duration <= 0:
            return 0.0

        if {"first", "beginning", "earliest", "initial"} & query_tokens:
            window = max(duration * 0.35, 1.0)
            score = max(0.0, 1.0 - (span.start / window))
            return round(score * 0.3, 4)

        if {"last", "final", "ending", "end"} & query_tokens:
            window = max(duration * 0.35, 1.0)
            distance_to_end = max(0.0, duration - span.end)
            score = max(0.0, 1.0 - (distance_to_end / window))
            return round(score * 0.3, 4)

        return 0.0

    def _embed_cached(self, cache_key: tuple[str, str], text: str) -> list[float]:
        if cache_key not in self._embedding_cache:
            self._embedding_cache[cache_key] = self.embedding_provider.embed_text(text)
        return self._embedding_cache[cache_key]

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            raise ValueError(
                f"Embedding dimension mismatch: left={len(left)} right={len(right)}"
            )
        left_norm = sum(value * value for value in left) ** 0.5
        right_norm = sum(value * value for value in right) ** 0.5
        if left_norm == 0 or right_norm == 0:
            return 0.0
        dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))
        similarity = dot_product / (left_norm * right_norm)
        return max(0.0, similarity)

    def _tokenize(self, text: str) -> set[str]:
        return {
            token
            for token in (match.group(0).lower() for match in TOKEN_PATTERN.finditer(text))
            if token not in STOPWORDS and len(token) > 1
        }
