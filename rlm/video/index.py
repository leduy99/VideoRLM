import os
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from rlm.video.adapters import EmbeddingProvider, ImageTextEmbeddingProvider
from rlm.video.types import FrontierItem, Modality, TimeSpan, VideoMemory, VideoNodeLevel

TOKEN_PATTERN = re.compile(r"\b\w+\b")
SearchMode = Literal["lexical", "graph"]
GRAPH_MODALITIES = {"visual", "ocr"}
FRAME_SIMILARITY_THRESHOLD = 0.88
SEMANTIC_FRAME_SIMILARITY_THRESHOLD = 0.16
SEMANTIC_FRAME_EDGE_THRESHOLD = 0.78
MAX_FRAME_SIMILARITY_NEIGHBORS = 3
MAX_SEMANTIC_FRAME_SIMILARITY_NEIGHBORS = 3
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
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


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


@dataclass
class FrameVectorRecord:
    node_id: str
    embedding: list[float]


@dataclass
class FrameVectorMatch:
    node_id: str
    score: float


class FrameVectorIndex:
    def __init__(self, records: list[FrameVectorRecord]):
        self.records = [record for record in records if record.embedding]
        self.backend = "scan"
        self.faiss_index: Any | None = None
        self.np: Any | None = None
        if self.records:
            self._try_build_faiss_index()

    def search(self, query_embedding: list[float], top_k: int) -> list[FrameVectorMatch]:
        if not query_embedding or top_k <= 0 or not self.records:
            return []
        if self.backend == "faiss" and self.faiss_index is not None and self.np is not None:
            return self._search_faiss(query_embedding, top_k)
        return self._search_scan(query_embedding, top_k)

    def _try_build_faiss_index(self) -> None:
        try:
            import faiss
            import numpy as np
        except ImportError:
            return

        dimension = len(self.records[0].embedding)
        kept_records = [record for record in self.records if len(record.embedding) == dimension]
        if not kept_records:
            return
        matrix = np.asarray([record.embedding for record in kept_records], dtype="float32")
        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(dimension)
        index.add(matrix)
        self.records = kept_records
        self.faiss_index = index
        self.np = np
        self.backend = "faiss"

    def _search_faiss(self, query_embedding: list[float], top_k: int) -> list[FrameVectorMatch]:
        dimension = len(self.records[0].embedding)
        if len(query_embedding) != dimension:
            return []
        query = self.np.asarray([query_embedding], dtype="float32")
        import faiss

        faiss.normalize_L2(query)
        scores, indices = self.faiss_index.search(query, min(top_k, len(self.records)))
        matches: list[FrameVectorMatch] = []
        for score, index in zip(scores[0], indices[0], strict=True):
            if int(index) < 0:
                continue
            matches.append(
                FrameVectorMatch(
                    node_id=self.records[int(index)].node_id,
                    score=max(0.0, float(score)),
                )
            )
        return matches

    def _search_scan(self, query_embedding: list[float], top_k: int) -> list[FrameVectorMatch]:
        matches = [
            FrameVectorMatch(
                node_id=record.node_id,
                score=_cosine_similarity(query_embedding, record.embedding),
            )
            for record in self.records
            if len(record.embedding) == len(query_embedding)
        ]
        matches.sort(key=lambda item: -item.score)
        return matches[:top_k]


class VideoMemoryIndex:
    def __init__(
        self,
        memory: VideoMemory,
        embedding_provider: EmbeddingProvider | None = None,
        image_text_embedding_provider: ImageTextEmbeddingProvider | None = None,
        lexical_weight: float = 0.7,
        semantic_weight: float = 0.3,
        search_mode: SearchMode = "lexical",
    ):
        if search_mode not in {"lexical", "graph"}:
            raise ValueError(f"Unsupported search mode: {search_mode}")
        self.memory = memory
        self.embedding_provider = embedding_provider
        self.image_text_embedding_provider = image_text_embedding_provider
        self.lexical_weight = lexical_weight
        self.semantic_weight = semantic_weight
        self.search_mode = search_mode
        self._embedding_cache: dict[tuple[str, str], list[float]] = {}
        self._image_text_embedding_cache: dict[str, list[float]] = {}
        self._semantic_frame_vector_index = self._build_semantic_frame_vector_index()

    def search(
        self,
        query: str,
        modality: Modality | None = None,
        top_k: int = 5,
        levels: Iterable[VideoNodeLevel] | None = None,
    ) -> list[SearchHit]:
        if self.search_mode == "graph" and (modality in GRAPH_MODALITIES or modality is None):
            return self.graph_search(
                query=query,
                modality=modality,
                top_k=top_k,
                levels=levels,
            )
        return self.lexical_search(query=query, modality=modality, top_k=top_k, levels=levels)

    def lexical_search(
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

    def graph_search(
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
        seed_hits = self.lexical_search(
            query=query,
            modality=modality,
            top_k=max(top_k * 4, 12),
            levels=levels,
        )

        candidates: dict[str, SearchHit] = {}
        score_sources: dict[str, list[float]] = {}
        source_terms: dict[str, set[str]] = {}
        source_reasons: dict[str, list[str]] = {}
        query_lower = query.lower()

        for seed in seed_hits:
            self._add_graph_candidate(
                candidates=candidates,
                score_sources=score_sources,
                source_terms=source_terms,
                source_reasons=source_reasons,
                node_id=seed.node_id,
                modality=seed.modality,
                score=seed.score + self._graph_level_bias(seed.level, seed.modality),
                matched_terms=seed.matched_terms,
                reason="seed",
            )
            for neighbor_id, edge_weight, edge_reason in self._graph_neighbors(seed.node_id):
                neighbor = self.memory.get_node(neighbor_id)
                if neighbor.level == "video":
                    continue
                if allowed_levels and neighbor.level not in allowed_levels:
                    continue
                neighbor_text = self._node_text(neighbor, seed.modality)
                direct_score, direct_overlap = self._lexical_score(
                    query_lower,
                    query_tokens,
                    neighbor_text,
                )
                graph_score = (seed.score * edge_weight) + (direct_score * 0.45)
                graph_score += self._temporal_score(query_tokens, neighbor.time_span)
                graph_score += self._graph_level_bias(neighbor.level, seed.modality)
                if graph_score <= 0:
                    continue
                self._add_graph_candidate(
                    candidates=candidates,
                    score_sources=score_sources,
                    source_terms=source_terms,
                    source_reasons=source_reasons,
                    node_id=neighbor.node_id,
                    modality=seed.modality,
                    score=round(graph_score, 4),
                    matched_terms=sorted(set(seed.matched_terms) | set(direct_overlap)),
                    reason=edge_reason,
                )

        if seed_hits:
            self._add_frame_similarity_candidates(
                seed_hits=seed_hits,
                query_lower=query_lower,
                query_tokens=query_tokens,
                allowed_levels=allowed_levels,
                candidates=candidates,
                score_sources=score_sources,
                source_terms=source_terms,
                source_reasons=source_reasons,
            )
        self._add_semantic_frame_candidates(
            query=query,
            query_lower=query_lower,
            query_tokens=query_tokens,
            modality=modality,
            top_k=top_k,
            allowed_levels=allowed_levels,
            candidates=candidates,
            score_sources=score_sources,
            source_terms=source_terms,
            source_reasons=source_reasons,
        )

        if not candidates:
            return []

        for node_id, hit in candidates.items():
            scores = score_sources[node_id]
            hit.score = round(max(scores), 4)
            hit.matched_terms = sorted(source_terms[node_id])
            hit.reason = self._build_graph_reason(
                modality=hit.modality,
                node_id=node_id,
                source_reasons=source_reasons[node_id],
                matched_terms=hit.matched_terms,
            )
            hit.score_breakdown = {
                "graph": hit.score,
                "sources": float(len(scores)),
                "mode": 1.0,
            }

        hits = list(candidates.values())
        hits.sort(
            key=lambda item: (
                -item.score,
                self._graph_level_rank(item.level),
                item.time_span.start,
                item.node_id,
            )
        )
        return hits[:top_k]

    def _add_graph_candidate(
        self,
        *,
        candidates: dict[str, SearchHit],
        score_sources: dict[str, list[float]],
        source_terms: dict[str, set[str]],
        source_reasons: dict[str, list[str]],
        node_id: str,
        modality: Modality,
        score: float,
        matched_terms: list[str],
        reason: str,
    ) -> None:
        node = self.memory.get_node(node_id)
        if node_id not in candidates:
            candidates[node_id] = SearchHit(
                node_id=node.node_id,
                time_span=node.time_span,
                level=node.level,
                score=score,
                reason=reason,
                modality=modality,
                matched_terms=list(matched_terms),
            )
            score_sources[node_id] = []
            source_terms[node_id] = set()
            source_reasons[node_id] = []
        score_sources[node_id].append(score)
        source_terms[node_id].update(matched_terms)
        if reason not in source_reasons[node_id]:
            source_reasons[node_id].append(reason)

    def _graph_neighbors(self, node_id: str) -> list[tuple[str, float, str]]:
        node = self.memory.get_node(node_id)
        neighbors: list[tuple[str, float, str]] = []

        for detail_node_id in node.metadata.get("visual_detail_node_ids", []):
            if detail_node_id in self.memory.nodes:
                neighbors.append((str(detail_node_id), 0.95, "visual-detail"))

        child_weight = (
            0.78 if node.metadata.get("visual_summary_mode") == "compact_parent_rollup" else 0.62
        )
        for child_id in node.children:
            neighbors.append((child_id, child_weight, "child"))

        if node.parent_id:
            neighbors.append((node.parent_id, 0.28, "parent"))
            parent = self.memory.get_node(node.parent_id)
            siblings = parent.children
            if node_id in siblings:
                index = siblings.index(node_id)
                for sibling_index in (index - 1, index + 1):
                    if 0 <= sibling_index < len(siblings):
                        neighbors.append((siblings[sibling_index], 0.36, "temporal-sibling"))

        for neighbor_id, similarity in self._frame_similarity_neighbors(node_id):
            edge_weight = 0.22 + (similarity * 0.18)
            neighbors.append((neighbor_id, edge_weight, f"frame-similarity:{similarity:.2f}"))

        for neighbor_id, similarity in self._semantic_frame_similarity_neighbors(node_id):
            edge_weight = 0.3 + (similarity * 0.2)
            neighbors.append((neighbor_id, edge_weight, f"semantic-frame-edge:{similarity:.2f}"))

        return self._dedupe_graph_neighbors(neighbors, node_id)

    def _add_frame_similarity_candidates(
        self,
        *,
        seed_hits: list[SearchHit],
        query_lower: str,
        query_tokens: set[str],
        allowed_levels: set[VideoNodeLevel] | None,
        candidates: dict[str, SearchHit],
        score_sources: dict[str, list[float]],
        source_terms: dict[str, set[str]],
        source_reasons: dict[str, list[str]],
    ) -> None:
        seed_groups: list[tuple[SearchHit, str, list[list[float]]]] = []
        for seed in seed_hits:
            if seed.modality not in GRAPH_MODALITIES:
                continue
            for seed_node_id in self._frame_seed_node_ids(seed.node_id):
                seed_embeddings = self._node_frame_embeddings(seed_node_id)
                if seed_embeddings:
                    seed_groups.append((seed, seed_node_id, seed_embeddings))

        if not seed_groups:
            return

        for node in self.memory.nodes.values():
            if node.level == "video":
                continue
            if allowed_levels and node.level not in allowed_levels:
                continue
            embeddings = self._node_frame_embeddings(node.node_id)
            if not embeddings:
                continue

            best_score = 0.0
            best_terms: set[str] = set()
            best_modality: Modality | None = None
            best_reason = ""
            for seed, seed_node_id, seed_embeddings in seed_groups:
                if node.node_id == seed_node_id:
                    continue
                similarity = self._max_frame_similarity(seed_embeddings, embeddings)
                if similarity < FRAME_SIMILARITY_THRESHOLD:
                    continue
                direct_text = self._node_text(node, seed.modality)
                direct_score, direct_overlap = self._lexical_score(
                    query_lower,
                    query_tokens,
                    direct_text,
                )
                frame_score = min(
                    1.0,
                    max(
                        0.0,
                        (similarity - FRAME_SIMILARITY_THRESHOLD)
                        / (1.0 - FRAME_SIMILARITY_THRESHOLD),
                    ),
                )
                graph_score = (seed.score * 0.22) + (frame_score * 0.62)
                graph_score += direct_score * 0.25
                graph_score += self._temporal_score(query_tokens, node.time_span)
                graph_score += self._graph_level_bias(node.level, seed.modality)
                if graph_score <= best_score:
                    continue
                best_score = graph_score
                best_terms = set(seed.matched_terms) | set(direct_overlap)
                best_modality = seed.modality
                best_reason = f"frame-similarity:{similarity:.2f}"

            if best_modality is None:
                continue
            self._add_graph_candidate(
                candidates=candidates,
                score_sources=score_sources,
                source_terms=source_terms,
                source_reasons=source_reasons,
                node_id=node.node_id,
                modality=best_modality,
                score=round(best_score, 4),
                matched_terms=sorted(best_terms),
                reason=best_reason,
            )

    def _add_semantic_frame_candidates(
        self,
        *,
        query: str,
        query_lower: str,
        query_tokens: set[str],
        modality: Modality | None,
        top_k: int,
        allowed_levels: set[VideoNodeLevel] | None,
        candidates: dict[str, SearchHit],
        score_sources: dict[str, list[float]],
        source_terms: dict[str, set[str]],
        source_reasons: dict[str, list[str]],
    ) -> None:
        if self.image_text_embedding_provider is None:
            return
        if modality not in GRAPH_MODALITIES and modality is not None:
            return

        query_embedding = self._image_text_embed_cached(query)
        if not query_embedding:
            return

        matches: list[tuple[float, str, Modality, list[str], str]] = []
        semantic_matches = self._semantic_frame_matches(
            query_embedding=query_embedding,
            top_k=max(top_k * 8, 24),
            allowed_levels=allowed_levels,
        )
        for node_id, semantic_score in semantic_matches:
            node = self.memory.get_node(node_id)
            if semantic_score < SEMANTIC_FRAME_SIMILARITY_THRESHOLD:
                continue

            visual_text = self._node_text(node, "visual")
            lexical_score, lexical_overlap = self._lexical_score(
                query_lower,
                query_tokens,
                visual_text,
            )
            atom_score, atom_overlap = self._ocr_atom_score(
                query_lower,
                query_tokens,
                node.node_id,
            )
            audio_score, audio_overlap = self._nearby_audio_score(
                query_lower,
                query_tokens,
                node.node_id,
            )
            temporal_score = self._temporal_score(query_tokens, node.time_span)
            score = (
                (semantic_score * 0.55)
                + (lexical_score * 0.2)
                + (atom_score * 0.15)
                + (audio_score * 0.15)
                + temporal_score
                + self._graph_level_bias(node.level, "visual")
            )
            reason_parts = [f"{self._semantic_frame_backend_reason()}:{semantic_score:.2f}"]
            if atom_score > 0:
                reason_parts.append("ocr-atoms")
            if audio_score > 0:
                reason_parts.append("audio-nearby")
            matches.append(
                (
                    round(score, 4),
                    node_id,
                    "visual",
                    sorted(set(lexical_overlap) | set(atom_overlap) | set(audio_overlap)),
                    ", ".join(reason_parts),
                )
            )

        matches.sort(
            key=lambda item: (-item[0], self.memory.get_node(item[1]).time_span.start, item[1])
        )
        for score, node_id, hit_modality, matched_terms, reason in matches[: max(top_k * 4, 12)]:
            if score <= 0:
                continue
            self._add_graph_candidate(
                candidates=candidates,
                score_sources=score_sources,
                source_terms=source_terms,
                source_reasons=source_reasons,
                node_id=node_id,
                modality=hit_modality,
                score=score,
                matched_terms=matched_terms,
                reason=reason,
            )

    def _semantic_frame_matches(
        self,
        *,
        query_embedding: list[float],
        top_k: int,
        allowed_levels: set[VideoNodeLevel] | None,
    ) -> list[tuple[str, float]]:
        if self._semantic_frame_vector_index is None:
            return []
        node_scores: dict[str, float] = {}
        for match in self._semantic_frame_vector_index.search(query_embedding, top_k):
            node = self.memory.get_node(match.node_id)
            if node.level == "video":
                continue
            if allowed_levels and node.level not in allowed_levels:
                continue
            node_scores[match.node_id] = max(node_scores.get(match.node_id, 0.0), match.score)
        return sorted(node_scores.items(), key=lambda item: (-item[1], item[0]))

    def _semantic_frame_backend_reason(self) -> str:
        if self._semantic_frame_vector_index is None:
            return "semantic-frame"
        if self._semantic_frame_vector_index.backend == "faiss":
            return "faiss-semantic-frame"
        return "semantic-frame"

    def _frame_seed_node_ids(self, node_id: str) -> list[str]:
        seed_ids: list[str] = []
        node = self.memory.get_node(node_id)
        if self._node_frame_embeddings(node_id):
            seed_ids.append(node_id)
        for detail_node_id in node.metadata.get("visual_detail_node_ids", []):
            detail_id = str(detail_node_id)
            if detail_id in self.memory.nodes and self._node_frame_embeddings(detail_id):
                seed_ids.append(detail_id)
        if not seed_ids:
            for descendant_id in self._descendant_clip_ids(node_id):
                if self._node_frame_embeddings(descendant_id):
                    seed_ids.append(descendant_id)
        return list(dict.fromkeys(seed_ids))

    def _semantic_frame_seed_node_ids(self, node_id: str) -> list[str]:
        seed_ids: list[str] = []
        node = self.memory.get_node(node_id)
        if self._node_semantic_frame_embeddings(node_id):
            seed_ids.append(node_id)
        for detail_node_id in node.metadata.get("visual_detail_node_ids", []):
            detail_id = str(detail_node_id)
            if detail_id in self.memory.nodes and self._node_semantic_frame_embeddings(detail_id):
                seed_ids.append(detail_id)
        if not seed_ids:
            for descendant_id in self._descendant_clip_ids(node_id):
                if self._node_semantic_frame_embeddings(descendant_id):
                    seed_ids.append(descendant_id)
        return list(dict.fromkeys(seed_ids))

    def _descendant_clip_ids(self, node_id: str) -> list[str]:
        descendants: list[str] = []
        stack = list(self.memory.get_node(node_id).children)
        while stack:
            current_id = stack.pop(0)
            current = self.memory.get_node(current_id)
            if current.level == "clip":
                descendants.append(current_id)
            stack.extend(current.children)
        return descendants

    def _frame_similarity_neighbors(
        self,
        node_id: str,
        *,
        limit: int = MAX_FRAME_SIMILARITY_NEIGHBORS,
    ) -> list[tuple[str, float]]:
        source_embeddings = self._node_frame_embeddings(node_id)
        if not source_embeddings:
            return []
        matches: list[tuple[str, float]] = []
        for candidate in self.memory.nodes.values():
            if candidate.node_id == node_id or candidate.level == "video":
                continue
            candidate_embeddings = self._node_frame_embeddings(candidate.node_id)
            if not candidate_embeddings:
                continue
            similarity = self._max_frame_similarity(source_embeddings, candidate_embeddings)
            if similarity >= FRAME_SIMILARITY_THRESHOLD:
                matches.append((candidate.node_id, similarity))
        matches.sort(
            key=lambda item: (-item[1], self.memory.get_node(item[0]).time_span.start, item[0])
        )
        return matches[:limit]

    def _semantic_frame_similarity_neighbors(
        self,
        node_id: str,
        *,
        limit: int = MAX_SEMANTIC_FRAME_SIMILARITY_NEIGHBORS,
    ) -> list[tuple[str, float]]:
        source_embeddings: list[list[float]] = []
        for seed_node_id in self._semantic_frame_seed_node_ids(node_id):
            source_embeddings.extend(self._node_semantic_frame_embeddings(seed_node_id))
        if not source_embeddings:
            return []

        source_node_ids = set(self._semantic_frame_seed_node_ids(node_id))
        if (
            self._semantic_frame_vector_index is not None
            and self._semantic_frame_vector_index.backend == "faiss"
        ):
            matches_by_node: dict[str, float] = {}
            for embedding in source_embeddings:
                for match in self._semantic_frame_vector_index.search(embedding, limit * 8):
                    if match.node_id == node_id or match.node_id in source_node_ids:
                        continue
                    candidate = self.memory.get_node(match.node_id)
                    if candidate.level == "video":
                        continue
                    if match.score >= SEMANTIC_FRAME_EDGE_THRESHOLD:
                        matches_by_node[match.node_id] = max(
                            matches_by_node.get(match.node_id, 0.0),
                            match.score,
                        )
            matches = list(matches_by_node.items())
            matches.sort(
                key=lambda item: (
                    -item[1],
                    self.memory.get_node(item[0]).time_span.start,
                    item[0],
                )
            )
            return matches[:limit]

        matches: list[tuple[str, float]] = []
        for candidate in self.memory.nodes.values():
            if candidate.node_id == node_id or candidate.node_id in source_node_ids:
                continue
            if candidate.level == "video":
                continue
            candidate_embeddings = self._node_semantic_frame_embeddings(candidate.node_id)
            if not candidate_embeddings:
                continue
            similarity = self._max_frame_similarity(source_embeddings, candidate_embeddings)
            if similarity >= SEMANTIC_FRAME_EDGE_THRESHOLD:
                matches.append((candidate.node_id, similarity))
        matches.sort(
            key=lambda item: (-item[1], self.memory.get_node(item[0]).time_span.start, item[0])
        )
        return matches[:limit]

    def _node_frame_embeddings(self, node_id: str) -> list[list[float]]:
        node = self.memory.get_node(node_id)
        raw_embeddings = node.metadata.get("pitome_frame_embeddings", [])
        if not isinstance(raw_embeddings, list):
            return []
        embeddings: list[list[float]] = []
        for raw_embedding in raw_embeddings:
            if not isinstance(raw_embedding, (list, tuple)):
                continue
            embeddings.append([float(value) for value in raw_embedding])
        return embeddings

    def _node_semantic_frame_embeddings(self, node_id: str) -> list[list[float]]:
        node = self.memory.get_node(node_id)
        raw_embeddings = node.metadata.get("semantic_frame_embeddings", [])
        if not isinstance(raw_embeddings, list):
            return []
        embeddings: list[list[float]] = []
        for raw_embedding in raw_embeddings:
            if not isinstance(raw_embedding, (list, tuple)):
                continue
            embeddings.append([float(value) for value in raw_embedding])
        return embeddings

    def _build_semantic_frame_vector_index(self) -> FrameVectorIndex | None:
        records: list[FrameVectorRecord] = []
        for node in self.memory.nodes.values():
            if node.level == "video":
                continue
            for embedding in self._node_semantic_frame_embeddings(node.node_id):
                records.append(FrameVectorRecord(node_id=node.node_id, embedding=embedding))
        if not records:
            return None
        return FrameVectorIndex(records)

    def _max_frame_similarity(
        self,
        left_embeddings: list[list[float]],
        right_embeddings: list[list[float]],
    ) -> float:
        best = 0.0
        for left in left_embeddings:
            for right in right_embeddings:
                if len(left) != len(right):
                    continue
                best = max(best, self._cosine_similarity(left, right))
        return round(best, 4)

    def _ocr_atom_score(
        self,
        query_lower: str,
        query_tokens: set[str],
        node_id: str,
    ) -> tuple[float, list[str]]:
        node = self.memory.get_node(node_id)
        text = " ".join(
            [
                " ".join(item.text for item in node.ocr_spans),
                " ".join(str(item) for item in node.metadata.get("visual_atoms", [])),
                " ".join(str(item) for item in node.metadata.get("visual_keywords", [])),
            ]
        )
        return self._lexical_score(query_lower, query_tokens, text)

    def _nearby_audio_score(
        self,
        query_lower: str,
        query_tokens: set[str],
        node_id: str,
    ) -> tuple[float, list[str]]:
        return self._lexical_score(query_lower, query_tokens, self._nearby_speech_text(node_id))

    def _nearby_speech_text(self, node_id: str) -> str:
        node = self.memory.get_node(node_id)
        clip_duration = float(self.memory.metadata.get("clip_duration_seconds") or 0.0)
        window_seconds = max(8.0, clip_duration)
        start = node.time_span.start - window_seconds
        end = node.time_span.end + window_seconds
        texts: list[str] = []
        seen: set[tuple[float, float, str]] = set()
        for candidate in self.memory.nodes.values():
            for span in candidate.speech_spans:
                if span.time_span.end < start or span.time_span.start > end:
                    continue
                key = (
                    round(span.time_span.start, 3),
                    round(span.time_span.end, 3),
                    span.text,
                )
                if key in seen:
                    continue
                seen.add(key)
                texts.append(span.text)
        return " ".join(texts)

    def _dedupe_graph_neighbors(
        self,
        neighbors: list[tuple[str, float, str]],
        node_id: str,
    ) -> list[tuple[str, float, str]]:
        best: dict[str, tuple[float, str]] = {}
        for neighbor_id, weight, reason in neighbors:
            if neighbor_id == node_id:
                continue
            current = best.get(neighbor_id)
            if current is None or weight > current[0]:
                best[neighbor_id] = (weight, reason)
        return [(neighbor_id, weight, reason) for neighbor_id, (weight, reason) in best.items()]

    def _graph_level_bias(self, level: VideoNodeLevel, modality: Modality) -> float:
        if modality in GRAPH_MODALITIES:
            return {
                "clip": 0.18,
                "segment": 0.06,
                "scene": -0.04,
                "video": -0.1,
            }.get(level, 0.0)
        return 0.0

    def _graph_level_rank(self, level: VideoNodeLevel) -> int:
        return {
            "clip": 0,
            "segment": 1,
            "scene": 2,
            "video": 3,
        }.get(level, 4)

    def _build_graph_reason(
        self,
        *,
        modality: Modality,
        node_id: str,
        source_reasons: list[str],
        matched_terms: list[str],
    ) -> str:
        parts = [f"Graph {modality} search"]
        if matched_terms:
            parts.append(f"matched terms {', '.join(matched_terms[:4])}")
        if source_reasons:
            parts.append(f"edges {', '.join(source_reasons[:3])}")
        return f"{'; '.join(parts)} in node {node_id}"

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
            parts = [
                node.visual_summary,
                " ".join(node.tags),
                " ".join(node.entities),
                " ".join(str(item) for item in node.metadata.get("visual_keywords", [])),
                " ".join(str(item) for item in node.metadata.get("visual_atoms", [])),
            ]
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

    def _image_text_embed_cached(self, text: str) -> list[float]:
        if self.image_text_embedding_provider is None:
            return []
        if text not in self._image_text_embedding_cache:
            self._image_text_embedding_cache[text] = self.image_text_embedding_provider.embed_text(
                text
            )
        return self._image_text_embedding_cache[text]

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        return _cosine_similarity(left, right)

    def _tokenize(self, text: str) -> set[str]:
        return {
            token
            for token in (match.group(0).lower() for match in TOKEN_PATTERN.finditer(text))
            if token not in STOPWORDS and len(token) > 1
        }


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f"Embedding dimension mismatch: left={len(left)} right={len(right)}")
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    dot_product = sum(
        left_value * right_value for left_value, right_value in zip(left, right, strict=True)
    )
    similarity = dot_product / (left_norm * right_norm)
    return max(0.0, similarity)
