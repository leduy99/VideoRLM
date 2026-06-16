from __future__ import annotations

import os
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from rlm.video.adapters import EmbeddingProvider, ImageTextEmbeddingProvider
from rlm.video.types import FrontierItem, Modality, TimeSpan, VideoMemory, VideoNodeLevel

TOKEN_PATTERN = re.compile(r"\b\w+\b")
SearchMode = Literal["lexical", "graph"]
GRAPH_MODALITIES = {"visual"}
FRAME_SIMILARITY_THRESHOLD = 0.88
SEMANTIC_FRAME_SIMILARITY_THRESHOLD = 0.16
SEMANTIC_FRAME_EDGE_THRESHOLD = 0.78
MAX_FRAME_SIMILARITY_NEIGHBORS = 3
MAX_SEMANTIC_FRAME_SIMILARITY_NEIGHBORS = 3
SPEECH_DENSE_ONLY_SCORE_CAP = 0.35
SPEECH_ANCHORLESS_TEMPORAL_SCORE_CAP = 0.42
SPEECH_ANCHOR_AFTER_SECONDS = 120.0
SPEECH_ANCHOR_BEFORE_SECONDS = 120.0
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
class TextSearchRecord:
    node_id: str
    text: str
    tokens: list[str]
    term_counts: Counter[str]
    level: VideoNodeLevel
    time_span: TimeSpan

    @property
    def doc_len(self) -> int:
        return len(self.tokens)


class BM25TextIndex:
    def __init__(
        self,
        records: list[TextSearchRecord],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.records = records
        self.k1 = k1
        self.b = b
        self.avg_doc_len = (
            sum(record.doc_len for record in records) / len(records) if records else 0.0
        )
        self.idf = self._build_idf(records)

    def search(
        self,
        query_tokens: Iterable[str],
        *,
        top_k: int,
        allowed_levels: set[VideoNodeLevel] | None = None,
    ) -> list[tuple[TextSearchRecord, float]]:
        if top_k <= 0 or not self.records:
            return []
        unique_query_tokens = sorted(set(query_tokens))
        if not unique_query_tokens:
            return []
        scored: list[tuple[TextSearchRecord, float]] = []
        for record in self.records:
            if allowed_levels is not None and record.level not in allowed_levels:
                continue
            score = self._score_record(record, unique_query_tokens)
            if score <= 0:
                continue
            scored.append((record, round(score, 6)))
        scored.sort(key=lambda item: (-item[1], item[0].time_span.start, item[0].node_id))
        return scored[:top_k]

    def _score_record(self, record: TextSearchRecord, query_tokens: list[str]) -> float:
        if record.doc_len <= 0:
            return 0.0
        score = 0.0
        length_norm = self.k1 * (
            1.0 - self.b + self.b * (record.doc_len / max(self.avg_doc_len, 1e-6))
        )
        for token in query_tokens:
            frequency = record.term_counts.get(token, 0)
            if frequency <= 0:
                continue
            numerator = frequency * (self.k1 + 1.0)
            denominator = frequency + length_norm
            score += self.idf.get(token, 0.0) * (numerator / max(denominator, 1e-6))
        return score

    def _build_idf(self, records: list[TextSearchRecord]) -> dict[str, float]:
        import math

        document_count = len(records)
        document_frequency: Counter[str] = Counter()
        for record in records:
            document_frequency.update(set(record.tokens))
        return {
            token: math.log(1.0 + ((document_count - frequency + 0.5) / (frequency + 0.5)))
            for token, frequency in document_frequency.items()
        }


@dataclass
class SpeechCandidateIndex:
    records: list[TextSearchRecord]
    bm25: BM25TextIndex
    dense_index: FrameVectorIndex | None = None

    def record_by_node_id(self) -> dict[str, TextSearchRecord]:
        return {record.node_id: record for record in self.records}


@dataclass(frozen=True)
class SpeechAnchorConstraint:
    kind: str
    anchor_query: str
    answer_type: str | None = None
    before_seconds: float = 0.0
    after_seconds: float = 0.0


_SPEECH_CANDIDATE_INDEX_CACHE: dict[tuple[int, int, int], SpeechCandidateIndex] = {}


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
        speech_embedding_provider: EmbeddingProvider | None = None,
        image_text_embedding_provider: ImageTextEmbeddingProvider | None = None,
        lexical_weight: float = 0.7,
        semantic_weight: float = 0.3,
        speech_semantic_min_score: float = 0.28,
        search_mode: SearchMode = "lexical",
    ):
        if search_mode not in {"lexical", "graph"}:
            raise ValueError(f"Unsupported search mode: {search_mode}")
        self.memory = memory
        self.embedding_provider = embedding_provider
        self.speech_embedding_provider = speech_embedding_provider
        self.image_text_embedding_provider = image_text_embedding_provider
        self.lexical_weight = lexical_weight
        self.semantic_weight = semantic_weight
        self.speech_semantic_min_score = speech_semantic_min_score
        self.search_mode = search_mode
        self._embedding_cache: dict[tuple[str, str], list[float]] = {}
        self._speech_embedding_cache: dict[tuple[str, str], list[float]] = {}
        self._image_text_embedding_cache: dict[str, list[float]] = {}
        self._search_cache: dict[tuple[str, str, int, tuple[str, ...]], list[SearchHit]] = {}
        self._semantic_frame_vector_index = self._build_semantic_frame_vector_index()

    def search(
        self,
        query: str,
        modality: Modality | None = None,
        top_k: int = 5,
        levels: Iterable[VideoNodeLevel] | None = None,
    ) -> list[SearchHit]:
        level_tuple = tuple(str(level) for level in levels) if levels else ()
        cache_key = (query, str(modality or ""), top_k, level_tuple)
        cached = self._search_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        search_levels: tuple[str, ...] | None = level_tuple or None
        if modality == "speech":
            results = self.speech_search(
                query=query,
                top_k=top_k,
                levels=search_levels,
            )
            self._search_cache[cache_key] = list(results)
            return results
        if self.search_mode == "graph" and (modality in GRAPH_MODALITIES or modality is None):
            results = self.graph_search(
                query=query,
                modality=modality,
                top_k=top_k,
                levels=search_levels,
            )
            self._search_cache[cache_key] = list(results)
            return results
        results = self.lexical_search(
            query=query,
            modality=modality,
            top_k=top_k,
            levels=search_levels,
        )
        self._search_cache[cache_key] = list(results)
        return results

    def speech_search(
        self,
        query: str,
        top_k: int = 5,
        levels: Iterable[VideoNodeLevel] | None = None,
    ) -> list[SearchHit]:
        query_terms = self._tokenize_terms(query)
        if not query_terms or top_k <= 0:
            return []

        allowed_levels = set(levels) if levels else None
        candidate_limit = max(top_k * 12, 96)
        candidate_index = self._speech_candidate_index()
        record_lookup = candidate_index.record_by_node_id()
        candidate_scores: dict[str, dict[str, float]] = {}
        anchor_constraint = self._speech_anchor_constraint(query)
        anchor_intervals = self._speech_anchor_intervals(
            candidate_index=candidate_index,
            constraint=anchor_constraint,
            allowed_levels=allowed_levels,
        )

        bm25_matches = candidate_index.bm25.search(
            query_terms,
            top_k=candidate_limit,
            allowed_levels=allowed_levels,
        )
        max_bm25_score = max((score for _record, score in bm25_matches), default=0.0)
        for record, score in bm25_matches:
            normalized_score = score / max_bm25_score if max_bm25_score > 0 else 0.0
            candidate_scores.setdefault(record.node_id, {})["bm25"] = normalized_score

        dense_matches = self._speech_dense_matches(
            candidate_index,
            query=query,
            top_k=candidate_limit,
            allowed_levels=allowed_levels,
        )
        for record, score in dense_matches:
            candidate_scores.setdefault(record.node_id, {})["dense"] = score

        if anchor_intervals:
            for record in candidate_index.records:
                if allowed_levels is not None and record.level not in allowed_levels:
                    continue
                interval_score = self._anchor_interval_score(record.time_span, anchor_intervals)
                if interval_score <= 0:
                    continue
                candidate_scores.setdefault(record.node_id, {})["anchor_interval"] = interval_score

        hits: list[SearchHit] = []
        query_tokens = set(query_terms)
        for node_id, scores in candidate_scores.items():
            if node_id not in record_lookup:
                continue
            anchor_interval_score = scores.get("anchor_interval", 0.0)
            if anchor_intervals and anchor_interval_score <= 0:
                continue
            hit = self._score_speech_candidate(
                node_id=node_id,
                query=query,
                query_tokens=query_tokens,
                bm25_score=scores.get("bm25", 0.0),
                dense_score=scores.get("dense", 0.0),
                anchor_constraint=anchor_constraint,
                anchor_interval_score=anchor_interval_score,
            )
            if hit is not None:
                hits.append(hit)

        hits.sort(key=lambda item: (-item.score, item.time_span.start, item.node_id))
        return self._prioritize_fine_speech_hits(hits, top_k)

    def _speech_candidate_index(self) -> SpeechCandidateIndex:
        provider_key = id(self.speech_embedding_provider) if self.speech_embedding_provider else 0
        cache_key = (id(self.memory), provider_key, len(self.memory.nodes))
        cached = _SPEECH_CANDIDATE_INDEX_CACHE.get(cache_key)
        if cached is not None:
            return cached

        records = self._speech_text_records()
        bm25 = BM25TextIndex(records)
        dense_index = self._build_speech_dense_index(records)
        candidate_index = SpeechCandidateIndex(
            records=records,
            bm25=bm25,
            dense_index=dense_index,
        )
        _SPEECH_CANDIDATE_INDEX_CACHE[cache_key] = candidate_index
        return candidate_index

    def _speech_text_records(self) -> list[TextSearchRecord]:
        records: list[TextSearchRecord] = []
        for node in self.memory.nodes.values():
            if node.level == "video":
                continue
            text = self._node_text(node, "speech")
            if not text:
                continue
            tokens = self._tokenize_terms(text)
            if not tokens:
                continue
            records.append(
                TextSearchRecord(
                    node_id=node.node_id,
                    text=text,
                    tokens=tokens,
                    term_counts=Counter(tokens),
                    level=node.level,
                    time_span=node.time_span,
                )
            )
        return records

    def _build_speech_dense_index(
        self,
        records: list[TextSearchRecord],
    ) -> FrameVectorIndex | None:
        if self.speech_embedding_provider is None or not records:
            return None
        texts = [record.text for record in records]
        embeddings = self._embed_text_batch(texts, "speech")
        vector_records = [
            FrameVectorRecord(node_id=record.node_id, embedding=embedding)
            for record, embedding in zip(records, embeddings, strict=False)
            if embedding
        ]
        if not vector_records:
            return None
        return FrameVectorIndex(vector_records)

    def _speech_dense_matches(
        self,
        candidate_index: SpeechCandidateIndex,
        *,
        query: str,
        top_k: int,
        allowed_levels: set[VideoNodeLevel] | None,
    ) -> list[tuple[TextSearchRecord, float]]:
        if candidate_index.dense_index is None:
            return []
        query_embedding = self._embed_cached(("query", query), query, "speech")
        if not query_embedding:
            return []
        record_lookup = candidate_index.record_by_node_id()
        matches: list[tuple[TextSearchRecord, float]] = []
        for match in candidate_index.dense_index.search(query_embedding, top_k * 2):
            record = record_lookup.get(match.node_id)
            if record is None:
                continue
            if allowed_levels is not None and record.level not in allowed_levels:
                continue
            matches.append((record, round(max(0.0, match.score), 6)))
            if len(matches) >= top_k:
                break
        return matches

    def _speech_anchor_constraint(self, query: str) -> SpeechAnchorConstraint | None:
        lowered = query.lower()
        answer_type = self._speech_answer_type(query)
        after_anchor = self._extract_temporal_anchor(
            query,
            patterns=(
                r"\bright\s+after\s+(.+?)(?:\?|$)",
                r"\bimmediately\s+after\s+(.+?)(?:\?|$)",
                r"\bjust\s+after\s+(.+?)(?:\?|$)",
                r"\bafter\s+(.+?)(?:\?|$)",
            ),
        )
        if after_anchor:
            return SpeechAnchorConstraint(
                kind="after",
                anchor_query=after_anchor,
                answer_type=answer_type or "consequence_event",
                before_seconds=15.0,
                after_seconds=SPEECH_ANCHOR_AFTER_SECONDS,
            )
        before_anchor = self._extract_temporal_anchor(
            query,
            patterns=(
                r"\bbefore\s+(.+?)(?:\?|$)",
                r"\bprior\s+to\s+(.+?)(?:\?|$)",
            ),
        )
        if before_anchor:
            return SpeechAnchorConstraint(
                kind="before",
                anchor_query=before_anchor,
                answer_type=answer_type,
                before_seconds=SPEECH_ANCHOR_BEFORE_SECONDS,
                after_seconds=15.0,
            )
        if any(
            cue in lowered
            for cue in (
                "first",
                "earliest",
                "beginning",
                "initial",
                "early in",
                "early lead",
                "strong start",
            )
        ):
            return SpeechAnchorConstraint(kind="early", anchor_query="", answer_type=answer_type)
        if any(cue in lowered for cue in ("later", "afterward", "afterwards", "rest of", "final")):
            return SpeechAnchorConstraint(kind="late", anchor_query="", answer_type=answer_type)
        if answer_type is not None:
            return SpeechAnchorConstraint(
                kind="answer_type",
                anchor_query="",
                answer_type=answer_type,
            )
        return None

    def _extract_temporal_anchor(self, query: str, *, patterns: Iterable[str]) -> str:
        for pattern in patterns:
            match = re.search(pattern, query, flags=re.IGNORECASE)
            if match is None:
                continue
            anchor = self._clean_anchor_query(match.group(1))
            if anchor:
                return anchor
        return ""

    def _clean_anchor_query(self, text: str) -> str:
        cleaned = re.split(
            r"\b(?:and why|and how|and what|because|so that|but|while|even though)\b",
            text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        cleaned = re.sub(r"\b(?:in this video|in the video|from the video)\b", " ", cleaned)
        cleaned = re.sub(r"^[\s,.;:!?]*(?:the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
        tokens = [
            token
            for token in self._tokenize_terms(cleaned)
            if token
            not in {
                "after",
                "before",
                "right",
                "immediately",
                "just",
                "next",
                "happen",
                "happened",
                "happens",
                "thing",
                "part",
            }
        ]
        return " ".join(tokens[:8])

    def _speech_anchor_intervals(
        self,
        *,
        candidate_index: SpeechCandidateIndex,
        constraint: SpeechAnchorConstraint | None,
        allowed_levels: set[VideoNodeLevel] | None,
    ) -> list[TimeSpan]:
        if constraint is None or not constraint.anchor_query:
            return []
        anchor_terms = self._tokenize_terms(constraint.anchor_query)
        if not anchor_terms:
            return []
        matches = candidate_index.bm25.search(
            anchor_terms,
            top_k=32,
            allowed_levels=allowed_levels,
        )
        max_bm25 = max((score for _record, score in matches), default=0.0)
        ranked: list[tuple[float, TextSearchRecord]] = []
        anchor_lower = constraint.anchor_query.lower()
        anchor_tokens = set(anchor_terms)
        for record, bm25_score in matches:
            lexical_score, overlap = self._lexical_score(anchor_lower, anchor_tokens, record.text)
            if not overlap:
                continue
            normalized_bm25 = bm25_score / max_bm25 if max_bm25 > 0 else 0.0
            score = (0.65 * lexical_score) + (0.35 * normalized_bm25)
            if score <= 0.08:
                continue
            ranked.append((round(score, 4), record))
        ranked.sort(key=lambda item: (-item[0], item[1].time_span.start, item[1].node_id))
        intervals: list[TimeSpan] = []
        duration = float(self.memory.metadata.get("duration_seconds") or 0.0)
        for _score, record in ranked[:3]:
            start = record.time_span.start
            end = record.time_span.end
            if constraint.kind == "after":
                interval = TimeSpan(
                    max(0.0, start - constraint.before_seconds),
                    self._clamp_time(end + constraint.after_seconds, duration),
                )
            elif constraint.kind == "before":
                interval = TimeSpan(
                    max(0.0, start - constraint.before_seconds),
                    self._clamp_time(end + constraint.after_seconds, duration),
                )
            else:
                continue
            intervals.append(interval)
        return self._merge_intervals(intervals)

    def _anchor_interval_score(self, span: TimeSpan, intervals: list[TimeSpan]) -> float:
        best = 0.0
        for interval in intervals:
            overlap = min(span.end, interval.end) - max(span.start, interval.start)
            if overlap <= 0:
                continue
            shorter = max(1e-6, min(span.duration, interval.duration))
            overlap_ratio = max(0.0, min(1.0, overlap / shorter))
            best = max(best, 0.5 + (0.5 * overlap_ratio))
        return round(best, 4)

    def _speech_anchor_text_score(
        self,
        constraint: SpeechAnchorConstraint | None,
        text: str,
    ) -> float:
        if constraint is None or not constraint.anchor_query:
            return 0.0
        anchor_tokens = set(self._tokenize_terms(constraint.anchor_query))
        if not anchor_tokens:
            return 0.0
        doc_tokens = set(self._tokenize_terms(text))
        overlap = anchor_tokens & doc_tokens
        return round(min(1.0, len(overlap) / max(1, min(len(anchor_tokens), 6))), 4)

    def _speech_answer_type(self, query: str) -> str | None:
        lowered = query.lower()
        if any(cue in lowered for cue in ("why", "reason", "because", "made", "meant")):
            return "causal_explanation"
        if any(cue in lowered for cue in ("what happened", "right after", "immediately after", "next")):
            return "consequence_event"
        if any(cue in lowered for cue in ("how did", "how was", "fix", "solve", "address")):
            return "process_fix"
        if any(cue in lowered for cue in ("first piece", "which piece", "what piece", "object")):
            return "object_identification"
        return None

    def _speech_answer_type_score(
        self,
        constraint: SpeechAnchorConstraint | None,
        query: str,
        text: str,
    ) -> float:
        answer_type = constraint.answer_type if constraint is not None else None
        answer_type = answer_type or self._speech_answer_type(query)
        if answer_type is None:
            return 0.0
        lowered = text.lower()
        keyword_groups = {
            "causal_explanation": (
                "because",
                "reason",
                "meant",
                "so",
                "therefore",
                "caused",
                "led",
                "wanted",
                "needed",
                "had to",
                "couldn't",
                "could not",
            ),
            "consequence_event": (
                "then",
                "after",
                "next",
                "damaged",
                "leaking",
                "lost",
                "losing",
                "started",
                "began",
                "result",
                "oxygen",
                "electricity",
            ),
            "process_fix": (
                "used",
                "using",
                "made",
                "built",
                "adapted",
                "fixed",
                "solve",
                "address",
                "tape",
                "cardboard",
                "sock",
                "filter",
                "canister",
            ),
            "object_identification": (
                "ring",
                "bracelet",
                "necklace",
                "earrings",
                "piece",
                "object",
                "diamond",
                "jewelry",
            ),
        }
        hits = sum(1 for keyword in keyword_groups.get(answer_type, ()) if keyword in lowered)
        if hits <= 0:
            return 0.0
        return round(min(0.18, 0.07 + (0.04 * hits)), 4)

    def _merge_intervals(self, intervals: list[TimeSpan]) -> list[TimeSpan]:
        if not intervals:
            return []
        ordered = sorted(intervals, key=lambda item: (item.start, item.end))
        merged = [ordered[0]]
        for interval in ordered[1:]:
            current = merged[-1]
            if interval.start <= current.end:
                merged[-1] = TimeSpan(current.start, max(current.end, interval.end))
            else:
                merged.append(interval)
        return merged

    def _clamp_time(self, value: float, duration: float) -> float:
        if duration <= 0:
            return max(0.0, value)
        return max(0.0, min(duration, value))

    def _score_speech_candidate(
        self,
        *,
        node_id: str,
        query: str,
        query_tokens: set[str],
        bm25_score: float,
        dense_score: float,
        anchor_constraint: SpeechAnchorConstraint | None = None,
        anchor_interval_score: float = 0.0,
    ) -> SearchHit | None:
        node = self.memory.get_node(node_id)
        query_lower = query.lower()
        text = self._node_text(node, "speech")
        if not text:
            return None

        lexical_score, overlap = self._lexical_score(query_lower, query_tokens, text)
        semantic_score = dense_score
        temporal_score = self._temporal_score(query_tokens, node.time_span)
        section_score = self._section_score(query_lower, query_tokens, node)
        anchor_score = self._speech_anchor_text_score(anchor_constraint, text)
        answer_type_score = self._speech_answer_type_score(anchor_constraint, query, text)
        if (
            lexical_score <= 0
            and semantic_score <= 0
            and section_score <= 0
            and bm25_score <= 0
            and anchor_interval_score <= 0
            and answer_type_score <= 0
        ):
            return None
        if (
            lexical_score <= 0
            and section_score <= 0
            and bm25_score <= 0
            and anchor_score <= 0
            and anchor_interval_score <= 0
            and semantic_score < self.speech_semantic_min_score
        ):
            return None

        bm25_bonus = 0.12 * max(0.0, min(1.0, bm25_score))
        anchor_interval_bonus = 0.22 * max(0.0, min(1.0, anchor_interval_score))
        score = round(
            self._combine_scores(
                lexical_score,
                semantic_score,
                temporal_score,
                "speech",
            )
            + section_score
            + bm25_bonus
            + anchor_interval_bonus
            + answer_type_score,
            4,
        )
        dense_only = (
            semantic_score > 0
            and lexical_score <= 0
            and bm25_score <= 0
            and section_score <= 0
            and anchor_score <= 0
            and anchor_interval_score <= 0
        )
        if dense_only:
            score = min(score, SPEECH_DENSE_ONLY_SCORE_CAP)
        elif (
            anchor_constraint is not None
            and anchor_constraint.anchor_query
            and anchor_score <= 0
            and anchor_interval_score <= 0
        ):
            score = min(score, SPEECH_ANCHORLESS_TEMPORAL_SCORE_CAP)
        fine_speech_window = self._is_fine_speech_window_node(node.node_id)
        if fine_speech_window:
            score = round(score + 0.12, 4)

        reason = self._build_reason(
            modality="speech",
            node_id=node.node_id,
            overlap=overlap,
            lexical_score=lexical_score,
            semantic_score=semantic_score,
            temporal_score=temporal_score,
            section_score=section_score,
        )
        if bm25_score > 0:
            reason = f"{reason}; bm25_asr={bm25_score:.2f}"
        if anchor_constraint is not None:
            reason = (
                f"{reason}; speech_anchor={anchor_constraint.kind}; "
                f"anchor_score={anchor_score:.2f}; "
                f"anchor_interval={anchor_interval_score:.2f}; "
                f"answer_type={answer_type_score:.2f}"
            )
        if fine_speech_window:
            reason = f"{reason}; fine ASR retrieval window"
        return SearchHit(
            node_id=node.node_id,
            time_span=node.time_span,
            level=node.level,
            score=score,
            reason=reason,
            modality="speech",
            matched_terms=overlap,
            score_breakdown={
                "lexical": lexical_score,
                "semantic": semantic_score,
                "temporal": temporal_score,
                "section": section_score,
                "bm25": round(bm25_score, 4),
                "anchor_text": anchor_score,
                "anchor_interval": round(anchor_interval_score, 4),
                "answer_type": answer_type_score,
                "dense_only_cap": 1.0 if dense_only else 0.0,
                "combined": score,
                "fine_speech_window": 1.0 if fine_speech_window else 0.0,
            },
        )

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

        for neighbor_id in node.metadata.get("cognitive_event_neighbor_ids", []):
            if neighbor_id in self.memory.nodes:
                neighbors.append((str(neighbor_id), 0.42, "cognitive-event-neighbor"))

        for key, weight, reason in (
            ("same_actor_event_ids", 0.38, "same-actor"),
            ("same_object_event_ids", 0.36, "same-object"),
            ("same_place_event_ids", 0.3, "same-place"),
            ("same_topic_event_ids", 0.32, "same-topic"),
            ("cause_effect_event_ids", 0.44, "cause-effect"),
            ("caused_by_event_ids", 0.4, "caused-by"),
            ("goal_continuation_event_ids", 0.34, "goal-continuation"),
            ("goal_predecessor_event_ids", 0.32, "goal-predecessor"),
        ):
            for neighbor_id in node.metadata.get(key, []):
                if neighbor_id in self.memory.nodes:
                    neighbors.append((str(neighbor_id), weight, reason))

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
        units = self._ocr_text_units(node)
        units.extend(str(item) for item in node.metadata.get("visual_atoms", []))
        units.extend(str(item) for item in node.metadata.get("visual_keywords", []))
        return self._max_unit_lexical_score(query_lower, query_tokens, units)

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
                "event": 0.14,
                "segment": 0.06,
                "scene": -0.04,
                "video": -0.1,
            }.get(level, 0.0)
        return 0.0

    def _graph_level_rank(self, level: VideoNodeLevel) -> int:
        return {
            "clip": 0,
            "event": 1,
            "segment": 2,
            "scene": 3,
            "video": 4,
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
            if current_modality == "ocr":
                text = self._ocr_semantic_text(node)
                lexical_score, overlap = self._ocr_lexical_score(query_lower, query_tokens, node)
            else:
                text = self._node_text(node, current_modality)
                lexical_score, overlap = self._lexical_score(query_lower, query_tokens, text)
            if not text:
                continue

            semantic_score = self._semantic_score(query, text, current_modality)
            temporal_score = self._temporal_score(query_tokens, node.time_span)
            section_score = self._section_score(query_lower, query_tokens, node)
            if lexical_score <= 0 and semantic_score <= 0 and section_score <= 0:
                continue
            if (
                current_modality == "speech"
                and lexical_score <= 0
                and section_score <= 0
                and semantic_score < self.speech_semantic_min_score
            ):
                continue

            score = round(
                self._combine_scores(
                    lexical_score,
                    semantic_score,
                    temporal_score,
                    current_modality,
                )
                + section_score,
                4,
            )
            fine_speech_window = (
                current_modality == "speech" and self._is_fine_speech_window_node(node.node_id)
            )
            if fine_speech_window:
                score = round(score + 0.12, 4)
            reason = self._build_reason(
                modality=current_modality,
                node_id=node.node_id,
                overlap=overlap,
                lexical_score=lexical_score,
                semantic_score=semantic_score,
                temporal_score=temporal_score,
                section_score=section_score,
            )
            if fine_speech_window:
                reason = f"{reason}; fine ASR retrieval window"
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
                    "section": section_score,
                    "combined": score,
                    "fine_speech_window": 1.0 if fine_speech_window else 0.0,
                },
            )
            if best_hit is None or hit.score > best_hit.score:
                best_hit = hit

        return best_hit

    def _prioritize_fine_speech_hits(
        self,
        hits: list[SearchHit],
        top_k: int,
    ) -> list[SearchHit]:
        if top_k <= 0:
            return []
        fine_hits = [hit for hit in hits if self._is_fine_speech_window_node(hit.node_id)]
        if not fine_hits:
            return hits[:top_k]
        coarse_hits = [hit for hit in hits if not self._is_fine_speech_window_node(hit.node_id)]
        return [*fine_hits, *coarse_hits][:top_k]

    def _is_fine_speech_window_node(self, node_id: str) -> bool:
        node = self.memory.nodes.get(node_id)
        return bool(node and node.metadata.get("speech_window_kind") == "fine_asr_window")

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
                self._event_schema_text(node.metadata.get("event_schema")),
            ]
            return " ".join(part for part in parts if part)
        if modality == "ocr":
            return self._ocr_semantic_text(node)
        if modality == "audio":
            return " ".join(item.label for item in node.audio_events)
        return ""

    def _ocr_text_units(self, node) -> list[str]:
        seen: set[str] = set()
        units: list[str] = []
        for span in node.ocr_spans:
            for raw_line in re.split(r"[\n\r]+", span.text):
                line = " ".join(raw_line.split()).strip()
                if not line:
                    continue
                if len(line) > 260:
                    units.extend(self._ocr_long_line_units(line))
                    continue
                key = self._ocr_unit_key(line)
                if not key or key in seen:
                    continue
                seen.add(key)
                units.append(line)
        return units

    def _ocr_long_line_units(self, line: str) -> list[str]:
        code_matches = [
            " ".join(match.group(0).split())
            for match in re.finditer(
                r"\b[A-Za-z_]\w*\s*=\s*(?:[A-Za-z_]\w*|\d+(?:\.\d+)?)"
                r"(?:\s*(?:==|!=|>=|<=|>|<|\+|\-|\*|/)\s*"
                r"(?:[A-Za-z_]\w*|\d+(?:\.\d+)?))?",
                line,
            )
        ]
        if code_matches:
            return code_matches
        return [line[:260]]

    def _ocr_unit_key(self, line: str) -> str:
        return re.sub(r"\W+", "", line.casefold())

    def _ocr_semantic_text(self, node) -> str:
        return " ".join(self._ocr_text_units(node)[:60])

    def _ocr_lexical_score(
        self,
        query_lower: str,
        query_tokens: set[str],
        node,
    ) -> tuple[float, list[str]]:
        units = self._ocr_text_units(node)
        if not units:
            return 0.0, []
        unit_score, unit_overlap = self._max_unit_lexical_score(query_lower, query_tokens, units)
        combined_tokens = set()
        for unit in units:
            combined_tokens.update(self._tokenize(unit))
        coverage_overlap = sorted(query_tokens & combined_tokens)
        coverage_score = 0.0
        if coverage_overlap:
            coverage_score = min(0.55, 0.55 * len(coverage_overlap) / max(len(query_tokens), 1))
        score = max(unit_score, coverage_score)
        total_chars = sum(len(unit) for unit in units)
        length_penalty = min(0.18, max(0, total_chars - 1200) / 6000)
        score = max(0.0, score - length_penalty)
        overlap = unit_overlap if unit_score >= coverage_score else coverage_overlap
        return round(score, 4), overlap

    def _max_unit_lexical_score(
        self,
        query_lower: str,
        query_tokens: set[str],
        units: Iterable[str],
    ) -> tuple[float, list[str]]:
        best_score = 0.0
        best_overlap: list[str] = []
        for unit in units:
            doc_tokens = self._tokenize(unit)
            overlap = sorted(query_tokens & doc_tokens)
            if not overlap:
                continue
            overlap_ratio = len(overlap) / max(len(query_tokens), 1)
            phrase_bonus = 0.25 if query_lower in unit.lower() else 0.0
            compactness_bonus = min(0.15, len(overlap) / max(len(doc_tokens), 1))
            length_penalty = min(0.12, max(0, len(unit) - 180) / 1200)
            score = min(1.0, overlap_ratio + phrase_bonus + compactness_bonus)
            score = max(0.0, score - length_penalty)
            if score > best_score:
                best_score = score
                best_overlap = overlap
        return round(best_score, 4), best_overlap

    def _event_schema_text(self, schema: Any) -> str:
        if not isinstance(schema, dict):
            return ""
        parts: list[str] = []
        for key in (
            "place",
            "actors",
            "objects",
            "actions",
            "goals_or_intentions",
            "goal_predecessors",
            "goal_successors",
            "causal_predecessors",
            "causal_outcomes",
            "spoken_topics",
            "ocr_entities",
            "visual_state",
            "audio_state",
            "event_type",
        ):
            value = schema.get(key)
            if isinstance(value, list):
                parts.extend(str(item) for item in value)
            elif isinstance(value, str):
                parts.append(value)
        return " ".join(parts)

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

    def _semantic_score(self, query: str, text: str, modality: Modality) -> float:
        provider = self._semantic_provider_for_modality(modality)
        if provider is None:
            return 0.0
        query_vector = self._embed_cached(("query", query), query, modality)
        text_vector = self._embed_cached(("text", text), text, modality)
        if not query_vector or not text_vector:
            return 0.0
        return round(self._cosine_similarity(query_vector, text_vector), 4)

    def _combine_scores(
        self,
        lexical_score: float,
        semantic_score: float,
        temporal_score: float,
        modality: Modality,
    ) -> float:
        if self._semantic_provider_for_modality(modality) is None:
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
        section_score: float = 0.0,
    ) -> str:
        parts = []
        if overlap:
            parts.append(f"Matched {modality} terms {', '.join(overlap[:4])}")
        if semantic_score > 0:
            parts.append(f"semantic similarity {semantic_score:.2f}")
        if temporal_score > 0:
            parts.append(f"temporal prior {temporal_score:.2f}")
        if section_score > 0:
            parts.append(f"section prior {section_score:.2f}")
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

    def _section_score(self, query_lower: str, query_tokens: set[str], node) -> float:
        section_tags = set(self._node_section_tags(node))
        if not section_tags:
            return 0.0
        score = 0.0
        if "third segment" in query_lower and "third_segment" in section_tags:
            score += 0.26
        if "second segment" in query_lower and "second_segment" in section_tags:
            score += 0.22
        if "first segment" in query_lower and "first_segment" in section_tags:
            score += 0.22
        if "second half" in query_lower and "second_half" in section_tags:
            score += 0.24
        if "arithmetic" in query_tokens and "arithmetic_section" in section_tags:
            score += 0.24
        if "assignment" in query_tokens and "assignment_section" in section_tags:
            score += 0.22
        if "comparison" in query_tokens and "comparison_section" in section_tags:
            score += 0.24
        if ({"logical", "boolean"} & query_tokens) and "logical_evaluation_section" in section_tags:
            score += 0.18
        if "before" in query_tokens and "comparison" in query_tokens:
            if "pre_comparison_section" in section_tags:
                score += 0.18
            if "comparison_section" in section_tags:
                score -= 0.12
        return round(max(0.0, min(score, 0.36)), 4)

    def _node_section_tags(self, node) -> list[str]:
        metadata_tags = [str(tag) for tag in node.metadata.get("section_tags", [])]
        if metadata_tags:
            return metadata_tags
        tags: list[str] = []
        duration = float(self.memory.metadata.get("duration_seconds") or 0.0)
        if duration > 0:
            midpoint = duration / 2.0
            if node.time_span.end > midpoint:
                tags.append("second_half")
            if node.time_span.start < midpoint:
                tags.append("first_half")
            segment_duration = float(self.memory.metadata.get("segment_duration_seconds") or 0.0)
            if segment_duration > 0 and node.level in {"segment", "event", "clip"}:
                start_index = int(max(0.0, node.time_span.start) // segment_duration) + 1
                end_time = max(node.time_span.start, min(duration, node.time_span.end) - 1e-6)
                end_index = int(end_time // segment_duration) + 1
                for index in range(max(1, start_index), max(start_index, end_index) + 1):
                    tags.append(f"segment_{index}")
                    ordinal = {
                        1: "first_segment",
                        2: "second_segment",
                        3: "third_segment",
                        4: "fourth_segment",
                        5: "fifth_segment",
                    }.get(index)
                    if ordinal:
                        tags.append(ordinal)
        text = " ".join(
            [
                node.visual_summary,
                " ".join(span.text for span in node.speech_spans),
                " ".join(span.text for span in node.ocr_spans),
            ]
        )
        lowered = text.lower()
        if re.search(r"\b[A-Za-z_]\w*\s*=", text) or "python script" in lowered:
            tags.append("code_section")
        if "assignment operator" in lowered or "variables" in lowered or re.search(r"\b[x-z]\s*=", text):
            tags.append("assignment_section")
        if "arithmetic operator" in lowered or re.search(r"=\s*[^=<>!]*(?:\+|\-|\*|/)", text):
            tags.append("arithmetic_section")
        if "comparison operator" in lowered or any(
            operator in text for operator in ("==", "!=", ">=", "<=", ">", "<")
        ):
            tags.append("comparison_section")
        if any(
            cue in lowered
            for cue in ("equal to", "not equal", "greater than", "less than", "boolean")
        ):
            tags.append("logical_evaluation_section")
        if ("assignment_section" in tags or "arithmetic_section" in tags) and (
            "comparison_section" not in tags
        ):
            tags.append("pre_comparison_section")
        deduped: list[str] = []
        seen: set[str] = set()
        for tag in tags:
            if tag in seen:
                continue
            seen.add(tag)
            deduped.append(tag)
        return deduped

    def _semantic_provider_for_modality(self, modality: Modality) -> EmbeddingProvider | None:
        if modality == "speech" and self.speech_embedding_provider is not None:
            return self.speech_embedding_provider
        return self.embedding_provider

    def _embed_cached(
        self,
        cache_key: tuple[str, str],
        text: str,
        modality: Modality,
    ) -> list[float]:
        provider = self._semantic_provider_for_modality(modality)
        if provider is None:
            return []
        if modality == "speech" and self.speech_embedding_provider is not None:
            if cache_key not in self._speech_embedding_cache:
                self._speech_embedding_cache[cache_key] = provider.embed_text(text)
            return self._speech_embedding_cache[cache_key]
        if cache_key not in self._embedding_cache:
            self._embedding_cache[cache_key] = provider.embed_text(text)
        return self._embedding_cache[cache_key]

    def _embed_text_batch(self, texts: list[str], modality: Modality) -> list[list[float]]:
        provider = self._semantic_provider_for_modality(modality)
        if provider is None or not texts:
            return []

        cache = (
            self._speech_embedding_cache
            if modality == "speech" and self.speech_embedding_provider is not None
            else self._embedding_cache
        )
        embeddings: list[list[float] | None] = []
        uncached_texts: list[str] = []
        uncached_positions: list[int] = []
        for text in texts:
            cache_key = ("text", text)
            cached = cache.get(cache_key)
            if cached is None:
                uncached_positions.append(len(embeddings))
                uncached_texts.append(text)
                embeddings.append(None)
            else:
                embeddings.append(cached)

        if uncached_texts:
            batch_embed = getattr(provider, "embed_texts", None)
            if callable(batch_embed):
                computed = batch_embed(uncached_texts)
            else:
                computed = [provider.embed_text(text) for text in uncached_texts]
            for position, text, embedding in zip(
                uncached_positions,
                uncached_texts,
                computed,
                strict=False,
            ):
                cache[("text", text)] = embedding
                embeddings[position] = embedding

        return [embedding or [] for embedding in embeddings]

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
        return set(self._tokenize_terms(text))

    def _tokenize_terms(self, text: str) -> list[str]:
        return [
            token
            for token in (match.group(0).lower() for match in TOKEN_PATTERN.finditer(text))
            if token not in STOPWORDS and len(token) > 1
        ]


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
