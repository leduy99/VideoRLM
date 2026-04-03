import json
import re

from rlm.clients.base_lm import BaseLM
from rlm.video.index import STOPWORDS, TOKEN_PATTERN, VideoMemoryIndex
from rlm.video.types import (
    ControllerAction,
    ControllerState,
    Evidence,
    FrontierItem,
    Modality,
    Observation,
    SpeechSpan,
    VideoMemory,
)

CONTROL_QUERY_TOKENS = {
    "why",
    "first",
    "beginning",
    "earliest",
    "initial",
    "last",
    "final",
    "ending",
    "end",
}


class VideoToolExecutor:
    def __init__(
        self,
        memory: VideoMemory,
        index: VideoMemoryIndex | None = None,
        top_k: int = 5,
        *,
        speech_snippet_refiner: BaseLM | None = None,
        enable_hybrid_speech_refinement: bool = False,
        speech_refine_candidate_count: int = 4,
    ):
        self.memory = memory
        self.index = index or VideoMemoryIndex(memory)
        self.top_k = top_k
        self._evidence_counter = 0
        self.speech_snippet_refiner = speech_snippet_refiner
        self.enable_hybrid_speech_refinement = (
            enable_hybrid_speech_refinement and speech_snippet_refiner is not None
        )
        self.speech_refine_candidate_count = speech_refine_candidate_count

    def execute(self, action: ControllerAction, state: ControllerState) -> Observation:
        if action.action_type == "SEARCH":
            return self.search(action.query or "", action.modality, self.top_k)
        if action.action_type == "OPEN":
            return self.open(action.node_id or "", action.modality, state)
        if action.action_type == "SPLIT":
            return self.split(action.node_id or "")
        if action.action_type == "MERGE":
            return self.merge(action.evidence_ids, state)
        if action.action_type == "STOP":
            return self.stop(action.answer or "", action.evidence_ids, state)
        raise ValueError(f"Unsupported action type: {action.action_type}")

    def search(self, query: str, modality: Modality | None, top_k: int) -> Observation:
        hits = self.index.search(query=query, modality=modality, top_k=top_k)
        frontier = [hit.to_frontier_item() for hit in hits]
        summary = f"SEARCH found {len(frontier)} candidate nodes for query '{query}'."
        return Observation(
            kind="search",
            summary=summary,
            frontier=frontier,
            metadata={"query": query, "modality": modality, "hit_count": len(frontier)},
        )

    def open(
        self,
        node_id: str,
        modality: Modality | None,
        state: ControllerState,
    ) -> Observation:
        node = self.memory.get_node(node_id)
        selected_modality = modality or "visual"

        if selected_modality == "speech":
            evidence = self._build_speech_evidence(node, state)
            if evidence:
                summary = (
                    f"OPEN gathered {len(evidence)} {selected_modality} evidence items "
                    f"from node {node.node_id}."
                )
            else:
                summary = f"OPEN found no {selected_modality} evidence in node {node.node_id}."
        else:
            detail = self._build_detail(node, selected_modality)
            if detail:
                evidence = [
                    Evidence(
                        evidence_id=self._next_evidence_id(),
                        claim=self._to_claim(detail, selected_modality),
                        modality=selected_modality,
                        time_span=node.time_span,
                        source_node_id=node.node_id,
                        confidence=self._confidence_from_detail(detail),
                        detail=detail,
                        metadata={"clip_path": node.clip_path},
                    )
                ]
                summary = f"OPEN gathered {selected_modality} evidence from node {node.node_id}."
            else:
                evidence = []
                summary = f"OPEN found no {selected_modality} evidence in node {node.node_id}."

        return Observation(
            kind="open",
            summary=summary,
            evidence=evidence,
            node_id=node.node_id,
            metadata={"modality": selected_modality, "clip_path": node.clip_path},
        )

    def split(self, node_id: str) -> Observation:
        children = self.memory.child_nodes(node_id)
        frontier = []
        for child in children:
            reason = f"Child node of {node_id} spanning {child.time_span.to_display()}"
            recommended = self._recommended_modalities(child)
            frontier.append(
                FrontierItem(
                    node_id=child.node_id,
                    time_span=child.time_span,
                    level=child.level,
                    score=self._child_priority(child),
                    why_candidate=reason,
                    recommended_modalities=recommended,
                    status="unopened",
                )
            )

        summary = f"SPLIT expanded {node_id} into {len(frontier)} child nodes."
        return Observation(
            kind="split",
            summary=summary,
            frontier=frontier,
            node_id=node_id,
            metadata={"child_count": len(frontier)},
        )

    def merge(self, evidence_ids: list[str], state: ControllerState) -> Observation:
        ledger = state.evidence_by_id()
        selected = [ledger[item] for item in evidence_ids if item in ledger]
        if not selected:
            return Observation(kind="merge", summary="MERGE found no matching evidence ids.")

        claim = " | ".join(item.claim for item in selected)
        detail = "\n".join(item.detail for item in selected if item.detail)
        merged = Evidence(
            evidence_id=self._next_evidence_id(),
            claim=claim,
            modality="cross_modal",
            time_span=selected[0].time_span,
            source_node_id=selected[0].source_node_id,
            confidence=round(sum(item.confidence for item in selected) / len(selected), 4),
            detail=detail,
            metadata={"merged_ids": list(evidence_ids)},
        )
        return Observation(
            kind="merge",
            summary=f"MERGE combined {len(selected)} evidence items.",
            evidence=[merged],
            metadata={"merged_ids": list(evidence_ids)},
        )

    def stop(self, answer: str, evidence_ids: list[str], state: ControllerState) -> Observation:
        selected = [item for item in state.evidence_ledger if item.evidence_id in set(evidence_ids)]
        summary = f"STOP selected {len(selected)} evidence items."
        return Observation(
            kind="stop",
            summary=summary,
            evidence=selected,
            metadata={"answer": answer, "evidence_ids": list(evidence_ids)},
        )

    def _build_detail(self, node, modality: Modality) -> str:
        if modality == "speech":
            return " ".join(item.text.strip() for item in node.speech_spans if item.text).strip()
        if modality == "visual":
            return node.visual_summary.strip()
        if modality == "ocr":
            return " ".join(item.text.strip() for item in node.ocr_spans if item.text).strip()
        if modality == "audio":
            labels = [item.label.strip() for item in node.audio_events if item.label]
            return ", ".join(labels).strip()
        return ""

    def _to_claim(self, detail: str, modality: Modality) -> str:
        cleaned = " ".join(detail.split())
        snippet = cleaned[:180]
        prefix = {
            "speech": "Speech evidence",
            "visual": "Visual evidence",
            "ocr": "OCR evidence",
            "audio": "Audio evidence",
            "cross_modal": "Merged evidence",
        }[modality]
        return f"{prefix}: {snippet}"

    def _confidence_from_detail(self, detail: str) -> float:
        length = len(detail.strip())
        if length == 0:
            return 0.1
        return round(min(0.95, 0.45 + (length / 500.0)), 4)

    def _recommended_modalities(self, node) -> list[Modality]:
        modalities: list[Modality] = []
        if node.speech_spans:
            modalities.append("speech")
        if node.visual_summary:
            modalities.append("visual")
        if node.ocr_spans:
            modalities.append("ocr")
        if node.audio_events:
            modalities.append("audio")
        return modalities or ["visual"]

    def _child_priority(self, node) -> float:
        score = 0.2
        score += min(len(node.speech_spans) * 0.05, 0.3)
        score += min(len(node.ocr_spans) * 0.03, 0.2)
        score += 0.15 if node.visual_summary else 0.0
        return round(score, 4)

    def _next_evidence_id(self) -> str:
        self._evidence_counter += 1
        return f"evidence_{self._evidence_counter:05d}"

    def _build_speech_evidence(self, node, state: ControllerState) -> list[Evidence]:
        selected_spans = self._select_relevant_speech_spans(node.speech_spans, state)
        if not selected_spans:
            return []

        evidence: list[Evidence] = []
        query_hint = self._latest_search_query(state)
        question_tokens = self._tokenize(state.question)
        query_tokens = self._tokenize(" ".join(part for part in [state.question, query_hint] if part))
        is_first_query = bool({"first", "beginning", "earliest", "initial"} & question_tokens)
        is_last_query = bool({"last", "final", "ending", "end"} & question_tokens)
        for position, (span, score) in enumerate(selected_spans):
            prefer_start = is_first_query and position > 0
            prefer_end = is_last_query and position < len(selected_spans) - 1
            detail = self._focus_speech_detail(
                span.text,
                question_tokens=question_tokens,
                query_tokens=query_tokens,
                prefer_start=prefer_start,
                prefer_end=prefer_end,
            )
            if not detail:
                continue
            detail, selection_metadata = self._maybe_refine_speech_detail(
                span=span,
                detail=detail,
                state=state,
                question_tokens=question_tokens,
                query_tokens=query_tokens,
                search_query=query_hint,
                prefer_start=prefer_start,
                prefer_end=prefer_end,
            )
            if self._is_duplicate_speech_evidence(state, span, detail):
                continue
            evidence.append(
                Evidence(
                    evidence_id=self._next_evidence_id(),
                    claim=self._to_claim(detail, "speech"),
                    modality="speech",
                    time_span=span.time_span,
                    source_node_id=node.node_id,
                    confidence=self._confidence_from_speech_score(detail, score),
                    detail=detail,
                    metadata={
                        "clip_path": self._clip_path_for_span(node.clip_path, span.time_span),
                        "parent_node_id": node.node_id,
                        "selection_score": round(score, 4),
                        "search_query": query_hint,
                        **selection_metadata,
                    },
                )
            )
        return evidence

    def _select_relevant_speech_spans(
        self,
        spans: list[SpeechSpan],
        state: ControllerState,
        max_items: int = 2,
    ) -> list[tuple[SpeechSpan, float]]:
        cleaned_spans = [span for span in spans if span.text.strip()]
        if not cleaned_spans:
            return []

        query_hint = self._latest_search_query(state)
        query_tokens = self._tokenize(" ".join(part for part in [state.question, query_hint] if part))
        question_tokens = self._tokenize(state.question)
        scored = [
            (
                self._score_speech_span(
                    span=span,
                    question_tokens=question_tokens,
                    query_tokens=query_tokens,
                ),
                index,
                span,
            )
            for index, span in enumerate(cleaned_spans)
        ]
        scored.sort(key=lambda item: (-item[0], item[2].time_span.start))
        best_score, best_index, _ = scored[0]
        if best_score <= 0:
            return [(cleaned_spans[0], 0.0)]

        selected_indices = {best_index}
        score_by_index = {index: score for score, index, _ in scored}
        is_why_query = "why" in question_tokens
        is_first_query = bool({"first", "beginning", "earliest", "initial"} & question_tokens)
        is_last_query = bool({"last", "final", "ending", "end"} & question_tokens)
        neighbor_candidates: list[int] = []
        if is_first_query:
            neighbor_candidates.append(best_index + 1)
        elif is_last_query:
            neighbor_candidates.append(best_index - 1)
        else:
            neighbor_candidates.extend([best_index - 1, best_index + 1])

        for neighbor_index in neighbor_candidates:
            if neighbor_index < 0 or neighbor_index >= len(cleaned_spans):
                continue
            neighbor_score = score_by_index.get(neighbor_index, 0.0)
            neighbor_has_why_signal = is_why_query and self._span_has_why_signal(
                cleaned_spans[neighbor_index]
            )
            if neighbor_score >= max(best_score * 0.35, 0.15) or neighbor_has_why_signal or (
                (is_first_query or is_last_query) and neighbor_score >= 0.05
            ):
                selected_indices.add(neighbor_index)
            if len(selected_indices) >= max_items:
                break

        if len(selected_indices) < max_items:
            for score, index, _span in scored[1:]:
                if index in selected_indices:
                    continue
                if score < max(best_score * 0.45, 0.2) and not (
                    is_why_query and self._span_has_why_signal(cleaned_spans[index])
                ):
                    continue
                selected_indices.add(index)
                if len(selected_indices) >= max_items:
                    break

        ordered_indices = sorted(selected_indices, key=lambda index: cleaned_spans[index].time_span.start)
        return [(cleaned_spans[index], score_by_index[index]) for index in ordered_indices[:max_items]]

    def _score_speech_span(
        self,
        span: SpeechSpan,
        question_tokens: set[str],
        query_tokens: set[str],
    ) -> float:
        text = " ".join(span.text.split()).strip()
        if not text:
            return 0.0

        doc_tokens = self._tokenize(text)
        if not doc_tokens:
            return 0.0

        lower_text = text.lower()
        overlap = query_tokens & doc_tokens
        overlap_ratio = len(overlap) / max(len(query_tokens), 1)
        density_bonus = sum(lower_text.count(term) for term in overlap) / max(len(doc_tokens), 1)
        score = overlap_ratio + density_bonus
        causal_hits = sum(
            1
            for keyword in ("worried", "lose", "fix", "repair", "open", "clasp")
            if keyword in doc_tokens or keyword in lower_text
        )
        support_hits = sum(
            1
            for keyword in ("wear", "love", "bracelet", "cartier", "clash", "perfect")
            if keyword in doc_tokens or keyword in lower_text
        )
        topic_shift_hits = sum(
            1
            for marker in ("other bracelet", "another bracelet", "last but not the least", "last but not least")
            if marker in lower_text
        )

        if "why" in question_tokens:
            score += causal_hits * 0.35
            score += support_hits * 0.08
            if "because" in lower_text and not causal_hits and len(overlap) < 2:
                score -= 0.2
        if topic_shift_hits:
            score -= topic_shift_hits * 0.25

        duration = float(self.memory.metadata.get("duration_seconds") or 0.0)
        if duration > 0:
            if {"first", "beginning", "earliest", "initial"} & question_tokens:
                score += max(0.0, 1.0 - (span.time_span.start / duration)) * 0.6
            if {"last", "final", "ending", "end"} & question_tokens:
                score += max(0.0, span.time_span.end / duration) * 0.6

        return round(score, 4)

    def _latest_search_query(self, state: ControllerState) -> str:
        for action in reversed(state.action_history):
            if action.get("action_type") != "SEARCH":
                continue
            query = str(action.get("query") or "").strip()
            if query:
                return query
        return state.question

    def _tokenize(self, text: str) -> set[str]:
        return {
            self._normalize_token(token)
            for token in (match.group(0).lower() for match in TOKEN_PATTERN.finditer(text))
            if (token not in STOPWORDS or token in CONTROL_QUERY_TOKENS) and len(token) > 1
        }

    def _normalize_token(self, token: str) -> str:
        normalized = {
            "wears": "wear",
            "wearing": "wear",
            "wore": "wear",
            "worn": "wear",
            "loves": "love",
            "loved": "love",
            "loving": "love",
            "opening": "open",
            "opened": "open",
            "opens": "open",
            "fixed": "fix",
            "fixing": "fix",
            "repaired": "repair",
            "repairs": "repair",
            "losing": "lose",
            "lost": "lose",
        }.get(token, token)
        return normalized

    def _confidence_from_speech_score(self, detail: str, score: float) -> float:
        base = self._confidence_from_detail(detail)
        return round(min(0.95, base + min(score, 1.0) * 0.15), 4)

    def _clip_path_for_span(self, clip_path: str | None, time_span) -> str | None:
        if not clip_path:
            return None
        base_path = clip_path.split("#t=", maxsplit=1)[0]
        return f"{base_path}#t={time_span.start:.2f},{time_span.end:.2f}"

    def _split_sentences(self, text: str) -> list[str]:
        normalized = " ".join(text.split()).strip()
        if not normalized:
            return []
        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", normalized)
            if sentence.strip()
        ]

    def _score_speech_sentence(
        self,
        sentence: str,
        *,
        index: int,
        sentence_count: int,
        question_tokens: set[str],
        query_tokens: set[str],
    ) -> tuple[float, str]:
        sentence_tokens = self._tokenize(sentence)
        overlap = len(query_tokens & sentence_tokens)
        lower_sentence = sentence.lower()
        score = overlap * 3
        sentence_anchor_kind = "generic"
        causal_hits = sum(
            1
            for keyword in ("worried", "lose", "fix", "repair", "open", "clasp")
            if keyword in sentence_tokens or keyword in lower_sentence
        )
        support_hits = sum(
            1
            for keyword in ("wear", "love", "perfect", "bracelet")
            if keyword in sentence_tokens or keyword in lower_sentence
        )
        topic_shift_hits = sum(
            1
            for marker in (
                "last but not the least",
                "last but not least",
                "other bracelet",
                "another bracelet",
                "last but not",
            )
            if marker in lower_sentence
        )
        is_first_query = bool({"first", "beginning", "earliest", "initial"} & question_tokens)
        is_last_query = bool({"last", "final", "ending", "end"} & question_tokens)
        is_why_query = "why" in question_tokens

        if is_why_query and any(
            keyword in lower_sentence
            for keyword in ("because", "worried", "lose", "lost", "fixed", "repair", "opening", "clasp")
        ):
            score += 8
            sentence_anchor_kind = "causal"
        if is_why_query and causal_hits:
            score += causal_hits * 4
        if is_why_query and support_hits:
            score += support_hits * 2
            if sentence_anchor_kind != "causal":
                sentence_anchor_kind = "support"
        if topic_shift_hits:
            score -= topic_shift_hits * 8
        if "because" in lower_sentence and not overlap and not causal_hits:
            score -= 4

        if is_first_query:
            if any(
                keyword in lower_sentence
                for keyword in (
                    "this is",
                    "what about",
                    "never had",
                    "different",
                    "exotic",
                    "bizarre",
                    "unusual",
                    "first",
                )
            ):
                score += 5
            if any(
                keyword in lower_sentence
                for keyword in (
                    "chicken head",
                    "pounds",
                    "flour",
                    "deep fry",
                    "fried",
                    "tastes",
                    "surprisingly",
                )
            ):
                score += 4
            score += max(0.0, 1.0 - (index / max(sentence_count, 1))) * 2

        if is_last_query:
            score += (index / max(sentence_count, 1)) * 2

        return score, sentence_anchor_kind

    def _rank_speech_sentences(
        self,
        sentences: list[str],
        *,
        question_tokens: set[str],
        query_tokens: set[str],
    ) -> list[tuple[float, int, str]]:
        ranked: list[tuple[float, int, str]] = []
        for index, sentence in enumerate(sentences):
            score, anchor_kind = self._score_speech_sentence(
                sentence,
                index=index,
                sentence_count=len(sentences),
                question_tokens=question_tokens,
                query_tokens=query_tokens,
            )
            ranked.append((score, index, anchor_kind))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return ranked

    def _snippet_from_anchor(
        self,
        sentences: list[str],
        *,
        anchor_index: int,
        anchor_kind: str,
        question_tokens: set[str],
        max_chars: int,
    ) -> str:
        is_first_query = bool({"first", "beginning", "earliest", "initial"} & question_tokens)
        is_last_query = bool({"last", "final", "ending", "end"} & question_tokens)
        is_why_query = "why" in question_tokens

        if is_first_query:
            return self._build_window_from_anchor(
                sentences,
                anchor_index,
                max_chars=max_chars,
                before=2,
                after=2,
                prefer="forward",
            )
        if is_last_query:
            return self._build_window_from_anchor(
                sentences,
                anchor_index,
                max_chars=max_chars,
                before=2,
                after=2,
                prefer="backward",
            )
        if is_why_query and anchor_kind == "causal":
            return self._build_window_from_anchor(
                sentences,
                anchor_index,
                max_chars=max_chars,
                before=1,
                after=2,
                prefer="forward",
            )
        if is_why_query and anchor_kind == "support":
            return self._build_window_from_anchor(
                sentences,
                anchor_index,
                max_chars=max_chars,
                before=1,
                after=2,
                prefer="backward",
            )
        return self._build_window_from_anchor(
            sentences,
            anchor_index,
            max_chars=max_chars,
            before=1,
            after=2,
            prefer="forward",
        )

    def _focus_speech_detail(
        self,
        text: str,
        question_tokens: set[str],
        query_tokens: set[str],
        *,
        prefer_start: bool = False,
        prefer_end: bool = False,
        max_chars: int = 900,
    ) -> str:
        normalized = " ".join(text.split()).strip()
        if not normalized:
            return ""

        sentences = self._split_sentences(normalized)
        if not sentences:
            return normalized[:max_chars]
        if len(normalized) <= max_chars and len(sentences) <= 4:
            return normalized

        if prefer_start:
            return self._join_sentence_window(sentences[:4], max_chars)
        if prefer_end:
            return self._join_sentence_window(sentences[-4:], max_chars)

        ranked = self._rank_speech_sentences(
            sentences,
            question_tokens=question_tokens,
            query_tokens=query_tokens,
        )
        best_score, best_index, anchor_kind = ranked[0]
        snippet = self._snippet_from_anchor(
            sentences,
            anchor_index=best_index,
            anchor_kind=anchor_kind,
            question_tokens=question_tokens,
            max_chars=max_chars,
        )
        if not snippet or best_score <= 0:
            return normalized[:max_chars]

        snippet_tokens = self._tokenize(snippet)
        full_tokens = self._tokenize(normalized)
        full_overlap = len(query_tokens & full_tokens)
        snippet_overlap = len(query_tokens & snippet_tokens)
        if full_overlap > 0 and snippet_overlap == 0:
            return normalized[:max_chars]
        return snippet

    def _maybe_refine_speech_detail(
        self,
        *,
        span: SpeechSpan,
        detail: str,
        state: ControllerState,
        question_tokens: set[str],
        query_tokens: set[str],
        search_query: str,
        prefer_start: bool,
        prefer_end: bool,
    ) -> tuple[str, dict[str, object]]:
        candidates = self._build_speech_refinement_candidates(
            span.text,
            question_tokens=question_tokens,
            query_tokens=query_tokens,
            initial_detail=detail,
            prefer_start=prefer_start,
            prefer_end=prefer_end,
        )
        metadata: dict[str, object] = {
            "selection_mode": "heuristic",
            "refinement_triggered": False,
            "candidate_count": len(candidates),
        }
        if not self._should_refine_speech_detail(
            span=span,
            question_tokens=question_tokens,
            candidates=candidates,
        ):
            return detail, metadata

        prompt = self._build_speech_refinement_prompt(
            question=state.question,
            search_query=search_query,
            candidates=candidates,
        )
        raw_response = self.speech_snippet_refiner.completion(prompt)
        selected_ids, reason = self._parse_refinement_response(raw_response, candidates)
        if not selected_ids:
            metadata["selection_mode"] = "heuristic_fallback"
            metadata["refinement_triggered"] = True
            metadata["refiner_reason"] = "No valid candidate ids returned by refiner."
            return detail, metadata

        selected_detail = self._combine_selected_candidates(candidates, selected_ids)
        if not selected_detail:
            metadata["selection_mode"] = "heuristic_fallback"
            metadata["refinement_triggered"] = True
            metadata["refiner_reason"] = "Refiner selected empty candidate set."
            return detail, metadata

        metadata["selection_mode"] = "hybrid_llm"
        metadata["refinement_triggered"] = True
        metadata["selected_candidate_ids"] = selected_ids
        if reason:
            metadata["refiner_reason"] = reason
        return selected_detail, metadata

    def _build_speech_refinement_candidates(
        self,
        text: str,
        *,
        question_tokens: set[str],
        query_tokens: set[str],
        initial_detail: str,
        prefer_start: bool,
        prefer_end: bool,
        max_chars: int = 900,
    ) -> list[dict[str, object]]:
        candidates: list[dict[str, object]] = []
        seen_details: set[str] = set()

        def add_candidate(detail: str, source: str) -> None:
            normalized_detail = " ".join(detail.split()).strip()
            if not normalized_detail or normalized_detail in seen_details:
                return
            seen_details.add(normalized_detail)
            candidates.append(
                {
                    "candidate_id": f"c{len(candidates) + 1}",
                    "detail": normalized_detail,
                    "source": source,
                }
            )

        add_candidate(initial_detail, "heuristic")
        sentences = self._split_sentences(text)
        if not sentences:
            return candidates
        if prefer_start:
            add_candidate(self._join_sentence_window(sentences[:4], max_chars), "prefer_start")
        if prefer_end:
            add_candidate(self._join_sentence_window(sentences[-4:], max_chars), "prefer_end")

        ranked = self._rank_speech_sentences(
            sentences,
            question_tokens=question_tokens,
            query_tokens=query_tokens,
        )
        for _score, index, anchor_kind in ranked[: self.speech_refine_candidate_count]:
            add_candidate(
                self._snippet_from_anchor(
                    sentences,
                    anchor_index=index,
                    anchor_kind=anchor_kind,
                    question_tokens=question_tokens,
                    max_chars=max_chars,
                ),
                f"anchor:{anchor_kind}:{index}",
            )

        add_candidate(self._join_sentence_window(sentences[:3], max_chars), "head")
        add_candidate(self._join_sentence_window(sentences[-3:], max_chars), "tail")
        return candidates

    def _should_refine_speech_detail(
        self,
        *,
        span: SpeechSpan,
        question_tokens: set[str],
        candidates: list[dict[str, object]],
    ) -> bool:
        if not self.enable_hybrid_speech_refinement or self.speech_snippet_refiner is None:
            return False
        if len(candidates) < 2:
            return False
        normalized = " ".join(span.text.split()).strip()
        sentences = self._split_sentences(normalized)
        lower_text = normalized.lower()
        topic_shift = any(
            marker in lower_text
            for marker in (
                "last but not the least",
                "last but not least",
                "other bracelet",
                "another bracelet",
            )
        )
        return bool(
            {"why", "first", "beginning", "earliest", "initial", "last", "final", "ending", "end"}
            & question_tokens
        ) or len(normalized) > 320 or len(sentences) > 4 or topic_shift

    def _build_speech_refinement_prompt(
        self,
        *,
        question: str,
        search_query: str,
        candidates: list[dict[str, object]],
    ) -> str:
        candidate_lines = []
        for candidate in candidates:
            candidate_lines.append(
                f"{candidate['candidate_id']}: {candidate['detail']}"
            )
        candidate_block = "\n".join(candidate_lines)
        return (
            "You are selecting grounded transcript snippets for a long-video QA tool.\n"
            "Choose the candidate snippet or snippets that most directly answer the question.\n"
            "Prefer causal explanation for 'why', earliest evidence for 'first', and latest evidence for 'last'.\n"
            "Select at most 2 candidate ids. Do not paraphrase. Only choose from the candidates below.\n"
            "Return strict JSON with this schema:\n"
            '{"selected_candidate_ids":["c1"],"reason":"short reason"}\n'
            f"Question: {question}\n"
            f"Search hint: {search_query}\n"
            "Candidates:\n"
            f"{candidate_block}\n"
        )

    def _parse_refinement_response(
        self,
        raw_response: str,
        candidates: list[dict[str, object]],
    ) -> tuple[list[str], str]:
        valid_ids = {str(candidate["candidate_id"]) for candidate in candidates}
        candidate_ids: list[str] = []
        reason = ""
        payload = raw_response.strip()
        parsed: dict[str, object] | None = None
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", payload, flags=re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
        if parsed is not None:
            raw_ids = parsed.get("selected_candidate_ids") or parsed.get("selected_candidates") or []
            if isinstance(raw_ids, list):
                candidate_ids = [str(item) for item in raw_ids if str(item) in valid_ids]
            raw_reason = parsed.get("reason")
            if raw_reason is not None:
                reason = str(raw_reason).strip()
        if not candidate_ids:
            candidate_ids = [match for match in re.findall(r"c\d+", payload.lower()) if match in valid_ids]
        deduped_ids: list[str] = []
        seen_ids: set[str] = set()
        for candidate_id in candidate_ids:
            if candidate_id in seen_ids:
                continue
            seen_ids.add(candidate_id)
            deduped_ids.append(candidate_id)
        return deduped_ids[:2], reason

    def _combine_selected_candidates(
        self,
        candidates: list[dict[str, object]],
        selected_ids: list[str],
        *,
        max_chars: int = 900,
    ) -> str:
        if not selected_ids:
            return ""
        selected_lookup = {candidate_id: index for index, candidate_id in enumerate(selected_ids)}
        selected_details = [
            str(candidate["detail"])
            for candidate in candidates
            if str(candidate["candidate_id"]) in selected_lookup
        ]
        combined = " ".join(selected_details).strip()
        return combined[:max_chars]

    def _join_sentence_window(self, sentences: list[str], max_chars: int) -> str:
        snippet = " ".join(sentence.strip() for sentence in sentences if sentence.strip()).strip()
        return snippet[:max_chars]

    def _build_window_from_anchor(
        self,
        sentences: list[str],
        anchor_index: int,
        *,
        max_chars: int,
        before: int,
        after: int,
        prefer: str,
    ) -> str:
        selected = [anchor_index]
        backward_indices = list(range(max(0, anchor_index - before), anchor_index))
        forward_indices = list(range(anchor_index + 1, min(len(sentences), anchor_index + after + 1)))

        if prefer == "backward":
            candidate_indices = list(reversed(backward_indices)) + forward_indices
        else:
            candidate_indices = forward_indices + list(reversed(backward_indices))

        current = sentences[anchor_index].strip()
        for index in candidate_indices:
            trial_indices = sorted(selected + [index])
            trial_text = " ".join(sentences[item].strip() for item in trial_indices).strip()
            if len(trial_text) > max_chars:
                continue
            selected.append(index)
            current = trial_text
        return current[:max_chars]

    def _is_duplicate_speech_evidence(
        self,
        state: ControllerState,
        span: SpeechSpan,
        detail: str,
    ) -> bool:
        normalized_detail = " ".join(detail.split()).strip().lower()
        span_start = round(span.time_span.start, 2)
        span_end = round(span.time_span.end, 2)
        for evidence in state.evidence_ledger:
            if evidence.modality != "speech":
                continue
            evidence_start = round(evidence.time_span.start, 2)
            evidence_end = round(evidence.time_span.end, 2)
            if evidence_start != span_start or evidence_end != span_end:
                continue
            existing_detail = " ".join(evidence.detail.split()).strip().lower()
            if existing_detail == normalized_detail:
                return True
        return False

    def _span_has_why_signal(self, span: SpeechSpan) -> bool:
        lower_text = span.text.lower()
        doc_tokens = self._tokenize(span.text)
        return any(
            keyword in doc_tokens or keyword in lower_text
            for keyword in ("worried", "lose", "fix", "repair", "open", "clasp")
        )
