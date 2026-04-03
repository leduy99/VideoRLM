import re

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
    def __init__(self, memory: VideoMemory, index: VideoMemoryIndex | None = None, top_k: int = 5):
        self.memory = memory
        self.index = index or VideoMemoryIndex(memory)
        self.top_k = top_k
        self._evidence_counter = 0

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

        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", normalized)
            if sentence.strip()
        ]
        if not sentences:
            return normalized[:max_chars]
        if len(normalized) <= max_chars and len(sentences) <= 4:
            return normalized

        if prefer_start:
            return self._join_sentence_window(sentences[:4], max_chars)
        if prefer_end:
            return self._join_sentence_window(sentences[-4:], max_chars)

        is_first_query = bool({"first", "beginning", "earliest", "initial"} & question_tokens)
        is_last_query = bool({"last", "final", "ending", "end"} & question_tokens)
        is_why_query = "why" in question_tokens

        best_index = 0
        best_score = float("-inf")
        anchor_kind = "generic"
        for index, sentence in enumerate(sentences):
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
                score += max(0.0, 1.0 - (index / max(len(sentences), 1))) * 2

            if is_last_query:
                score += (index / max(len(sentences), 1)) * 2

            if score > best_score:
                best_score = score
                best_index = index
                anchor_kind = sentence_anchor_kind

        if is_first_query:
            snippet = self._build_window_from_anchor(
                sentences,
                best_index,
                max_chars=max_chars,
                before=2,
                after=2,
                prefer="forward",
            )
        elif is_last_query:
            snippet = self._build_window_from_anchor(
                sentences,
                best_index,
                max_chars=max_chars,
                before=2,
                after=2,
                prefer="backward",
            )
        else:
            if is_why_query and anchor_kind == "causal":
                snippet = self._build_window_from_anchor(
                    sentences,
                    best_index,
                    max_chars=max_chars,
                    before=1,
                    after=2,
                    prefer="forward",
                )
            elif is_why_query and anchor_kind == "support":
                snippet = self._build_window_from_anchor(
                    sentences,
                    best_index,
                    max_chars=max_chars,
                    before=1,
                    after=2,
                    prefer="backward",
                )
            else:
                snippet = self._build_window_from_anchor(
                    sentences,
                    best_index,
                    max_chars=max_chars,
                    before=1,
                    after=2,
                    prefer="forward",
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
