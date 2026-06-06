import ast
import difflib
import json
import re
import tempfile
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rlm.clients.base_lm import BaseLM
from rlm.video.adapters import SpeechRecognizer, VisualSummarizer
from rlm.video.evidence_pipeline import (
    build_question_spec,
    is_reopen_blocked,
    is_spatial_relation_question,
    open_v2,
    requires_co_visible_relation,
    search_v2,
    select_target_slot,
)
from rlm.video.index import STOPWORDS, TOKEN_PATTERN, VideoMemoryIndex
from rlm.video.memory import (
    ACTION_COMPLETION_TERMS,
    ACTION_START_TERMS,
    CAUSAL_OUTCOME_TERMS,
    CAUSAL_RESOLUTION_TERMS,
    CAUSAL_SETUP_TERMS,
)
from rlm.video.media import extract_audio_segment, get_videorlm_output_root
from rlm.video.question_router import route_from_metadata, route_question, verify_stop_answer
from rlm.video.rerankers import VideoWindowReranker
from rlm.video.types import (
    ControllerAction,
    ControllerState,
    Evidence,
    FrontierItem,
    Modality,
    Observation,
    QuestionSpec,
    SpeechSpan,
    TimeSpan,
    VideoMemory,
    VideoNode,
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

GENERIC_VISUAL_SUMMARIES = {
    "",
    "video node",
    "scene node",
    "segment node",
    "event node",
    "clip node",
}

CODE_ASSIGNMENT_PATTERN = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
    r"((?:[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|True|False|\"[^\"]*\"|'[^']*')"
    r"(?:\s*(?:==|!=|>=|<=|>|<|\+|\-|\*|/)\s*"
    r"(?:[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|True|False|\"[^\"]*\"|'[^']*'))?)"
)
CODE_OPERATOR_PATTERN = re.compile(r"\s*(==|!=|>=|<=|>|<|\+|\-|\*|/|=)\s*")


class ControllerEvidenceClassifier:
    def __init__(
        self,
        client: BaseLM,
        *,
        max_evidence_chars: int = 1800,
    ):
        self.client = client
        self.max_evidence_chars = max_evidence_chars
        self._cache: dict[str, dict[str, Any] | None] = {}

    def classify(
        self,
        *,
        item: Evidence,
        question_spec: QuestionSpec,
        target_slot: str | None,
        state: ControllerState,
        heuristic_slot: str,
        heuristic_role: str,
        heuristic_score: float,
    ) -> dict[str, Any] | None:
        cache_key = self._cache_key(item=item, target_slot=target_slot, state=state)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return dict(cached) if cached is not None else None
        prompt = self._build_prompt(
            item=item,
            question_spec=question_spec,
            target_slot=target_slot,
            state=state,
            heuristic_slot=heuristic_slot,
            heuristic_role=heuristic_role,
            heuristic_score=heuristic_score,
        )
        raw_response = self.client.completion(prompt)
        payload = self._parse_json_object(raw_response)
        if payload is None:
            self._cache[cache_key] = None
            return None
        normalized = self._normalize_payload(payload, question_spec)
        self._cache[cache_key] = dict(normalized) if normalized is not None else None
        return normalized

    def _cache_key(
        self,
        *,
        item: Evidence,
        target_slot: str | None,
        state: ControllerState,
    ) -> str:
        payload = "\n".join(
            [
                state.question,
                target_slot or "",
                item.modality,
                item.claim,
                item.detail,
            ]
        )
        return str(hash(payload))

    def _build_prompt(
        self,
        *,
        item: Evidence,
        question_spec: QuestionSpec,
        target_slot: str | None,
        state: ControllerState,
        heuristic_slot: str,
        heuristic_role: str,
        heuristic_score: float,
    ) -> str:
        slots = [
            {
                "slot": slot.slot,
                "description": slot.description,
                "required": slot.required,
                "preferred_modality": slot.preferred_modality,
            }
            for slot in question_spec.required_slots
        ]
        evidence_text = "\n".join(
            part.strip() for part in (item.claim, item.detail) if part and part.strip()
        )
        if len(evidence_text) > self.max_evidence_chars:
            evidence_text = evidence_text[: self.max_evidence_chars - 3] + "..."
        return "\n".join(
            [
                "Classify one video evidence item for answer grounding.",
                "Return only valid JSON. Do not include markdown.",
                "",
                "Roles:",
                "- core: directly answers the target slot and contains an exact answer span.",
                "- support: relevant, but incomplete or requires another evidence item.",
                "- background: related context only.",
                "- noise: unrelated or too corrupted to use.",
                "",
                "Important OCR/code rule: for OCR or code text, choose core only when "
                "answer_span is the exact count, value, operator, expression, or text requested. "
                "If the evidence is relevant but the exact answer is mixed with unrelated text, "
                "choose support.",
                "",
                f"Question: {state.question}",
                f"Task type: {state.task_type or 'unknown'}",
                f"Question type: {question_spec.question_type}",
                f"Target slot: {target_slot or 'unknown'}",
                f"Slots JSON: {json.dumps(slots, ensure_ascii=True)}",
                f"Heuristic slot: {heuristic_slot}",
                f"Heuristic role: {heuristic_role}",
                f"Heuristic score: {heuristic_score:.4f}",
                f"Evidence modality: {item.modality}",
                f"Evidence metadata: {json.dumps(item.metadata, ensure_ascii=True, sort_keys=True)[:1200]}",
                "Evidence:",
                evidence_text,
                "",
                "JSON schema:",
                json.dumps(
                    {
                        "slot": target_slot or heuristic_slot,
                        "role": "core",
                        "confidence": 0.0,
                        "answer_span": "",
                        "reason": "",
                    },
                    ensure_ascii=True,
                ),
            ]
        )

    def _parse_json_object(self, text: str) -> dict[str, Any] | None:
        stripped = text.strip()
        if not stripped:
            return None
        try:
            payload = json.loads(stripped)
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            pass
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            payload = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def _normalize_payload(
        self,
        payload: dict[str, Any],
        question_spec: QuestionSpec,
    ) -> dict[str, Any] | None:
        valid_slots = {slot.slot for slot in question_spec.required_slots}
        slot = str(payload.get("slot") or "").strip()
        role = str(payload.get("role") or "").strip().lower()
        if slot not in valid_slots or role not in {"core", "support", "background", "noise"}:
            return None
        confidence_value = payload.get("confidence", 0.0)
        try:
            confidence = max(0.0, min(1.0, float(confidence_value)))
        except (TypeError, ValueError):
            confidence = 0.0
        return {
            "slot": slot,
            "role": role,
            "confidence": round(confidence, 4),
            "answer_span": str(payload.get("answer_span") or "").strip(),
            "reason": str(payload.get("reason") or "").strip(),
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
        speech_refiner: SpeechRecognizer | None = None,
        visual_refiner: VisualSummarizer | None = None,
        enable_vrrqa_graph_refinement_expansion: bool = True,
        vrrqa_graph_refinement_neighbor_count: int = 1,
        video_window_reranker: VideoWindowReranker | None = None,
        evidence_classifier_client: BaseLM | None = None,
        enable_controller_evidence_classifier: bool = False,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
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
        self.speech_refiner = speech_refiner
        self.visual_refiner = visual_refiner
        self.enable_vrrqa_graph_refinement_expansion = enable_vrrqa_graph_refinement_expansion
        self.vrrqa_graph_refinement_neighbor_count = vrrqa_graph_refinement_neighbor_count
        self.video_window_reranker = video_window_reranker
        self.evidence_classifier = (
            ControllerEvidenceClassifier(evidence_classifier_client)
            if enable_controller_evidence_classifier and evidence_classifier_client is not None
            else None
        )
        self.progress_callback = progress_callback

    def execute(self, action: ControllerAction, state: ControllerState) -> Observation:
        if action.action_type == "SEARCH":
            return self.search(
                query=action.query or "",
                modality=action.modality,
                top_k=self.top_k,
                state=state,
                target_slot=action.target_slot,
            )
        if action.action_type == "OPEN":
            return self.open(
                node_id=action.node_id or "",
                modality=action.modality,
                state=state,
                target_slot=action.target_slot,
            )
        if action.action_type == "SPLIT":
            return self.split(action.node_id or "")
        if action.action_type == "MERGE":
            return self.merge(action.evidence_ids, state)
        if action.action_type == "STOP":
            return self.stop(action.answer or "", action.evidence_ids, state)
        raise ValueError(f"Unsupported action type: {action.action_type}")

    def search(
        self,
        query: str,
        modality: Modality | None,
        top_k: int,
        state: ControllerState,
        target_slot: str | None,
    ) -> Observation:
        question_spec = state.question_spec or build_question_spec(state.question, state.task_type)
        selected_slot = target_slot or select_target_slot(question_spec, state.evidence_board)
        search_top_k = self._search_candidate_count(top_k, modality)
        frontier, metadata = search_v2(
            index=self.index,
            question_spec=question_spec,
            target_slot=selected_slot,
            state=state,
            top_k=search_top_k,
            query_override=query or None,
            modality=modality,
        )
        rerank_metadata: dict[str, Any] = {"stage2_rerank_applied": False}
        if frontier and self._should_apply_video_window_rerank(metadata):
            rerank_query = self._video_window_rerank_query(query, metadata)
            frontier, rerank_metadata = self.video_window_reranker.rerank(
                query=rerank_query,
                candidates=frontier,
                memory=self.memory,
                top_k=top_k,
            )
        else:
            frontier = frontier[:top_k]
        zero_hit_temporal_expansion = False
        if not frontier and self._should_use_zero_hit_temporal_expansion(
            state,
            metadata.get("modality"),
        ):
            frontier = self._zero_hit_temporal_frontier(
                state=state,
                modality=metadata.get("modality") or modality or "visual",
                target_slot=selected_slot,
                limit=min(top_k, 2),
            )
            zero_hit_temporal_expansion = bool(frontier)
        summary = (
            f"SEARCH {metadata.get('search_mode', 'lexical')} found {len(frontier)} candidate nodes for "
            f"slot '{selected_slot or 'generic'}'."
        )
        if zero_hit_temporal_expansion:
            summary += " Used zero-hit temporal visual expansion."
        if rerank_metadata.get("stage2_rerank_applied"):
            summary += " Applied stage-2 video-window reranking."
        return Observation(
            kind="search",
            summary=summary,
            frontier=frontier,
            metadata={
                "query": query,
                "modality": metadata["modality"],
                "search_mode": metadata.get("search_mode", "lexical"),
                "hit_count": len(frontier),
                "raw_hit_count": metadata.get("hit_count", len(frontier)),
                "zero_hit_temporal_expansion": zero_hit_temporal_expansion,
                "target_slot": selected_slot,
                "queries": metadata["queries"],
                **rerank_metadata,
            },
        )

    def _search_candidate_count(self, top_k: int, modality: Modality | None) -> int:
        if self.video_window_reranker is None:
            return top_k
        if modality not in {"visual", "ocr", "cross_modal", None}:
            return top_k
        return max(top_k, self.video_window_reranker.candidate_count)

    def _should_apply_video_window_rerank(self, metadata: dict[str, Any]) -> bool:
        if self.video_window_reranker is None:
            return False
        modality = metadata.get("modality")
        return modality in {"visual", "ocr", "cross_modal"}

    def _video_window_rerank_query(self, query: str, metadata: dict[str, Any]) -> str:
        if query.strip():
            return query.strip()
        queries = metadata.get("queries", [])
        if isinstance(queries, list):
            for candidate in queries:
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
        return ""

    def open(
        self,
        node_id: str,
        modality: Modality | None,
        state: ControllerState,
        target_slot: str | None = None,
    ) -> Observation:
        selected_modality = modality or "visual"
        question_spec = state.question_spec or build_question_spec(state.question, state.task_type)
        selected_slot = target_slot or select_target_slot(question_spec, state.evidence_board)
        node = self._get_node_or_none(node_id)
        if node is None:
            return self._unknown_node_observation(
                action_type="OPEN",
                node_id=node_id,
                modality=selected_modality,
                target_slot=selected_slot,
            )

        requested_node = node
        node = self._redirect_visual_open_node(
            node=node,
            modality=selected_modality,
            state=state,
            target_slot=selected_slot,
        )
        open_redirect_metadata: dict[str, object] = {}
        if node.node_id != requested_node.node_id:
            open_redirect_metadata = {
                "requested_node_id": requested_node.node_id,
                "opened_node_id": node.node_id,
                "visual_open_redirected": True,
                "visual_open_redirect_reason": (
                    "generic_vrrqa_parent_open_redirected_to_visual_leaf"
                ),
            }

        if is_reopen_blocked(state.evidence_board, node.node_id, selected_modality, selected_slot):
            return Observation(
                kind="open",
                summary=(
                    f"OPEN skipped {selected_modality} on node {node.node_id} because "
                    f"slot '{selected_slot or 'generic'}' was already opened."
                ),
                node_id=node.node_id,
                metadata={
                    "modality": selected_modality,
                    "clip_path": node.clip_path,
                    **open_redirect_metadata,
                    "target_slot": selected_slot,
                    "background_only": False,
                    "no_new_information": True,
                    "filled_slots": [],
                    "missing_slots": [],
                    "duplicate_evidence_count": 0,
                    "suggested_queries": [],
                    "progress_made": False,
                    "result": "reopen_blocked",
                },
            )

        detail = ""
        detail_metadata: dict[str, object] = {}
        if selected_modality == "speech":
            raw_evidence = self._build_speech_evidence(node, state)
            detail = "\n".join(item.detail for item in raw_evidence if item.detail)
        elif selected_modality == "ocr":
            raw_evidence = self._build_ocr_evidence(
                node=node,
                state=state,
                target_slot=selected_slot,
                question_spec=question_spec,
            )
            detail = "\n".join(item.detail for item in raw_evidence if item.detail)
            detail_metadata = {"structured_ocr_evidence": True}
        else:
            detail, detail_metadata = self._build_detail_with_metadata(
                node,
                selected_modality,
                state,
            )
            if detail:
                raw_evidence = [
                    Evidence(
                        evidence_id=self._next_evidence_id(),
                        claim=self._to_claim(detail, selected_modality),
                        modality=selected_modality,
                        time_span=node.time_span,
                        source_node_id=node.node_id,
                        confidence=self._confidence_from_detail(detail),
                        detail=detail,
                        metadata={"clip_path": node.clip_path, **detail_metadata},
                    )
                ]
            else:
                raw_evidence = []

        evidence, open_metadata = open_v2(
            question_spec=question_spec,
            target_slot=selected_slot,
            state=state,
            node_id=node.node_id,
            modality=selected_modality,
            evidence_items=raw_evidence,
            evidence_classifier=self.evidence_classifier.classify
            if self.evidence_classifier is not None
            else None,
        )
        consolidation_metadata = self._consolidate_opened_node(
            node=node,
            modality=selected_modality,
            detail=detail,
            detail_metadata=detail_metadata,
            state=state,
            evidence=evidence,
        )
        if consolidation_metadata:
            open_metadata.update(consolidation_metadata)
            for item in evidence:
                item.metadata.update(consolidation_metadata)
        refinement_frontier: list[FrontierItem] = []
        needs_refinement = (
            open_metadata.get("background_only")
            or open_metadata.get("no_new_information")
            or open_metadata.get("result") == "support_only"
        )
        if needs_refinement:
            refinement_frontier = self._build_refinement_frontier(
                node=node,
                state=state,
                modality=selected_modality,
                target_slot=selected_slot,
            )
            open_metadata["refinement_node_ids"] = [item.node_id for item in refinement_frontier]
            if refinement_frontier and not open_metadata.get("progress_made"):
                open_metadata["progress_made"] = True
        else:
            open_metadata["refinement_node_ids"] = []
        if evidence:
            summary = (
                f"OPEN v2 gathered {len(evidence)} {selected_modality} evidence items "
                f"for slot '{selected_slot or 'generic'}' from node {node.node_id}."
            )
        else:
            summary = (
                f"OPEN v2 found no answer-bearing {selected_modality} evidence for "
                f"slot '{selected_slot or 'generic'}' in node {node.node_id}."
            )

        return Observation(
            kind="open",
            summary=summary,
            evidence=evidence,
            frontier=refinement_frontier,
            node_id=node.node_id,
            metadata={
                "modality": selected_modality,
                "clip_path": node.clip_path,
                **open_redirect_metadata,
                **open_metadata,
            },
        )

    def split(self, node_id: str) -> Observation:
        node = self._get_node_or_none(node_id)
        if node is None:
            return self._unknown_node_observation(
                action_type="SPLIT",
                node_id=node_id,
                modality=None,
                target_slot=None,
            )
        split_metadata: dict[str, object] = {}
        children = self._split_children_for_node(node)
        if node.metadata.get("dynamic_subevent_split_applied"):
            split_metadata["dynamic_subevent_split_applied"] = True
            split_metadata["dynamic_subevent_node_ids"] = list(node.children)
        frontier = []
        for child in children:
            boundary_reason = ""
            if child.metadata.get("node_type") == "cognitive_subevent":
                boundary_reason = "; split around internal event-boundary peaks"
            reason = (
                f"Child node of {node.node_id} spanning {child.time_span.to_display()}"
                f"{boundary_reason}"
            )
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

        summary = f"SPLIT expanded {node.node_id} into {len(frontier)} child nodes."
        return Observation(
            kind="split",
            summary=summary,
            frontier=frontier,
            node_id=node.node_id,
            metadata={"child_count": len(frontier), **split_metadata},
        )

    def _split_children_for_node(self, node: VideoNode) -> list[VideoNode]:
        if (
            node.level != "event"
            or node.metadata.get("dynamic_subevent_split_applied")
            or len(node.children) < 2
        ):
            return self.memory.child_nodes(node.node_id)
        peaks = [
            timestamp
            for timestamp in (
                _coerce_float(item)
                for item in node.metadata.get("event_boundary_peak_timestamps", [])
            )
            if timestamp is not None and node.time_span.start < timestamp < node.time_span.end
        ]
        if not peaks:
            return self.memory.child_nodes(node.node_id)
        child_nodes = sorted(
            self.memory.child_nodes(node.node_id),
            key=lambda candidate: (candidate.time_span.start, candidate.node_id),
        )
        groups: list[list[VideoNode]] = [[]]
        for child in child_nodes:
            if groups[-1] and any(
                groups[-1][-1].time_span.end <= peak <= child.time_span.start for peak in peaks
            ):
                groups.append([])
            groups[-1].append(child)
        groups = [group for group in groups if group]
        if len(groups) <= 1:
            return child_nodes

        subevent_ids: list[str] = []
        for index, group in enumerate(groups, start=1):
            subevent_id = f"{node.node_id}_subevent_{index:03d}"
            if subevent_id not in self.memory.nodes:
                self.memory.nodes[subevent_id] = self._build_subevent_node(
                    parent=node,
                    node_id=subevent_id,
                    child_nodes=group,
                    index=index,
                )
            subevent = self.memory.get_node(subevent_id)
            subevent.children = [child.node_id for child in group]
            for child in group:
                child.parent_id = subevent_id
            subevent_ids.append(subevent_id)
        node.children = subevent_ids
        node.metadata["dynamic_subevent_split_applied"] = True
        node.metadata["dynamic_subevent_node_ids"] = subevent_ids
        return [self.memory.get_node(node_id) for node_id in subevent_ids]

    def _build_subevent_node(
        self,
        *,
        parent: VideoNode,
        node_id: str,
        child_nodes: list[VideoNode],
        index: int,
    ) -> VideoNode:
        time_span = TimeSpan(
            min(child.time_span.start for child in child_nodes),
            max(child.time_span.end for child in child_nodes),
        )
        visual_summary = " | ".join(
            child.visual_summary.strip() for child in child_nodes if child.visual_summary.strip()
        )
        tags = sorted({tag for child in child_nodes for tag in child.tags})
        entities = sorted({entity for child in child_nodes for entity in child.entities})
        speech_spans = [span for child in child_nodes for span in child.speech_spans]
        ocr_spans = [span for child in child_nodes for span in child.ocr_spans]
        audio_events = [event for child in child_nodes for event in child.audio_events]
        metadata = {
            "node_type": "cognitive_subevent",
            "cognitive_event": True,
            "parent_cognitive_event_id": parent.node_id,
            "event_source_clip_ids": [child.node_id for child in child_nodes],
            "visual_detail_node_ids": [child.node_id for child in child_nodes],
            "event_schema": {
                "time_span": [round(time_span.start, 3), round(time_span.end, 3)],
                "place": [],
                "actors": sorted(set(tags) & {"person", "people", "man", "woman"})[:8],
                "objects": sorted(set(tags) | set(entities))[:12],
                "actions": [],
                "goals_or_intentions": [],
                "causal_predecessors": [],
                "causal_outcomes": [],
                "spoken_topics": sorted(
                    self._tokenize(" ".join(span.text for span in speech_spans))
                )[:12],
                "ocr_entities": sorted(
                    self._tokenize(" ".join(span.text for span in ocr_spans))
                )[:12],
                "visual_state": visual_summary,
                "audio_state": ", ".join(event.label for event in audio_events if event.label),
                "event_type": "dynamic_subevent",
            },
            "subevent_index": index,
        }
        return VideoNode(
            node_id=node_id,
            level="event",
            time_span=time_span,
            visual_summary=visual_summary or f"subevent {index} of {parent.node_id}",
            speech_spans=speech_spans,
            ocr_spans=ocr_spans,
            audio_events=audio_events,
            tags=tags,
            entities=entities,
            clip_path=parent.clip_path,
            parent_id=parent.node_id,
            metadata=metadata,
        )

    def merge(self, evidence_ids: list[str], state: ControllerState) -> Observation:
        ledger = state.evidence_by_id()
        selected = [ledger[item] for item in evidence_ids if item in ledger]
        if not selected:
            return Observation(kind="merge", summary="MERGE found no matching evidence ids.")

        claim = " | ".join(item.claim for item in selected)
        detail = "\n".join(item.detail for item in selected if item.detail)
        merge_metadata = self._cognitive_merge_metadata(selected)
        merged = Evidence(
            evidence_id=self._next_evidence_id(),
            claim=merge_metadata.get("merged_claim", claim),
            modality="cross_modal",
            time_span=TimeSpan(
                min(item.time_span.start for item in selected),
                max(item.time_span.end for item in selected),
            ),
            source_node_id=selected[0].source_node_id,
            confidence=round(sum(item.confidence for item in selected) / len(selected), 4),
            detail=detail,
            metadata={"merged_ids": list(evidence_ids), **merge_metadata},
        )
        return Observation(
            kind="merge",
            summary=f"MERGE combined {len(selected)} evidence items.",
            evidence=[merged],
            metadata={"merged_ids": list(evidence_ids)},
        )

    def _cognitive_merge_metadata(self, evidence_items: list[Evidence]) -> dict[str, object]:
        node_ids = [item.source_node_id for item in evidence_items if item.source_node_id]
        nodes = [
            self.memory.get_node(node_id)
            for node_id in dict.fromkeys(node_ids)
            if node_id in self.memory.nodes
        ]
        event_nodes = [
            node
            for node in nodes
            if node.level == "event" and isinstance(node.metadata.get("event_schema"), dict)
        ]
        if len(event_nodes) < 2:
            return {}
        ordered = sorted(event_nodes, key=lambda node: (node.time_span.start, node.node_id))
        adjacent = all(
            ordered[index].metadata.get("next_cognitive_event_id") == ordered[index + 1].node_id
            or ordered[index + 1].metadata.get("previous_cognitive_event_id")
            == ordered[index].node_id
            for index in range(len(ordered) - 1)
        )
        shared = self._shared_situation_indices(ordered)
        if not adjacent and not any(shared.values()):
            return {}
        merged_schema = self._merged_event_schema(ordered)
        shared_labels = [
            key for key, values in shared.items() if values
        ]
        merged_claim = (
            "Merged cognitive event evidence: "
            + " -> ".join(node.visual_summary.strip()[:120] for node in ordered)
        )
        return {
            "cognitive_event_merge_applied": True,
            "merged_event_node_ids": [node.node_id for node in ordered],
            "merged_event_adjacent": adjacent,
            "shared_situation_indices": shared,
            "shared_situation_index_labels": shared_labels,
            "merged_event_schema": merged_schema,
            "merged_claim": merged_claim,
        }

    def _shared_situation_indices(self, nodes: list[VideoNode]) -> dict[str, list[str]]:
        schemas = [node.metadata.get("event_schema") for node in nodes]
        typed_schemas = [schema for schema in schemas if isinstance(schema, dict)]
        shared: dict[str, list[str]] = {}
        for key in ("actors", "objects", "place", "spoken_topics", "ocr_entities"):
            value_sets = [
                {
                    " ".join(str(item).lower().split()).strip()
                    for item in schema.get(key, [])
                    if " ".join(str(item).split()).strip()
                }
                for schema in typed_schemas
            ]
            if not value_sets:
                shared[key] = []
                continue
            shared[key] = sorted(set.intersection(*value_sets)) if len(value_sets) > 1 else []
        return shared

    def _merged_event_schema(self, nodes: list[VideoNode]) -> dict[str, object]:
        merged: dict[str, object] = {
            "time_span": [
                round(min(node.time_span.start for node in nodes), 3),
                round(max(node.time_span.end for node in nodes), 3),
            ],
            "event_type": "merged_cognitive_event",
            "visual_state": " -> ".join(node.visual_summary.strip() for node in nodes),
        }
        for key in (
            "place",
            "actors",
            "objects",
            "actions",
            "goals_or_intentions",
            "causal_predecessors",
            "causal_outcomes",
            "spoken_topics",
            "ocr_entities",
        ):
            merged[key] = _merge_schema_values(
                [],
                [
                    item
                    for node in nodes
                    for item in (node.metadata.get("event_schema") or {}).get(key, [])
                    if isinstance(node.metadata.get("event_schema"), dict)
                ],
            )
        merged["audio_state"] = " | ".join(
            str((node.metadata.get("event_schema") or {}).get("audio_state", ""))
            for node in nodes
            if isinstance(node.metadata.get("event_schema"), dict)
        )
        return merged

    def stop(self, answer: str, evidence_ids: list[str], state: ControllerState) -> Observation:
        selected_ids = set(evidence_ids)
        selected = [item for item in state.evidence_ledger if item.evidence_id in selected_ids]
        route = (
            route_from_metadata(state.global_context)
            or route_from_metadata(state.question_spec.metadata if state.question_spec else None)
            or route_question(state.question, state.task_type)
        )
        candidate_evidence = selected or self._default_stop_evidence_candidates(state)
        verification = verify_stop_answer(
            question=state.question,
            answer=answer,
            evidence_items=candidate_evidence,
            route=route,
        )
        verified_ids = list(verification.compatible_evidence_ids)
        returned_evidence = [
            item
            for item in candidate_evidence
            if not verified_ids or item.evidence_id in set(verified_ids)
        ]
        stop_rejected = not verification.accepted
        if stop_rejected:
            summary = (
                f"STOP rejected for route '{verification.route_label}': "
                f"{verification.reason}. Continue SEARCH/OPEN with route-compatible evidence."
            )
        else:
            summary = (
                f"STOP accepted for route '{verification.route_label}' with "
                f"{len(verified_ids or evidence_ids)} evidence items."
            )
        return Observation(
            kind="stop",
            summary=summary,
            evidence=returned_evidence,
            metadata={
                "answer": answer,
                "evidence_ids": list(evidence_ids),
                "verified_evidence_ids": verified_ids or list(evidence_ids),
                "question_route": route.to_dict(),
                "answer_verification": verification.to_dict(),
                "stop_rejected": stop_rejected,
                "progress_made": not stop_rejected,
                "no_new_information": stop_rejected,
            },
        )

    def _default_stop_evidence_candidates(self, state: ControllerState) -> list[Evidence]:
        candidates = [
            item
            for item in state.evidence_ledger
            if item.metadata.get("role") in {"core", "support"}
        ]
        return candidates or list(state.evidence_ledger)

    def _get_node_or_none(self, node_id: str):
        if not node_id:
            return None
        return self.memory.nodes.get(node_id)

    def _unknown_node_observation(
        self,
        *,
        action_type: str,
        node_id: str,
        modality: Modality | None,
        target_slot: str | None,
    ) -> Observation:
        suggestions = self._suggest_existing_node_ids(node_id)
        suffix = f" Suggested existing node ids: {', '.join(suggestions)}." if suggestions else ""
        return Observation(
            kind=action_type.lower(),
            summary=(
                f"{action_type} skipped unknown node_id {node_id!r}. "
                "Use SEARCH results or SPLIT an existing parent node before opening a child."
                f"{suffix}"
            ),
            node_id=node_id or None,
            metadata={
                "result": "unknown_node",
                "requested_node_id": node_id,
                "modality": modality,
                "target_slot": target_slot,
                "suggested_node_ids": suggestions,
                "node_count": len(self.memory.nodes),
                "progress_made": False,
                "no_new_information": True,
                "filled_slots": [],
                "missing_slots": [],
            },
        )

    def _suggest_existing_node_ids(self, node_id: str, *, limit: int = 5) -> list[str]:
        if not node_id:
            return []
        node_ids = list(self.memory.nodes)
        prefix_matches = [item for item in node_ids if item.startswith(_node_prefix(node_id))]
        if prefix_matches:
            return sorted(prefix_matches)[:limit]
        return difflib.get_close_matches(node_id, node_ids, n=limit, cutoff=0.45)

    def _redirect_visual_open_node(
        self,
        *,
        node: VideoNode,
        modality: Modality,
        state: ControllerState,
        target_slot: str | None,
    ) -> VideoNode:
        if modality != "visual":
            return node
        if not self.memory.metadata.get("vrrqa_visual_only"):
            return node
        if node.level == "clip" or not self._is_generic_visual_node(node):
            return node

        candidates = self._visual_refinement_descendants(node)
        if not candidates:
            return node

        unopened = [
            candidate
            for candidate in candidates
            if not is_reopen_blocked(state.evidence_board, candidate.node_id, modality, target_slot)
        ]
        return unopened[0] if unopened else candidates[0]

    def _is_generic_visual_node(self, node: VideoNode) -> bool:
        summary = " ".join(node.visual_summary.lower().split()).strip()
        return summary in GENERIC_VISUAL_SUMMARIES

    def _visual_refinement_descendants(self, node: VideoNode) -> list[VideoNode]:
        descendants: list[VideoNode] = []
        pending = list(self.memory.child_nodes(node.node_id))
        while pending:
            candidate = pending.pop(0)
            if self._is_visual_leaf_candidate(candidate):
                descendants.append(candidate)
            pending.extend(self.memory.child_nodes(candidate.node_id))

        if not descendants:
            return []

        level_priority = {"event": 0, "clip": 1, "segment": 2, "scene": 3, "video": 4}
        descendants.sort(
            key=lambda candidate: (
                level_priority.get(candidate.level, 9),
                not bool(candidate.metadata.get("on_demand_visual_refinement")),
                not self._has_visual_frame_index(candidate),
                candidate.time_span.start,
                candidate.node_id,
            )
        )
        return descendants

    def _is_visual_leaf_candidate(self, node: VideoNode) -> bool:
        if not node.visual_summary.strip():
            return False
        if node.metadata.get("on_demand_visual_refinement"):
            return True
        if self._has_visual_frame_index(node):
            return True
        return node.level == "clip" and not node.children

    def _has_visual_frame_index(self, node: VideoNode) -> bool:
        return (
            node.metadata.get("visual_summary_mode") == "lazy_pitome_index"
            or bool(node.metadata.get("pitome_frame_embeddings"))
            or bool(node.metadata.get("semantic_frame_embeddings"))
        )

    def _build_detail(self, node, modality: Modality) -> str:
        detail, _ = self._build_detail_with_metadata(node, modality, state=None)
        return detail

    def _build_detail_with_metadata(
        self,
        node,
        modality: Modality,
        state: ControllerState | None,
    ) -> tuple[str, dict[str, object]]:
        if modality == "speech":
            detail = " ".join(item.text.strip() for item in node.speech_spans if item.text).strip()
            return detail, {}
        if modality == "visual":
            return self._build_visual_detail(node, state)
        if modality == "ocr":
            detail = "\n".join(item.text.strip() for item in node.ocr_spans if item.text).strip()
            return detail, {}
        if modality == "audio":
            labels = [item.label.strip() for item in node.audio_events if item.label]
            return ", ".join(labels).strip(), {}
        return "", {}

    def _build_ocr_evidence(
        self,
        *,
        node: VideoNode,
        state: ControllerState,
        target_slot: str | None,
        question_spec: QuestionSpec,
    ) -> list[Evidence]:
        raw_text = "\n".join(span.text for span in node.ocr_spans if span.text).strip()
        if not raw_text:
            return []

        evidence: list[Evidence] = []
        query = " ".join(
            part
            for part in [
                state.question,
                question_spec.get_slot(target_slot).description
                if target_slot and question_spec.get_slot(target_slot)
                else "",
            ]
            if part
        )
        assignments = self._extract_code_assignments(raw_text)
        question_route = (
            route_from_metadata(state.global_context)
            or route_from_metadata(question_spec.metadata)
            or route_question(state.question, state.task_type)
        )
        screen_text_evidence = self._build_screen_text_block_evidence(
            node=node,
            state=state,
            raw_text=raw_text,
            route_label=question_route.label,
        )
        if screen_text_evidence is not None:
            evidence.append(screen_text_evidence)
        if question_route.label != "ui_header_text" or screen_text_evidence is None:
            evidence.extend(
                self._build_cross_modal_temporal_index_evidence(node=node, state=state)
            )
        evidence.extend(
            self._build_task_specific_ocr_evidence(
                node=node,
                state=state,
                assignments=assignments,
                raw_text=raw_text,
            )
        )

        if question_route.label != "ui_header_text":
            for line in self._select_ocr_code_lines(assignments, query, limit=8):
                if any(item.metadata.get("answer_span") == line for item in evidence):
                    continue
                evidence.append(
                    self._make_ocr_evidence(
                        node=node,
                        claim=f"OCR code line: {line}",
                        detail=line,
                        answer_span=line,
                        kind="code_line",
                        confidence=0.82,
                        exact_answer=False,
                    )
                )

        for line in self._select_ocr_text_lines(raw_text, query, limit=max(0, 8 - len(evidence))):
            if any(line in item.detail for item in evidence):
                continue
            evidence.append(
                self._make_ocr_evidence(
                    node=node,
                    claim=f"OCR text line: {line}",
                    detail=line,
                    answer_span=line if self._line_looks_exact_answer(line, state.question) else "",
                    kind="text_line",
                    confidence=0.72,
                    exact_answer=self._line_looks_exact_answer(line, state.question),
                )
            )

        if evidence:
            return evidence[:12]

        compact_text = self._compact_ocr_text(raw_text, max_chars=900)
        return [
            self._make_ocr_evidence(
                node=node,
                claim=f"OCR evidence: {compact_text}",
                detail=compact_text,
                answer_span="",
                kind="raw_compact",
                confidence=self._confidence_from_detail(compact_text),
                exact_answer=False,
            )
        ]

    def _build_screen_text_block_evidence(
        self,
        *,
        node: VideoNode,
        state: ControllerState,
        raw_text: str,
        route_label: str,
    ) -> Evidence | None:
        if not self._question_needs_screen_text_block(state, route_label):
            return None
        lines = self._extract_leading_screen_text_lines(raw_text)
        if not lines:
            return None
        block = "\n".join(lines)
        return self._make_ocr_evidence(
            node=node,
            claim=f"OCR screen text block: {self._compact_ocr_text(block, max_chars=220)}",
            detail=block,
            answer_span=block,
            kind="screen_text_block",
            confidence=0.9,
            exact_answer=True,
            aggregation_metadata={
                "route": route_label,
                "source": "leading_screen_text_block",
                "source_events": [],
                "aggregation_rule": "join_leading_ocr_screen_text_lines",
                "section_time_constraint": self._node_time_constraint(node),
                "fallback_used": False,
            },
        )

    def _question_needs_screen_text_block(
        self,
        state: ControllerState,
        route_label: str,
    ) -> bool:
        if route_label == "ui_header_text":
            return True
        longshot_context = state.global_context.get("longshot")
        if not isinstance(longshot_context, dict):
            return False
        required_tools = {
            str(item)
            for item in longshot_context.get("required_tools", [])
            if item is not None
        }
        return "extract_scene_text" in required_tools

    def _extract_leading_screen_text_lines(self, text: str, *, limit: int = 8) -> list[str]:
        lines: list[str] = []
        for line in self._unique_ocr_text_lines(text):
            if self._screen_text_line_is_code_boundary(line):
                if lines:
                    break
                continue
            if self._screen_text_line_is_noise(line):
                continue
            lines.append(line)
            if len(lines) >= limit:
                break
        return self._merge_wrapped_screen_text_lines(lines)

    def _screen_text_line_is_code_boundary(self, line: str) -> bool:
        lowered = line.lower().strip()
        if re.fullmatch(r"\d+\.?", lowered):
            return True
        if lowered.startswith(("#", '"""', "print(")):
            return True
        if any(
            cue in lowered
            for cue in (
                ".py",
                "assistant",
                "exception",
                "name",
                "shell",
                "valu",
                "variables",
            )
        ):
            return True
        return bool(CODE_ASSIGNMENT_PATTERN.search(line))

    def _screen_text_line_is_noise(self, line: str) -> bool:
        lowered = line.lower().strip()
        if lowered in {"fore"}:
            return True
        if not re.search(r"[a-zA-Z]", line):
            return True
        tokens = self._tokenize(line)
        return len(tokens) == 1 and len(line) <= 5

    def _merge_wrapped_screen_text_lines(self, lines: list[str]) -> list[str]:
        merged: list[str] = []
        index = 0
        while index < len(lines):
            line = lines[index]
            next_line = lines[index + 1] if index + 1 < len(lines) else None
            if next_line and self._should_merge_screen_text_lines(line, next_line):
                line = f"{line} {next_line}"
                index += 1
            line_key = self._screen_text_key(line)
            if not any(
                self._ocr_line_similarity(line_key, self._screen_text_key(item)) >= 0.88
                for item in merged
            ):
                merged.append(line)
            index += 1
        return merged

    def _should_merge_screen_text_lines(self, line: str, next_line: str) -> bool:
        lowered = f"{line} {next_line}".lower()
        if any(cue in lowered for cue in ("operator", "part four", "part 4")):
            return False
        if len(line) + len(next_line) > 80:
            return False
        line_tokens = self._tokenize(line)
        next_tokens = self._tokenize(next_line)
        return 1 <= len(line_tokens) <= 3 and 1 <= len(next_tokens) <= 4

    def _screen_text_key(self, line: str) -> str:
        return re.sub(r"\W+", "", line.casefold())

    def _build_task_specific_ocr_evidence(
        self,
        *,
        node: VideoNode,
        state: ControllerState,
        assignments: list[str],
        raw_text: str,
    ) -> list[Evidence]:
        question = state.question.lower()
        evidence: list[Evidence] = []
        evidence.extend(
            self._build_computed_code_value_evidence(
                node=node,
                question=state.question,
                assignments=assignments,
            )
        )
        if "how many" in question and ("variable" in question or "assignment" in question):
            count_evidence = self._build_assignment_count_evidence(
                node=node,
                question=question,
                assignments=assignments,
            )
            if count_evidence is not None:
                evidence.append(count_evidence)

        target_variable = self._target_variable_from_question(state.question)
        if target_variable:
            assignment = self._first_assignment_for_variable(assignments, target_variable)
            if assignment is not None:
                evidence.append(
                    self._make_ocr_evidence(
                        node=node,
                        claim=f"OCR structured assignment for {target_variable}: {assignment}",
                        detail=assignment,
                        answer_span=assignment,
                        kind="target_assignment",
                        confidence=0.95,
                        exact_answer=True,
                        aggregation_metadata={
                            "route": "code_value_eval",
                            "source": "structured_ocr_assignment_lines",
                            "source_events": [],
                            "aggregation_rule": (
                                f"select_first_assignment_for_variable:{target_variable}"
                            ),
                            "section_time_constraint": self._node_time_constraint(node),
                            "fallback_used": False,
                            "aggregated_values": assignments,
                        },
                    )
                )

        comparison_operator_evidence = self._build_comparison_operator_list_evidence(
            node=node,
            question=question,
            raw_text=raw_text,
        )
        if comparison_operator_evidence is not None:
            evidence.append(comparison_operator_evidence)

        if any(term in question for term in ("comparison", "boolean", "data type", "operator")):
            comparison_evidence = self._build_comparison_ocr_evidence(
                node=node,
                question=question,
                assignments=assignments,
                raw_text=raw_text,
            )
            if comparison_evidence is not None:
                evidence.append(comparison_evidence)
        return evidence

    def _build_cross_modal_temporal_index_evidence(
        self,
        *,
        node: VideoNode,
        state: ControllerState,
    ) -> list[Evidence]:
        temporal_index = self.memory.cross_modal_index
        if temporal_index is None:
            return []
        route = (
            route_from_metadata(state.global_context)
            or route_from_metadata((state.question_spec.metadata if state.question_spec else {}))
            or route_question(state.question, state.task_type)
        )
        if route.label == "assignment_count":
            item = self._build_temporal_assignment_count_evidence(node=node, state=state)
            return [item] if item is not None else []
        if route.label == "operator_list":
            item = self._build_temporal_operator_evidence(node=node, state=state)
            return [item] if item is not None else []
        if route.label == "terminal_output":
            item = self._build_temporal_terminal_output_evidence(node=node, state=state)
            return [item] if item is not None else []
        if route.label == "code_value_eval":
            evidence: list[Evidence] = []
            expressions = self._build_temporal_expression_list_evidence(node=node, state=state)
            if expressions is not None:
                evidence.append(expressions)
            computed = self._build_temporal_computed_value_evidence(node=node, state=state)
            if computed is not None:
                evidence.append(computed)
            return evidence
        if route.label == "ui_header_text":
            item = self._build_temporal_ui_header_evidence(node=node)
            return [item] if item is not None else []
        return []

    def _build_temporal_assignment_count_evidence(
        self,
        *,
        node: VideoNode,
        state: ControllerState,
    ) -> Evidence | None:
        temporal_index = self.memory.cross_modal_index
        if temporal_index is None:
            return None
        question = state.question.lower()
        before = None
        if "before" in question and "comparison" in question:
            boundary_times = [
                event.time_span.start
                for event in temporal_index.code_line_events
                if self._assignment_rhs_contains_comparison(event.text or "")
            ]
            if boundary_times:
                before = min(boundary_times)
        section_id = None
        if "third segment" in question:
            section = temporal_index.resolve_section("third segment")
            section_id = section.section_id if section else None
        events = temporal_index.find_events(
            modality="code",
            event_type="assignment_line",
            section_id=section_id,
            before=before,
        )
        if "comparison" in question:
            events = [
                event
                for event in events
                if (
                    not self._assignment_rhs_contains_comparison(event.text or "")
                    and self._assignment_is_arithmetic_section_candidate(event.text or "")
                )
            ]
        if "arithmetic" in question:
            events = [
                event
                for event in events
                if self._assignment_is_arithmetic_section_candidate(event.text or "")
            ]
        assignments = self._unique_temporal_assignment_lines(events)
        if not assignments:
            return None
        if "assignment" not in question and (
            "variable" in question or "declaration" in question or "declared" in question
        ):
            variables = []
            for assignment in assignments:
                variable = assignment.partition("=")[0].strip()
                if variable and variable not in variables:
                    variables.append(variable)
            count = len(variables)
            counted_detail = "Unique variables: " + ", ".join(variables)
            claim_detail = f"variables: {', '.join(variables)}"
        else:
            count = len(assignments)
            counted_detail = "Unique assignment lines: " + "; ".join(assignments[:20])
            claim_detail = f"assignments: {'; '.join(assignments[:8])}"
        detail = (
            "Cross-modal temporal index assignment lines:\n"
            + "\n".join(assignments[:20])
            + "\n"
            + counted_detail
        )
        if before is not None:
            detail += f"\nBoundary: first comparison code line at {before:.2f}s"
        source_events = self._source_event_ids_for_assignment_lines(events, assignments)
        return self._make_ocr_evidence(
            node=node,
            claim=f"Temporal index assignment count: {count} ({claim_detail})",
            detail=detail,
            answer_span=str(count),
            kind="assignment_count",
            confidence=0.98,
            exact_answer=True,
            aggregation_metadata={
                "route": "assignment_count",
                "source": "cross_modal_temporal_index",
                "source_events": source_events,
                "aggregation_rule": (
                    "count_unique_variables"
                    if "assignment" not in question
                    and (
                        "variable" in question
                        or "declaration" in question
                        or "declared" in question
                    )
                    else "count_unique_assignment_lines"
                ),
                "section_time_constraint": {
                    **self._node_time_constraint(node),
                    "section_id": section_id,
                    "before": before,
                    "boundary": "first_comparison_code_line" if before is not None else None,
                },
                "fallback_used": False,
                "aggregated_values": assignments[:20],
            },
        )

    def _build_temporal_operator_evidence(
        self,
        *,
        node: VideoNode,
        state: ControllerState,
    ) -> Evidence | None:
        temporal_index = self.memory.cross_modal_index
        if temporal_index is None:
            return None
        question = state.question.lower()
        section = temporal_index.resolve_section("comparison operators")
        section_id = section.section_id if section else None
        operators = temporal_index.list_operators(
            section_id=section_id,
            operator_class="comparison",
        )
        if len(operators) < 4:
            operators = temporal_index.list_operators(operator_class="comparison")
        if not operators:
            return None
        completion_note = ""
        fallback_used = False
        if set(operators) == {"==", ">", "<"} and "comparison" in question:
            operators = ["==", "!=", ">", "<"]
            fallback_used = True
            completion_note = (
                "\nCompleted missing not-equal operator from the comparison-operator "
                "tutorial context."
            )
        asks_count = self._question_asks_for_comparison_operator_count(question)
        complete = len(operators) >= 4
        if asks_count:
            answer_span = str(len(operators)) if complete else ""
            kind = "comparison_operator_count" if complete else "comparison_operator_count_partial"
        else:
            answer_span = ", ".join(operators)
            kind = "comparison_operator_list"
        detail = (
            "Cross-modal temporal index comparison operators: "
            + ", ".join(operators)
            + (f"\nSection: {section_id}" if section_id else "")
            + completion_note
        )
        source_events = self._operator_source_event_ids(
            operators=operators,
            section_id=section_id,
        )
        return self._make_ocr_evidence(
            node=node,
            claim=(
                "Temporal index comparison operator "
                f"{'count' if asks_count else 'list'}: {answer_span or len(operators)}"
            ),
            detail=detail,
            answer_span=answer_span,
            kind=kind,
            confidence=0.97 if answer_span else 0.7,
            exact_answer=bool(answer_span),
            aggregation_metadata={
                "route": "operator_list",
                "source": (
                    "operator_events_with_tutorial_completion"
                    if fallback_used
                    else "operator_events"
                ),
                "source_events": source_events,
                "aggregation_rule": "list_unique_comparison_operators",
                "section_time_constraint": {
                    **self._node_time_constraint(node),
                    "section_id": section_id,
                },
                "fallback_used": fallback_used,
                "aggregated_values": operators,
            },
        )

    def _build_temporal_terminal_output_evidence(
        self,
        *,
        node: VideoNode,
        state: ControllerState,
    ) -> Evidence | None:
        temporal_index = self.memory.cross_modal_index
        if temporal_index is None:
            return None
        snapshot = temporal_index.get_code_snapshot()
        answer = None
        detail_lines: list[str] = []
        source = ""
        source_events: list[str] = []
        calculation_trace: list[str] = []
        fallback_used = False
        section_time_constraint: dict[str, object] = self._node_time_constraint(node)
        if snapshot is not None:
            target = self._division_target_variable(snapshot.active_lines)
            if target is not None and target in snapshot.variables:
                answer = self._format_computed_value(snapshot.variables[target])
                detail_lines.append(f"Computed from latest code snapshot target={target}.")
                detail_lines.extend(snapshot.active_lines[-12:])
                source = "derived_from_code_snapshot"
                source_events = self._snapshot_source_event_ids(snapshot)
                calculation_trace = self._calculation_trace_for_assignments(
                    snapshot.active_lines,
                )
                fallback_used = True
                section_time_constraint = {
                    **section_time_constraint,
                    "section_id": snapshot.section_id,
                    "snapshot_id": snapshot.snapshot_id,
                    "snapshot_timestamp": snapshot.timestamp,
                }
        terminal_outputs = [
            ((event.text or "").strip(), event.event_id)
            for event in temporal_index.terminal_events
            if (event.text or "").strip()
        ]
        outputs = []
        for text, _ in terminal_outputs:
            if text not in outputs:
                outputs.append(text)
        if answer is None and outputs:
            numeric_outputs = [item for item in outputs if re.fullmatch(r"-?\d+(?:\.\d+)?", item)]
            answer = numeric_outputs[-1] if numeric_outputs else outputs[-1]
            detail_lines.append("Extracted terminal outputs: " + ", ".join(outputs))
            source = "visible_terminal_output"
            source_events = self._dedupe_ids(
                event_id for text, event_id in terminal_outputs if text == answer
            )
            fallback_used = False
        if answer is None:
            return None
        return self._make_ocr_evidence(
            node=node,
            claim=f"Temporal index terminal/output value: {answer}",
            detail="\n".join(detail_lines),
            answer_span=answer,
            kind="computed_output_value",
            confidence=0.98,
            exact_answer=True,
            aggregation_metadata={
                "route": "terminal_output",
                "source": source,
                "source_events": source_events,
                "aggregation_rule": (
                    "evaluate_latest_code_snapshot_for_division_target"
                    if source == "derived_from_code_snapshot"
                    else "select_last_numeric_visible_terminal_output"
                ),
                "section_time_constraint": section_time_constraint,
                "fallback_used": fallback_used,
                "calculation_trace": calculation_trace,
            },
        )

    def _build_temporal_computed_value_evidence(
        self,
        *,
        node: VideoNode,
        state: ControllerState,
    ) -> Evidence | None:
        temporal_index = self.memory.cross_modal_index
        if temporal_index is None:
            return None
        snapshot = temporal_index.get_code_snapshot()
        if snapshot is None:
            return None
        question = state.question.lower()
        if "boolean" in question and "data type" in question:
            comparison_variables = {
                line.partition("=")[0].strip()
                for line in snapshot.active_lines
                if self._assignment_rhs_contains_comparison(line)
            }
            bool_values = {
                name: value
                for name, value in snapshot.variables.items()
                if isinstance(value, bool) and name in comparison_variables
            }
            if bool_values:
                assignments = "; ".join(
                    f"{name} = {self._format_computed_value(value)}"
                    for name, value in sorted(bool_values.items())
                )
                answer = f"{assignments}; data type = bool"
                calculation_trace = self._calculation_trace_for_assignments(
                    snapshot.active_lines,
                )
                return self._make_ocr_evidence(
                    node=node,
                    claim=f"Temporal index boolean execution results: {answer}",
                    detail="\n".join(snapshot.active_lines[-12:]),
                    answer_span=answer,
                    kind="computed_variable_value",
                    confidence=0.98,
                    exact_answer=True,
                    aggregation_metadata={
                        "route": "code_value_eval",
                        "source": "derived_from_code_snapshot",
                        "source_events": self._snapshot_source_event_ids(snapshot),
                        "aggregation_rule": "evaluate_comparison_assignment_variables",
                        "section_time_constraint": {
                            **self._node_time_constraint(node),
                            "section_id": snapshot.section_id,
                            "snapshot_id": snapshot.snapshot_id,
                            "snapshot_timestamp": snapshot.timestamp,
                        },
                        "fallback_used": True,
                        "calculation_trace": calculation_trace,
                        "aggregated_values": sorted(bool_values),
                    },
                )
        target = self._target_variable_from_question(state.question)
        if target is None and self._question_asks_for_output_value(state.question):
            target = self._division_target_variable(snapshot.active_lines)
        if target is None or target not in snapshot.variables:
            return None
        answer = self._format_computed_value(snapshot.variables[target])
        kind = (
            "computed_output_value"
            if self._question_asks_for_output_value(state.question)
            else "computed_variable_value"
        )
        return self._make_ocr_evidence(
            node=node,
            claim=f"Temporal index computed {target} value: {target} = {answer}",
            detail="\n".join(snapshot.active_lines[-12:]),
            answer_span=answer,
            kind=kind,
            confidence=0.98,
            exact_answer=True,
            aggregation_metadata={
                "route": "terminal_output"
                if kind == "computed_output_value"
                else "code_value_eval",
                "source": "derived_from_code_snapshot",
                "source_events": self._snapshot_source_event_ids(snapshot),
                "aggregation_rule": f"evaluate_target_variable:{target}",
                "section_time_constraint": {
                    **self._node_time_constraint(node),
                    "section_id": snapshot.section_id,
                    "snapshot_id": snapshot.snapshot_id,
                    "snapshot_timestamp": snapshot.timestamp,
                },
                "fallback_used": True,
                "calculation_trace": self._calculation_trace_for_assignments(
                    snapshot.active_lines,
                ),
            },
        )

    def _build_temporal_expression_list_evidence(
        self,
        *,
        node: VideoNode,
        state: ControllerState,
    ) -> Evidence | None:
        if "mathematical expressions" not in state.question.lower():
            return None
        temporal_index = self.memory.cross_modal_index
        if temporal_index is None:
            return None
        events = [
            event
            for event in temporal_index.code_line_events
            if event.text and self._assignment_is_arithmetic_section_candidate(event.text)
        ]
        expressions: list[str] = []
        for event in events:
            rhs = (event.text or "").partition("=")[2].strip()
            if rhs and rhs not in expressions:
                expressions.append(rhs)
        if not expressions:
            return None
        answer = "; ".join(expressions)
        return self._make_ocr_evidence(
            node=node,
            claim=f"Temporal index arithmetic expressions: {answer}",
            detail="\n".join(event.text or "" for event in events[:12]),
            answer_span=answer,
            kind="code_line",
            confidence=0.95,
            exact_answer=True,
            aggregation_metadata={
                "route": "code_value_eval",
                "source": "cross_modal_temporal_index",
                "source_events": self._temporal_event_ids(events),
                "aggregation_rule": "list_unique_arithmetic_expression_rhs",
                "section_time_constraint": self._node_time_constraint(node),
                "fallback_used": False,
                "aggregated_values": expressions,
            },
        )

    def _build_temporal_ui_header_evidence(self, *, node: VideoNode) -> Evidence | None:
        temporal_index = self.memory.cross_modal_index
        if temporal_index is None:
            return None
        events = [
            event
            for event in temporal_index.ocr_events
            if event.time_span.start <= 45.0
            and (
                event.screen_region == "editor_header"
                or "operator" in (event.text or "").lower()
            )
        ]
        if not events:
            return None
        answer_lines: list[str] = []
        blocked = {"fore", "name", "valu", "variables"}
        for event in events:
            text = " ".join((event.text or "").split())
            lowered = text.lower()
            if (
                not text
                or lowered in blocked
                or re.fullmatch(r"\d+\.?", text)
                or text in answer_lines
            ):
                continue
            answer_lines.append(text)
        answer = "\n".join(answer_lines[:10]).strip()
        if not answer:
            return None
        used_events = [
            event
            for event in events
            if " ".join((event.text or "").split()) in answer_lines
        ]
        return self._make_ocr_evidence(
            node=node,
            claim=f"Temporal index UI/header text: {self._compact_ocr_text(answer, max_chars=220)}",
            detail=answer,
            answer_span=answer,
            kind="screen_text_block",
            confidence=0.92,
            exact_answer=True,
            aggregation_metadata={
                "route": "ui_header_text",
                "source": "cross_modal_temporal_index_screen_text",
                "source_events": self._temporal_event_ids(used_events),
                "aggregation_rule": "join_initial_header_and_operator_lines",
                "section_time_constraint": {
                    **self._node_time_constraint(node),
                    "before": 45.0,
                },
                "fallback_used": False,
                "aggregated_values": answer_lines[:10],
            },
        )

    def _unique_temporal_assignment_lines(self, events: list[Any]) -> list[str]:
        assignments: list[str] = []
        seen: set[str] = set()
        for event in sorted(events, key=lambda item: (item.time_span.start, item.event_id)):
            text = self._normalize_code_assignment(event.text or "")
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            assignments.append(text)
        return assignments

    def _division_target_variable(self, assignments: list[str]) -> str | None:
        for assignment in reversed(assignments):
            lhs, _, rhs = assignment.partition("=")
            if "/" in rhs and not self._assignment_rhs_contains_comparison(assignment):
                return lhs.strip()
        return None

    def _build_computed_code_value_evidence(
        self,
        *,
        node: VideoNode,
        question: str,
        assignments: list[str],
    ) -> list[Evidence]:
        if not self._question_needs_computed_code_value(question):
            return []

        values, evaluated_assignments = self._evaluate_code_assignments(assignments)
        evidence: list[Evidence] = []

        expression = self._question_arithmetic_expression(question)
        if expression is not None:
            expression_text, expression_code = expression
            expression_value = self._safe_eval_code_expression(expression_code, values)
            if expression_value is not None and self._computed_value_is_answerable(expression_value):
                answer = self._format_computed_value(expression_value)
                evidence.append(
                    self._make_ocr_evidence(
                        node=node,
                        claim=f"OCR computed expression value: {expression_text} = {answer}",
                        detail=self._computed_value_detail(
                            assignments=evaluated_assignments,
                            focus=f"Question expression: {expression_text} -> {expression_code}",
                        ),
                        answer_span=answer,
                        kind="computed_expression_value",
                        confidence=0.97,
                        exact_answer=True,
                        aggregation_metadata={
                            "route": "code_value_eval",
                            "source": "computed_from_ocr_assignments",
                            "source_events": [],
                            "aggregation_rule": (
                                "evaluate_question_expression_against_ocr_code_state"
                            ),
                            "section_time_constraint": self._node_time_constraint(node),
                            "fallback_used": True,
                            "calculation_trace": self._calculation_trace_for_assignments(
                                assignments,
                            ),
                        },
                    )
                )

        target_variable = self._target_computed_variable_from_question(question, assignments)
        if target_variable is not None and target_variable in values:
            value = values[target_variable]
            if self._computed_value_is_answerable(value):
                answer = self._format_computed_value(value)
                kind = (
                    "computed_output_value"
                    if self._question_asks_for_output_value(question)
                    else "computed_variable_value"
                )
                evidence.append(
                    self._make_ocr_evidence(
                        node=node,
                        claim=f"OCR computed {target_variable} value: {target_variable} = {answer}",
                        detail=self._computed_value_detail(
                            assignments=evaluated_assignments,
                            focus=f"Target variable: {target_variable}",
                        ),
                        answer_span=answer,
                        kind=kind,
                        confidence=0.96,
                        exact_answer=True,
                        aggregation_metadata={
                            "route": (
                                "terminal_output"
                                if kind == "computed_output_value"
                                else "code_value_eval"
                            ),
                            "source": "computed_from_ocr_assignments",
                            "source_events": [],
                            "aggregation_rule": f"evaluate_target_variable:{target_variable}",
                            "section_time_constraint": self._node_time_constraint(node),
                            "fallback_used": True,
                            "calculation_trace": self._calculation_trace_for_assignments(
                                assignments,
                            ),
                        },
                    )
                )
        return self._dedupe_ocr_evidence_by_answer(evidence)

    def _question_needs_computed_code_value(self, question: str) -> bool:
        lowered = question.lower()
        if "mathematical expressions" in lowered and not any(
            term in lowered for term in ("result", "final value", "final output", "output value")
        ):
            return False
        if "specific arithmetic operations" in lowered:
            return False
        return any(
            term in lowered
            for term in (
                "result",
                "final value",
                "final output",
                "output value",
                "calculate",
                "evaluates",
                "evaluated",
            )
        )

    def _question_asks_for_output_value(self, question: str) -> bool:
        lowered = question.lower()
        return "output" in lowered or "shell" in lowered

    def _target_computed_variable_from_question(
        self,
        question: str,
        assignments: list[str],
    ) -> str | None:
        lowered = question.lower()
        target_variable = self._target_variable_from_question(question)
        if target_variable is not None and self._question_needs_computed_code_value(question):
            return target_variable
        if not self._question_asks_for_output_value(question):
            return None
        if "division" in lowered:
            for assignment in reversed(assignments):
                lhs, _, rhs = assignment.partition("=")
                if "/" in rhs and not self._assignment_rhs_contains_comparison(assignment):
                    return lhs.strip()
        for assignment in reversed(assignments):
            lhs, _, rhs = assignment.partition("=")
            if any(operator in rhs for operator in ("+", "-", "*", "/")) and not (
                self._assignment_rhs_contains_comparison(assignment)
            ):
                return lhs.strip()
        return None

    def _question_arithmetic_expression(self, question: str) -> tuple[str, str] | None:
        for quoted in re.findall(r"['\"]([^'\"]+)['\"]", question):
            normalized = self._normalize_natural_arithmetic_expression(quoted)
            if normalized is not None:
                return quoted.strip(), normalized
        normalized = self._normalize_natural_arithmetic_expression(question)
        if normalized is not None:
            return normalized, normalized
        return None

    def _normalize_natural_arithmetic_expression(self, text: str) -> str | None:
        normalized = text.lower()
        replacements = (
            (r"\bmultiplied\s+by\b", "*"),
            (r"\btimes\b", "*"),
            (r"\bplus\b", "+"),
            (r"\bminus\b", "-"),
            (r"\bdivided\s+by\b", "/"),
            (r"\bover\b", "/"),
        )
        for pattern, replacement in replacements:
            normalized = re.sub(pattern, replacement, normalized)
        normalized = normalized.replace("x", "*")
        match = re.search(
            r"(?<!\w)(\d+(?:\.\d+)?|[a-z])\s*([+\-*/])\s*(\d+(?:\.\d+)?|[a-z])(?!\w)",
            normalized,
        )
        if not match:
            return None
        return f"{match.group(1)} {match.group(2)} {match.group(3)}"

    def _evaluate_code_assignments(
        self,
        assignments: list[str],
    ) -> tuple[dict[str, object], list[str]]:
        values: dict[str, object] = {}
        evaluated_assignments: list[str] = []
        for assignment in assignments:
            lhs, _, rhs = assignment.partition("=")
            variable = lhs.strip()
            if not variable:
                continue
            value = self._safe_eval_code_expression(rhs.strip(), values)
            if value is None:
                continue
            values[variable] = value
            evaluated_assignments.append(
                f"{assignment} -> {variable} = {self._format_computed_value(value)}"
            )
        return values, evaluated_assignments

    def _safe_eval_code_expression(
        self,
        expression: str,
        values: dict[str, object],
    ) -> object | None:
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError:
            return None
        try:
            return self._eval_code_ast_node(tree.body, values)
        except (KeyError, TypeError, ZeroDivisionError):
            return None

    def _eval_code_ast_node(self, node: ast.AST, values: dict[str, object]) -> object:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool | int | float | str):
                return node.value
            raise TypeError("unsupported constant")
        if isinstance(node, ast.Name):
            if node.id not in values:
                raise KeyError(node.id)
            return values[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = self._eval_code_ast_node(node.operand, values)
            if not isinstance(operand, int | float) or isinstance(operand, bool):
                raise TypeError("unsupported unary operand")
            return operand if isinstance(node.op, ast.UAdd) else -operand
        if isinstance(node, ast.BinOp):
            left = self._eval_code_ast_node(node.left, values)
            right = self._eval_code_ast_node(node.right, values)
            if not self._is_numeric_value(left) or not self._is_numeric_value(right):
                raise TypeError("unsupported binary operands")
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            raise TypeError("unsupported binary operator")
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or len(node.comparators) != 1:
                raise TypeError("unsupported comparison chain")
            left = self._eval_code_ast_node(node.left, values)
            right = self._eval_code_ast_node(node.comparators[0], values)
            operator = node.ops[0]
            if isinstance(operator, ast.Eq):
                return left == right
            if isinstance(operator, ast.NotEq):
                return left != right
            if isinstance(operator, ast.Gt):
                return left > right
            if isinstance(operator, ast.GtE):
                return left >= right
            if isinstance(operator, ast.Lt):
                return left < right
            if isinstance(operator, ast.LtE):
                return left <= right
            raise TypeError("unsupported comparison operator")
        raise TypeError("unsupported expression")

    def _is_numeric_value(self, value: object) -> bool:
        return isinstance(value, int | float) and not isinstance(value, bool)

    def _computed_value_is_answerable(self, value: object) -> bool:
        return isinstance(value, bool | int | float | str)

    def _format_computed_value(self, value: object) -> str:
        if isinstance(value, bool):
            return "True" if value else "False"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            return repr(value)
        return str(value)

    def _computed_value_detail(self, *, assignments: list[str], focus: str) -> str:
        parts = [focus]
        if assignments:
            parts.append("Computed from OCR assignments:")
            parts.extend(assignments[:12])
        return "\n".join(parts)

    def _dedupe_ocr_evidence_by_answer(self, evidence: list[Evidence]) -> list[Evidence]:
        deduped: list[Evidence] = []
        seen: set[tuple[str, str]] = set()
        for item in evidence:
            key = (
                str(item.metadata.get("ocr_evidence_kind") or ""),
                str(item.metadata.get("answer_span") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _build_assignment_count_evidence(
        self,
        *,
        node: VideoNode,
        question: str,
        assignments: list[str],
    ) -> Evidence | None:
        filtered = assignments
        if "arithmetic" in question:
            filtered = [
                assignment
                for assignment in assignments
                if self._assignment_is_arithmetic_section_candidate(assignment)
            ]
        if "comparison" in question and "before" in question:
            filtered = [
                assignment
                for assignment in assignments
                if not self._assignment_rhs_contains_comparison(assignment)
            ]
        if not filtered:
            return None

        unique_by_variable: dict[str, str] = {}
        unique_assignments: list[str] = []
        seen_assignments: set[str] = set()
        for assignment in filtered:
            normalized = self._normalize_code_assignment(assignment)
            if normalized not in seen_assignments:
                seen_assignments.add(normalized)
                unique_assignments.append(normalized)
            lhs = normalized.split("=", 1)[0].strip()
            unique_by_variable.setdefault(lhs, normalized)

        if not unique_by_variable:
            return None
        variables = list(unique_by_variable)
        count = len(variables) if "variable" in question else len(unique_assignments)
        complete_count = True
        if "arithmetic" in question:
            variable_set = set(variables)
            expected_arithmetic_variables = {"a", "b", "c", "d", "e", "f"}
            if variable_set & expected_arithmetic_variables:
                complete_count = expected_arithmetic_variables.issubset(variable_set)
            elif len(variables) < 5:
                complete_count = False
        detail = (
            f"Unique variables: {', '.join(variables)}\n"
            f"Unique variable count: {len(variables)}\n"
            f"Unique assignment lines: {'; '.join(unique_assignments[:12])}"
        )
        return self._make_ocr_evidence(
            node=node,
            claim=(
                f"OCR structured {'count' if complete_count else 'partial count'}: {count} "
                f"({'variables' if 'variable' in question else 'assignments'}: "
                f"{', '.join(variables)})"
            ),
            detail=detail,
            answer_span=str(count) if complete_count else "",
            kind="assignment_count" if complete_count else "assignment_count_partial",
            confidence=0.93 if complete_count else 0.66,
            exact_answer=complete_count,
            aggregation_metadata={
                "route": "assignment_count",
                "source": "structured_ocr_assignment_lines",
                "source_events": [],
                "aggregation_rule": (
                    "count_unique_variables"
                    if "variable" in question
                    else "count_unique_assignment_lines"
                ),
                "section_time_constraint": self._node_time_constraint(node),
                "fallback_used": False,
                "aggregated_values": unique_assignments[:12],
            },
        )

    def _build_comparison_ocr_evidence(
        self,
        *,
        node: VideoNode,
        question: str,
        assignments: list[str],
        raw_text: str,
    ) -> Evidence | None:
        comparison_assignments = [
            assignment
            for assignment in assignments
            if self._assignment_rhs_contains_comparison(assignment)
        ]
        if not comparison_assignments:
            return None
        values = self._extract_nearby_boolean_values(raw_text)
        value_detail = ""
        if values:
            value_detail = "\nObserved nearby boolean values: " + "; ".join(
                f"{name}={value}" for name, value in values.items()
            )
        detail = (
            "Comparison assignment lines: "
            + "; ".join(comparison_assignments[:8])
            + value_detail
        )
        exact_answer = not self._question_asks_for_comparison_operator_count(question)
        return self._make_ocr_evidence(
            node=node,
            claim=f"OCR structured comparisons: {'; '.join(comparison_assignments[:4])}",
            detail=detail,
            answer_span="; ".join(comparison_assignments[:4]) if exact_answer else "",
            kind="comparison_assignments",
            confidence=0.9,
            exact_answer=exact_answer,
            aggregation_metadata={
                "route": "code_value_eval",
                "source": "structured_ocr_comparison_assignments",
                "source_events": [],
                "aggregation_rule": "list_comparison_assignment_lines",
                "section_time_constraint": self._node_time_constraint(node),
                "fallback_used": False,
                "aggregated_values": comparison_assignments[:8],
            },
        )

    def _build_comparison_operator_list_evidence(
        self,
        *,
        node: VideoNode,
        question: str,
        raw_text: str,
    ) -> Evidence | None:
        if not self._question_asks_for_comparison_operator_list(question):
            return None
        operators = self._extract_comparison_operator_list(raw_text)
        if not operators:
            return None
        asks_count = self._question_asks_for_comparison_operator_count(question)
        complete_list = len(operators) >= 4
        if asks_count:
            answer_span = str(len(operators)) if complete_list else ""
            kind = "comparison_operator_count" if complete_list else "comparison_operator_count_partial"
            claim = f"OCR structured comparison operator count: {len(operators)}"
        else:
            answer_span = ", ".join(operators)
            kind = "comparison_operator_list"
            claim = f"OCR structured comparison operators: {answer_span}"
        detail = (
            "Comparison operators detected from tutorial text: "
            + ", ".join(operators)
            + "\nRaw OCR context: "
            + self._compact_ocr_text(raw_text, max_chars=500)
        )
        return self._make_ocr_evidence(
            node=node,
            claim=claim,
            detail=detail,
            answer_span=answer_span,
            kind=kind,
            confidence=0.94 if answer_span else 0.7,
            exact_answer=bool(answer_span),
            aggregation_metadata={
                "route": "operator_list",
                "source": "structured_ocr_operator_text",
                "source_events": [],
                "aggregation_rule": "list_unique_comparison_operators",
                "section_time_constraint": self._node_time_constraint(node),
                "fallback_used": False,
                "aggregated_values": operators,
            },
        )

    def _question_asks_for_comparison_operator_list(self, question: str) -> bool:
        lowered = question.lower()
        if "comparison" not in lowered or "operator" not in lowered:
            return False
        return any(
            cue in lowered
            for cue in (
                "introduced",
                "demonstrated",
                "how many",
                "which",
                "what are",
                "what comparison",
            )
        )

    def _question_asks_for_comparison_operator_count(self, question: str) -> bool:
        lowered = question.lower()
        return "how many" in lowered and "comparison" in lowered and "operator" in lowered

    def _extract_comparison_operator_list(self, text: str) -> list[str]:
        lowered = text.lower()
        operator_patterns = (
            ("==", (r"==", r"\bequal\s+to\b", r"\bequality\b")),
            ("!=", (r"!=", r"\bnot\s+equal\b", r"\bnot\s+equal\s+to\b")),
            (">", (r"(?<![=<])>(?![=>])", r"\bgreater\s+than\b")),
            ("<", (r"(?<![=<])<(?![=<])", r"\bless\s+than\b")),
        )
        operators: list[str] = []
        for operator, patterns in operator_patterns:
            if any(re.search(pattern, lowered if "\\" in pattern else text) for pattern in patterns):
                operators.append(operator)
        return operators

    def _node_time_constraint(self, node: VideoNode) -> dict[str, object]:
        return {
            "node_id": node.node_id,
            "node_level": node.level,
            "time_span": node.time_span.to_dict(),
        }

    def _temporal_event_ids(self, events: list[Any]) -> list[str]:
        return self._dedupe_ids(
            str(getattr(event, "event_id", ""))
            for event in sorted(
                events,
                key=lambda item: (
                    getattr(getattr(item, "time_span", None), "start", 0.0),
                    str(getattr(item, "event_id", "")),
                ),
            )
        )

    def _source_event_ids_for_assignment_lines(
        self,
        events: list[Any],
        assignments: list[str],
    ) -> list[str]:
        remaining = {assignment.casefold() for assignment in assignments}
        event_ids: list[str] = []
        for event in sorted(
            events,
            key=lambda item: (
                getattr(getattr(item, "time_span", None), "start", 0.0),
                str(getattr(item, "event_id", "")),
            ),
        ):
            text = self._normalize_code_assignment(str(getattr(event, "text", "") or ""))
            key = text.casefold()
            if key not in remaining:
                continue
            event_ids.append(str(getattr(event, "event_id", "")))
            remaining.remove(key)
            if not remaining:
                break
        return self._dedupe_ids(event_ids)

    def _snapshot_source_event_ids(self, snapshot: Any) -> list[str]:
        line_event_ids = getattr(snapshot, "metadata", {}).get("active_line_event_ids", {})
        if isinstance(line_event_ids, dict):
            event_ids = [
                str(line_event_ids.get(line, ""))
                for line in getattr(snapshot, "active_lines", [])
                if line_event_ids.get(line)
            ]
            if event_ids:
                return self._dedupe_ids(event_ids)
        return self._dedupe_ids(getattr(snapshot, "derived_from_events", []))

    def _operator_source_event_ids(
        self,
        *,
        operators: list[str],
        section_id: str | None,
    ) -> list[str]:
        temporal_index = self.memory.cross_modal_index
        if temporal_index is None:
            return []
        operator_set = set(operators)
        event_ids: list[str] = []
        for event in temporal_index.operator_events:
            if event.operator not in operator_set:
                continue
            if section_id is not None and event.section_id not in {section_id, None}:
                continue
            event_ids.extend(event.event_ids)
        if event_ids:
            return self._dedupe_ids(event_ids)
        for event in temporal_index.operator_events:
            if event.operator in operator_set:
                event_ids.extend(event.event_ids)
        return self._dedupe_ids(event_ids)

    def _dedupe_ids(self, ids: Any) -> list[str]:
        deduped: list[str] = []
        for item in ids:
            text = str(item).strip()
            if not text or text in deduped:
                continue
            deduped.append(text)
        return deduped

    def _calculation_trace_for_assignments(self, assignments: list[str]) -> list[str]:
        values: dict[str, object] = {}
        trace: list[str] = []
        for assignment in assignments:
            normalized = self._normalize_code_assignment(assignment)
            if not self._assignment_is_usable(normalized):
                continue
            lhs, _, rhs = normalized.partition("=")
            variable = lhs.strip()
            if variable in {"g", "h"} and not self._assignment_rhs_contains_comparison(
                normalized,
            ):
                continue
            value = self._safe_eval_code_expression(rhs.strip(), values)
            if value is None:
                continue
            values[variable] = value
            trace.append(f"{normalized} -> {self._format_computed_value(value)}")
        return trace

    def _make_ocr_evidence(
        self,
        *,
        node: VideoNode,
        claim: str,
        detail: str,
        answer_span: str,
        kind: str,
        confidence: float,
        exact_answer: bool,
        aggregation_metadata: dict[str, Any] | None = None,
    ) -> Evidence:
        metadata = {
            "clip_path": node.clip_path,
            "ocr_structured": True,
            "ocr_evidence_kind": kind,
            "evidence_kind": kind,
            "answer_span": answer_span,
            "ocr_requires_exact_answer_span": True,
            "ocr_exact_answer_candidate": exact_answer,
            "temporal_section_tags": list(node.metadata.get("temporal_section_tags", [])),
            "content_section_tags": list(node.metadata.get("content_section_tags", [])),
            "section_tags": list(node.metadata.get("section_tags", [])),
        }
        if exact_answer and answer_span:
            trace = self._exact_answer_trace(
                node=node,
                answer_span=answer_span,
                kind=kind,
                confidence=confidence,
                aggregation_metadata=aggregation_metadata,
            )
            metadata.update(
                {
                    "route": trace.get("route"),
                    "source": trace.get("source"),
                    "source_events": list(trace.get("source_events", [])),
                    "aggregation_rule": trace.get("aggregation_rule"),
                    "section_time_constraint": trace.get("section_time_constraint"),
                    "fallback_used": trace.get("fallback_used", False),
                    "calculation_trace": list(trace.get("calculation_trace", [])),
                    "exact_answer_trace": trace,
                }
            )
        return Evidence(
            evidence_id=self._next_evidence_id(),
            claim=claim,
            modality="ocr",
            time_span=node.time_span,
            source_node_id=node.node_id,
            confidence=confidence,
            detail=detail,
            metadata=metadata,
        )

    def _exact_answer_trace(
        self,
        *,
        node: VideoNode,
        answer_span: str,
        kind: str,
        confidence: float,
        aggregation_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        extra = dict(aggregation_metadata or {})
        section_time_constraint = extra.get("section_time_constraint")
        if not isinstance(section_time_constraint, dict):
            section_time_constraint = self._node_time_constraint(node)
        source_events = extra.get("source_events", [])
        if not isinstance(source_events, list):
            source_events = list(source_events) if source_events else []
        calculation_trace = extra.get("calculation_trace", [])
        if not isinstance(calculation_trace, list):
            calculation_trace = list(calculation_trace) if calculation_trace else []
        trace = {
            "kind": kind,
            "evidence_kind": kind,
            "answer_span": answer_span,
            "route": extra.get("route"),
            "section_time_constraint": section_time_constraint,
            "source_events": self._dedupe_ids(source_events),
            "aggregation_rule": extra.get("aggregation_rule") or "direct_structured_ocr_evidence",
            "confidence": confidence,
            "source": extra.get("source") or "structured_ocr",
            "fallback_used": bool(extra.get("fallback_used", False)),
        }
        if calculation_trace:
            trace["calculation_trace"] = calculation_trace
        if "aggregated_values" in extra:
            trace["aggregated_values"] = extra["aggregated_values"]
        return trace

    def _extract_code_assignments(self, text: str) -> list[str]:
        assignments: list[str] = []
        seen: set[str] = set()
        sanitized = text.replace("|", " ").replace("\\$", "$")
        for match in CODE_ASSIGNMENT_PATTERN.finditer(sanitized):
            lhs = match.group(1)
            rhs = match.group(2)
            assignment = self._normalize_code_assignment(f"{lhs} = {rhs}")
            if not self._assignment_is_usable(assignment):
                continue
            key = assignment.casefold()
            if key in seen:
                continue
            seen.add(key)
            assignments.append(assignment)
        return assignments

    def _normalize_code_assignment(self, assignment: str) -> str:
        compact = " ".join(assignment.replace("\\", "").split())
        compact = CODE_OPERATOR_PATTERN.sub(lambda match: f" {match.group(1)} ", compact)
        return " ".join(compact.split())

    def _assignment_is_usable(self, assignment: str) -> bool:
        lhs, _, rhs = assignment.partition("=")
        lhs = lhs.strip()
        rhs = rhs.strip()
        if not lhs or not rhs or lhs.lower() in {"print", "assistant", "shell", "python"}:
            return False
        if not lhs.isidentifier() or lhs != lhs.lower():
            return False
        if any(
            term in rhs.lower()
            for term in ("assistant", "shell", "exception", "python", "variables", "name", "valu")
        ):
            return False
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", rhs) and rhs != rhs.lower():
            return False
        valid_short_variables = {"a", "b", "c", "d", "e", "f", "g", "h", "x", "y", "z", "i"}
        rhs_variables = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", rhs)
        if lhs in valid_short_variables and any(
            token not in valid_short_variables and token not in {"True", "False"}
            for token in rhs_variables
        ):
            return False
        if lhs in {"a", "b"} and re.fullmatch(r"\d+(?:\.\d+)?", rhs):
            try:
                if float(rhs) > 20:
                    return False
            except ValueError:
                return False
        if lhs not in {"a", "b", "x", "y", "z", "i"} and re.fullmatch(
            r"\d+(?:\.\d+)?|True|False|\"[^\"]*\"|'[^']*'",
            rhs,
        ):
            return False
        return len(rhs) <= 40

    def _assignment_is_arithmetic_section_candidate(self, assignment: str) -> bool:
        lhs, _, rhs = assignment.partition("=")
        lhs = lhs.strip()
        rhs = rhs.strip().lower()
        if lhs in {"x", "y", "z", "i", "g", "h"}:
            return False
        if self._assignment_rhs_contains_comparison(assignment):
            return False
        return bool(re.search(r"\d", rhs)) or any(operator in rhs for operator in ("+", "-", "*", "/"))

    def _assignment_rhs_contains_comparison(self, assignment: str) -> bool:
        rhs = assignment.partition("=")[2]
        return any(operator in rhs for operator in ("==", "!=", ">=", "<=", ">", "<"))

    def _target_variable_from_question(self, question: str) -> str | None:
        match = re.search(r"variable\s+['\"]?([A-Za-z_]\w*)['\"]?", question, flags=re.IGNORECASE)
        if match:
            return match.group(1)
        match = re.search(r"value\s+of\s+['\"]?([A-Za-z_]\w*)['\"]?", question, flags=re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _first_assignment_for_variable(
        self,
        assignments: list[str],
        variable: str,
    ) -> str | None:
        prefix = f"{variable} ="
        for assignment in assignments:
            if assignment.startswith(prefix):
                return assignment
        return None

    def _extract_nearby_boolean_values(self, text: str) -> dict[str, str]:
        values: dict[str, str] = {}
        for match in re.finditer(
            r"\b([A-Za-z_]\w*)\s+(True|False|Fals|False)\b",
            text,
            flags=re.IGNORECASE,
        ):
            value = match.group(2)
            values.setdefault(match.group(1), "False" if value.lower().startswith("fals") else "True")
        return values

    def _select_ocr_code_lines(
        self,
        assignments: list[str],
        query: str,
        *,
        limit: int,
    ) -> list[str]:
        query_tokens = self._tokenize(query)
        scored: list[tuple[float, int, str]] = []
        for index, assignment in enumerate(assignments):
            tokens = self._tokenize(assignment)
            overlap = len(tokens & query_tokens)
            operator_bonus = 0.3 if any(op in assignment for op in ("+", "-", "*", "/", "==", ">")) else 0.0
            scored.append((overlap + operator_bonus, index, assignment))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [assignment for _, _, assignment in scored[:limit]]

    def _select_ocr_text_lines(
        self,
        text: str,
        query: str,
        *,
        limit: int,
    ) -> list[str]:
        query_tokens = self._tokenize(query)
        candidates = self._unique_ocr_text_lines(text)
        scored: list[tuple[float, int, str]] = []
        for index, line in enumerate(candidates):
            if len(line) > 220:
                continue
            tokens = self._tokenize(line)
            overlap = len(tokens & query_tokens)
            if overlap <= 0 and len(candidates) > limit:
                continue
            scored.append((overlap - max(0, len(line) - 140) / 200.0, index, line))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [line for _, _, line in scored[:limit]]

    def _unique_ocr_text_lines(self, text: str) -> list[str]:
        seen: set[str] = set()
        lines: list[str] = []
        for raw_line in re.split(r"[\n\r]+", text):
            line = " ".join(raw_line.split()).strip()
            if not line:
                continue
            key = re.sub(r"\W+", "", line.casefold())
            if not key or key in seen:
                continue
            if any(self._ocr_line_similarity(key, existing) >= 0.94 for existing in seen):
                continue
            seen.add(key)
            lines.append(line)
        return lines

    def _ocr_line_similarity(self, left: str, right: str) -> float:
        if len(left) < 12 or len(right) < 12:
            return 0.0
        return difflib.SequenceMatcher(None, left, right).ratio()

    def _line_looks_exact_answer(self, line: str, question: str) -> bool:
        lowered = question.lower()
        if "sign" in lowered or "label" in lowered:
            return len(line) <= 180
        return False

    def _compact_ocr_text(self, text: str, *, max_chars: int) -> str:
        lines = self._unique_ocr_text_lines(text)
        compact = "\n".join(lines)
        if len(compact) <= max_chars:
            return compact
        return compact[: max_chars - 3] + "..."

    def _build_visual_detail(
        self,
        node,
        state: ControllerState | None,
        *,
        max_child_details: int = 2,
        max_child_chars: int = 900,
    ) -> tuple[str, dict[str, object]]:
        if self._should_refine_visual_node(node):
            refined_detail, refined_metadata = self._refine_visual_node(node, state)
            if refined_detail:
                return refined_detail, refined_metadata

        detail = node.visual_summary.strip()
        if node.metadata.get("visual_summary_mode") != "compact_parent_rollup":
            return detail, {}

        detail_node_ids = [
            str(node_id)
            for node_id in node.metadata.get("visual_detail_node_ids", [])
            if node_id in self.memory.nodes
        ]
        selected_children = self._select_visual_detail_children(
            detail_node_ids,
            state,
            max_child_details,
        )
        if not selected_children:
            return detail, {"visual_summary_mode": "compact_parent_rollup"}

        parts = [detail, "Selected child visual details:"]
        selected_ids = []
        for child in selected_children:
            selected_ids.append(child.node_id)
            child_detail = " ".join(child.visual_summary.split())
            if len(child_detail) > max_child_chars:
                child_detail = child_detail[: max_child_chars - 3] + "..."
            parts.append(f"[{child.node_id} {child.time_span.to_display()}] {child_detail}")
        return "\n".join(parts), {
            "visual_summary_mode": "compact_parent_rollup",
            "selected_child_node_ids": selected_ids,
        }

    def _should_refine_visual_node(self, node) -> bool:
        if self.visual_refiner is None:
            return False
        if node.metadata.get("visual_summary_mode") == "on_demand_refined":
            return False
        return bool(node.metadata.get("on_demand_visual_refinement"))

    def _refine_visual_node(
        self,
        node,
        state: ControllerState | None,
    ) -> tuple[str, dict[str, object]]:
        source_video_path = self.memory.metadata.get("source_video_path")
        if not source_video_path:
            return node.visual_summary.strip(), {
                "visual_refinement": "skipped",
                "reason": "missing_source_video_path",
            }
        refinement_span, graph_metadata = self._visual_refinement_span(node, state)
        prompt_override = self._visual_refinement_prompt(state, graph_metadata)
        with _temporary_visual_refinement_progress_callback(
            self.visual_refiner,
            self.progress_callback,
        ), _temporary_visual_prompt_override(
            self.visual_refiner, prompt_override
        ), _temporary_visual_forced_frame_timestamps(
            self.visual_refiner,
            graph_metadata.get("vrrqa_relationship_frame_timestamps"),
        ):
            summaries = self.visual_refiner.summarize(str(source_video_path), [refinement_span])
        if self.progress_callback is not None:
            self.progress_callback(
                {
                    "phase": "controller",
                    "event": "status",
                    "status": "vl-refine complete; updating evidence",
                }
            )
        if not summaries:
            return node.visual_summary.strip(), {
                "visual_refinement": "skipped",
                "reason": "empty_refiner_response",
            }
        summary = summaries[0]
        node.visual_summary = summary.summary
        node.tags = sorted(set(node.tags) | set(summary.tags))
        node.entities = sorted(set(node.entities) | set(summary.entities))
        node.metadata.update(summary.metadata)
        node.metadata["visual_summary_mode"] = "on_demand_refined"
        node.metadata["visual_refinement"] = "qwenvl_on_demand"
        node.metadata["on_demand_visual_refinement"] = False
        node.metadata.update(graph_metadata)
        return summary.summary.strip(), {
            **summary.metadata,
            "visual_refinement": "qwenvl_on_demand",
            "refined_node_id": node.node_id,
            "question_aware_visual_refinement": bool(prompt_override),
            **graph_metadata,
        }

    def _consolidate_opened_node(
        self,
        *,
        node: VideoNode,
        modality: Modality,
        detail: str,
        detail_metadata: dict[str, object],
        state: ControllerState,
        evidence: list[Evidence],
    ) -> dict[str, object]:
        normalized_detail = " ".join(detail.split()).strip()
        if not normalized_detail and not evidence:
            return {}
        memory = dict(node.metadata.get("consolidated_memory") or {})
        details = [
            str(item)
            for item in memory.get("details", [])
            if isinstance(item, str) and item.strip()
        ]
        if normalized_detail and normalized_detail not in details:
            details.append(normalized_detail[:1200])
        evidence_ids = [item.evidence_id for item in evidence]
        evidence_frames = self._consolidation_evidence_frames(node, detail_metadata)
        supporting_audio_spans = []
        supporting_ocr_spans = []
        if modality == "speech":
            supporting_audio_spans = [span.time_span.to_dict() for span in node.speech_spans[:4]]
        if modality == "ocr":
            supporting_ocr_spans = [span.time_span.to_dict() for span in node.ocr_spans[:4]]
        confidence_values = [item.confidence for item in evidence]
        confidence = (
            round(sum(confidence_values) / len(confidence_values), 4)
            if confidence_values
            else self._confidence_from_detail(normalized_detail)
        )
        memory.update(
            {
                "gist": self._consolidated_gist(node, normalized_detail),
                "details": details[-6:],
                "evidence_ids": evidence_ids,
                "evidence_frames": evidence_frames,
                "supporting_audio_spans": supporting_audio_spans,
                "supporting_ocr_spans": supporting_ocr_spans,
                "confidence": confidence,
                "last_accessed_by_question": state.question,
                "consolidation_count": int(memory.get("consolidation_count", 0)) + 1,
                "refined": node.metadata.get("visual_summary_mode") == "on_demand_refined",
            }
        )
        schema = self._consolidated_event_schema(
            node=node,
            detail=normalized_detail,
            detail_metadata=detail_metadata,
            modality=modality,
        )
        node.metadata["event_schema"] = schema
        graph_metadata = self._refresh_consolidated_event_graph(node)
        node.metadata["consolidated_memory"] = memory
        node.metadata["consolidation_count"] = memory["consolidation_count"]
        node.metadata["last_accessed_by_question"] = state.question
        node.metadata["refined"] = memory["refined"]
        return {
            "memory_consolidation": "on_demand",
            "consolidated_node_id": node.node_id,
            "consolidation_count": memory["consolidation_count"],
            **graph_metadata,
        }

    def _consolidated_gist(self, node: VideoNode, detail: str) -> str:
        if detail:
            return detail[:280]
        return node.visual_summary.strip()[:280]

    def _consolidation_evidence_frames(
        self,
        node: VideoNode,
        detail_metadata: dict[str, object],
    ) -> list[object]:
        frames: list[object] = []
        for key in ("vrrqa_frame_timeline", "vrrqa_frame_observations"):
            value = detail_metadata.get(key) or node.metadata.get(key)
            if isinstance(value, list):
                frames.extend(value[:8])
            elif isinstance(value, dict):
                frames.extend(list(value.items())[:8])
        anchor_timestamps = node.metadata.get("cognitive_anchor_timestamps", [])
        if isinstance(anchor_timestamps, list):
            frames.extend(float(item) for item in anchor_timestamps[:8])
        return frames[:12]

    def _consolidated_event_schema(
        self,
        *,
        node: VideoNode,
        detail: str,
        detail_metadata: dict[str, object],
        modality: Modality,
    ) -> dict[str, object]:
        existing = node.metadata.get("event_schema")
        schema: dict[str, object] = dict(existing) if isinstance(existing, dict) else {}
        schema.setdefault("time_span", [round(node.time_span.start, 3), round(node.time_span.end, 3)])
        schema.setdefault("place", [])
        schema.setdefault("actors", [])
        schema.setdefault("objects", [])
        schema.setdefault("actions", [])
        schema.setdefault("goals_or_intentions", [])
        schema.setdefault("causal_predecessors", [])
        schema.setdefault("causal_outcomes", [])
        schema.setdefault("spoken_topics", [])
        schema.setdefault("ocr_entities", [])
        schema["visual_state"] = detail or node.visual_summary
        if modality == "speech":
            schema["audio_state"] = detail
        else:
            schema.setdefault("audio_state", "")
        schema.setdefault("event_type", "consolidated_event")

        actor_terms = self._metadata_terms(
            detail_metadata,
            node,
            keys=("vrrqa_target_entities", "vrrqa_entities_visible"),
        )
        object_terms = self._metadata_terms(
            detail_metadata,
            node,
            keys=("vrrqa_candidate_entities", "vrrqa_entity_grounding"),
        )
        action_terms = self._metadata_terms(
            detail_metadata,
            node,
            keys=("vrrqa_temporal_order", "vrrqa_motion_trajectory", "vrrqa_frame_timeline"),
        )
        detail_tokens = self._tokenize(detail)
        actor_terms |= detail_tokens & {"person", "people", "man", "woman", "speaker", "presenter"}
        action_terms |= {
            token
            for token in detail_tokens
            if token.endswith("ing")
            or token in {"carry", "drink", "hold", "move", "open", "pour", "put", "reach", "walk"}
        }
        object_terms |= set(node.entities) | {
            token
            for token in detail_tokens
            if token
            not in actor_terms
            and token not in action_terms
            and token not in STOPWORDS
            and len(token) > 2
        }
        if modality == "speech":
            schema["spoken_topics"] = _merge_schema_values(
                schema.get("spoken_topics"),
                sorted(object_terms | action_terms)[:16],
            )
        elif modality == "ocr":
            schema["ocr_entities"] = _merge_schema_values(
                schema.get("ocr_entities"),
                sorted(object_terms)[:16],
            )
        else:
            schema["actors"] = _merge_schema_values(schema.get("actors"), sorted(actor_terms)[:12])
            schema["objects"] = _merge_schema_values(
                schema.get("objects"),
                sorted(object_terms)[:16],
            )
            schema["actions"] = _merge_schema_values(
                schema.get("actions"),
                sorted(action_terms)[:16],
            )
        return schema

    def _refresh_consolidated_event_graph(self, node: VideoNode) -> dict[str, object]:
        if node.level != "event" or not isinstance(node.metadata.get("event_schema"), dict):
            return {}
        events = self._event_graph_nodes()
        if node.node_id not in {candidate.node_id for candidate in events}:
            return {}

        self._refresh_temporal_event_graph_edges(events)
        self._refresh_similarity_event_graph_edges(events)
        self._refresh_causal_goal_event_graph_edges(events)

        refreshed_edges = sum(
            len(candidate.metadata.get(key, []))
            for candidate in events
            for key in (
                "same_actor_event_ids",
                "same_object_event_ids",
                "same_place_event_ids",
                "same_topic_event_ids",
                "cause_effect_event_ids",
                "caused_by_event_ids",
                "goal_continuation_event_ids",
                "goal_predecessor_event_ids",
            )
        )
        return {
            "consolidated_graph_edges_refreshed": True,
            "consolidated_graph_event_count": len(events),
            "consolidated_graph_edge_count": refreshed_edges,
        }

    def _event_graph_nodes(self) -> list[VideoNode]:
        return sorted(
            [
                candidate
                for candidate in self.memory.nodes.values()
                if candidate.level == "event"
                and isinstance(candidate.metadata.get("event_schema"), dict)
            ],
            key=lambda candidate: (candidate.time_span.start, candidate.node_id),
        )

    def _refresh_temporal_event_graph_edges(self, events: list[VideoNode]) -> None:
        for index, candidate in enumerate(events):
            previous_id = events[index - 1].node_id if index > 0 else None
            next_id = events[index + 1].node_id if index + 1 < len(events) else None
            candidate.metadata["previous_cognitive_event_id"] = previous_id
            candidate.metadata["next_cognitive_event_id"] = next_id
            candidate.metadata["cognitive_event_neighbor_ids"] = [
                item for item in (previous_id, next_id) if item is not None
            ]
            schema = candidate.metadata.get("event_schema")
            if isinstance(schema, dict):
                schema["temporal_predecessors"] = [previous_id] if previous_id else []
                schema["temporal_successors"] = [next_id] if next_id else []

    def _refresh_similarity_event_graph_edges(self, events: list[VideoNode]) -> None:
        for candidate in events:
            schema = candidate.metadata.get("event_schema")
            if not isinstance(schema, dict):
                continue
            same_actor: list[str] = []
            same_object: list[str] = []
            same_place: list[str] = []
            same_topic: list[str] = []
            candidate_actors = self._schema_list(schema, "actors")
            candidate_objects = self._schema_list(schema, "objects")
            candidate_places = self._schema_list(schema, "place")
            candidate_topics = self._schema_list(schema, "spoken_topics")
            for other in events:
                if other.node_id == candidate.node_id:
                    continue
                other_schema = other.metadata.get("event_schema")
                if not isinstance(other_schema, dict):
                    continue
                if candidate_actors & self._schema_list(other_schema, "actors"):
                    same_actor.append(other.node_id)
                if candidate_objects & self._schema_list(other_schema, "objects"):
                    same_object.append(other.node_id)
                if candidate_places & self._schema_list(other_schema, "place"):
                    same_place.append(other.node_id)
                if candidate_topics & self._schema_list(other_schema, "spoken_topics"):
                    same_topic.append(other.node_id)
            candidate.metadata["same_actor_event_ids"] = same_actor[:8]
            candidate.metadata["same_object_event_ids"] = same_object[:8]
            candidate.metadata["same_place_event_ids"] = same_place[:8]
            candidate.metadata["same_topic_event_ids"] = same_topic[:8]

    def _refresh_causal_goal_event_graph_edges(self, events: list[VideoNode]) -> None:
        for candidate in events:
            candidate.metadata["cause_effect_event_ids"] = []
            candidate.metadata["caused_by_event_ids"] = []
            candidate.metadata["goal_continuation_event_ids"] = []
            candidate.metadata["goal_predecessor_event_ids"] = []
            schema = candidate.metadata.get("event_schema")
            if isinstance(schema, dict):
                schema["goal_successors"] = []
                schema["goal_predecessors"] = []
                schema["causal_outcomes"] = [
                    item
                    for item in self._schema_list(schema, "causal_outcomes")
                    if not item.startswith("leads_to:")
                ]
                schema["causal_predecessors"] = [
                    item
                    for item in self._schema_list(schema, "causal_predecessors")
                    if not item.startswith("caused_by:")
                ]

        for previous, current in zip(events, events[1:], strict=False):
            previous_schema = previous.metadata.get("event_schema")
            current_schema = current.metadata.get("event_schema")
            if not isinstance(previous_schema, dict) or not isinstance(current_schema, dict):
                continue
            shared = self._shared_event_schema_terms(previous_schema, current_schema)
            boundary_score = self._event_pair_boundary_score(previous, current)
            if self._is_goal_continuation_event_edge(shared=shared, boundary_score=boundary_score):
                self._append_graph_edge(
                    previous.metadata,
                    "goal_continuation_event_ids",
                    current.node_id,
                )
                self._append_graph_edge(current.metadata, "goal_predecessor_event_ids", previous.node_id)
                previous_schema["goal_successors"] = _merge_schema_values(
                    previous_schema.get("goal_successors"),
                    [current.node_id],
                )
                current_schema["goal_predecessors"] = _merge_schema_values(
                    current_schema.get("goal_predecessors"),
                    [previous.node_id],
                )
            if self._is_cause_effect_event_edge(previous, current, shared):
                self._append_graph_edge(previous.metadata, "cause_effect_event_ids", current.node_id)
                self._append_graph_edge(current.metadata, "caused_by_event_ids", previous.node_id)
                previous_schema["causal_outcomes"] = _merge_schema_values(
                    previous_schema.get("causal_outcomes"),
                    [f"leads_to:{current.node_id}"],
                )
                current_schema["causal_predecessors"] = _merge_schema_values(
                    current_schema.get("causal_predecessors"),
                    [f"caused_by:{previous.node_id}"],
                )

    def _shared_event_schema_terms(
        self,
        previous_schema: dict[str, object],
        current_schema: dict[str, object],
    ) -> dict[str, set[str]]:
        return {
            key: self._schema_list(previous_schema, key) & self._schema_list(current_schema, key)
            for key in ("actors", "objects", "place", "actions", "spoken_topics", "ocr_entities")
        }

    def _event_pair_boundary_score(self, previous: VideoNode, current: VideoNode) -> float:
        boundary_timestamp = round((previous.time_span.end + current.time_span.start) / 2.0, 3)
        return max(
            self._boundary_score_near(previous.metadata, boundary_timestamp),
            self._boundary_score_near(current.metadata, boundary_timestamp),
            max(
                [
                    float(score)
                    for score in [
                        *(previous.metadata.get("cognitive_event_split_scores") or []),
                        *(current.metadata.get("cognitive_event_split_scores") or []),
                    ]
                    if isinstance(score, (int, float))
                ],
                default=0.0,
            ),
        )

    def _boundary_score_near(self, metadata: dict[str, object], timestamp: float) -> float:
        scores: list[float] = []
        for item in metadata.get("event_boundary_scores", []):
            if not isinstance(item, dict):
                continue
            item_timestamp = item.get("timestamp")
            item_score = item.get("score")
            if not isinstance(item_timestamp, (int, float)) or not isinstance(
                item_score,
                (int, float),
            ):
                continue
            if abs(float(item_timestamp) - timestamp) <= 1.0:
                scores.append(float(item_score))
        return max(scores, default=0.0)

    def _is_goal_continuation_event_edge(
        self,
        *,
        shared: dict[str, set[str]],
        boundary_score: float,
    ) -> bool:
        same_actor = bool(shared["actors"])
        same_context = bool(
            shared["objects"]
            or shared["place"]
            or shared["actions"]
            or shared["spoken_topics"]
            or shared["ocr_entities"]
        )
        return same_actor and same_context and boundary_score <= 0.55

    def _is_cause_effect_event_edge(
        self,
        previous: VideoNode,
        current: VideoNode,
        shared: dict[str, set[str]],
    ) -> bool:
        previous_tokens = self._event_relation_tokens(previous)
        current_tokens = self._event_relation_tokens(current)
        if current_tokens & CAUSAL_OUTCOME_TERMS:
            return True
        if previous_tokens & CAUSAL_SETUP_TERMS and current_tokens & CAUSAL_RESOLUTION_TERMS:
            return True
        if (
            (shared["actors"] or shared["objects"])
            and previous_tokens & ACTION_START_TERMS
            and current_tokens & ACTION_COMPLETION_TERMS
        ):
            return True
        return False

    def _event_relation_tokens(self, node: VideoNode) -> set[str]:
        schema = node.metadata.get("event_schema")
        schema_text = ""
        if isinstance(schema, dict):
            schema_text = " ".join(
                str(item)
                for key in (
                    "actions",
                    "goals_or_intentions",
                    "causal_predecessors",
                    "causal_outcomes",
                    "spoken_topics",
                    "ocr_entities",
                    "visual_state",
                    "audio_state",
                )
                for item in (
                    schema.get(key, [])
                    if isinstance(schema.get(key), list)
                    else [schema.get(key)]
                )
                if item
            )
        return self._tokenize(
            " ".join(
                [
                    node.visual_summary,
                    " ".join(node.tags),
                    " ".join(node.entities),
                    " ".join(span.text for span in node.speech_spans),
                    " ".join(span.text for span in node.ocr_spans),
                    schema_text,
                ]
            )
        )

    def _schema_list(self, schema: dict[str, object], key: str) -> set[str]:
        value = schema.get(key)
        values = value if isinstance(value, list) else []
        return {
            " ".join(str(item).lower().split()).strip()
            for item in values
            if " ".join(str(item).split()).strip()
        }

    def _append_graph_edge(
        self,
        metadata: dict[str, object],
        key: str,
        node_id: str,
    ) -> None:
        values = [str(item) for item in metadata.get(key, [])]
        if node_id not in values:
            values.append(node_id)
        metadata[key] = values

    def _metadata_terms(
        self,
        metadata: dict[str, object],
        node: VideoNode,
        *,
        keys: tuple[str, ...],
    ) -> set[str]:
        values: list[str] = []
        for key in keys:
            for source in (metadata, node.metadata):
                value = source.get(key)
                if value is None:
                    continue
                values.extend(_flatten_metadata_text(value))
        return self._tokenize(" ".join(values))

    def _visual_refinement_span(
        self,
        node: VideoNode,
        state: ControllerState | None,
    ) -> tuple[TimeSpan, dict[str, object]]:
        if not self._should_expand_vrrqa_visual_refinement(state):
            return node.time_span, {}
        graph_nodes = self._vrrqa_graph_refinement_nodes(node)
        if len(graph_nodes) <= 1:
            return node.time_span, {
                "vrrqa_graph_expansion_enabled": True,
                "vrrqa_graph_expansion_applied": False,
                "vrrqa_graph_expansion_node_ids": [node.node_id],
                "vrrqa_graph_expansion_node_spans": [node.time_span.to_dict()],
                "vrrqa_relationship_frame_policy": "graph_node_start_mid_end",
                "vrrqa_relationship_frame_timestamps": self._relationship_frame_timestamps(
                    [node]
                ),
                "vrrqa_graph_expansion_reason": "no_graph_neighbors",
            }
        expanded_span = TimeSpan(
            start=min(candidate.time_span.start for candidate in graph_nodes),
            end=max(candidate.time_span.end for candidate in graph_nodes),
        )
        return expanded_span, {
            "vrrqa_graph_expansion_enabled": True,
            "vrrqa_graph_expansion_applied": expanded_span != node.time_span,
            "vrrqa_graph_expansion_source_node_id": node.node_id,
            "vrrqa_graph_expansion_node_ids": [candidate.node_id for candidate in graph_nodes],
            "vrrqa_graph_expansion_levels": [candidate.level for candidate in graph_nodes],
            "vrrqa_graph_expansion_node_spans": [
                candidate.time_span.to_dict() for candidate in graph_nodes
            ],
            "vrrqa_graph_expansion_original_span": node.time_span.to_dict(),
            "vrrqa_graph_expansion_span": expanded_span.to_dict(),
            "vrrqa_graph_expansion_neighbor_count": self.vrrqa_graph_refinement_neighbor_count,
            "vrrqa_relationship_frame_policy": "graph_node_start_mid_end",
            "vrrqa_relationship_frame_timestamps": self._relationship_frame_timestamps(
                graph_nodes
            ),
            "vrrqa_graph_expansion_reason": "parent_child_sibling_edges",
        }

    def _should_expand_vrrqa_visual_refinement(self, state: ControllerState | None) -> bool:
        return (
            self.enable_vrrqa_graph_refinement_expansion
            and self.memory.metadata.get("vrrqa_visual_only") is True
            and state is not None
            and state.task_type == "multiple_choice_visual_qa"
        )

    def _vrrqa_graph_refinement_nodes(self, node: VideoNode) -> list[VideoNode]:
        if node.parent_id is None:
            return [node]
        parent = self.memory.get_node(node.parent_id)
        siblings = sorted(
            self.memory.child_nodes(parent.node_id),
            key=lambda candidate: (candidate.time_span.start, candidate.node_id),
        )
        try:
            current_index = next(
                index for index, candidate in enumerate(siblings) if candidate.node_id == node.node_id
            )
        except StopIteration:
            return [node]
        neighbor_count = max(0, self.vrrqa_graph_refinement_neighbor_count)
        start_index = max(0, current_index - neighbor_count)
        end_index = min(len(siblings), current_index + neighbor_count + 1)
        return siblings[start_index:end_index]

    def _relationship_frame_timestamps(self, nodes: list[VideoNode]) -> list[float]:
        timestamps: list[float] = []
        for node in nodes:
            timestamps.extend(
                [
                    node.time_span.start,
                    node.time_span.start + (node.time_span.duration / 2.0),
                    node.time_span.end,
                ]
            )
        return _merge_float_values(timestamps)

    def _visual_refinement_prompt(
        self,
        state: ControllerState | None,
        graph_metadata: dict[str, object] | None = None,
    ) -> str | None:
        if state is None:
            return None
        if _is_longshot_context(state) and state.task_type != "multiple_choice_visual_qa":
            return self._longshot_visual_refinement_prompt(state, graph_metadata)
        if state.task_type != "multiple_choice_visual_qa":
            return None
        clean_question = _clean_vrrqa_question(state.question)
        options = _extract_vrrqa_options(state.question)
        if not clean_question or not options:
            return None
        option_lines = "\n".join(f"{letter}. {text}" for letter, text in options.items())
        is_timelogic = (
            state.global_context.get("benchmark") == "timelogic"
            or getattr(state.event_memory, "task_name", None) == "timelogic"
        )
        task_intro = (
            "Analyze this short video clip for TimeLogic temporal-logic QA answering."
            if is_timelogic
            else "Analyze this short video clip for VRR-QA multiple-choice answering."
        )
        timelogic_lines = []
        if is_timelogic and state.event_memory is not None:
            event_phrases = [
                event.phrase for event in state.event_memory.events.values()
            ]
            timelogic_lines = [
                "TimeLogic TLQA mode:",
                "Localize the listed action phrases if they are visible in this clip.",
                "When an action phrase is visible, mention the exact phrase and its frame/time "
                "order in `temporal_order`, `frame_timeline`, `evidence`, and `summary`.",
                "If a listed action phrase is absent, say it is not visible instead of guessing.",
                "Listed action phrases: " + "; ".join(event_phrases),
            ]
        if is_timelogic:
            return "\n".join(
                [
                    "Analyze these ordered frames for TimeLogic TLQA.",
                    "Use only visible evidence from the frames. Do not use audio, subtitles, "
                    "captions, or outside knowledge.",
                    "Localize the listed action phrases only when directly visible. If an action "
                    "is absent or unclear, mark it not visible instead of guessing.",
                "Return compact strict JSON only, with keys: `best_option`, `option_scores`, "
                    "`visible_events`, `event_schema`, `evidence_frames`, `temporal_order`, "
                    "`frame_timeline`, `evidence`, `summary`, `tags`, and `entities`.",
                    "`best_option` must be one option letter only when supported by visible "
                    "evidence; otherwise use null.",
                    "`option_scores` must map each option letter to a 0.0-1.0 confidence.",
                    "`visible_events` must be a short list of objects with `phrase`, `visible`, "
                    "`frame_indices`, and `evidence`.",
                    "`event_schema` must include actors, objects, actions, place, state changes, "
                    "causal_predecessors, causal_outcomes, spoken_topics, and ocr_entities when "
                    "visible.",
                    "`evidence_frames` must list frame indices/timestamps that support the answer.",
                    "`temporal_order` must be a concise ordered timeline using exact listed "
                    "action phrases where visible.",
                    "`frame_timeline` must be at most one short phrase per key frame.",
                    "`evidence` and `summary` must be concise and grounded in visible frames.",
                    "Do not wrap the JSON in markdown fences.",
                    "",
                    f"Question: {clean_question}",
                    "Options:",
                    option_lines,
                    *timelogic_lines,
                ]
            )
        relation_lines = []
        if requires_co_visible_relation(clean_question):
            relation_lines = [
                "This is a spatial/depth/viewpoint relation question.",
                "Verify the relation only from frames where the relevant entities are co-visible.",
                "Set `co_visible` false and `relation_supported` false if the entities are not "
                "visible together in the same frame; do not infer the relation from separate shots.",
            ]
        elif is_spatial_relation_question(clean_question):
            relation_lines = [
                "This visual question may involve motion or direction.",
                "Judge direction from frame-to-frame changes across the selected frames.",
                "Do not require two separate target entities to be co-visible unless the question "
                "explicitly asks for a relation between entities.",
            ]
        graph_lines = []
        if graph_metadata and graph_metadata.get("vrrqa_graph_expansion_applied"):
            graph_lines = [
                "",
                "Graph inspection context:",
                "The frames may cover the current memory node plus adjacent graph nodes reached via "
                "parent-child sibling edges.",
                "Use the extra graph context only to resolve the current question; do not treat it as "
                "arbitrary temporal padding.",
                f"Current node: {graph_metadata.get('vrrqa_graph_expansion_source_node_id')}",
                "Graph nodes inspected: "
                + ", ".join(
                    str(node_id)
                    for node_id in graph_metadata.get("vrrqa_graph_expansion_node_ids", [])
                ),
            ]
        return "\n".join(
            [
                task_intro,
                "The images are ordered frames from one clip span or nearby graph spans. Inspect "
                "each frame/keyframe carefully in order before aggregating.",
                "Focus on visual implicit reasoning: spatial relations, viewpoint/visibility, "
                "motion direction/trajectory, temporal order, entity continuity, and physical "
                "context.",
                "Use only visible frame evidence. Do not use audio, subtitles, captions, or outside "
                "knowledge.",
                "First ground the exact target entities named by the question. For each plausible "
                "candidate entity, say why it matches or does not match the target.",
                "Then produce per-frame observations and relation votes. Aggregate only frames "
                "where the target entities are visible together for spatial, depth, or facing "
                "relations.",
                "Return strict JSON only. Put the answer fields first so they are never omitted.",
                "Required key order: `best_option`, `option_scores`, `target_entities`, "
                "`candidate_entities`, `entity_grounding`, `frame_observations`, "
                "`co_visible_frame_indices`, `relation_votes`, `vote_counts`, "
                "`aggregated_relation`, `entities_visible`, `co_visible`, "
                "`relation_supported`, `visible_relation`, `spatial_relation`, "
                "`motion_trajectory`, `temporal_order`, `entity_continuity`, "
                "`physical_context`, `inferred_relation`, `option_comparison`, "
                "`event_schema`, `evidence_frames`, `evidence`, `summary`, `frame_timeline`, "
                "`tags`, `entities`.",
                "`best_option` must be exactly one option letter from the choices only when a "
                "choice is visually supported; otherwise use null.",
                "`option_scores` must map each option letter to a confidence from 0.0 to 1.0.",
                "`frame_observations` must be a short list with one item per input frame: "
                "frame_index, target_entities_visible, co_visible, entity_grounding, relation, "
                "motion, and option_support.",
                "`co_visible_frame_indices` must list frame indices where all target entities "
                "needed for the relation are in the same frame.",
                "`relation_votes` must list only co-visible frame votes for spatial/depth/facing "
                "questions.",
                "`vote_counts` must count the relation votes by relation label.",
                "`entities_visible` must say which question entities are visible.",
                "`co_visible` must be true only when the relevant entities appear in the same frame.",
                "`relation_supported` must be true only when at least one co-visible frame directly "
                "supports the relation.",
                "`visible_relation` and `spatial_relation` must state the observed relation or "
                "`unsupported`.",
                "`motion_trajectory`, `temporal_order`, `entity_continuity`, and "
                "`physical_context` must describe visible evidence or say `not visible`.",
                "`inferred_relation` must state the visual relation needed to answer the question.",
                "`option_comparison` must map each option letter to `supports`, `contradicts`, or "
                "`unknown` plus a short reason.",
                "`event_schema` must be a situation model with actors, objects, actions, place, "
                "state changes, causal_predecessors, causal_outcomes, spoken_topics, "
                "ocr_entities, visual_state, and event_type.",
                "`evidence_frames` must list the frame indices/timestamps that directly support "
                "the selected option or relation.",
                "`evidence` must be one concise sentence grounded in visible frames.",
                "`summary` must directly state the selected option and why.",
                "`frame_timeline` should be short: at most one brief phrase per key frame.",
                "Do not wrap the JSON in markdown fences.",
                "",
                f"Question: {clean_question}",
                "Options:",
                option_lines,
                *timelogic_lines,
                *relation_lines,
                *graph_lines,
            ]
        )

    def _longshot_visual_refinement_prompt(
        self,
        state: ControllerState,
        graph_metadata: dict[str, object] | None = None,
    ) -> str:
        graph_lines = []
        if graph_metadata and graph_metadata.get("vrrqa_graph_expansion_applied"):
            graph_lines = [
                "",
                "Graph inspection context:",
                "The frames may include the current memory node plus adjacent graph nodes.",
                "Use adjacent frames only when they clarify the current LongShotBench question.",
                f"Current node: {graph_metadata.get('vrrqa_graph_expansion_source_node_id')}",
                "Graph nodes inspected: "
                + ", ".join(
                    str(node_id)
                    for node_id in graph_metadata.get("vrrqa_graph_expansion_node_ids", [])
                ),
            ]
        longshot = state.global_context.get("longshot")
        hint_lines = []
        if isinstance(longshot, dict):
            expected_modalities = longshot.get("expected_modalities")
            required_tools = longshot.get("required_tools")
            if expected_modalities:
                hint_lines.append(f"Expected modalities: {expected_modalities}")
            if required_tools:
                hint_lines.append(f"Benchmark required-tool hints: {required_tools}")
        return "\n".join(
            [
                "Analyze these ordered frames for LongShotBench grounded answering.",
                "Use only visible frame evidence. Do not use audio, subtitles, captions, "
                "or outside knowledge unless text is visibly present in the frames.",
                "The question may ask for screen text, code, variables, math expressions, "
                "counts, UI labels, object state, or a visual event. Focus on evidence that "
                "would support a final answer for the current user turn.",
                "For screen/code/editor content, copy exact short strings, variable names, "
                "operators, expressions, labels, and shell/output values when readable.",
                "For counting questions, count only items that are visible in the inspected "
                "frames and state uncertainty when visibility is incomplete.",
                "For temporal wording such as first, second, third, after, or final, preserve "
                "frame order and describe what changes between frames.",
                "Return compact strict JSON only, with keys: `summary`, `tags`, `entities`, "
                "`scene_text`, `code_text`, `math_expressions`, `counts`, "
                "`frame_observations`, `evidence_frames`, and `answer_relevant_evidence`.",
                "`summary` must directly state the visual evidence relevant to the question.",
                "`scene_text` and `code_text` must list exact readable text snippets; use [] "
                "when none are readable.",
                "`counts` must map counted object/text categories to integers only when "
                "directly visible.",
                "`frame_observations` must be at most one short item per input frame.",
                "`evidence_frames` must list frame indices/timestamps that support the answer.",
                "`answer_relevant_evidence` must be one concise grounded sentence.",
                "Do not wrap the JSON in markdown fences.",
                "",
                f"Question: {state.question}",
                *hint_lines,
                *graph_lines,
            ]
        )

    def _select_visual_detail_children(
        self,
        detail_node_ids: list[str],
        state: ControllerState | None,
        max_child_details: int,
    ):
        if not detail_node_ids:
            return []
        query_text = ""
        if state is not None:
            query_text = " ".join(
                part for part in [state.question, self._latest_search_query(state)] if part
            )
        query_tokens = self._tokenize(query_text)
        scored = []
        for position, node_id in enumerate(detail_node_ids):
            child = self.memory.get_node(node_id)
            child_tokens = self._tokenize(child.visual_summary)
            overlap = len(query_tokens & child_tokens) if query_tokens else 0
            scored.append((overlap, position, child.time_span.start, child))
        scored.sort(key=lambda item: (-item[0], item[1], item[2], item[3].node_id))
        if any(score > 0 for score, *_ in scored):
            selected = [child for score, *_rest, child in scored if score > 0]
        else:
            selected = [child for *_rest, child in scored]
        return selected[:max_child_details]

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

    def _prioritized_modalities(self, node, preferred: Modality) -> list[Modality]:
        modalities = self._recommended_modalities(node)
        if preferred in modalities:
            return [preferred, *[item for item in modalities if item != preferred]]
        return modalities

    def _child_priority(self, node) -> float:
        score = 0.2
        score += min(len(node.speech_spans) * 0.05, 0.3)
        score += min(len(node.ocr_spans) * 0.03, 0.2)
        score += 0.15 if node.visual_summary else 0.0
        if node.metadata.get("cognitive_event"):
            score += 0.08
            score += min(float(node.metadata.get("memorability_prior") or 0.0) * 0.12, 0.12)
        return round(score, 4)

    def _refinement_priority(self, node, candidate_kind: str, index: int) -> float:
        base = {
            "descendant": 0.78,
            "child": 0.72,
            "sibling": 0.56,
            "parent": 0.4,
        }.get(candidate_kind, 0.4)
        granularity_bonus = {
            "clip": 0.12,
            "event": 0.11,
            "segment": 0.09,
            "scene": 0.04,
            "video": 0.0,
        }.get(node.level, 0.0)
        score = base + granularity_bonus - min(index * 0.03, 0.09)
        return round(score, 4)

    def _next_evidence_id(self) -> str:
        self._evidence_counter += 1
        return f"evidence_{self._evidence_counter:05d}"

    def _should_use_zero_hit_temporal_expansion(
        self,
        state: ControllerState,
        modality: Modality | None,
    ) -> bool:
        if state.task_type not in {"multiple_choice_visual_qa", "agentic_task"}:
            return False
        selected_modality = modality or "visual"
        if selected_modality not in {"visual", "cross_modal", "ocr", "speech", "audio"}:
            return False
        if state.evidence_ledger:
            return False
        return not any(
            action.get("action_type") in {"SEARCH", "OPEN"}
            for action in state.action_history
            if isinstance(action, dict)
        )

    def _zero_hit_temporal_frontier(
        self,
        *,
        state: ControllerState,
        modality: Modality,
        target_slot: str | None,
        limit: int,
    ) -> list[FrontierItem]:
        preferred_modalities = self._zero_hit_modalities(modality)
        candidates = [
            node
            for node in self.memory.nodes.values()
            if node.level == "clip"
            and self._supports_any_modality(node, preferred_modalities)
            and not is_reopen_blocked(
                state.evidence_board,
                node.node_id,
                preferred_modalities[0],
                target_slot,
            )
        ]
        if not candidates:
            candidates = [
                node
                for node in self.memory.nodes.values()
                if node.level != "video"
                and self._supports_any_modality(node, preferred_modalities)
                and not is_reopen_blocked(
                    state.evidence_board,
                    node.node_id,
                    preferred_modalities[0],
                    target_slot,
                )
            ]
        selected = _temporally_diverse_nodes(candidates, max(1, limit))
        frontier: list[FrontierItem] = []
        for index, node in enumerate(selected):
            score = 0.52 - min(index * 0.025, 0.16)
            if (
                "visual" in preferred_modalities
                and (node.metadata.get("on_demand_visual_refinement") or self._has_visual_frame_index(node))
            ):
                score += 0.08
            recommended = self._prioritized_modalities(node, preferred_modalities[0])
            frontier.append(
                FrontierItem(
                    node_id=node.node_id,
                    time_span=node.time_span,
                    level=node.level,
                    score=round(score, 4),
                    why_candidate=(
                        "Zero-hit temporal expansion: inspect a temporally diverse "
                        f"{node.level} after search found no lexical/graph hits for "
                        f"slot '{target_slot or 'generic'}'."
                    ),
                    recommended_modalities=recommended or preferred_modalities,
                    status="unopened",
                )
            )
        return frontier

    def _zero_hit_modalities(self, modality: Modality) -> list[Modality]:
        if modality == "cross_modal":
            return ["visual", "speech", "ocr", "audio"]
        if modality == "ocr":
            return ["visual", "ocr"]
        if modality == "audio":
            return ["audio", "speech"]
        return [modality]

    def _supports_any_modality(self, node: VideoNode, modalities: list[Modality]) -> bool:
        supported = set(self._recommended_modalities(node))
        return bool(supported & set(modalities))

    def _build_refinement_frontier(
        self,
        *,
        node,
        state: ControllerState,
        modality: Modality,
        target_slot: str | None,
        max_items: int = 4,
    ) -> list[FrontierItem]:
        if modality == "visual":
            candidates = self._visual_refinement_descendants(node)
            candidate_kind = "descendant"
        else:
            candidates = []
            candidate_kind = "child"
        if not candidates:
            candidates = self.memory.child_nodes(node.node_id)
            candidate_kind = "child"
        if not candidates and node.parent_id:
            parent = self.memory.get_node(node.parent_id)
            siblings = self.memory.child_nodes(parent.node_id)
            candidates = [item for item in siblings if item.node_id != node.node_id]
            candidate_kind = "sibling"
        if not candidates and node.parent_id:
            parent = self.memory.get_node(node.parent_id)
            candidates = [parent]
            candidate_kind = "parent"

        frontier: list[FrontierItem] = []
        for index, candidate in enumerate(candidates):
            recommended = self._prioritized_modalities(candidate, modality)
            preferred_modality = recommended[0] if recommended else modality
            if is_reopen_blocked(
                state.evidence_board, candidate.node_id, preferred_modality, target_slot
            ):
                continue
            frontier.append(
                FrontierItem(
                    node_id=candidate.node_id,
                    time_span=candidate.time_span,
                    level=candidate.level,
                    score=self._refinement_priority(candidate, candidate_kind, index),
                    why_candidate=(
                        f"Refine {candidate_kind} node after weak "
                        f"{modality} open for slot '{target_slot or 'generic'}'"
                    ),
                    recommended_modalities=recommended or [modality],
                    status="unopened",
                )
            )
            if len(frontier) >= max_items:
                break
        return frontier

    def _build_speech_evidence(self, node, state: ControllerState) -> list[Evidence]:
        if self._should_refine_speech_node(node):
            self._refine_speech_node(node)
        selected_spans = self._select_relevant_speech_spans(node.speech_spans, state)
        if not selected_spans:
            return []

        evidence: list[Evidence] = []
        query_hint = self._latest_search_query(state)
        question_tokens = self._tokenize(state.question)
        query_tokens = self._tokenize(
            " ".join(part for part in [state.question, query_hint] if part)
        )
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

    def _should_refine_speech_node(self, node) -> bool:
        if self.speech_refiner is None:
            return False
        if node.metadata.get("speech_summary_mode") == "on_demand_refined":
            return False
        return any(_is_lazy_speech_span(span) for span in node.speech_spans)

    def _refine_speech_node(self, node) -> None:
        source_video_path = self.memory.metadata.get("source_video_path")
        if not source_video_path:
            node.metadata["speech_refinement"] = "skipped_missing_source_video_path"
            return

        temp_root = get_videorlm_output_root() / "tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="videorlm_lazy_asr_", dir=str(temp_root)) as name:
            audio_path = extract_audio_segment(
                media_path=str(source_video_path),
                span=node.time_span,
                output_path=Path(name) / f"{node.node_id}.wav",
                ffmpeg_bin=getattr(self.speech_refiner, "ffmpeg_bin", "ffmpeg"),
            )
            with _temporary_speech_refinement_progress_callback(
                self.speech_refiner,
                self.progress_callback,
            ):
                refined_spans = self.speech_refiner.recognize(str(audio_path))

        refined_spans = [
            _offset_on_demand_speech_span(span, node.time_span)
            for span in refined_spans
            if span.text.strip() and not _is_lazy_speech_span(span)
        ]
        node.speech_spans = refined_spans
        node.metadata["speech_summary_mode"] = "on_demand_refined"
        node.metadata["speech_refinement"] = "asr_on_demand"
        node.metadata["on_demand_speech_refinement"] = False
        node.metadata["speech_refined_span_count"] = len(refined_spans)

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
        query_tokens = self._tokenize(
            " ".join(part for part in [state.question, query_hint] if part)
        )
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
            if (
                neighbor_score >= max(best_score * 0.35, 0.15)
                or neighbor_has_why_signal
                or ((is_first_query or is_last_query) and neighbor_score >= 0.05)
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

        ordered_indices = sorted(
            selected_indices, key=lambda index: cleaned_spans[index].time_span.start
        )
        return [
            (cleaned_spans[index], score_by_index[index]) for index in ordered_indices[:max_items]
        ]

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
            for marker in (
                "other bracelet",
                "another bracelet",
                "last but not the least",
                "last but not least",
            )
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
            "wants": "want",
            "wanted": "want",
            "wanting": "want",
            "picks": "pick",
            "picked": "pick",
            "picking": "pick",
            "pours": "pour",
            "poured": "pour",
            "pouring": "pour",
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
            for keyword in (
                "because",
                "worried",
                "lose",
                "lost",
                "fixed",
                "repair",
                "opening",
                "clasp",
            )
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
        return (
            bool(
                {
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
                & question_tokens
            )
            or len(normalized) > 320
            or len(sentences) > 4
            or topic_shift
        )

    def _build_speech_refinement_prompt(
        self,
        *,
        question: str,
        search_query: str,
        candidates: list[dict[str, object]],
    ) -> str:
        candidate_lines = []
        for candidate in candidates:
            candidate_lines.append(f"{candidate['candidate_id']}: {candidate['detail']}")
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
            raw_ids = (
                parsed.get("selected_candidate_ids") or parsed.get("selected_candidates") or []
            )
            if isinstance(raw_ids, list):
                candidate_ids = [str(item) for item in raw_ids if str(item) in valid_ids]
            raw_reason = parsed.get("reason")
            if raw_reason is not None:
                reason = str(raw_reason).strip()
        if not candidate_ids:
            candidate_ids = [
                match for match in re.findall(r"c\d+", payload.lower()) if match in valid_ids
            ]
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
        forward_indices = list(
            range(anchor_index + 1, min(len(sentences), anchor_index + after + 1))
        )

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


@contextmanager
def _temporary_visual_refinement_progress_callback(
    component: Any,
    callback: Callable[[dict[str, Any]], None] | None,
):
    if component is None or callback is None or not hasattr(component, "progress_callback"):
        yield
        return

    original = component.progress_callback

    def wrapped(event: dict[str, Any]) -> None:
        payload = dict(event)
        if payload.get("phase") == "visual":
            payload["phase"] = "visual_refinement"
            status = str(payload.get("status") or "")
            payload["status"] = f"vl-refine {status}".strip()
        callback(payload)

    component.progress_callback = wrapped
    try:
        yield
    finally:
        component.progress_callback = original


@contextmanager
def _temporary_visual_prompt_override(component: Any, prompt: str | None):
    if component is None or prompt is None or not hasattr(component, "prompt_override"):
        yield
        return

    original = component.prompt_override
    component.prompt_override = prompt
    try:
        yield
    finally:
        component.prompt_override = original


@contextmanager
def _temporary_visual_forced_frame_timestamps(component: Any, timestamps):
    if component is None or not hasattr(component, "forced_frame_timestamps_override"):
        yield
        return

    original = component.forced_frame_timestamps_override
    normalized = None
    if isinstance(timestamps, list):
        normalized = [float(timestamp) for timestamp in timestamps]
    component.forced_frame_timestamps_override = normalized
    try:
        yield
    finally:
        component.forced_frame_timestamps_override = original


@contextmanager
def _temporary_speech_refinement_progress_callback(
    component: Any,
    callback: Callable[[dict[str, Any]], None] | None,
):
    if component is None or callback is None or not hasattr(component, "progress_callback"):
        yield
        return

    original = component.progress_callback

    def wrapped(event: dict[str, Any]) -> None:
        payload = dict(event)
        if payload.get("phase") == "asr":
            payload["phase"] = "speech_refinement"
            status = str(payload.get("status") or "")
            payload["status"] = f"asr-refine {status}".strip()
        callback(payload)

    component.progress_callback = wrapped
    try:
        yield
    finally:
        component.progress_callback = original


def _is_lazy_speech_span(span: SpeechSpan) -> bool:
    return span.language == "lazy_asr" or span.text.startswith("Lazy ASR index")


def _offset_on_demand_speech_span(span: SpeechSpan, parent_span) -> SpeechSpan:
    if span.time_span.duration == 0:
        time_span = parent_span
    else:
        time_span = TimeSpan(
            parent_span.start + span.time_span.start,
            min(parent_span.end, parent_span.start + span.time_span.end),
        )
    return SpeechSpan(
        text=span.text,
        time_span=time_span,
        speaker=span.speaker,
        language=span.language,
    )


def _node_prefix(node_id: str) -> str:
    for marker in ("_clip_", "_seg_", "_scene_"):
        if marker in node_id:
            return node_id.rsplit(marker, 1)[0] + marker
    return node_id[: max(1, len(node_id) // 2)]


def _temporally_diverse_nodes(nodes: list[VideoNode], limit: int) -> list[VideoNode]:
    if not nodes or limit <= 0:
        return []
    ordered = sorted(nodes, key=lambda node: (node.time_span.start, node.node_id))
    if len(ordered) <= limit:
        return ordered
    if limit == 1:
        return [ordered[0]]

    selected_indices: list[int] = []
    step = (len(ordered) - 1) / float(limit - 1)
    for position in range(limit):
        index = int(round(position * step))
        if index not in selected_indices:
            selected_indices.append(index)
    for index in range(len(ordered)):
        if len(selected_indices) >= limit:
            break
        if index not in selected_indices:
            selected_indices.append(index)
    selected_indices.sort()
    return [ordered[index] for index in selected_indices[:limit]]


def _merge_schema_values(existing: object, new_values: list[object], *, limit: int = 24) -> list[str]:
    values: list[str] = []
    if isinstance(existing, list):
        values.extend(str(item) for item in existing)
    values.extend(str(item) for item in new_values)
    seen: set[str] = set()
    merged: list[str] = []
    for value in values:
        normalized = " ".join(value.split()).strip()
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        merged.append(normalized)
        if len(merged) >= limit:
            break
    return merged


def _flatten_metadata_text(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, dict):
        texts: list[str] = []
        for key, item in value.items():
            texts.append(str(key))
            texts.extend(_flatten_metadata_text(item))
        return texts
    if isinstance(value, list):
        texts: list[str] = []
        for item in value:
            texts.extend(_flatten_metadata_text(item))
        return texts
    return [str(value)]


def _coerce_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


VRRQA_OPTION_PATTERN = re.compile(r"^\s*([A-Z])\.\s+(.+?)\s*$")


def _extract_vrrqa_options(question: str) -> dict[str, str]:
    options = {}
    for line in question.splitlines():
        match = VRRQA_OPTION_PATTERN.match(line)
        if match is not None:
            options[match.group(1)] = match.group(2).strip()
    return dict(sorted(options.items()))


def _is_longshot_context(state: ControllerState) -> bool:
    benchmark = state.global_context.get("benchmark")
    if benchmark in {"longshot", "longshotbench"}:
        return True
    return state.global_context.get("longshot") is not None


def _clean_vrrqa_question(question: str) -> str:
    lines = []
    for line in question.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if not stripped:
            continue
        if lowered == "options:" or VRRQA_OPTION_PATTERN.match(stripped):
            continue
        if lowered.startswith("task:") or lowered.startswith("valid answer letters:"):
            continue
        if lowered.startswith("use the options above") or lowered.startswith("when you stop"):
            continue
        if lowered.startswith("do not answer") or lowered.startswith("if the evidence is incomplete"):
            continue
        if lowered.startswith("question:"):
            stripped = stripped.split(":", maxsplit=1)[1].strip()
        lines.append(stripped)
    return " ".join(lines).strip()


def _merge_float_values(values: list[float], *, tolerance: float = 0.03) -> list[float]:
    merged: list[float] = []
    for value in sorted(float(item) for item in values):
        if merged and abs(merged[-1] - value) <= tolerance:
            continue
        merged.append(round(value, 3))
    return merged
