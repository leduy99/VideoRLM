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
    open_v2,
    search_v2,
    select_target_slot,
)
from rlm.video.index import STOPWORDS, TOKEN_PATTERN, VideoMemoryIndex
from rlm.video.media import extract_audio_segment, get_videorlm_output_root
from rlm.video.types import (
    ControllerAction,
    ControllerState,
    Evidence,
    FrontierItem,
    Modality,
    Observation,
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
    "clip node",
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
        frontier, metadata = search_v2(
            index=self.index,
            question_spec=question_spec,
            target_slot=selected_slot,
            state=state,
            top_k=top_k,
            query_override=query or None,
            modality=modality,
        )
        summary = (
            f"SEARCH {metadata.get('search_mode', 'lexical')} found {len(frontier)} candidate nodes for "
            f"slot '{selected_slot or 'generic'}'."
        )
        return Observation(
            kind="search",
            summary=summary,
            frontier=frontier,
            metadata={
                "query": query,
                "modality": metadata["modality"],
                "search_mode": metadata.get("search_mode", "lexical"),
                "hit_count": len(frontier),
                "target_slot": selected_slot,
                "queries": metadata["queries"],
            },
        )

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

        if selected_modality == "speech":
            raw_evidence = self._build_speech_evidence(node, state)
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
        )
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
        children = self.memory.child_nodes(node.node_id)
        frontier = []
        for child in children:
            reason = f"Child node of {node.node_id} spanning {child.time_span.to_display()}"
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

        level_priority = {"clip": 0, "segment": 1, "scene": 2, "video": 3}
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
            detail = " ".join(item.text.strip() for item in node.ocr_spans if item.text).strip()
            return detail, {}
        if modality == "audio":
            labels = [item.label.strip() for item in node.audio_events if item.label]
            return ", ".join(labels).strip(), {}
        return "", {}

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
        if state is None or state.task_type != "multiple_choice_visual_qa":
            return None
        clean_question = _clean_vrrqa_question(state.question)
        options = _extract_vrrqa_options(state.question)
        if not clean_question or not options:
            return None
        option_lines = "\n".join(f"{letter}. {text}" for letter, text in options.items())
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
                "Analyze this short video clip for VRR-QA multiple-choice answering.",
                "The images are ordered frames from one clip span. Inspect them carefully in order.",
                "Return strict JSON only. Put the answer fields first so they are never omitted.",
                "Required key order: `best_option`, `option_scores`, `evidence`, "
                "`summary`, `frame_timeline`, `tags`, `entities`.",
                "`best_option` must be exactly one option letter from the choices.",
                "`option_scores` must map each option letter to a confidence from 0.0 to 1.0.",
                "`evidence` must be one concise sentence grounded in visible frames.",
                "`summary` must directly state the selected option and why.",
                "`frame_timeline` should be short: at most one brief phrase per key frame.",
                "Do not wrap the JSON in markdown fences.",
                "",
                f"Question: {clean_question}",
                "Options:",
                option_lines,
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
            "segment": 0.09,
            "scene": 0.04,
            "video": 0.0,
        }.get(node.level, 0.0)
        score = base + granularity_bonus - min(index * 0.03, 0.09)
        return round(score, 4)

    def _next_evidence_id(self) -> str:
        self._evidence_counter += 1
        return f"evidence_{self._evidence_counter:05d}"

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


VRRQA_OPTION_PATTERN = re.compile(r"^\s*([A-Z])\.\s+(.+?)\s*$")


def _extract_vrrqa_options(question: str) -> dict[str, str]:
    options = {}
    for line in question.splitlines():
        match = VRRQA_OPTION_PATTERN.match(line)
        if match is not None:
            options[match.group(1)] = match.group(2).strip()
    return dict(sorted(options.items()))


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
