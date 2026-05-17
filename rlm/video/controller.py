import copy
import json
import re
import time
from collections.abc import Callable
from typing import Any, Literal

from rlm.clients import get_client
from rlm.clients.base_lm import BaseLM
from rlm.core.types import ClientBackend
from rlm.video.adapters import (
    EmbeddingProvider,
    ImageTextEmbeddingProvider,
    SpeechRecognizer,
    VisualSummarizer,
)
from rlm.video.evidence_pipeline import (
    build_evidence_board,
    build_question_spec,
    build_slot_queries,
    search_v2,
    select_target_slot,
    update_evidence_board,
)
from rlm.video.index import STOPWORDS, TOKEN_PATTERN, VideoMemoryIndex
from rlm.video.logger import VideoRLMLogger
from rlm.video.prompts import build_controller_prompt
from rlm.video.tools import VideoToolExecutor
from rlm.video.types import (
    BudgetState,
    ControllerAction,
    ControllerState,
    FrontierItem,
    TraceStep,
    VideoMemory,
    VideoRLMResult,
)


class VideoRLM:
    def __init__(
        self,
        controller_backend: ClientBackend = "openai",
        controller_backend_kwargs: dict[str, Any] | None = None,
        controller_client: BaseLM | None = None,
        logger: VideoRLMLogger | None = None,
        max_steps: int = 8,
        search_top_k: int = 5,
        max_frontier_items: int = 8,
        enable_hybrid_speech_refinement: bool = False,
        speech_snippet_refiner_client: BaseLM | None = None,
        speech_refine_candidate_count: int = 4,
        search_mode: Literal["lexical", "graph"] = "lexical",
        embedding_provider: EmbeddingProvider | None = None,
        image_text_embedding_provider: ImageTextEmbeddingProvider | None = None,
        speech_refiner: SpeechRecognizer | None = None,
        visual_refiner: VisualSummarizer | None = None,
        enable_vrrqa_graph_refinement_expansion: bool = True,
        vrrqa_graph_refinement_neighbor_count: int = 1,
    ):
        if search_mode not in {"lexical", "graph"}:
            raise ValueError(f"Unsupported search mode: {search_mode}")
        self.controller_backend = controller_backend
        self.controller_backend_kwargs = controller_backend_kwargs or {}
        self.controller_client = controller_client or get_client(
            controller_backend, self.controller_backend_kwargs
        )
        self.logger = logger
        self.max_steps = max_steps
        self.search_top_k = search_top_k
        self.max_frontier_items = max_frontier_items
        self.enable_hybrid_speech_refinement = enable_hybrid_speech_refinement
        self.speech_snippet_refiner_client = speech_snippet_refiner_client or (
            self.controller_client if enable_hybrid_speech_refinement else None
        )
        self.speech_refine_candidate_count = speech_refine_candidate_count
        self.search_mode = search_mode
        self.embedding_provider = embedding_provider
        self.image_text_embedding_provider = image_text_embedding_provider
        self.speech_refiner = speech_refiner
        self.visual_refiner = visual_refiner
        self.enable_vrrqa_graph_refinement_expansion = enable_vrrqa_graph_refinement_expansion
        self.vrrqa_graph_refinement_neighbor_count = vrrqa_graph_refinement_neighbor_count

    def run(
        self,
        question: str,
        memory: VideoMemory,
        dialogue_context: list[dict[str, str]] | None = None,
        task_type: str | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> VideoRLMResult:
        start_time = time.perf_counter()
        index = VideoMemoryIndex(
            memory,
            embedding_provider=self.embedding_provider,
            image_text_embedding_provider=self.image_text_embedding_provider,
            search_mode=self.search_mode,
        )
        tools = VideoToolExecutor(
            memory=memory,
            index=index,
            top_k=self.search_top_k,
            speech_snippet_refiner=self.speech_snippet_refiner_client,
            enable_hybrid_speech_refinement=self.enable_hybrid_speech_refinement,
            speech_refine_candidate_count=self.speech_refine_candidate_count,
            speech_refiner=self.speech_refiner,
            visual_refiner=self.visual_refiner,
            enable_vrrqa_graph_refinement_expansion=(
                self.enable_vrrqa_graph_refinement_expansion
            ),
            vrrqa_graph_refinement_neighbor_count=self.vrrqa_graph_refinement_neighbor_count,
            progress_callback=progress_callback,
        )
        state = self._build_initial_state(
            question=question,
            memory=memory,
            index=index,
            dialogue_context=dialogue_context or [],
            task_type=task_type,
        )

        if self.logger:
            self.logger.clear_steps()
            self.logger.log_metadata(
                {
                    "controller_model": self.controller_client.model_name,
                    "video_id": memory.video_id,
                    "max_steps": self.max_steps,
                    "search_top_k": self.search_top_k,
                    "search_mode": self.search_mode,
                    "semantic_frame_embeddings": self.image_text_embedding_provider is not None,
                    "hybrid_speech_refinement": self.enable_hybrid_speech_refinement,
                    "speech_refinement": self.speech_refiner is not None,
                    "visual_refinement": self.visual_refiner is not None,
                    "vrrqa_graph_refinement_expansion": (
                        self.enable_vrrqa_graph_refinement_expansion
                    ),
                }
            )

        trace_steps: list[dict[str, Any]] = []
        answer: str | None = None
        consecutive_empty_open_steps = 0

        while state.budget.steps_remaining > 0:
            if self._should_use_multiple_choice_final_step(state):
                previous_state = copy.deepcopy(state.to_dict())
                answer, raw_response = self._multiple_choice_completion_from_state(state)
                action = ControllerAction(
                    action_type="STOP",
                    answer=answer,
                    evidence_ids=self._final_answer_evidence_ids(state),
                )
                observation = tools.execute(action, state)
                state = self._apply_observation(state, action, observation)
                next_state = state.to_dict()
                trace_step = TraceStep(
                    step_index=state.budget.steps_used,
                    state=previous_state,
                    action=action.to_dict(),
                    observation=observation.to_dict(),
                    next_state=next_state,
                    raw_model_response=raw_response,
                )
                trace_steps.append(trace_step.to_dict())
                if self.logger:
                    self.logger.log_step(trace_step)
                break

            prompt = build_controller_prompt(
                state,
                max_frontier_items=self.max_frontier_items,
            )
            raw_response = self.controller_client.completion(prompt)
            action = self._parse_action(raw_response, state)
            if action.target_slot is None:
                action.target_slot = select_target_slot(state.question_spec, state.evidence_board)
            previous_state = copy.deepcopy(state.to_dict())
            observation = tools.execute(action, state)
            state = self._apply_observation(state, action, observation)
            next_state = state.to_dict()

            trace_step = TraceStep(
                step_index=state.budget.steps_used,
                state=previous_state,
                action=action.to_dict(),
                observation=observation.to_dict(),
                next_state=next_state,
                raw_model_response=raw_response,
            )
            trace_steps.append(trace_step.to_dict())
            if self.logger:
                self.logger.log_step(trace_step)

            if action.action_type == "STOP":
                answer = self._answer_from_stop_action(action, state)
                break
            if action.action_type == "OPEN" and not observation.evidence:
                consecutive_empty_open_steps += 1
            else:
                consecutive_empty_open_steps = 0
            if consecutive_empty_open_steps >= 2 and state.evidence_ledger:
                answer = self._fallback_answer_from_state(state)
                break
            if state.no_progress_steps >= 2:
                answer = self._fallback_answer_from_state(state)
                break

        if answer is None:
            answer = self._fallback_answer_from_state(state)

        usage = self.controller_client.get_usage_summary()
        return VideoRLMResult(
            answer=answer,
            state=state,
            trace=trace_steps,
            usage_summary=usage,
            execution_time=time.perf_counter() - start_time,
        )

    def _build_initial_state(
        self,
        question: str,
        memory: VideoMemory,
        index: VideoMemoryIndex,
        dialogue_context: list[dict[str, str]],
        task_type: str | None,
    ) -> ControllerState:
        clean_question = _clean_controller_question(question, task_type)
        answer_options = _extract_multiple_choice_options(question)
        question_spec = build_question_spec(
            question=clean_question,
            task_type=task_type,
            dialogue_context=dialogue_context,
        )
        evidence_board = build_evidence_board(question_spec)
        scene_summaries = []
        for node in memory.top_level_nodes()[:6]:
            summary = node.visual_summary or node.node_id
            scene_summaries.append(summary[:120])

        budget = BudgetState(
            steps_used=0,
            steps_remaining=self.max_steps,
            tool_calls_used=0,
            max_depth=0,
            current_depth=0,
            clips_opened=0,
            tokens_spent=0,
        )
        global_context = {
            "video_id": memory.video_id,
            "video_length_seconds": memory.metadata.get("duration_seconds"),
            "node_count": len(memory.nodes),
            "available_modalities": self._available_modalities(memory),
            "search_mode": self.search_mode,
            "semantic_frame_embeddings": self.image_text_embedding_provider is not None,
            "speech_refinement": self.speech_refiner is not None,
            "visual_refinement": self.visual_refiner is not None,
            "vrrqa_graph_refinement_expansion": self.enable_vrrqa_graph_refinement_expansion,
            "clean_question": clean_question,
            "answer_options": answer_options,
            "valid_answer_letters": sorted(answer_options),
            "topical_index": scene_summaries,
            "evidence_metrics": {
                "slot_fill_rate": 0.0,
                "background_only_open_rate": 0.0,
                "duplicate_evidence_rate": 0.0,
                "no_progress_rate": 0.0,
                "tokens_per_step": 0.0,
            },
        }
        state = ControllerState(
            question=question,
            task_type=task_type,
            dialogue_context=dialogue_context,
            question_spec=question_spec,
            evidence_board=evidence_board,
            budget=budget,
            global_context=global_context,
        )
        frontier, _ = search_v2(
            index=index,
            question_spec=question_spec,
            target_slot=select_target_slot(question_spec, evidence_board),
            state=state,
            top_k=self.max_frontier_items,
        )
        if not frontier:
            frontier = [
                FrontierItem(
                    node_id=node.node_id,
                    time_span=node.time_span,
                    level=node.level,
                    score=0.1,
                    why_candidate=f"Top-level node {node.node_id}",
                    recommended_modalities=["visual", "speech"],
                )
                for node in memory.top_level_nodes()
            ]
        state.frontier = frontier[: self.max_frontier_items]
        return state

    def _answer_from_stop_action(
        self,
        action: ControllerAction,
        state: ControllerState,
    ) -> str:
        multiple_choice_answer = self._multiple_choice_answer_from_state(state)
        if multiple_choice_answer is not None:
            return multiple_choice_answer
        return action.answer or self._fallback_answer_from_state(state)

    def _apply_observation(
        self,
        state: ControllerState,
        action: ControllerAction,
        observation,
    ) -> ControllerState:
        state.budget.steps_used += 1
        state.budget.steps_remaining = max(0, state.budget.steps_remaining - 1)
        state.budget.tool_calls_used += 1
        if action.action_type == "OPEN":
            state.budget.clips_opened += 1

        usage = self.controller_client.get_usage_summary()
        state.budget.tokens_spent = usage.total_input_tokens + usage.total_output_tokens
        state.action_history.append(action.to_dict())
        if state.evidence_board is not None:
            state.evidence_board = update_evidence_board(
                state.evidence_board,
                state.question_spec,
                observation,
                state.budget.steps_used,
            )

        if action.action_type == "SEARCH":
            state.frontier = self._merge_frontier(state.frontier, observation.frontier)
        elif action.action_type == "OPEN":
            state.frontier = self._set_frontier_status(state.frontier, action.node_id, "opened")
            state.frontier = self._remove_frontier_node(state.frontier, action.node_id)
            state.evidence_ledger.extend(observation.evidence)
            if observation.frontier:
                state.frontier = self._merge_frontier(state.frontier, observation.frontier)
        elif action.action_type == "SPLIT":
            state.frontier = self._set_frontier_status(state.frontier, action.node_id, "expanded")
            state.frontier = self._merge_frontier(state.frontier, observation.frontier)
        elif action.action_type == "MERGE":
            state.evidence_ledger.extend(observation.evidence)
        elif action.action_type == "STOP":
            selected = set(action.evidence_ids)
            for evidence in state.evidence_ledger:
                if evidence.evidence_id in selected:
                    evidence.used_in_final_answer = True

        progress_made = bool(observation.metadata.get("progress_made"))
        if progress_made:
            state.no_progress_steps = 0
        else:
            state.no_progress_steps += 1
            if state.evidence_board is not None:
                state.evidence_board.no_progress_count += 1

        state.global_context["evidence_metrics"] = self._build_evidence_metrics(state)

        return state

    def _merge_frontier(
        self,
        existing: list[FrontierItem],
        new_items: list[FrontierItem],
    ) -> list[FrontierItem]:
        merged = {item.node_id: item for item in existing}
        for item in new_items:
            current = merged.get(item.node_id)
            if current is None or item.score >= current.score:
                merged[item.node_id] = item
            elif current is not None:
                current.recommended_modalities = sorted(
                    set(current.recommended_modalities) | set(item.recommended_modalities)
                )

        ordered = sorted(merged.values(), key=lambda item: (-item.score, item.time_span.start))
        return ordered[: self.max_frontier_items]

    def _set_frontier_status(
        self,
        frontier: list[FrontierItem],
        node_id: str | None,
        status: str,
    ) -> list[FrontierItem]:
        if not node_id:
            return frontier
        updated = []
        for item in frontier:
            if item.node_id == node_id:
                item.status = status
            updated.append(item)
        return updated

    def _remove_frontier_node(
        self,
        frontier: list[FrontierItem],
        node_id: str | None,
    ) -> list[FrontierItem]:
        if not node_id:
            return frontier
        return [item for item in frontier if item.node_id != node_id]

    def _parse_action(
        self, raw_response: str, state: ControllerState | None = None
    ) -> ControllerAction:
        candidate = raw_response.strip()
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                extracted = self._extract_first_json_object(candidate)
                payload = json.loads(extracted)
            except (json.JSONDecodeError, ValueError):
                payload = self._recover_partial_action_payload(candidate, state)
        if not isinstance(payload, dict):
            payload = self._recover_partial_action_payload(candidate, state)
        payload = self._sanitize_action_payload(payload)
        if state is not None:
            payload = self._repair_action_payload(payload, state)
        return ControllerAction.from_dict(payload)

    def _sanitize_action_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        cleaned = dict(payload)
        action_type = str(cleaned.get("action_type") or "").upper()
        if action_type in {"SEARCH", "OPEN", "SPLIT", "MERGE", "STOP"}:
            cleaned["action_type"] = action_type

        evidence_ids = cleaned.get("evidence_ids")
        if isinstance(evidence_ids, list):
            cleaned["evidence_ids"] = [
                item
                for item in evidence_ids
                if isinstance(item, str) and item.startswith("evidence_")
            ]
        elif isinstance(evidence_ids, str) and evidence_ids.startswith("evidence_"):
            cleaned["evidence_ids"] = [evidence_ids]
        else:
            cleaned["evidence_ids"] = []
        return cleaned

    def _recover_partial_action_payload(
        self,
        text: str,
        state: ControllerState | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "action_type": self._extract_json_scalar_field(text, "action_type") or "SEARCH",
            "query": self._extract_json_scalar_field(text, "query"),
            "modality": self._extract_json_scalar_field(text, "modality"),
            "node_id": self._extract_json_scalar_field(text, "node_id"),
            "target_slot": self._extract_json_scalar_field(text, "target_slot"),
            "evidence_ids": sorted(set(re.findall(r"evidence_\d+", text))),
            "answer": self._extract_json_scalar_field(text, "answer"),
            "rationale": self._extract_json_scalar_field(text, "rationale"),
        }
        action_type = str(payload["action_type"]).upper()
        if action_type not in {"SEARCH", "OPEN", "SPLIT", "MERGE", "STOP"}:
            action_type = "SEARCH"
        payload["action_type"] = action_type

        if action_type == "SEARCH" and not payload["query"] and state is not None:
            target_slot = payload["target_slot"] or select_target_slot(
                state.question_spec,
                state.evidence_board,
            )
            payload["target_slot"] = target_slot
            payload["query"] = self._default_search_query(state, target_slot)
        return payload

    def _extract_json_scalar_field(self, text: str, field_name: str) -> str | None:
        pattern = re.compile(
            rf'"{re.escape(field_name)}"\s*:\s*(null|"(?:\\.|[^"\\])*")',
            re.DOTALL,
        )
        match = pattern.search(text)
        if match is None:
            return None
        token = match.group(1)
        if token == "null":
            return None
        try:
            value = json.loads(token)
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, str) else None

    def _repair_action_payload(
        self,
        payload: dict[str, Any],
        state: ControllerState,
    ) -> dict[str, Any]:
        action_type = payload.get("action_type")
        if action_type == "SPLIT":
            self._repair_node_id_payload(payload, state)
            return payload
        if action_type not in {"SEARCH", "OPEN"}:
            return payload

        target_slot = payload.get("target_slot") or select_target_slot(
            state.question_spec,
            state.evidence_board,
        )
        preferred_modality = self._preferred_search_modality(state, target_slot)
        current_modality = payload.get("modality")
        resolved_modality = (
            self._resolve_available_modality(current_modality, state) if current_modality else None
        )
        if action_type == "SEARCH":
            if self._should_open_frontier_instead_of_search(state):
                payload["action_type"] = "OPEN"
                payload["node_id"] = state.frontier[0].node_id
                payload["query"] = None
                resolved_modality = self._resolve_available_modality(
                    state.frontier[0].recommended_modalities[0]
                    if state.frontier[0].recommended_modalities
                    else preferred_modality,
                    state,
                )
                payload["modality"] = resolved_modality
                payload["target_slot"] = target_slot
                self._repair_node_id_payload(payload, state)
                return payload
            if not payload.get("query"):
                payload["query"] = self._default_search_query(state, target_slot)
            else:
                payload["query"] = self._clean_search_query(str(payload["query"]), state)
            if resolved_modality is None or self._should_override_open_modality(
                current_modality=resolved_modality,
                preferred_modality=preferred_modality,
            ):
                payload["modality"] = preferred_modality
            else:
                payload["modality"] = resolved_modality
        elif resolved_modality is None or self._should_override_open_modality(
            current_modality=resolved_modality,
            preferred_modality=preferred_modality,
        ):
            payload["modality"] = preferred_modality
        else:
            payload["modality"] = resolved_modality
        payload["target_slot"] = target_slot
        if action_type == "OPEN":
            self._repair_node_id_payload(payload, state)
        return payload

    def _default_search_query(self, state: ControllerState, target_slot: str | None) -> str:
        question = _clean_controller_question(state.question, state.task_type)
        if state.question_spec is None:
            return question
        return build_slot_queries(question, state.question_spec, target_slot)[0]

    def _clean_search_query(self, query: str, state: ControllerState) -> str:
        if state.task_type != "multiple_choice_visual_qa":
            return query
        clean_question = _clean_controller_question(state.question, state.task_type)
        query_text = " ".join(query.split())
        if "Valid answer letters:" in query_text or "Do not answer with None" in query_text:
            return clean_question
        if len(query_text) > max(len(clean_question) * 2, 180):
            return clean_question
        return query

    def _should_open_frontier_instead_of_search(self, state: ControllerState) -> bool:
        if state.task_type != "multiple_choice_visual_qa":
            return False
        if not state.frontier:
            return False
        if state.evidence_ledger:
            return False
        if not state.action_history:
            return False
        last_action = state.action_history[-1]
        if last_action.get("action_type") != "SEARCH":
            return False
        return any(item.status == "unopened" for item in state.frontier)

    def _repair_node_id_payload(
        self,
        payload: dict[str, Any],
        state: ControllerState,
    ) -> None:
        node_id = payload.get("node_id")
        frontier_ids = state.frontier_ids()
        if node_id in frontier_ids:
            return
        if state.frontier:
            payload["node_id"] = state.frontier[0].node_id
            return
        payload["action_type"] = "SEARCH"
        payload["node_id"] = None
        payload["query"] = payload.get("query") or _clean_controller_question(
            state.question, state.task_type
        )

    def _preferred_search_modality(
        self,
        state: ControllerState,
        target_slot: str | None,
    ) -> str:
        if state.question_spec is None:
            return "speech"
        if target_slot is not None:
            slot = state.question_spec.get_slot(target_slot)
            if slot is not None and slot.preferred_modality is not None:
                return self._resolve_available_modality(slot.preferred_modality, state)
        return self._resolve_available_modality(
            state.question_spec.preferred_modality or "speech", state
        )

    def _resolve_available_modality(self, modality: str, state: ControllerState) -> str:
        available = state.global_context.get("available_modalities", {})
        if modality == "ocr" and not available.get("ocr") and available.get("visual"):
            return "visual"
        if modality == "audio" and not available.get("audio") and available.get("speech"):
            return "speech"
        return modality

    def _should_override_open_modality(
        self,
        current_modality: str | None,
        preferred_modality: str,
    ) -> bool:
        if current_modality == preferred_modality:
            return False
        if current_modality == "speech" and preferred_modality in {"visual", "ocr", "audio"}:
            return True
        if current_modality == "visual" and preferred_modality in {"ocr", "audio"}:
            return True
        return False

    def _available_modalities(self, memory: VideoMemory) -> dict[str, bool]:
        nodes = [node for node in memory.nodes.values() if node.level != "video"]
        return {
            "speech": any(node.speech_spans for node in nodes),
            "visual": any(node.visual_summary.strip() for node in nodes),
            "ocr": any(node.ocr_spans for node in nodes),
            "audio": any(node.audio_events for node in nodes),
        }

    def _extract_first_json_object(self, text: str) -> str:
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(text[index:])
                return json.dumps(payload)
            except json.JSONDecodeError:
                continue
        raise ValueError(f"Could not parse controller action JSON from: {text}")

    def _fallback_answer_from_state(self, state: ControllerState) -> str:
        multiple_choice_answer = self._multiple_choice_answer_from_state(state)
        if multiple_choice_answer is not None:
            return multiple_choice_answer
        if state.evidence_board is not None and state.evidence_board.missing_required_slots:
            return self._diagnostic_abstain_from_state(state)
        if state.evidence_ledger:
            return self._synthesize_answer_from_evidence(state)
        return "Controller exhausted its budget before collecting grounded evidence."

    def _synthesize_answer_from_evidence(self, state: ControllerState) -> str:
        multiple_choice_answer = self._multiple_choice_answer_from_state(state)
        if multiple_choice_answer is not None:
            return multiple_choice_answer
        if state.evidence_board is not None:
            allowed_ids = set(
                state.evidence_board.core_evidence_ids + state.evidence_board.support_evidence_ids
            )
        else:
            allowed_ids = set()
        filtered_evidence = [
            item
            for item in state.evidence_ledger
            if not allowed_ids or item.evidence_id in allowed_ids
        ]
        top_evidence = sorted(
            filtered_evidence or state.evidence_ledger,
            key=lambda item: (-item.confidence, item.time_span.start),
        )[:4]
        evidence_lines = []
        for item in top_evidence:
            evidence_lines.append(
                json.dumps(
                    {
                        "evidence_id": item.evidence_id,
                        "slot": item.metadata.get("slot"),
                        "role": item.metadata.get("role"),
                        "modality": item.modality,
                        "time_span": item.time_span.to_dict(),
                        "excerpt": _focus_evidence_detail(item.detail, state.question),
                    },
                    ensure_ascii=True,
                )
            )

        prompt = (
            "You are a grounded answerer for a long-video reasoning system.\n"
            "Answer the user's question using only the evidence below.\n"
            "Prefer the most direct causal explanation supported by the evidence.\n"
            "If the evidence includes both a problem and a later fix or repair, mention both.\n"
            "If the evidence includes concrete numbers, preparation details, or quoted reactions that directly support the answer, include the most relevant ones.\n"
            "If the question asks about the first or last thing, identify the earliest or latest relevant item or event from the evidence rather than a later summary.\n"
            "Be concise and specific. If the evidence is insufficient, say that clearly.\n"
            "Do not mention internal ids or budget exhaustion.\n\n"
            f"Question: {state.question}\n\n"
            "Evidence:\n" + "\n".join(evidence_lines)
        )
        return self.controller_client.completion(prompt).strip()

    def _should_use_multiple_choice_final_step(self, state: ControllerState) -> bool:
        return (
            state.task_type == "multiple_choice_visual_qa"
            and state.budget.steps_remaining == 1
            and bool(state.evidence_ledger)
            and bool(_extract_multiple_choice_options(state.question))
        )

    def _multiple_choice_answer_from_state(self, state: ControllerState) -> str | None:
        completion = self._multiple_choice_completion_from_state(state)
        if completion is None:
            return None
        answer, _raw_response = completion
        return answer

    def _multiple_choice_completion_from_state(
        self,
        state: ControllerState,
    ) -> tuple[str, str] | None:
        if state.task_type != "multiple_choice_visual_qa":
            return None
        options = _extract_multiple_choice_options(state.question)
        if not options:
            return None
        evidence_best_option = self._best_verified_option_from_evidence(state, options)
        if evidence_best_option is not None:
            return evidence_best_option, json.dumps(
                {
                    "source": "vrrqa_visual_verification",
                    "best_option": evidence_best_option,
                }
            )

        evidence_lines = []
        for item in sorted(
            state.evidence_ledger,
            key=lambda evidence: (-evidence.confidence, evidence.time_span.start),
        )[:8]:
            evidence_lines.append(
                json.dumps(
                    {
                        "slot": item.metadata.get("slot"),
                        "role": item.metadata.get("role"),
                        "modality": item.modality,
                        "time_span": item.time_span.to_dict(),
                        "detail": _focus_evidence_detail(item.detail or item.claim, state.question),
                    },
                    ensure_ascii=True,
                )
            )

        option_lines = [f"{letter}. {text}" for letter, text in options.items()]
        prompt = "\n".join(
            [
                "You are answering a VRR-QA multiple-choice question.",
                "Return exactly one option letter from the valid choices. Do not explain.",
                "If the evidence is incomplete or ambiguous, still choose the best-supported option.",
                "",
                f"Question: {_strip_options_from_question(state.question)}",
                "Options:",
                *option_lines,
                "",
                "Evidence:",
                *(evidence_lines or ["No direct evidence was collected."]),
                "",
                f"Valid answer letters: {', '.join(options)}",
                "Final answer letter:",
            ]
        )
        response = self.controller_client.completion(prompt).strip()
        parsed = _parse_multiple_choice_letter(response, options)
        return parsed or sorted(options)[0], response

    def _best_verified_option_from_evidence(
        self,
        state: ControllerState,
        options: dict[str, str],
    ) -> str | None:
        verified: list[tuple[float, float, str]] = []
        valid_choices = set(options)
        for item in state.evidence_ledger:
            best_option = item.metadata.get("vrrqa_best_option")
            if not isinstance(best_option, str) or best_option not in valid_choices:
                continue
            score = _verified_option_score(best_option, item.metadata.get("vrrqa_option_scores"))
            verified.append((score, item.confidence, best_option))
        if not verified:
            return None
        verified.sort(reverse=True)
        return verified[0][2]

    def _final_answer_evidence_ids(self, state: ControllerState, max_items: int = 4) -> list[str]:
        if state.evidence_board is not None:
            allowed_ids = set(
                state.evidence_board.core_evidence_ids + state.evidence_board.support_evidence_ids
            )
        else:
            allowed_ids = set()
        evidence = [
            item
            for item in state.evidence_ledger
            if not allowed_ids or item.evidence_id in allowed_ids
        ]
        if not evidence:
            evidence = list(state.evidence_ledger)
        evidence.sort(key=lambda item: (-item.confidence, item.time_span.start))
        return [item.evidence_id for item in evidence[:max_items]]

    def _diagnostic_abstain_from_state(self, state: ControllerState) -> str:
        if state.evidence_board is None:
            return "I could not collect enough grounded evidence to answer safely."
        missing = ", ".join(state.evidence_board.missing_required_slots) or "unknown"
        background_slots = [
            slot_name
            for slot_name, slot in state.evidence_board.slots.items()
            if slot.status == "background_only"
        ]
        if background_slots:
            return (
                "I found related background evidence, but the required answer-bearing slots are "
                f"still missing: {missing}. Background-only slots: {', '.join(background_slots)}."
            )
        return (
            "I could not fill all required evidence slots needed for a grounded answer. "
            f"Missing slots: {missing}."
        )

    def _build_evidence_metrics(self, state: ControllerState) -> dict[str, float]:
        board = state.evidence_board
        if board is None:
            return {
                "slot_fill_rate": 0.0,
                "background_only_open_rate": 0.0,
                "duplicate_evidence_rate": 0.0,
                "no_progress_rate": 0.0,
                "tokens_per_step": 0.0,
            }
        opened_count = max(1, len(board.opened_targets))
        total_slots = max(1, len(board.slots))
        return {
            "slot_fill_rate": round(board.slot_fill_count / total_slots, 4),
            "background_only_open_rate": round(board.background_only_open_count / opened_count, 4),
            "duplicate_evidence_rate": round(board.duplicate_evidence_count / opened_count, 4),
            "no_progress_rate": round(board.no_progress_count / max(1, state.budget.steps_used), 4),
            "tokens_per_step": round(
                state.budget.tokens_spent / max(1, state.budget.steps_used),
                4,
            ),
        }


def _focus_evidence_detail(detail: str, question: str, max_chars: int = 1200) -> str:
    normalized = " ".join(detail.split())
    if len(normalized) <= max_chars:
        return normalized

    focus_keywords = (
        "because",
        "why",
        "reason",
        "clasp",
        "opening",
        "opened",
        "open",
        "worried",
        "lose",
        "lost",
        "fixed",
        "repair",
        "repaired",
        "brought it back",
    )
    question_tokens = {
        token
        for token in (match.group(0).lower() for match in TOKEN_PATTERN.finditer(question))
        if token not in STOPWORDS and len(token) > 1
    }
    if {"first", "beginning", "earliest", "initial"} & question_tokens:
        early_window = min(len(normalized), max(max_chars, 1800))
        return normalized[:early_window].strip()
    if {"last", "final", "ending", "end"} & question_tokens:
        late_window = min(len(normalized), max(max_chars, 1800))
        return normalized[-late_window:].strip()
    sentences = [
        sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", normalized) if sentence.strip()
    ]

    candidates: list[tuple[str, int]] = []
    if sentences:
        for index in range(len(sentences)):
            start = max(0, index - 1)
            end = min(len(sentences), index + 3)
            snippet = " ".join(sentences[start:end]).strip()
            if snippet:
                candidates.append((snippet, normalized.find(snippet)))

    lower_detail = normalized.lower()
    for keyword in focus_keywords:
        search_start = 0
        while True:
            match_index = lower_detail.find(keyword, search_start)
            if match_index == -1:
                break
            snippet_start = max(0, match_index - max_chars // 4)
            snippet_end = min(len(normalized), match_index + (3 * max_chars) // 4)
            snippet = normalized[snippet_start:snippet_end].strip()
            if snippet:
                candidates.append((snippet, snippet_start))
            search_start = match_index + len(keyword)

    if not candidates:
        return normalized[:max_chars]

    best_snippet = candidates[0][0]
    best_score = -1
    for snippet, start_index in candidates:
        snippet_tokens = {
            token
            for token in (match.group(0).lower() for match in TOKEN_PATTERN.finditer(snippet))
            if token not in STOPWORDS and len(token) > 1
        }
        overlap = len(question_tokens & snippet_tokens)
        keyword_hits = sum(1 for keyword in focus_keywords if keyword in snippet.lower())
        score = overlap * 3 + keyword_hits * 5
        if any(
            keyword in snippet.lower()
            for keyword in ("worried", "lose", "lost", "fixed", "repair", "clasp", "opening")
        ):
            score += 8
        if {"first", "beginning", "earliest", "initial"} & question_tokens:
            position = max(0.0, 1.0 - (max(start_index, 0) / max(len(normalized), 1)))
            score += position * 10
        if {"last", "final", "ending", "end"} & question_tokens:
            position = max(0.0, max(start_index, 0) / max(len(normalized), 1))
            score += position * 10
        if score > best_score:
            best_score = score
            best_snippet = snippet

    return best_snippet[:max_chars]


MULTIPLE_CHOICE_OPTION_PATTERN = re.compile(r"^\s*([A-Z])\.\s+(.+?)\s*$")
MULTIPLE_CHOICE_LETTER_PATTERN = re.compile(
    r"^\s*(?:OPTION\s*)?([A-Z])(?:[\).:]|\s|$)"
    r"|\b(?:ANSWER|CHOICE)\s*(?:IS|=|:)\s*\(?([A-Z])\)?\b"
)


def _extract_multiple_choice_options(question: str) -> dict[str, str]:
    options: dict[str, str] = {}
    for line in question.splitlines():
        match = MULTIPLE_CHOICE_OPTION_PATTERN.match(line)
        if match is None:
            continue
        options[match.group(1)] = match.group(2).strip()
    return dict(sorted(options.items()))


def _strip_options_from_question(question: str) -> str:
    lines = []
    for line in question.splitlines():
        if line.strip().lower() == "options:":
            continue
        if MULTIPLE_CHOICE_OPTION_PATTERN.match(line):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _clean_controller_question(question: str, task_type: str | None = None) -> str:
    if task_type != "multiple_choice_visual_qa":
        return question
    cleaned = _strip_options_from_question(question)
    lines = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if lowered.startswith("task:"):
            continue
        if lowered.startswith("valid answer letters:"):
            continue
        if lowered.startswith("use the options above"):
            continue
        if lowered.startswith("when you stop"):
            continue
        if lowered.startswith("do not answer"):
            continue
        if lowered.startswith("if the evidence is incomplete"):
            continue
        if lowered.startswith("question:"):
            stripped = stripped.split(":", maxsplit=1)[1].strip()
        lines.append(stripped)
    return " ".join(lines).strip() or question


def _parse_multiple_choice_letter(response: str, options: dict[str, str]) -> str | None:
    valid_letters = set(options)
    normalized = response.strip().upper()
    if normalized in valid_letters:
        return normalized
    for match in MULTIPLE_CHOICE_LETTER_PATTERN.finditer(response.upper()):
        letter = match.group(1) or match.group(2)
        if letter in valid_letters:
            return letter
    response_text = _normalize_choice_text(response)
    contained = [
        letter
        for letter, option_text in options.items()
        if _normalize_choice_text(option_text) and _normalize_choice_text(option_text) in response_text
    ]
    if len(contained) == 1:
        return contained[0]
    return None


def _verified_option_score(choice: str, option_scores: Any) -> float:
    if not isinstance(option_scores, dict):
        return 1.0
    raw_score = option_scores.get(choice)
    try:
        return float(raw_score)
    except (TypeError, ValueError):
        return 1.0


def _normalize_choice_text(text: str) -> str:
    return " ".join(
        token.group(0).lower()
        for token in TOKEN_PATTERN.finditer(text)
        if token.group(0).lower() not in STOPWORDS
    )
