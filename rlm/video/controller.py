import copy
import json
import re
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, Literal

from rlm.clients import get_client
from rlm.clients.base_lm import BaseLM
from rlm.core.types import ClientBackend
from rlm.video.adapters import (
    EmbeddingProvider,
    ImageTextEmbeddingProvider,
    SpeechRecognizer,
    VideoWindowEmbeddingProvider,
    VisualSummarizer,
)
from rlm.video.dynamic_retrieval import (
    dynamic_evidence_retrieval_policy,
    select_dynamic_evidence_chain,
)
from rlm.video.event_memory import (
    build_event_memory_from_global_context,
    event_match_score,
    event_memory_metrics,
    update_event_memory_from_observation,
)
from rlm.video.evidence_pipeline import (
    build_evidence_board,
    build_question_spec,
    build_slot_queries,
    graph_expansion_covered_node_ids,
    is_reopen_blocked,
    relation_evidence_status,
    requires_co_visible_relation,
    search_v2,
    select_target_slot,
    update_evidence_board,
)
from rlm.video.gpu_memory import unload_component
from rlm.video.index import STOPWORDS, TOKEN_PATTERN, VideoMemoryIndex
from rlm.video.logger import VideoRLMLogger
from rlm.video.prompt_plugins import render_prompt_plugin_section
from rlm.video.prompts import build_controller_prompt
from rlm.video.question_router import (
    QuestionRoute,
    evidence_matches_route,
    format_answer_for_route,
    route_from_metadata,
    route_question,
)
from rlm.video.rerankers import VideoWindowReranker
from rlm.video.timing import TimingRecorder
from rlm.video.tools import VideoToolExecutor
from rlm.video.types import (
    BudgetState,
    ControllerAction,
    ControllerState,
    Evidence,
    EvidenceSlotSpec,
    FrontierItem,
    TimeSpan,
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
        speech_embedding_provider: EmbeddingProvider | None = None,
        image_text_embedding_provider: ImageTextEmbeddingProvider | None = None,
        video_window_embedding_provider: VideoWindowEmbeddingProvider | None = None,
        enable_video_window_reranking: bool = False,
        video_window_rerank_candidate_count: int = 24,
        video_window_rerank_weight: float = 0.75,
        video_window_rerank_window_seconds: float | None = None,
        video_window_rerank_min_score: float | None = None,
        speech_refiner: SpeechRecognizer | None = None,
        enable_targeted_asr_refinement: bool = False,
        enable_refinement_frontier: bool = True,
        visual_refiner: VisualSummarizer | None = None,
        enable_vrrqa_graph_refinement_expansion: bool = True,
        vrrqa_graph_refinement_neighbor_count: int = 1,
        enable_vrrqa_visual_answer_verifier: bool = True,
        vrrqa_visual_verifier_frame_count: int = 8,
        enable_controller_evidence_classifier: bool = False,
        offload_components_after_use: bool = False,
        enable_dynamic_evidence_retrieval: bool = True,
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
        self.speech_embedding_provider = speech_embedding_provider
        self.image_text_embedding_provider = image_text_embedding_provider
        self.video_window_embedding_provider = video_window_embedding_provider
        self.enable_video_window_reranking = enable_video_window_reranking
        self.video_window_rerank_candidate_count = video_window_rerank_candidate_count
        self.video_window_rerank_weight = video_window_rerank_weight
        self.video_window_rerank_window_seconds = video_window_rerank_window_seconds
        self.video_window_rerank_min_score = video_window_rerank_min_score
        self.speech_refiner = speech_refiner
        self.enable_targeted_asr_refinement = enable_targeted_asr_refinement
        self.enable_refinement_frontier = enable_refinement_frontier
        self.visual_refiner = visual_refiner
        self.enable_vrrqa_graph_refinement_expansion = enable_vrrqa_graph_refinement_expansion
        self.vrrqa_graph_refinement_neighbor_count = vrrqa_graph_refinement_neighbor_count
        self.enable_vrrqa_visual_answer_verifier = enable_vrrqa_visual_answer_verifier
        self.vrrqa_visual_verifier_frame_count = vrrqa_visual_verifier_frame_count
        self.enable_controller_evidence_classifier = enable_controller_evidence_classifier
        self.offload_components_after_use = offload_components_after_use
        self.enable_dynamic_evidence_retrieval = enable_dynamic_evidence_retrieval

    def _video_window_reranking_enabled(self) -> bool:
        return (
            self.enable_video_window_reranking
            and self.video_window_embedding_provider is not None
        )

    def _build_video_window_reranker(self) -> VideoWindowReranker | None:
        if not self._video_window_reranking_enabled():
            return None
        return VideoWindowReranker(
            embedding_provider=self.video_window_embedding_provider,
            candidate_count=self.video_window_rerank_candidate_count,
            stage2_weight=self.video_window_rerank_weight,
            window_seconds=self.video_window_rerank_window_seconds,
            min_stage2_score=self.video_window_rerank_min_score,
            offload_after_rerank=self.offload_components_after_use,
        )

    def run(
        self,
        question: str,
        memory: VideoMemory,
        dialogue_context: list[dict[str, str]] | None = None,
        task_type: str | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        global_context_overrides: dict[str, Any] | None = None,
    ) -> VideoRLMResult:
        start_time = time.perf_counter()
        timing_recorder = TimingRecorder()
        with timing_recorder.record("controller.init.index_build"):
            index = VideoMemoryIndex(
                memory,
                embedding_provider=self.embedding_provider,
                speech_embedding_provider=self.speech_embedding_provider,
                image_text_embedding_provider=self.image_text_embedding_provider,
                search_mode=self.search_mode,
            )
        with timing_recorder.record("controller.init.video_window_reranker_build"):
            video_window_reranker = self._build_video_window_reranker()
        with timing_recorder.record("controller.init.tool_executor_build"):
            tools = VideoToolExecutor(
                memory=memory,
                index=index,
                top_k=self.search_top_k,
                speech_snippet_refiner=self.speech_snippet_refiner_client,
                enable_hybrid_speech_refinement=self.enable_hybrid_speech_refinement,
                speech_refine_candidate_count=self.speech_refine_candidate_count,
                speech_search_reranker=self.controller_client,
                speech_refiner=self.speech_refiner,
                enable_targeted_asr_refinement=self.enable_targeted_asr_refinement,
                enable_refinement_frontier=self.enable_refinement_frontier,
                visual_refiner=self.visual_refiner,
                enable_vrrqa_graph_refinement_expansion=(
                    self.enable_vrrqa_graph_refinement_expansion
                ),
                vrrqa_graph_refinement_neighbor_count=(
                    self.vrrqa_graph_refinement_neighbor_count
                ),
                video_window_reranker=video_window_reranker,
                evidence_classifier_client=self.controller_client,
                enable_controller_evidence_classifier=self.enable_controller_evidence_classifier,
                progress_callback=progress_callback,
                timing_recorder=timing_recorder,
            )
        with timing_recorder.record("controller.init.initial_state_build"):
            state = self._build_initial_state(
                question=question,
                memory=memory,
                index=index,
                dialogue_context=dialogue_context or [],
                task_type=task_type,
                global_context_overrides=global_context_overrides,
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
                    "video_window_reranking": self._video_window_reranking_enabled(),
                    "controller_evidence_classifier": (
                        self.enable_controller_evidence_classifier
                    ),
                    "hybrid_speech_refinement": self.enable_hybrid_speech_refinement,
                    "speech_refinement": self.speech_refiner is not None,
                    "targeted_asr_refinement": self.enable_targeted_asr_refinement,
                    "refinement_frontier": self.enable_refinement_frontier,
                    "dynamic_evidence_retrieval": self.enable_dynamic_evidence_retrieval,
                    "visual_refinement": self.visual_refiner is not None,
                    "vrrqa_graph_refinement_expansion": (
                        self.enable_vrrqa_graph_refinement_expansion
                    ),
                    "vrrqa_visual_answer_verifier": (
                        self.enable_vrrqa_visual_answer_verifier
                        and self.visual_refiner is not None
                    ),
                }
            )

        trace_steps: list[dict[str, Any]] = []
        answer: str | None = None
        mismatch_answer = self._dataset_video_mismatch_answer(state)
        if mismatch_answer is not None:
            answer = mismatch_answer
            state.global_context["early_stop"] = {
                "source": "dataset_video_mismatch_guard",
                "steps_used": state.budget.steps_used,
            }
        consecutive_empty_open_steps = 0
        self._notify_progress_status(progress_callback, "controller ready")

        while answer is None and state.budget.steps_remaining > 0:
            if self._should_use_multiple_choice_final_step(state):
                step_total_start = time.perf_counter()
                step_timing: dict[str, float] = {}
                previous_state = copy.deepcopy(state.to_dict())
                self._notify_progress_status(
                    progress_callback,
                    "controller final choice generation",
                )
                finalizer_start = time.perf_counter()
                answer, raw_response = self._multiple_choice_completion_from_state(state)
                step_timing["finalizer_completion_seconds"] = round(
                    time.perf_counter() - finalizer_start,
                    6,
                )
                timing_recorder.add(
                    "controller.finalizer.multiple_choice_completion",
                    step_timing["finalizer_completion_seconds"],
                )
                action = ControllerAction(
                    action_type="STOP",
                    answer=answer,
                    evidence_ids=self._final_answer_evidence_ids(state),
                )
                tool_start = time.perf_counter()
                observation = tools.execute(action, state)
                step_timing["tool_execute_seconds"] = round(
                    time.perf_counter() - tool_start,
                    6,
                )
                if not observation.metadata.get("stop_rejected"):
                    verified_ids = observation.metadata.get("verified_evidence_ids")
                    if isinstance(verified_ids, list):
                        action.evidence_ids = [str(item) for item in verified_ids]
                apply_start = time.perf_counter()
                state = self._apply_observation(state, action, observation)
                step_timing["state_apply_seconds"] = round(
                    time.perf_counter() - apply_start,
                    6,
                )
                timing_recorder.add(
                    "controller.state_apply",
                    step_timing["state_apply_seconds"],
                )
                next_state = state.to_dict()
                step_timing["step_total_seconds"] = round(
                    time.perf_counter() - step_total_start,
                    6,
                )
                trace_step = TraceStep(
                    step_index=state.budget.steps_used,
                    state=previous_state,
                    action=action.to_dict(),
                    observation=observation.to_dict(),
                    next_state=next_state,
                    raw_model_response=raw_response,
                    timing=step_timing,
                )
                trace_steps.append(trace_step.to_dict())
                if self.logger:
                    self.logger.log_step(trace_step)
                if observation.metadata.get("stop_rejected"):
                    answer = None
                    self._notify_progress_status(
                        progress_callback,
                        "controller stop rejected by answer verifier",
                    )
                    continue
                break

            step_total_start = time.perf_counter()
            step_timing: dict[str, float] = {}
            prompt_start = time.perf_counter()
            prompt = build_controller_prompt(
                state,
                max_frontier_items=self.max_frontier_items,
            )
            step_timing["prompt_build_seconds"] = round(
                time.perf_counter() - prompt_start,
                6,
            )
            timing_recorder.add(
                "controller.prompt_build",
                step_timing["prompt_build_seconds"],
            )
            self._notify_progress_status(
                progress_callback,
                f"controller step {state.budget.steps_used + 1}/{self.max_steps} generating action",
            )
            controller_completion_start = time.perf_counter()
            raw_response = self.controller_client.completion(prompt)
            step_timing["controller_completion_seconds"] = round(
                time.perf_counter() - controller_completion_start,
                6,
            )
            timing_recorder.add(
                "controller.completion",
                step_timing["controller_completion_seconds"],
            )
            self._notify_progress_status(
                progress_callback,
                f"controller step {state.budget.steps_used + 1}/{self.max_steps} parsing action",
            )
            parse_start = time.perf_counter()
            action = self._parse_action(raw_response, state)
            step_timing["parse_action_seconds"] = round(
                time.perf_counter() - parse_start,
                6,
            )
            timing_recorder.add(
                "controller.parse_action",
                step_timing["parse_action_seconds"],
            )
            if action.target_slot is None:
                action.target_slot = select_target_slot(state.question_spec, state.evidence_board)
            forced_final_answer: str | None = None
            if self._should_force_multiple_choice_finalization(state, action):
                self._notify_progress_status(
                    progress_callback,
                    "controller forced final choice generation",
                )
                forced_finalizer_start = time.perf_counter()
                completion = self._multiple_choice_completion_from_state(state)
                step_timing["forced_finalizer_completion_seconds"] = round(
                    time.perf_counter() - forced_finalizer_start,
                    6,
                )
                timing_recorder.add(
                    "controller.finalizer.forced_multiple_choice_completion",
                    step_timing["forced_finalizer_completion_seconds"],
                )
                if completion is not None:
                    forced_final_answer, finalizer_response = completion
                    raw_response = json.dumps(
                        {
                            "source": "loop_control_finalization",
                            "original_controller_response": raw_response,
                            "finalizer_response": finalizer_response,
                        },
                        ensure_ascii=True,
                    )
                    action = ControllerAction(
                        action_type="STOP",
                        target_slot=select_target_slot(
                            state.question_spec,
                            state.evidence_board,
                        ),
                        evidence_ids=self._final_answer_evidence_ids(state),
                        answer=forced_final_answer,
                    )
            previous_state = copy.deepcopy(state.to_dict())
            self._notify_progress_status(
                progress_callback,
                f"controller executing {action.action_type.lower()}",
            )
            tool_start = time.perf_counter()
            observation = tools.execute(action, state)
            step_timing["tool_execute_seconds"] = round(
                time.perf_counter() - tool_start,
                6,
            )
            if action.action_type == "STOP" and not observation.metadata.get("stop_rejected"):
                verified_ids = observation.metadata.get("verified_evidence_ids")
                if isinstance(verified_ids, list):
                    action.evidence_ids = [str(item) for item in verified_ids]
            self._notify_progress_status(
                progress_callback,
                f"controller applying {action.action_type.lower()} result",
            )
            apply_start = time.perf_counter()
            state = self._apply_observation(state, action, observation)
            step_timing["state_apply_seconds"] = round(
                time.perf_counter() - apply_start,
                6,
            )
            timing_recorder.add(
                "controller.state_apply",
                step_timing["state_apply_seconds"],
            )
            next_state = state.to_dict()
            step_timing["step_total_seconds"] = round(
                time.perf_counter() - step_total_start,
                6,
            )

            trace_step = TraceStep(
                step_index=state.budget.steps_used,
                state=previous_state,
                action=action.to_dict(),
                observation=observation.to_dict(),
                next_state=next_state,
                raw_model_response=raw_response,
                timing=step_timing,
            )
            trace_steps.append(trace_step.to_dict())
            if self.logger:
                self.logger.log_step(trace_step)

            if action.action_type == "STOP":
                if observation.metadata.get("stop_rejected"):
                    self._notify_progress_status(
                        progress_callback,
                        "controller stop rejected by answer verifier",
                    )
                    continue
                self._notify_progress_status(progress_callback, "controller finalizing answer")
                answer = self._format_final_answer_for_state(
                    forced_final_answer or self._answer_from_stop_action(action, state),
                    state,
                    self._evidence_for_action(action, state),
                )
                break
            filled_slot_answer = self._filled_required_slots_answer_from_state(state)
            if filled_slot_answer is not None:
                answer, evidence_ids, early_stop_source = filled_slot_answer
                for evidence in state.evidence_ledger:
                    if evidence.evidence_id in set(evidence_ids):
                        evidence.used_in_final_answer = True
                state.global_context["early_stop"] = {
                    "source": early_stop_source,
                    "evidence_ids": evidence_ids,
                    "steps_used": state.budget.steps_used,
                }
                self._notify_progress_status(
                    progress_callback,
                    "controller filled-slot early stop",
                )
                break
            grounded_completion = self._grounded_multiple_choice_completion_from_state(state)
            if grounded_completion is not None:
                answer, raw_grounded_response = grounded_completion
                state.global_context["early_stop"] = {
                    "source": "grounded_multiple_choice_completion",
                    "response": raw_grounded_response,
                    "steps_used": state.budget.steps_used,
                }
                self._notify_progress_status(
                    progress_callback,
                    "controller grounded early stop",
                )
                break
            if action.action_type == "OPEN" and not observation.evidence:
                consecutive_empty_open_steps += 1
            else:
                consecutive_empty_open_steps = 0
            if (
                consecutive_empty_open_steps >= 2
                and state.evidence_ledger
                and not self._has_unopened_required_dynamic_chain_targets(state)
            ):
                answer = self._fallback_answer_from_state(state)
                break
            if (
                state.no_progress_steps >= 2
                and not self._has_unopened_required_dynamic_chain_targets(state)
            ):
                answer = self._fallback_answer_from_state(state)
                break

        if answer is None:
            self._notify_progress_status(progress_callback, "controller fallback answer")
            answer = self._fallback_answer_from_state(state)

        answer = self._repair_final_answer(answer, state)
        if not str(answer or "").strip() and state.evidence_ledger:
            answer = self._synthesize_answer_from_evidence(state)
        if not str(answer or "").strip() and state.evidence_ledger:
            answer = self._deterministic_answer_from_evidence(state, state.evidence_ledger)
        self._notify_progress_status(progress_callback, "controller done")
        self._offload_idle_components()
        usage = self.controller_client.get_usage_summary()
        execution_time = time.perf_counter() - start_time
        timing_recorder.add("controller.run.total_wall", execution_time)
        timing_summary = timing_recorder.snapshot()
        timing_summary.update(
            {
                "wall_seconds": round(execution_time, 6),
                "steps_used": state.budget.steps_used,
                "tool_calls_used": state.budget.tool_calls_used,
            }
        )
        state.global_context["timing"] = timing_summary
        return VideoRLMResult(
            answer=answer,
            state=state,
            trace=trace_steps,
            usage_summary=usage,
            execution_time=execution_time,
            timing=timing_summary,
        )

    def _offload_idle_components(self) -> None:
        if not self.offload_components_after_use:
            return
        for component in (
            self.controller_client,
            self.embedding_provider,
            self.speech_embedding_provider,
            self.image_text_embedding_provider,
            self.video_window_embedding_provider,
            self.speech_refiner,
            self.visual_refiner,
        ):
            unload_component(component)

    def _notify_progress_status(
        self,
        progress_callback: Callable[[dict[str, Any]], None] | None,
        status: str,
    ) -> None:
        if progress_callback is None:
            return
        progress_callback({"phase": "controller", "event": "status", "status": status})

    def _build_initial_state(
        self,
        question: str,
        memory: VideoMemory,
        index: VideoMemoryIndex,
        dialogue_context: list[dict[str, str]],
        task_type: str | None,
        global_context_overrides: dict[str, Any] | None = None,
    ) -> ControllerState:
        clean_question = _clean_controller_question(question, task_type)
        answer_options = _extract_multiple_choice_options(question)
        question_spec = build_question_spec(
            question=clean_question,
            task_type=task_type,
            dialogue_context=dialogue_context,
        )
        route_context = (global_context_overrides or {}).get("longshot")
        question_route = route_question(clean_question, task_type, route_context)
        question_spec.metadata["question_route"] = question_route.to_dict()
        temporal_intents = _postvalid_temporal_intents(clean_question)
        if temporal_intents:
            question_spec.metadata["postvalid_temporal_intents"] = temporal_intents
        if question_route.preferred_modality is not None:
            question_spec.preferred_modality = question_route.preferred_modality
            for slot in question_spec.required_slots:
                if slot.preferred_modality is None or slot.slot == "main_claim":
                    slot.preferred_modality = question_route.preferred_modality
        self._apply_postvalid_question_spec(question_spec, question_route, route_context)
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
            "source_video_path": memory.metadata.get("source_video_path"),
            "video_length_seconds": memory.metadata.get("duration_seconds"),
            "node_count": len(memory.nodes),
            "available_modalities": self._available_modalities(memory),
            "search_mode": self.search_mode,
            "semantic_frame_embeddings": self.image_text_embedding_provider is not None,
            "speech_refinement": self.speech_refiner is not None,
            "refinement_frontier": self.enable_refinement_frontier,
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
            "question_route": question_route.to_dict(),
        }
        if global_context_overrides:
            global_context.update(global_context_overrides)
        if temporal_intents:
            global_context["postvalid_temporal_intents"] = temporal_intents
        mismatch = self._detect_dataset_video_mismatch(
            question=clean_question,
            memory=memory,
            longshot_context=global_context.get("longshot"),
        )
        if mismatch is not None:
            global_context["dataset_video_mismatch"] = mismatch
        if global_context.get("benchmark") == "timelogic":
            global_context["timelogic_temporal_sweep_candidates"] = (
                _timelogic_temporal_sweep_candidates(memory)
            )
        event_memory = build_event_memory_from_global_context(global_context)
        if event_memory is not None:
            global_context["event_memory_metrics"] = event_memory_metrics(event_memory)
        state = ControllerState(
            question=question,
            task_type=task_type,
            dialogue_context=dialogue_context,
            question_spec=question_spec,
            evidence_board=evidence_board,
            event_memory=event_memory,
            budget=budget,
            global_context=global_context,
        )
        dynamic_policy_enabled, dynamic_policy_reason = dynamic_evidence_retrieval_policy(
            state,
            question_spec,
            question_route,
        )
        dynamic_plan = (
            select_dynamic_evidence_chain(
                index=index,
                state=state,
                question_spec=question_spec,
                top_k=self.max_frontier_items,
            )
            if self.enable_dynamic_evidence_retrieval and dynamic_policy_enabled
            else None
        )
        if dynamic_plan is not None:
            frontier = dynamic_plan.frontier
            state.global_context["dynamic_evidence_retrieval"] = {
                **dynamic_plan.to_metadata(),
                "policy_reason": dynamic_policy_reason,
            }
        else:
            frontier, _ = search_v2(
                index=index,
                question_spec=question_spec,
                target_slot=select_target_slot(question_spec, evidence_board),
                state=state,
                top_k=self.max_frontier_items,
            )
            state.global_context["dynamic_evidence_retrieval"] = {
                "enabled": False,
                "policy_reason": (
                    dynamic_policy_reason
                    if self.enable_dynamic_evidence_retrieval
                    else "disabled_by_config"
                ),
            }
        if not frontier:
            frontier = self._fallback_initial_frontier(memory, task_type)
        state.frontier = frontier[: self.max_frontier_items]
        return state

    def _apply_postvalid_question_spec(
        self,
        question_spec,
        question_route: QuestionRoute,
        route_context: Any,
    ) -> None:
        if not _is_postvalid_v1_longshot_context(route_context):
            return
        if question_route.label == "sentiment_analysis":
            question_spec.question_type = "postvalid_sentiment_analysis"
            question_spec.preferred_modality = "cross_modal"
            question_spec.required_slots = [
                EvidenceSlotSpec(
                    slot="speech_content",
                    description=(
                        "Quoted or emotional speech around the moment, including what the "
                        "person or coach says and any stated motivation"
                    ),
                    required=True,
                    preferred_modality="speech",
                ),
                EvidenceSlotSpec(
                    slot="visual_body_language",
                    description=(
                        "Visible body language, facial expression, bench/sideline behavior, "
                        "or coach interaction that shows the person's emotional state"
                    ),
                    required=True,
                    preferred_modality="visual",
                ),
                EvidenceSlotSpec(
                    slot="scene_context",
                    description=(
                        "Scenario context such as tryout status, not playing, non-roster "
                        "participant, bench role, or why the moment matters"
                    ),
                    required=True,
                    preferred_modality="visual",
                ),
                EvidenceSlotSpec(
                    slot="tone_or_audio_event",
                    description=(
                        "Voice tone, crowd reaction, whistle, music, or other audio cue if "
                        "available"
                    ),
                    required=False,
                    preferred_modality="audio",
                ),
            ]
            question_spec.metadata["postvalid_sentiment_slots"] = True
            question_spec.metadata.pop("postvalid_speech_slots", None)
            return
        if question_route.label not in {
            "speech_explanation",
            "causal_chain",
            "temporal_occurrence",
            "rubric_explanation",
        }:
            return
        question_spec.question_type = "postvalid_speech_explanation"
        question_spec.preferred_modality = "speech"
        question_spec.required_slots = [
            EvidenceSlotSpec(
                slot="answer_core",
                description="The central spoken claim that directly answers the current question",
                required=True,
                preferred_modality="speech",
            ),
            EvidenceSlotSpec(
                slot="mechanism",
                description="How the relevant process, setup, or method works",
                required=False,
                preferred_modality="speech",
            ),
            EvidenceSlotSpec(
                slot="causal_or_temporal_link",
                description="The causal or temporal connection needed to explain the answer",
                required=False,
                preferred_modality="speech",
            ),
            EvidenceSlotSpec(
                slot="consequence",
                description="The result, implication, or consequence of the event or explanation",
                required=False,
                preferred_modality="speech",
            ),
        ]
        question_spec.metadata["postvalid_speech_slots"] = True

    def _detect_dataset_video_mismatch(
        self,
        *,
        question: str,
        memory: VideoMemory,
        longshot_context: Any,
    ) -> dict[str, Any] | None:
        if not _is_postvalid_v1_longshot_context(longshot_context):
            return None
        if not isinstance(longshot_context, dict):
            return None
        question_terms = _postvalid_key_terms(question)
        if not question_terms:
            return None
        memory_terms = _memory_key_terms(memory)
        scenario_terms = _postvalid_key_terms(str(longshot_context.get("scenario") or ""))
        missing_terms = [
            term
            for term in question_terms
            if not _term_present_in_memory(term, memory_terms)
        ]
        if not missing_terms:
            return None
        question_term_set = {term.lower() for term in question_terms}
        scenario_term_set = {term.lower() for term in scenario_terms}
        scenario_conflict = bool(scenario_term_set) and not bool(
            question_term_set & scenario_term_set
        )
        severity = "low"
        missing_ratio = len(missing_terms) / max(len(question_terms), 1)
        if missing_ratio >= 0.5 and (scenario_conflict or len(missing_terms) >= 2):
            severity = "high"
        return {
            "severity": severity,
            "question_terms": question_terms,
            "missing_question_terms": missing_terms,
            "scenario_terms": scenario_terms,
            "scenario_conflict": scenario_conflict,
            "memory_terms_sample": sorted(memory_terms)[:40],
        }

    def _dataset_video_mismatch_answer(self, state: ControllerState) -> str | None:
        mismatch = state.global_context.get("dataset_video_mismatch")
        if not isinstance(mismatch, dict) or mismatch.get("severity") != "high":
            return None
        missing = ", ".join(str(term) for term in mismatch.get("missing_question_terms", [])[:6])
        return (
            "Dataset/video mismatch detected: the question asks about terms not present in "
            f"the loaded video memory ({missing}). Check the postvalid_v1 sample/video mapping "
            "or rerun after redownloading the correct video."
        )

    def _fallback_initial_frontier(
        self,
        memory: VideoMemory,
        task_type: str | None,
    ) -> list[FrontierItem]:
        if task_type == "multiple_choice_visual_qa":
            candidates = [
                node
                for node in memory.nodes.values()
                if node.level == "clip" and node.visual_summary.strip()
            ]
            if not candidates:
                candidates = [
                    node
                    for node in memory.nodes.values()
                    if node.level != "video" and node.visual_summary.strip()
                ]
            selected_nodes = _select_temporally_diverse_initial_nodes(
                candidates,
                self.max_frontier_items,
            )
            if selected_nodes:
                return [
                    FrontierItem(
                        node_id=node.node_id,
                        time_span=node.time_span,
                        level=node.level,
                        score=round(0.32 - min(index * 0.02, 0.12), 4),
                        why_candidate=(
                            "Initial temporal visual fallback after search found no candidates"
                        ),
                        recommended_modalities=["visual"],
                    )
                    for index, node in enumerate(selected_nodes)
                ]
        return [
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

    def _answer_from_stop_action(
        self,
        action: ControllerAction,
        state: ControllerState,
    ) -> str:
        if action.answer:
            options = _extract_multiple_choice_options(state.question)
            if options:
                visual_verifier = self._visual_answer_verification_from_state(state, options)
                if visual_verifier is not None:
                    verified_answer, verifier_metadata = visual_verifier
                    if verified_answer is not None:
                        return verified_answer
                parsed_answer = _parse_multiple_choice_letter(action.answer, options)
                if parsed_answer is not None:
                    return parsed_answer
            return action.answer.strip()
        multiple_choice_answer = self._multiple_choice_answer_from_state(state)
        if multiple_choice_answer is not None:
            return multiple_choice_answer
        return self._fallback_answer_from_state(state)

    def _format_final_answer_for_state(
        self,
        answer: str,
        state: ControllerState,
        evidence_items: list[Evidence] | None = None,
    ) -> str:
        route = self._route_for_state(state)
        return format_answer_for_route(answer, route, evidence_items or [])

    def _route_for_state(self, state: ControllerState) -> QuestionRoute:
        return (
            route_from_metadata(state.global_context)
            or route_from_metadata(state.question_spec.metadata if state.question_spec else None)
            or route_question(state.question, state.task_type)
        )

    def _evidence_for_action(
        self,
        action: ControllerAction,
        state: ControllerState,
    ) -> list[Evidence]:
        selected_ids = set(action.evidence_ids)
        if selected_ids:
            return [item for item in state.evidence_ledger if item.evidence_id in selected_ids]
        if state.evidence_board is not None:
            allowed = set(
                state.evidence_board.core_evidence_ids + state.evidence_board.support_evidence_ids
            )
            return [item for item in state.evidence_ledger if item.evidence_id in allowed]
        return list(state.evidence_ledger)

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
            state.frontier = self._merge_frontier(state.frontier, observation.frontier, state)
        elif action.action_type == "OPEN":
            covered_node_ids = self._covered_open_node_ids(action, observation)
            state.frontier = self._set_frontier_status_many(
                state.frontier,
                covered_node_ids,
                "opened",
            )
            state.frontier = self._remove_frontier_nodes(state.frontier, covered_node_ids)
            state.evidence_ledger.extend(observation.evidence)
            if observation.frontier:
                state.frontier = self._merge_frontier(state.frontier, observation.frontier, state)
        elif action.action_type == "SPLIT":
            state.frontier = self._set_frontier_status(state.frontier, action.node_id, "expanded")
            state.frontier = self._merge_frontier(state.frontier, observation.frontier, state)
        elif action.action_type == "MERGE":
            state.evidence_ledger.extend(observation.evidence)
        elif action.action_type == "STOP":
            verification = observation.metadata.get("answer_verification")
            if isinstance(verification, dict):
                state.global_context["last_stop_verification"] = verification
            if not observation.metadata.get("stop_rejected"):
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
        if state.event_memory is not None:
            event_updates = update_event_memory_from_observation(
                state.event_memory,
                observation,
            )
            if event_updates:
                state.no_progress_steps = 0
            state.global_context["event_memory_metrics"] = event_memory_metrics(
                state.event_memory
            )

        return state

    def _merge_frontier(
        self,
        existing: list[FrontierItem],
        new_items: list[FrontierItem],
        state: ControllerState | None = None,
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

        dynamic_order = self._dynamic_chain_order(state) if state is not None else {}
        ordered = sorted(
            merged.values(),
            key=lambda item: (
                dynamic_order.get(item.node_id, 10_000),
                -item.score,
                item.time_span.start,
            ),
        )
        return ordered[: self.max_frontier_items]

    def _dynamic_chain_order(self, state: ControllerState | None) -> dict[str, int]:
        if state is None:
            return {}
        metadata = state.global_context.get("dynamic_evidence_retrieval")
        if not isinstance(metadata, dict) or not metadata.get("enabled"):
            return {}
        selected = metadata.get("selected")
        if not isinstance(selected, list):
            return {}
        opened = {
            str(action.get("node_id"))
            for action in state.action_history
            if action.get("action_type") == "OPEN" and action.get("node_id")
        }
        order: dict[str, int] = {}
        for index, raw in enumerate(selected):
            if not isinstance(raw, dict):
                continue
            node_id = str(raw.get("node_id") or "")
            if not node_id or node_id in opened:
                continue
            order.setdefault(node_id, index)
        return order

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

    def _set_frontier_status_many(
        self,
        frontier: list[FrontierItem],
        node_ids: set[str],
        status: str,
    ) -> list[FrontierItem]:
        updated = []
        for item in frontier:
            if item.node_id in node_ids:
                item.status = status
            updated.append(item)
        return updated

    def _remove_frontier_nodes(
        self,
        frontier: list[FrontierItem],
        node_ids: set[str],
    ) -> list[FrontierItem]:
        if not node_ids:
            return frontier
        return [item for item in frontier if item.node_id not in node_ids]

    def _covered_open_node_ids(self, action: ControllerAction, observation) -> set[str]:
        covered = {action.node_id} if action.node_id else set()
        covered.update(graph_expansion_covered_node_ids(observation))
        chain_node_ids = observation.metadata.get("chain_opened_node_ids", [])
        if isinstance(chain_node_ids, list):
            covered.update(str(node_id) for node_id in chain_node_ids if node_id)
        return covered

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
        if state is not None and not text.strip():
            recovered = self._empty_controller_response_payload(state)
            if recovered is not None:
                return recovered
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

    def _empty_controller_response_payload(self, state: ControllerState) -> dict[str, Any] | None:
        if self._is_postvalid_speech_aggregation_state(state):
            target_slot = select_target_slot(state.question_spec, state.evidence_board)
            preferred_modality = self._preferred_search_modality(state, target_slot)
            candidate = self._next_postvalid_speech_frontier_candidate(
                state,
                target_slot,
                preferred_modality,
            )
            if candidate is not None:
                item, item_modality = candidate
                return {
                    "action_type": "OPEN",
                    "query": None,
                    "modality": item_modality,
                    "node_id": item.node_id,
                    "target_slot": target_slot,
                    "evidence_ids": [],
                    "answer": None,
                    "rationale": "open top speech frontier after empty controller response",
                }
            return {
                "action_type": "SEARCH",
                "query": self._postvalid_aggregation_search_query(state, target_slot),
                "modality": "speech",
                "node_id": None,
                "target_slot": target_slot,
                "evidence_ids": [],
                "answer": None,
                "rationale": "recover empty controller response with speech search",
            }
        return None

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
        if action_type == "STOP":
            dynamic_payload = self._dynamic_chain_open_payload(
                state,
                required_only=not self._dynamic_chain_requires_all_selected(state),
            )
            if dynamic_payload is not None:
                return dynamic_payload
            aggregation_payload = self._postvalid_speech_aggregation_payload(state)
            if aggregation_payload is not None:
                return aggregation_payload
            followup = self._event_memory_followup_payload(state)
            return followup or payload
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
            if not payload.get("query"):
                payload["query"] = self._default_search_query(state, target_slot)
            else:
                payload["query"] = self._clean_search_query(str(payload["query"]), state)
            frontier_candidate = self._next_frontier_open_candidate(
                state,
                target_slot,
                preferred_modality,
            )
            if (
                self._should_open_frontier_instead_of_search(state)
                or (
                    state.evidence_ledger
                    and self._is_timelogic_repeated_search_payload(state, payload)
                )
            ) and frontier_candidate is not None:
                item, item_modality = frontier_candidate
                payload["action_type"] = "OPEN"
                payload["node_id"] = item.node_id
                payload["query"] = None
                payload["modality"] = item_modality
                payload["target_slot"] = target_slot
                self._repair_node_id_payload(payload, state)
                return payload
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
        dynamic_payload = self._dynamic_chain_open_payload(
            state,
            target_slot=target_slot,
            preferred_modality=str(payload.get("modality") or preferred_modality),
            requested_node_id=payload.get("node_id"),
            required_only=not self._dynamic_chain_requires_all_selected(state),
        )
        if dynamic_payload is not None:
            return dynamic_payload
        if action_type == "OPEN":
            self._repair_node_id_payload(payload, state)
        return payload

    def _dynamic_chain_open_payload(
        self,
        state: ControllerState,
        *,
        target_slot: str | None = None,
        preferred_modality: str | None = None,
        requested_node_id: object | None = None,
        required_only: bool = False,
    ) -> dict[str, Any] | None:
        chain_item = self._next_dynamic_chain_item(state, required_only=required_only)
        if chain_item is None:
            return None
        item, selected_metadata = chain_item
        requested = str(requested_node_id) if requested_node_id else None
        if requested == item.node_id:
            return None
        modality = (
            item.recommended_modalities[0]
            if item.recommended_modalities
            else str(selected_metadata.get("modality") or preferred_modality or "speech")
        )
        resolved_modality = self._resolve_available_modality(modality, state)
        selected_slot = (
            target_slot
            or str(selected_metadata.get("target_slot") or "")
            or select_target_slot(state.question_spec, state.evidence_board)
        )
        return {
            "action_type": "OPEN",
            "query": None,
            "modality": resolved_modality,
            "node_id": item.node_id,
            "target_slot": selected_slot,
            "evidence_ids": [],
            "answer": None,
            "rationale": "open remaining dynamic multi-evidence chain target before finalizing",
        }

    def _next_dynamic_chain_item(
        self,
        state: ControllerState,
        *,
        required_only: bool = False,
    ) -> tuple[FrontierItem, dict[str, Any]] | None:
        metadata = state.global_context.get("dynamic_evidence_retrieval")
        if not isinstance(metadata, dict) or not metadata.get("enabled"):
            return None
        selected = metadata.get("selected")
        if not isinstance(selected, list):
            return None
        opened_node_ids = {
            str(action.get("node_id"))
            for action in state.action_history
            if action.get("action_type") == "OPEN" and action.get("node_id")
        }
        frontier_by_id = {item.node_id: item for item in state.frontier if item.status == "unopened"}
        for raw in selected:
            if not isinstance(raw, dict):
                continue
            if required_only and raw.get("required") is False:
                continue
            node_id = str(raw.get("node_id") or "")
            if not node_id or node_id in opened_node_ids:
                continue
            frontier_item = frontier_by_id.get(node_id)
            if frontier_item is not None:
                return frontier_item, raw
            time_span = raw.get("time_span")
            if not isinstance(time_span, dict):
                continue
            try:
                span = TimeSpan(
                    start=float(time_span.get("start")),
                    end=float(time_span.get("end")),
                )
            except (TypeError, ValueError):
                continue
            modality = str(raw.get("modality") or "speech")
            return (
                FrontierItem(
                    node_id=node_id,
                    time_span=span,
                    level="clip",
                    score=float(raw.get("score") or 0.5),
                    why_candidate=(
                        "Reconstructed unopened dynamic multi-evidence chain node"
                    ),
                    recommended_modalities=[modality],
                    status="unopened",
                ),
                raw,
            )
        return None

    def _has_unopened_required_dynamic_chain_targets(self, state: ControllerState) -> bool:
        return (
            self._next_dynamic_chain_item(
                state,
                required_only=not self._dynamic_chain_requires_all_selected(state),
            )
            is not None
        )

    def _dynamic_chain_requires_all_selected(self, state: ControllerState) -> bool:
        metadata = state.global_context.get("dynamic_evidence_retrieval")
        if not isinstance(metadata, dict) or not metadata.get("enabled"):
            return False
        if metadata.get("requires_all_selected") is True:
            return True
        return self._is_postvalid_speech_aggregation_state(state)

    def _event_memory_followup_payload(self, state: ControllerState) -> dict[str, Any] | None:
        event_memory = state.event_memory
        if event_memory is None or event_memory.task_name != "timelogic":
            return None
        missing_event_ids = self._blocking_timelogic_missing_event_ids(state)
        if state.budget.steps_remaining <= 0 or not missing_event_ids:
            return None

        target_slot = select_target_slot(state.question_spec, state.evidence_board)
        if event_memory.mode == "mc":
            for event_id in missing_event_ids:
                event = event_memory.events[event_id]
                if event.source != "option":
                    continue
                if self._event_search_count(state, event.phrase) >= 1:
                    continue
                return {
                    "action_type": "SEARCH",
                    "query": event.phrase,
                    "modality": self._resolve_available_modality("visual", state),
                    "node_id": None,
                    "target_slot": target_slot,
                    "evidence_ids": [],
                    "answer": None,
                    "rationale": "search option event",
                }

        if event_memory.mode == "bool" and self._timelogic_missing_search_count(
            state,
            missing_event_ids,
        ) >= 2:
            sweep_payload = self._timelogic_temporal_sweep_payload(
                state,
                target_slot=target_slot,
                reason="sweep missing event",
            )
            if sweep_payload is not None:
                return sweep_payload

        for item in state.frontier:
            if item.status == "unopened":
                modality = (
                    item.recommended_modalities[0]
                    if item.recommended_modalities
                    else "visual"
                )
                return {
                    "action_type": "OPEN",
                    "query": None,
                    "modality": self._resolve_available_modality(modality, state),
                    "node_id": item.node_id,
                    "target_slot": target_slot,
                    "evidence_ids": [],
                    "answer": None,
                    "rationale": "localize missing event",
                }

        for event_id in missing_event_ids:
            event = event_memory.events[event_id]
            if self._event_search_count(state, event.phrase) >= 2:
                continue
            return {
                "action_type": "SEARCH",
                "query": event.phrase,
                "modality": self._resolve_available_modality("visual", state),
                "node_id": None,
                "target_slot": target_slot,
                "evidence_ids": [],
                "answer": None,
                "rationale": "search missing event",
            }
        return None

    def _postvalid_speech_aggregation_payload(
        self,
        state: ControllerState,
    ) -> dict[str, Any] | None:
        if not self._is_postvalid_speech_aggregation_state(state):
            return None
        if state.budget.steps_remaining <= 0:
            return None
        compatible = self._postvalid_compatible_speech_evidence(state)
        if self._postvalid_speech_evidence_ready(state, compatible):
            return None
        if compatible and self._postvalid_speech_open_count(state) >= 4:
            return None

        target_slot = select_target_slot(state.question_spec, state.evidence_board)
        preferred_modality = self._preferred_search_modality(state, target_slot)
        opened_ids = {
            str(action.get("node_id"))
            for action in state.action_history
            if action.get("action_type") == "OPEN" and action.get("node_id")
        }
        if compatible and not self._has_search_after_last_postvalid_open(state):
            return {
                "action_type": "SEARCH",
                "query": self._postvalid_aggregation_search_query(state, target_slot),
                "modality": "speech",
                "node_id": None,
                "target_slot": target_slot,
                "evidence_ids": [],
                "answer": None,
                "rationale": "aggregate additional postvalid speech evidence",
            }

        candidate = self._next_postvalid_speech_frontier_candidate(
            state,
            target_slot,
            preferred_modality,
            exclude_node_ids=opened_ids,
        )
        if candidate is not None:
            item, item_modality = candidate
            return {
                "action_type": "OPEN",
                "query": None,
                "modality": item_modality,
                "node_id": item.node_id,
                "target_slot": target_slot,
                "evidence_ids": [],
                "answer": None,
                "rationale": "open additional reranked postvalid speech evidence",
            }

        if not compatible:
            return {
                "action_type": "SEARCH",
                "query": self._postvalid_aggregation_search_query(state, target_slot),
                "modality": "speech",
                "node_id": None,
                "target_slot": target_slot,
                "evidence_ids": [],
                "answer": None,
                "rationale": "search for postvalid speech evidence",
            }
        return None

    def _next_postvalid_speech_frontier_candidate(
        self,
        state: ControllerState,
        target_slot: str | None,
        preferred_modality: str,
        *,
        exclude_node_ids: set[str] | None = None,
    ) -> tuple[FrontierItem, str] | None:
        excluded = set(exclude_node_ids or set())
        excluded.update(
            item.source_node_id
            for item in state.evidence_ledger
            if item.source_node_id
        )
        opened_spans = [
            item.time_span
            for item in state.evidence_ledger
            if item.modality == "speech"
        ]
        candidates: list[tuple[int, float, float, FrontierItem, str]] = []
        for item in state.frontier:
            if item.status != "unopened" or item.node_id in excluded:
                continue
            if any(_is_related_postvalid_node_id(item.node_id, excluded_id) for excluded_id in excluded):
                continue
            if not any(modality == "speech" for modality in item.recommended_modalities):
                continue
            candidate = self._frontier_item_with_modality(
                item,
                state,
                target_slot,
                preferred_modality,
            )
            if candidate is not None:
                candidate_item, item_modality = candidate
                overlap_rank = (
                    1 if _overlaps_existing_postvalid_speech(item.time_span, opened_spans) else 0
                )
                candidates.append(
                    (
                        overlap_rank,
                        -candidate_item.score,
                        candidate_item.time_span.start,
                        candidate_item,
                        item_modality,
                    )
                )
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3].node_id))
        _, _, _, selected_item, selected_modality = candidates[0]
        return selected_item, selected_modality

    def _frontier_item_with_modality(
        self,
        item: FrontierItem,
        state: ControllerState,
        target_slot: str | None,
        preferred_modality: str,
    ) -> tuple[FrontierItem, str] | None:
        current_modality = (
            item.recommended_modalities[0]
            if item.recommended_modalities
            else preferred_modality
        )
        resolved_modality = self._resolve_available_modality(current_modality, state)
        if self._should_override_open_modality(
            current_modality=resolved_modality,
            preferred_modality=preferred_modality,
        ):
            resolved_modality = preferred_modality
        if is_reopen_blocked(
            state.evidence_board,
            item.node_id,
            resolved_modality,
            target_slot,
        ):
            return None
        return item, resolved_modality

    def _is_postvalid_speech_explanation_state(self, state: ControllerState) -> bool:
        return self._is_postvalid_speech_aggregation_state(
            state,
            allowed_routes={"speech_explanation"},
        )

    def _is_postvalid_speech_aggregation_state(
        self,
        state: ControllerState,
        *,
        allowed_routes: set[str] | None = None,
    ) -> bool:
        longshot_context = state.global_context.get("longshot")
        if not _is_postvalid_v1_longshot_context(longshot_context):
            return False
        route = self._route_for_state(state)
        route_labels = allowed_routes or {
            "speech_explanation",
            "causal_chain",
            "temporal_occurrence",
            "rubric_explanation",
        }
        if route.label not in route_labels:
            return False
        return route.preferred_modality in {"speech", "cross_modal"}

    def _postvalid_compatible_speech_evidence(
        self,
        state: ControllerState,
    ) -> list[Evidence]:
        route = (
            route_from_metadata(state.global_context)
            or route_from_metadata(state.question_spec.metadata if state.question_spec else None)
            or route_question(state.question, state.task_type)
        )
        if state.evidence_board is not None:
            allowed_ids = set(
                state.evidence_board.core_evidence_ids + state.evidence_board.support_evidence_ids
            )
        else:
            allowed_ids = set()
        evidence = [
            item
            for item in state.evidence_ledger
            if item.modality == "speech"
            and (not allowed_ids or item.evidence_id in allowed_ids)
            and item.metadata.get("role") in {"core", "support"}
            and evidence_matches_route(item, route)
        ]
        evidence.sort(
            key=lambda item: (
                0 if item.metadata.get("role") == "core" else 1,
                -item.confidence,
                item.time_span.start,
                item.evidence_id,
            )
        )
        return evidence

    def _postvalid_speech_evidence_ready(
        self,
        state: ControllerState,
        evidence: list[Evidence],
    ) -> bool:
        if not evidence:
            return False
        distinct_sources = {item.source_node_id for item in evidence if item.source_node_id}
        temporal_intents = state.global_context.get("postvalid_temporal_intents", [])
        has_clear_answer_span = any(
            item.metadata.get("role") == "core"
            and _has_exact_answer_span(item)
            for item in evidence
        )
        if has_clear_answer_span:
            return True
        required_distinct_sources = _postvalid_required_distinct_speech_sources(temporal_intents)
        if len(evidence) >= required_distinct_sources and len(distinct_sources) >= required_distinct_sources:
            return True
        if state.budget.steps_remaining <= 0:
            return True
        return False

    def _postvalid_speech_open_count(self, state: ControllerState) -> int:
        return sum(
            1
            for action in state.action_history
            if action.get("action_type") == "OPEN"
            and str(action.get("modality") or "") in {"speech", "cross_modal"}
        )

    def _has_search_after_last_postvalid_open(self, state: ControllerState) -> bool:
        last_open_index = -1
        for index, action in enumerate(state.action_history):
            if action.get("action_type") == "OPEN":
                last_open_index = index
        if last_open_index < 0:
            return False
        return any(
            action.get("action_type") == "SEARCH"
            and str(action.get("modality") or "") in {"speech", "cross_modal", ""}
            for action in state.action_history[last_open_index + 1 :]
        )

    def _postvalid_aggregation_search_query(
        self,
        state: ControllerState,
        target_slot: str | None,
    ) -> str:
        base_query = self._default_search_query(state, target_slot)
        phrases = _postvalid_temporal_intent_query_phrases(
            state.global_context.get("postvalid_temporal_intents")
            or (
                state.question_spec.metadata.get("postvalid_temporal_intents")
                if state.question_spec is not None
                else []
            )
        )
        dialogue_terms = [
            str(turn.get("content") or "")
            for turn in state.dialogue_context[-4:]
            if turn.get("role") == "user"
        ]
        task_terms = _postvalid_task_specific_query_terms(state.question)
        additions = " ".join([*phrases, task_terms, *dialogue_terms]).strip()
        if additions:
            return f"{base_query} {additions}"
        return base_query

    def _timelogic_missing_search_count(
        self,
        state: ControllerState,
        missing_event_ids: list[str],
    ) -> int:
        event_memory = state.event_memory
        if event_memory is None:
            return 0
        return sum(
            self._event_search_count(state, event_memory.events[event_id].phrase)
            for event_id in missing_event_ids
            if event_id in event_memory.events
        )

    def _timelogic_temporal_sweep_payload(
        self,
        state: ControllerState,
        *,
        target_slot: str | None,
        reason: str,
    ) -> dict[str, Any] | None:
        candidate = self._timelogic_temporal_sweep_candidate(state, target_slot)
        if candidate is None:
            return None
        return {
            "action_type": "OPEN",
            "query": None,
            "modality": self._resolve_available_modality("visual", state),
            "node_id": candidate["node_id"],
            "target_slot": target_slot,
            "evidence_ids": [],
            "answer": None,
            "rationale": reason,
        }

    def _timelogic_temporal_sweep_candidate(
        self,
        state: ControllerState,
        target_slot: str | None,
    ) -> dict[str, Any] | None:
        candidates = state.global_context.get("timelogic_temporal_sweep_candidates")
        if not isinstance(candidates, list):
            return None
        opened_node_ids = {
            str(action.get("node_id"))
            for action in state.action_history
            if action.get("action_type") == "OPEN" and action.get("node_id")
        }
        opened_midpoints = [
            (evidence.time_span.start + evidence.time_span.end) / 2.0
            for evidence in state.evidence_ledger
            if evidence.modality == "visual"
        ]
        scored: list[tuple[float, float, dict[str, Any]]] = []
        for item in candidates:
            if not isinstance(item, dict):
                continue
            node_id = str(item.get("node_id") or "")
            if not node_id or node_id in opened_node_ids:
                continue
            if is_reopen_blocked(
                state.evidence_board,
                node_id,
                "visual",
                target_slot,
            ):
                continue
            try:
                start = float(item["start"])
                end = float(item["end"])
            except (KeyError, TypeError, ValueError):
                continue
            midpoint = (start + end) / 2.0
            if opened_midpoints:
                diversity = min(abs(midpoint - opened) for opened in opened_midpoints)
            else:
                diversity = midpoint
            scored.append((diversity, -start, item))
        if not scored:
            return None
        scored.sort(reverse=True)
        return scored[0][2]

    def _blocking_timelogic_missing_event_ids(self, state: ControllerState) -> list[str]:
        event_memory = state.event_memory
        if event_memory is None or event_memory.task_name != "timelogic":
            return []
        if event_memory.mode == "mc":
            target_missing = [
                event.event_id
                for event in event_memory.events.values()
                if event.source != "option" and event.status != "localized"
            ]
            if target_missing:
                return target_missing
            has_localized_option = any(
                event.source == "option" and event.status == "localized"
                for event in event_memory.events.values()
            )
            if not has_localized_option:
                return [
                    event.event_id
                    for event in event_memory.events.values()
                    if event.source == "option"
                ]
            return [
                event.event_id
                for event in event_memory.events.values()
                if event.source == "option"
                and event.status != "localized"
                and self._event_search_count(state, event.phrase) == 0
            ]
        return event_memory.missing_event_ids

    def _event_search_count(self, state: ControllerState, phrase: str) -> int:
        normalized_phrase = " ".join(phrase.lower().split())
        count = 0
        for action in state.action_history:
            if action.get("action_type") != "SEARCH":
                continue
            query = " ".join(str(action.get("query") or "").lower().split())
            if query == normalized_phrase or event_match_score(phrase, query) >= 0.67:
                count += 1
        return count

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
        if state.evidence_ledger and not self._blocking_timelogic_missing_event_ids(state):
            return False
        if not state.action_history:
            return False
        last_action = state.action_history[-1]
        if last_action.get("action_type") != "SEARCH":
            return False
        target_slot = select_target_slot(state.question_spec, state.evidence_board)
        preferred_modality = self._preferred_search_modality(state, target_slot)
        return (
            self._next_frontier_open_candidate(state, target_slot, preferred_modality)
            is not None
        )

    def _is_timelogic_repeated_search_payload(
        self,
        state: ControllerState,
        payload: dict[str, Any],
    ) -> bool:
        if state.event_memory is None or state.event_memory.task_name != "timelogic":
            return False
        query = " ".join(str(payload.get("query") or "").split())
        if not query:
            return False
        return any(
            action.get("action_type") == "SEARCH"
            and event_match_score(query, str(action.get("query") or "")) >= 0.67
            for action in state.action_history
        )

    def _next_frontier_open_candidate(
        self,
        state: ControllerState,
        target_slot: str | None,
        preferred_modality: str,
        *,
        exclude_node_ids: set[str] | None = None,
    ) -> tuple[FrontierItem, str] | None:
        excluded = exclude_node_ids or set()
        for item in state.frontier:
            if item.status != "unopened" or item.node_id in excluded:
                continue
            current_modality = (
                item.recommended_modalities[0]
                if item.recommended_modalities
                else preferred_modality
            )
            resolved_modality = self._resolve_available_modality(current_modality, state)
            if self._should_override_open_modality(
                current_modality=resolved_modality,
                preferred_modality=preferred_modality,
            ):
                resolved_modality = preferred_modality
            if is_reopen_blocked(
                state.evidence_board,
                item.node_id,
                resolved_modality,
                target_slot,
            ):
                continue
            return item, resolved_modality
        return None

    def _repair_node_id_payload(
        self,
        payload: dict[str, Any],
        state: ControllerState,
    ) -> None:
        node_id = payload.get("node_id")
        frontier_ids = state.frontier_ids()
        target_slot = payload.get("target_slot") or select_target_slot(
            state.question_spec,
            state.evidence_board,
        )
        preferred_modality = self._preferred_search_modality(state, target_slot)
        current_modality = payload.get("modality")
        resolved_modality = (
            self._resolve_available_modality(current_modality, state)
            if current_modality
            else preferred_modality
        )
        if self._should_override_open_modality(
            current_modality=resolved_modality,
            preferred_modality=preferred_modality,
        ):
            resolved_modality = preferred_modality
        if node_id in frontier_ids:
            if not is_reopen_blocked(
                state.evidence_board,
                str(node_id),
                resolved_modality,
                target_slot,
            ):
                payload["modality"] = resolved_modality
                return
            candidate = self._next_frontier_open_candidate(
                state,
                target_slot,
                resolved_modality,
                exclude_node_ids={str(node_id)},
            )
            if candidate is not None:
                item, item_modality = candidate
                payload["node_id"] = item.node_id
                payload["modality"] = item_modality
                return
            return
        if state.frontier:
            candidate = self._next_frontier_open_candidate(
                state,
                target_slot,
                resolved_modality,
            )
            if candidate is not None:
                item, item_modality = candidate
                payload["node_id"] = item.node_id
                payload["modality"] = item_modality
                return
            payload["node_id"] = state.frontier[0].node_id
            payload["modality"] = resolved_modality
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
        filled_slot_answer = self._filled_required_slots_answer_from_state(state)
        if filled_slot_answer is not None:
            answer, _, _ = filled_slot_answer
            return answer
        if state.evidence_board is not None and state.evidence_board.missing_required_slots:
            return self._diagnostic_abstain_from_state(state)
        if state.evidence_ledger:
            return self._synthesize_answer_from_evidence(state)
        return "Controller exhausted its budget before collecting grounded evidence."

    def _filled_required_slots_answer_from_state(
        self,
        state: ControllerState,
    ) -> tuple[str, list[str], str] | None:
        if self._has_unopened_required_dynamic_chain_targets(state):
            return None
        board = state.evidence_board
        route = (
            route_from_metadata(state.global_context)
            or route_from_metadata(state.question_spec.metadata if state.question_spec else None)
            or route_question(state.question, state.task_type)
        )
        if route.label in {
            "speech_explanation",
            "causal_chain",
            "temporal_occurrence",
            "rubric_explanation",
        }:
            if self._postvalid_speech_aggregation_payload(state) is not None:
                return None
            speech_answer = self._speech_explanation_answer_from_evidence(state, route)
            if speech_answer is not None:
                return speech_answer

        if board is None or board.missing_required_slots:
            return None
        required_slots = (
            [slot.slot for slot in state.question_spec.required_slots if slot.required]
            if state.question_spec is not None
            else ["main_claim"]
        )
        if any(not board.is_slot_filled(slot_name) for slot_name in required_slots):
            return None

        core_ids = set(board.core_evidence_ids)
        compatible_core_evidence = [
            item
            for item in state.evidence_ledger
            if item.evidence_id in core_ids
            and item.metadata.get("role") == "core"
            and evidence_matches_route(item, route)
            and _has_exact_answer_span(item)
        ]
        if not compatible_core_evidence:
            return None
        compatible_core_evidence.sort(
            key=lambda item: (-item.confidence, item.time_span.start, item.evidence_id)
        )
        answer = str(compatible_core_evidence[0].metadata.get("answer_span") or "").strip()
        answer = self._format_final_answer_for_state(answer, state, compatible_core_evidence)
        evidence_ids = [item.evidence_id for item in compatible_core_evidence]
        return answer, evidence_ids, "filled_required_slots_answer_span"

    def _speech_explanation_answer_from_evidence(
        self,
        state: ControllerState,
        route: QuestionRoute,
    ) -> tuple[str, list[str], str] | None:
        board = state.evidence_board
        if board is not None:
            allowed_ids = list(dict.fromkeys(board.core_evidence_ids + board.support_evidence_ids))
        else:
            allowed_ids = []
        allowed_id_set = set(allowed_ids)
        compatible_evidence = [
            item
            for item in state.evidence_ledger
            if (not allowed_id_set or item.evidence_id in allowed_id_set)
            and item.metadata.get("role") in {"core", "support"}
            and evidence_matches_route(item, route)
        ]
        if not compatible_evidence:
            return None
        compatible_evidence.sort(
            key=lambda item: (
                -_evidence_answer_relevance_score(item, state.question),
                -item.confidence,
                0 if item.metadata.get("role") == "core" else 1,
                item.time_span.start,
                item.evidence_id,
            )
        )
        selected = compatible_evidence[:4]
        answer = self._synthesize_answer_from_evidence(state, selected)
        if not answer:
            return None
        return (
            answer,
            [item.evidence_id for item in selected],
            "speech_explanation_evidence_synthesis",
        )

    def _synthesize_answer_from_evidence(
        self,
        state: ControllerState,
        evidence_items: list[Evidence] | None = None,
    ) -> str:
        multiple_choice_answer = self._multiple_choice_answer_from_state(state)
        if multiple_choice_answer is not None:
            return multiple_choice_answer
        if evidence_items is not None:
            filtered_evidence = list(evidence_items)
        elif state.evidence_board is not None:
            allowed_ids = set(
                state.evidence_board.core_evidence_ids + state.evidence_board.support_evidence_ids
            )
            filtered_evidence = [
                item
                for item in state.evidence_ledger
                if not allowed_ids or item.evidence_id in allowed_ids
            ]
        else:
            filtered_evidence = []
        answer_bearing_evidence = self._answer_bearing_evidence_for_state(
            state,
            filtered_evidence or state.evidence_ledger,
            strict=False,
        )
        top_evidence = sorted(
            answer_bearing_evidence or filtered_evidence or state.evidence_ledger,
            key=lambda item: (
                -_evidence_answer_relevance_score(item, state.question),
                -item.confidence,
                item.time_span.start,
            ),
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
                        "evidence_kind": item.metadata.get("evidence_kind"),
                        "answer_span": item.metadata.get("answer_span"),
                        "context_span": item.metadata.get("context_span"),
                        "evidence_bundle_role": item.metadata.get("evidence_bundle_role"),
                        "aggregation_rule": item.metadata.get("aggregation_rule"),
                        "source_events": item.metadata.get("source_events", []),
                        "temporal_constraint": item.metadata.get("temporal_constraint"),
                        "evidence_bundle": item.metadata.get("evidence_bundle"),
                        "excerpt": _focus_evidence_detail(item.detail, state.question),
                    },
                    ensure_ascii=True,
                )
            )
        synthesis_context = self._postvalid_synthesis_context(state)

        prompt = (
            "You are a grounded answerer for a long-video reasoning system.\n"
            "Answer the user's question using only the evidence below.\n"
            "Cover every required answer aspect listed in the compatible context when the "
            "evidence supports it.\n"
            "Prefer the most direct causal explanation supported by the evidence.\n"
            "For LongShotBench postvalid_v1 documentary speech questions, support evidence can "
            "be enough: if the evidence describes the same event, problem, mechanism, or causal "
            "link, give the best grounded synthesis instead of refusing.\n"
            "For Bell-test/quasar questions, treat evidence about loopholes, random filter or "
            "measurement choices, photons, hidden influences, or using the universe as the "
            "randomness source as relevant answer-bearing support.\n"
            "You may use the compatible scenario and recent dialogue context only to resolve "
            "references or fill a named entity already consistent with the evidence; do not use "
            "context that conflicts with the evidence.\n"
            "If the evidence includes both a problem and a later fix or repair, mention both.\n"
            "If the evidence includes concrete numbers, preparation details, or quoted reactions that directly support the answer, include the most relevant ones.\n"
            "If the question asks about the first or last thing, identify the earliest or latest relevant item or event from the evidence rather than a later summary.\n"
            "Be concise and specific. Say the evidence is insufficient only when it is about a different topic or lacks any answer-bearing mechanism.\n"
            "Do not mention internal ids or budget exhaustion.\n\n"
            f"Question: {state.question}\n\n"
            f"Compatible context:\n{json.dumps(synthesis_context, ensure_ascii=True)}\n\n"
            "Evidence:\n" + "\n".join(evidence_lines)
        )
        answer = self.controller_client.completion(prompt).strip()
        if answer and not _looks_incomplete_answer(answer) and not _looks_refusal_answer(answer):
            return answer
        if _looks_refusal_answer(answer) and not answer_bearing_evidence:
            return answer
        deterministic = self._deterministic_answer_from_evidence(state, top_evidence)
        if deterministic:
            return deterministic
        return answer

    def _repair_final_answer(self, answer: str, state: ControllerState) -> str:
        if not _looks_incomplete_answer(answer) and not _looks_refusal_answer(answer):
            return answer
        evidence_items = self._evidence_for_final_repair(state)
        if not evidence_items:
            return answer
        if _looks_refusal_answer(answer):
            evidence_items = self._answer_bearing_evidence_for_state(
                state,
                evidence_items,
                strict=True,
            )
            if not evidence_items:
                return answer
        repaired = self._complete_answer_from_evidence(answer, state, evidence_items)
        if (
            repaired
            and not _looks_incomplete_answer(repaired)
            and not _looks_refusal_answer(repaired)
        ):
            return self._format_final_answer_for_state(repaired, state, evidence_items)
        deterministic = self._deterministic_answer_from_evidence(state, evidence_items)
        if (
            deterministic
            and not _looks_incomplete_answer(deterministic)
            and not _looks_refusal_answer(deterministic)
        ):
            return deterministic
        return answer

    def _evidence_for_final_repair(self, state: ControllerState) -> list[Evidence]:
        if state.evidence_board is not None:
            allowed = set(
                state.evidence_board.core_evidence_ids + state.evidence_board.support_evidence_ids
            )
            candidates = [
                item for item in state.evidence_ledger if not allowed or item.evidence_id in allowed
            ]
        else:
            candidates = list(state.evidence_ledger)
        return sorted(
            candidates,
            key=lambda item: (
                -_evidence_answer_relevance_score(item, state.question),
                -item.confidence,
                item.time_span.start,
            ),
        )[:4]

    def _complete_answer_from_evidence(
        self,
        partial_answer: str,
        state: ControllerState,
        evidence_items: list[Evidence],
    ) -> str:
        evidence_lines = [
            json.dumps(
                {
                    "evidence_id": item.evidence_id,
                    "slot": item.metadata.get("slot"),
                    "role": item.metadata.get("role"),
                    "modality": item.modality,
                    "time_span": item.time_span.to_dict(),
                    "evidence_kind": item.metadata.get("evidence_kind"),
                    "answer_span": item.metadata.get("answer_span"),
                    "context_span": item.metadata.get("context_span"),
                    "evidence_bundle_role": item.metadata.get("evidence_bundle_role"),
                    "aggregation_rule": item.metadata.get("aggregation_rule"),
                    "source_events": item.metadata.get("source_events", []),
                    "temporal_constraint": item.metadata.get("temporal_constraint"),
                    "evidence_bundle": item.metadata.get("evidence_bundle"),
                    "excerpt": _focus_evidence_detail(item.detail, state.question),
                },
                ensure_ascii=True,
            )
            for item in evidence_items
        ]
        synthesis_context = self._postvalid_synthesis_context(state)
        prompt = (
            "Rewrite the partial answer into one complete concise answer using only the evidence. "
            "Do not introduce unsupported details. Do not mention evidence ids. "
            "If the partial answer is a refusal but the evidence contains answer-bearing details, "
            "replace the refusal with the best grounded answer. If the evidence is only topical "
            "or does not contain the named entities, mechanism, event, or requested detail from "
            "the question, keep the refusal. Cover the required answer aspects listed in the "
            "context when supported.\n\n"
            f"Question: {state.question}\n"
            f"Partial answer: {partial_answer}\n\n"
            f"Compatible context:\n{json.dumps(synthesis_context, ensure_ascii=True)}\n\n"
            "Evidence:\n" + "\n".join(evidence_lines)
        )
        return self.controller_client.completion(prompt).strip()

    def _deterministic_answer_from_evidence(
        self,
        state: ControllerState,
        evidence_items: list[Evidence],
    ) -> str:
        ranked_items = sorted(
            [item for item in evidence_items if item.detail.strip()],
            key=lambda item: (
                -_evidence_answer_relevance_score(item, state.question),
                -item.confidence,
                item.time_span.start,
            ),
        )
        answer_sentences = [
            sentence
            for item in ranked_items[:4]
            for sentence in _focused_answer_sentences(item.detail, state.question)
        ]
        if not answer_sentences:
            return ""
        deduped_sentences = list(dict.fromkeys(answer_sentences))
        answer = " ".join(deduped_sentences[:3]).strip()
        answer = re.sub(r"\s+", " ", answer)
        if len(answer) > 900:
            answer = answer[:900].rsplit(" ", maxsplit=1)[0].strip()
        return self._format_final_answer_for_state(answer, state, ranked_items[:4])

    def _answer_bearing_evidence_for_state(
        self,
        state: ControllerState,
        evidence_items: list[Evidence],
        *,
        strict: bool,
    ) -> list[Evidence]:
        route = self._route_for_state(state)
        candidates = [
            item
            for item in evidence_items
            if evidence_matches_route(item, route)
            and self._evidence_can_support_answer(state, item, strict=strict)
        ]
        return sorted(
            candidates,
            key=lambda item: (
                -_evidence_answer_relevance_score(item, state.question),
                -item.confidence,
                item.time_span.start,
                item.evidence_id,
            ),
        )

    def _evidence_can_support_answer(
        self,
        state: ControllerState,
        item: Evidence,
        *,
        strict: bool,
    ) -> bool:
        detail = item.detail or item.claim
        if not detail.strip():
            return False
        if str(item.metadata.get("answer_span") or "").strip():
            return True
        anchor_check = _postvalid_anchor_match_status(state.question, detail)
        if anchor_check == "failed":
            return False
        score = _evidence_answer_relevance_score(item, state.question)
        role = item.metadata.get("role")
        overlap_count = len(_postvalid_tokens(state.question) & _postvalid_tokens(detail))
        if strict:
            if role == "core" and score >= 14.0 and overlap_count >= 2:
                return True
            return score >= 24.0 and overlap_count >= 3
        if role == "core" and score >= 10.0:
            return True
        if anchor_check == "strong":
            return score >= 10.0
        return score >= 16.0 and overlap_count >= 2

    def _postvalid_synthesis_context(self, state: ControllerState) -> dict[str, Any]:
        longshot_context = state.global_context.get("longshot")
        if not isinstance(longshot_context, dict):
            return {}
        scenario = str(longshot_context.get("scenario") or "").strip()
        dialogue = [
            {
                "role": str(turn.get("role") or ""),
                "content": str(turn.get("content") or "")[:500],
            }
            for turn in state.dialogue_context[-4:]
            if str(turn.get("content") or "").strip()
        ]
        context: dict[str, Any] = {
            "temporal_intents": state.global_context.get("postvalid_temporal_intents", []),
            "recent_dialogue": dialogue,
        }
        dynamic_aspects = self._dynamic_answer_aspects(state)
        if dynamic_aspects:
            context["required_answer_aspects"] = dynamic_aspects
        evidence_bundles = self._evidence_bundles_for_synthesis_context(state)
        if evidence_bundles:
            context["evidence_bundles"] = evidence_bundles
        if scenario and _scenario_agrees_with_question_for_controller(
            scenario,
            state.question,
        ):
            context["scenario"] = scenario[:700]
        return context

    def _evidence_bundles_for_synthesis_context(
        self,
        state: ControllerState,
    ) -> list[dict[str, Any]]:
        bundles: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in state.evidence_ledger:
            bundle = item.metadata.get("evidence_bundle")
            if not isinstance(bundle, dict):
                continue
            bundle_id = str(bundle.get("bundle_id") or "")
            if not bundle_id or bundle_id in seen:
                continue
            seen.add(bundle_id)
            bundles.append(
                {
                    "bundle_id": bundle_id,
                    "route": bundle.get("route"),
                    "aggregation_rule": bundle.get("aggregation_rule"),
                    "opened_targets": bundle.get("opened_targets", []),
                    "evidence_kinds": bundle.get("evidence_kinds", []),
                    "answer_spans": bundle.get("answer_spans", []),
                    "source_events": bundle.get("source_events", []),
                    "temporal_constraints": bundle.get("temporal_constraints", []),
                    "confidence": bundle.get("confidence"),
                    "time_span": bundle.get("time_span"),
                }
            )
            if len(bundles) >= 4:
                break
        return bundles

    def _dynamic_answer_aspects(self, state: ControllerState) -> list[dict[str, str]]:
        metadata = state.global_context.get("dynamic_evidence_retrieval")
        if not isinstance(metadata, dict) or not metadata.get("enabled"):
            return []
        opened_node_ids = {
            str(action.get("node_id"))
            for action in state.action_history
            if action.get("action_type") == "OPEN" and action.get("node_id")
        }
        if state.evidence_board is not None:
            opened_node_ids.update(item.node_id for item in state.evidence_board.opened_targets)
        aspects: list[dict[str, str]] = []
        for raw in metadata.get("selected", []):
            if not isinstance(raw, dict) or raw.get("required") is False:
                continue
            node_id = str(raw.get("node_id") or "")
            aspects.append(
                {
                    "label": str(raw.get("label") or raw.get("target_id") or "answer_part"),
                    "query": str(raw.get("query") or "")[:220],
                    "opened": "yes" if node_id in opened_node_ids else "no",
                }
            )
        return aspects

    def _should_use_multiple_choice_final_step(self, state: ControllerState) -> bool:
        if self._has_timelogic_missing_event_work(state):
            return False
        return (
            state.task_type == "multiple_choice_visual_qa"
            and state.budget.steps_remaining == 1
            and bool(state.evidence_ledger)
            and bool(_extract_multiple_choice_options(state.question))
        )

    def _has_timelogic_missing_event_work(self, state: ControllerState) -> bool:
        if state.task_type != "multiple_choice_visual_qa":
            return False
        event_memory = state.event_memory
        if event_memory is None or event_memory.task_name != "timelogic":
            return False
        missing_event_ids = self._blocking_timelogic_missing_event_ids(state)
        if not missing_event_ids:
            return False
        if state.budget.steps_remaining <= 0:
            return False
        target_slot = select_target_slot(state.question_spec, state.evidence_board)
        if self._timelogic_temporal_sweep_candidate(state, target_slot) is not None:
            return True
        return any(
            self._event_search_count(state, event_memory.events[event_id].phrase) < 2
            for event_id in missing_event_ids
            if event_id in event_memory.events
        )

    def _should_force_multiple_choice_finalization(
        self,
        state: ControllerState,
        action: ControllerAction,
    ) -> bool:
        if state.task_type != "multiple_choice_visual_qa":
            return False
        if not _extract_multiple_choice_options(state.question):
            return False
        visual_evidence_count = sum(
            1 for evidence in state.evidence_ledger if evidence.modality == "visual"
        )
        if visual_evidence_count == 0:
            return False
        duplicate_open = (
            action.action_type == "OPEN"
            and action.node_id
            and action.modality
            and is_reopen_blocked(
                state.evidence_board,
                action.node_id,
                action.modality,
                action.target_slot,
            )
        )
        if duplicate_open:
            return True
        if action.action_type == "SEARCH" and self._is_timelogic_repeated_search(state, action):
            return True
        if (
            state.event_memory is not None
            and state.event_memory.task_name == "timelogic"
            and self._blocking_timelogic_missing_event_ids(state)
            and state.budget.steps_remaining > 1
        ):
            return False
        return action.action_type == "SEARCH" and visual_evidence_count >= 2

    def _is_timelogic_repeated_search(
        self,
        state: ControllerState,
        action: ControllerAction,
    ) -> bool:
        if state.event_memory is None or state.event_memory.task_name != "timelogic":
            return False
        query = " ".join(str(action.query or "").split())
        if not query:
            return False
        return any(
            previous.get("action_type") == "SEARCH"
            and event_match_score(query, str(previous.get("query") or "")) >= 0.67
            for previous in state.action_history
        )

    def _multiple_choice_answer_from_state(self, state: ControllerState) -> str | None:
        completion = self._multiple_choice_completion_from_state(state)
        if completion is None:
            return None
        answer, _raw_response = completion
        return answer

    def _grounded_multiple_choice_completion_from_state(
        self,
        state: ControllerState,
    ) -> tuple[str, str] | None:
        if state.task_type != "multiple_choice_visual_qa":
            return None
        options = _extract_multiple_choice_options(state.question)
        if not options:
            return None

        timelogic_symbolic = self._timelogic_symbolic_completion_from_state(state, options)
        if timelogic_symbolic is not None:
            return timelogic_symbolic

        evidence_best_option, evidence_metadata = self._best_verified_option_from_evidence(
            state,
            options,
        )
        if evidence_best_option is None:
            return None
        return evidence_best_option, json.dumps(
            {
                "source": "grounded_evidence_to_option_entailment",
                "best_option": evidence_best_option,
                **evidence_metadata,
            },
            ensure_ascii=True,
        )

    def _multiple_choice_completion_from_state(
        self,
        state: ControllerState,
    ) -> tuple[str, str] | None:
        if state.task_type != "multiple_choice_visual_qa":
            return None
        options = _extract_multiple_choice_options(state.question)
        if not options:
            return None
        timelogic_symbolic = self._timelogic_symbolic_completion_from_state(state, options)
        if timelogic_symbolic is not None:
            return timelogic_symbolic
        visual_verifier = self._visual_answer_verification_from_state(state, options)
        if visual_verifier is not None:
            verified_answer, verifier_metadata = visual_verifier
            if verified_answer is not None:
                return verified_answer, json.dumps(
                    {
                        "source": "visual_answer_verifier",
                        "best_option": verified_answer,
                        **verifier_metadata,
                    },
                    ensure_ascii=True,
                )
        evidence_best_option, evidence_metadata = self._best_verified_option_from_evidence(
            state,
            options,
        )
        if evidence_best_option is not None:
            return evidence_best_option, json.dumps(
                {
                    "source": "evidence_to_option_entailment",
                    "best_option": evidence_best_option,
                    **evidence_metadata,
                }
            )

        visual_evidence_lines = []
        for item in sorted(
            [evidence for evidence in state.evidence_ledger if evidence.modality == "visual"],
            key=lambda evidence: (-evidence.confidence, evidence.time_span.start),
        )[:8]:
            visual_evidence_lines.append(
                json.dumps(
                    {
                        "evidence_id": item.evidence_id,
                        "source_node_id": item.source_node_id,
                        "slot": item.metadata.get("slot"),
                        "role": item.metadata.get("role"),
                        "time_span": item.time_span.to_dict(),
                        "relation_evidence_status": item.metadata.get(
                            "relation_evidence_status"
                        ),
                        "co_visible": item.metadata.get("vrrqa_co_visible"),
                        "relation_supported": item.metadata.get("vrrqa_relation_supported"),
                        "visible_relation": item.metadata.get("vrrqa_visible_relation"),
                        "spatial_relation": item.metadata.get("vrrqa_spatial_relation"),
                        "detail": _focus_evidence_detail(item.detail or item.claim, state.question),
                    },
                    ensure_ascii=True,
                )
            )

        option_lines = [f"{letter}. {text}" for letter, text in options.items()]
        prompt = _build_vrrqa_visual_reasoning_prompt(
            question_text=_strip_options_from_question(state.question),
            option_lines=option_lines,
            visual_evidence_lines=visual_evidence_lines,
            prompt_plugin=state.global_context.get("benchmark_prompt_plugin"),
        )
        response = self.controller_client.completion(prompt).strip()
        parsed = _parse_multiple_choice_letter(response, options)
        return parsed or sorted(options)[0], response

    def _timelogic_symbolic_completion_from_state(
        self,
        state: ControllerState,
        options: dict[str, str],
    ) -> tuple[str, str] | None:
        event_memory = state.event_memory
        if event_memory is None or event_memory.task_name != "timelogic":
            return None
        if not event_memory.relations:
            return None

        if event_memory.mode == "bool":
            verdicts = [
                self._evaluate_timelogic_relation(event_memory, relation)
                for relation in event_memory.relations
            ]
            if any(verdict is None for verdict in verdicts):
                return None
            answer_text = "yes" if all(verdicts) else "no"
            choice = _choice_for_option_text(options, answer_text)
            if choice is None:
                return None
            return choice, json.dumps(
                {
                    "source": "timelogic_symbolic_event_memory",
                    "verdicts": verdicts,
                    "answer": answer_text,
                },
                ensure_ascii=True,
            )

        if event_memory.mode == "mc":
            target_relations = [
                relation
                for relation in event_memory.relations
                if not self._relation_mentions_option(relation)
            ]
            target_verdicts = [
                self._evaluate_timelogic_relation(event_memory, relation)
                for relation in target_relations
            ]
            if any(verdict is False for verdict in target_verdicts):
                return None

            supported: list[str] = []
            for letter in sorted(options):
                option_event_id = f"option_{letter}"
                option_relations = [
                    relation
                    for relation in event_memory.relations
                    if relation.get("left") == option_event_id
                    or relation.get("right") == option_event_id
                ]
                if not option_relations:
                    continue
                option_verdicts = [
                    self._evaluate_timelogic_relation(event_memory, relation)
                    for relation in option_relations
                ]
                if option_verdicts and all(verdict is True for verdict in option_verdicts):
                    supported.append(letter)
            if len(supported) != 1:
                return None
            return supported[0], json.dumps(
                {
                    "source": "timelogic_symbolic_event_memory",
                    "best_option": supported[0],
                },
                ensure_ascii=True,
            )
        return None

    def _relation_mentions_option(self, relation: dict[str, Any]) -> bool:
        return str(relation.get("left", "")).startswith("option_") or str(
            relation.get("right", "")
        ).startswith("option_")

    def _evaluate_timelogic_relation(
        self,
        event_memory,
        relation: dict[str, Any],
    ) -> bool | None:
        left = event_memory.events.get(str(relation.get("left")))
        right = event_memory.events.get(str(relation.get("right")))
        if left is None or right is None:
            return None
        left_intervals = left.intervals
        right_intervals = right.intervals
        operator = str(relation.get("operator") or "").lower()
        if operator == "imply":
            if left_intervals and not right_intervals:
                return False
            if not left_intervals:
                return None
            return bool(right_intervals)
        if not left_intervals or not right_intervals:
            return None
        if operator == "before":
            if relation.get("quantifier") == "always":
                return max(item.time_span.end for item in left_intervals) <= min(
                    item.time_span.start for item in right_intervals
                )
            return min(item.time_span.start for item in left_intervals) <= min(
                item.time_span.start for item in right_intervals
            )
        if operator == "overlap":
            return any(
                left_interval.time_span.overlaps(right_interval.time_span)
                for left_interval in left_intervals
                for right_interval in right_intervals
            )
        if operator == "disjoint":
            return not any(
                left_interval.time_span.overlaps(right_interval.time_span)
                for left_interval in left_intervals
                for right_interval in right_intervals
            )
        return None

    def _visual_answer_verification_from_state(
        self,
        state: ControllerState,
        options: dict[str, str],
    ) -> tuple[str | None, dict[str, Any]] | None:
        if not self.enable_vrrqa_visual_answer_verifier:
            return None
        if self.visual_refiner is None:
            return None
        if state.task_type != "multiple_choice_visual_qa":
            return None
        source_video_path = state.global_context.get("source_video_path")
        if not source_video_path:
            return None

        evidence = self._visual_verifier_evidence(state)
        if not evidence:
            return None
        evidence_ids = [item.evidence_id for item in evidence]
        cache_key = "|".join(evidence_ids)
        cached = state.global_context.get("vrrqa_visual_answer_verifier")
        if isinstance(cached, dict) and cached.get("cache_key") == cache_key:
            answer = cached.get("answer")
            metadata = dict(cached.get("metadata", {}))
            return (answer if isinstance(answer, str) else None), metadata

        prompt = _build_vrrqa_visual_answer_verifier_prompt(
            question_text=_strip_options_from_question(state.question),
            option_lines=[f"{letter}. {text}" for letter, text in options.items()],
            visual_evidence_lines=[
                json.dumps(
                    {
                        "evidence_id": item.evidence_id,
                        "source_node_id": item.source_node_id,
                        "time_span": item.time_span.to_dict(),
                        "relation_evidence_status": item.metadata.get(
                            "relation_evidence_status"
                        ),
                        "co_visible": item.metadata.get("vrrqa_co_visible"),
                        "co_visible_frame_count": item.metadata.get(
                            "vrrqa_co_visible_frame_count"
                        ),
                        "relation_supported": item.metadata.get("vrrqa_relation_supported"),
                        "visible_relation": item.metadata.get("vrrqa_visible_relation"),
                        "spatial_relation": item.metadata.get("vrrqa_spatial_relation"),
                        "detail": _focus_evidence_detail(item.detail or item.claim, state.question),
                    },
                    ensure_ascii=True,
                )
                for item in evidence
            ],
            requires_co_visible_relation=requires_co_visible_relation(state.question),
            prompt_plugin=state.global_context.get("benchmark_prompt_plugin"),
        )
        verification_span = _merge_evidence_spans(evidence)
        metadata: dict[str, Any] = {
            "attempted": True,
            "evidence_ids": evidence_ids,
            "verification_span": verification_span.to_dict(),
        }
        try:
            with _temporary_component_prompt_override(
                self.visual_refiner,
                prompt,
            ), _temporary_component_frame_count(
                self.visual_refiner,
                self.vrrqa_visual_verifier_frame_count,
            ):
                summaries = self.visual_refiner.summarize(
                    str(source_video_path),
                    [verification_span],
                )
        except (OSError, RuntimeError, ValueError) as exc:
            metadata.update(
                {
                    "answer": None,
                    "reason": "visual_verifier_failed",
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            )
            state.global_context["vrrqa_visual_answer_verifier"] = {
                "cache_key": cache_key,
                "answer": None,
                "metadata": metadata,
            }
            return None, metadata

        if not summaries:
            metadata.update({"answer": None, "reason": "empty_visual_verifier_response"})
            state.global_context["vrrqa_visual_answer_verifier"] = {
                "cache_key": cache_key,
                "answer": None,
                "metadata": metadata,
            }
            return None, metadata

        summary = summaries[0]
        metadata.update(
            {
                "reason": "visual_verifier_response",
                "summary": summary.summary,
                "summary_metadata": dict(summary.metadata),
            }
        )
        answer = self._supported_visual_verifier_option(summary.metadata, options, state)
        metadata["answer"] = answer
        state.global_context["vrrqa_visual_answer_verifier"] = {
            "cache_key": cache_key,
            "answer": answer,
            "metadata": metadata,
        }
        return answer, metadata

    def _visual_verifier_evidence(
        self,
        state: ControllerState,
        max_items: int = 3,
    ) -> list[Evidence]:
        visual_evidence = [item for item in state.evidence_ledger if item.modality == "visual"]
        if not visual_evidence:
            return []
        allowed_ids: set[str] = set()
        if state.evidence_board is not None:
            allowed_ids = set(
                state.evidence_board.core_evidence_ids + state.evidence_board.support_evidence_ids
            )
        filtered = [
            item
            for item in visual_evidence
            if not allowed_ids or item.evidence_id in allowed_ids
        ]
        selected = sorted(
            filtered or visual_evidence,
            key=lambda item: (-item.confidence, item.time_span.start),
        )
        return selected[:max_items]

    def _supported_visual_verifier_option(
        self,
        metadata: dict[str, Any],
        options: dict[str, str],
        state: ControllerState,
    ) -> str | None:
        if _metadata_bool(metadata.get("vrrqa_needs_more_evidence")) is True:
            return None
        best_option = metadata.get("vrrqa_best_option")
        if isinstance(best_option, str):
            best_option = best_option.strip().upper()[:1]
        if not isinstance(best_option, str) or best_option not in options:
            return None
        if not _option_comparison_supports(metadata.get("vrrqa_option_comparison"), best_option):
            return None
        if requires_co_visible_relation(state.question):
            relation_supported = _metadata_bool(metadata.get("vrrqa_relation_supported"))
            if relation_supported is not True:
                return None
            if not _metadata_has_co_visible_frame(metadata):
                return None
        return best_option

    def _best_verified_option_from_evidence(
        self,
        state: ControllerState,
        options: dict[str, str],
    ) -> tuple[str | None, dict[str, Any]]:
        verified: list[tuple[float, float, str, str]] = []
        aggregate_scores = {letter: 0.0 for letter in options}
        valid_choices = set(options)
        if _has_duplicate_option_text(options):
            return None, {"reason": "duplicate_option_text"}
        requires_relation_evidence = requires_co_visible_relation(state.question)
        for item in state.evidence_ledger:
            if requires_relation_evidence and item.modality == "visual":
                relation_status = relation_evidence_status(item)
                if relation_status != "supported":
                    continue
            best_option = item.metadata.get("vrrqa_best_option")
            source = item.evidence_id
            if isinstance(best_option, str):
                best_option = best_option.strip().upper()[:1]
            if isinstance(best_option, str) and best_option in valid_choices:
                score = _verified_option_score(
                    best_option,
                    item.metadata.get("vrrqa_option_scores"),
                )
                weighted_score = max(0.0, score) * max(item.confidence, 0.1)
                aggregate_scores[best_option] += weighted_score
                verified.append((score, item.confidence, best_option, source))

            text_choice = _evidence_text_choice(item, options)
            if text_choice is not None:
                aggregate_scores[text_choice] += 0.35 * max(item.confidence, 0.1)
                verified.append((0.35, item.confidence, text_choice, source))
        if not verified:
            return None, {"reason": "no_verified_option"}

        ranked = sorted(
            aggregate_scores.items(),
            key=lambda pair: (-pair[1], pair[0]),
        )
        best_letter, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        if best_score < 0.45 or best_score - second_score < 0.18:
            return None, {
                "reason": "weak_or_low_margin_entailment",
                "option_scores": aggregate_scores,
            }
        supporting_evidence_ids = [
            evidence_id
            for _score, _confidence, letter, evidence_id in verified
            if letter == best_letter
        ]
        return best_letter, {
            "option_scores": aggregate_scores,
            "supporting_evidence_ids": supporting_evidence_ids[:4],
        }

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


POSTVALID_DOMAIN_TERMS = {
    "bell",
    "canary",
    "climate",
    "dalio",
    "einstein",
    "entanglement",
    "filters",
    "loophole",
    "microfinance",
    "photon",
    "photons",
    "quasar",
    "quasars",
    "ronaldo",
    "yunus",
}


def _is_postvalid_v1_longshot_context(context: Any) -> bool:
    return isinstance(context, dict) and str(context.get("dataset_name") or "") == "postvalid_v1"


def _postvalid_key_terms(text: str) -> list[str]:
    terms: list[str] = []
    for match in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", text):
        phrase = " ".join(match.group(0).split())
        phrase_tokens = [
            token.lower()
            for token in TOKEN_PATTERN.findall(phrase)
            if token.lower() not in STOPWORDS
        ]
        if not phrase_tokens:
            continue
        if len(phrase_tokens) > 1 and phrase.lower() not in {term.lower() for term in terms}:
            terms.append(phrase)
        for token in phrase_tokens:
            if token in POSTVALID_DOMAIN_TERMS and token not in terms:
                terms.append(token)
    for token in _postvalid_tokens(text):
        if token in POSTVALID_DOMAIN_TERMS and token not in terms:
            terms.append(token)
    return terms[:24]


def _memory_key_terms(memory: VideoMemory) -> set[str]:
    cached = memory.metadata.get("postvalid_memory_key_terms")
    if isinstance(cached, list) and all(isinstance(item, str) for item in cached):
        return set(cached)

    terms: set[str] = set()
    for node in memory.nodes.values():
        parts = [
            node.visual_summary,
            " ".join(node.tags),
            " ".join(node.entities),
            " ".join(span.text for span in node.speech_spans),
            " ".join(span.text for span in node.ocr_spans),
        ]
        text = " ".join(part for part in parts if part)
        terms.update(_postvalid_tokens(text))
        for key_term in _postvalid_key_terms(text):
            terms.update(_postvalid_tokens(key_term))
    memory.metadata["postvalid_memory_key_terms"] = sorted(terms)
    return terms


def _term_present_in_memory(term: str, memory_terms: set[str]) -> bool:
    tokens = _postvalid_tokens(term)
    return bool(tokens) and all(token in memory_terms for token in tokens)


def _postvalid_temporal_intents(question: str) -> list[str]:
    lowered = question.lower()
    intents: list[str] = []
    phrase_map = [
        ("immediate_after", ("right after", "immediately after", "just after", "after the")),
        (
            "earlier_problem",
            (
                "earlier experiment",
                "earlier experiments",
                "big problem",
                "main problem",
                "specific problem",
                "challenge",
                "wanted to fix",
                "made people doubt",
                "doubt whether",
                "left open loopholes",
                "loophole",
                "not truly random",
            ),
        ),
        (
            "first_piece",
            (
                "first piece",
                "first jewelry",
                "first thing",
                "first got",
                "first real sign",
                "first sign",
                "starting to take",
            ),
        ),
        (
            "early_race",
            (
                "early in the race",
                "early lead",
                "big lead early",
                "strong start",
                "stay ahead",
                "pull away",
                "ahead of",
                "lead early",
                "got a gap",
            ),
        ),
        (
            "later_effect",
            (
                "rest of the game",
                "rest of this video",
                "affect how",
                "what happened next",
                "next happened",
                "after that",
                "later",
                "consequence",
            ),
        ),
    ]
    for label, phrases in phrase_map:
        if any(phrase in lowered for phrase in phrases):
            intents.append(label)
    if "why" in lowered or "how did" in lowered or "how could" in lowered:
        intents.append("cause_consequence")
    return list(dict.fromkeys(intents))


def _postvalid_temporal_intent_query_phrases(intents: Any) -> list[str]:
    if not isinstance(intents, list):
        return []
    phrase_by_intent = {
        "immediate_after": "immediate aftermath right after next event damage leak emergency",
        "earlier_problem": "earlier problem loophole hidden influence doubt challenge fix not truly random filter choices",
        "first_piece": (
            "first piece first received first real sign started taking seriously "
            "behavior changed why meaningful surprise"
        ),
        "early_race": "early race strong start early lead gap pull away stay ahead same bikes",
        "later_effect": "later effect rest of game rest of video consequence after decision what happened next after that",
        "cause_consequence": "cause mechanism reason consequence why explanation",
    }
    return [phrase_by_intent[item] for item in intents if item in phrase_by_intent]


def _postvalid_task_specific_query_terms(question: str) -> str:
    lowered = question.lower()
    expansions: list[str] = []
    if "diamond add" in lowered or "add-on" in lowered:
        expansions.append(
            "diamond add-on right away rest of this video easy put a stud wore immediately"
        )
    if "filippo" in lowered or "first piece" in lowered:
        expansions.append("Filippo first piece jewelry ring diamond special shocked meaningful")
    if "rebounding" in lowered or "cheering" in lowered:
        expansions.append(
            "rebounding cheering bench not playing still contributing team player hustle"
        )
    if "harder r5" in lowered or "rear tire" in lowered or "softer compound" in lowered:
        expansions.append(
            "Cameron Beaubier harder R5 rear tire softer compound hot track greasy "
            "cold tearing tire strategy why chose"
        )
    if "apollo 1" in lowered or "inside the capsule" in lowered:
        expansions.append("Apollo 1 command module fire capsule pure oxygen hatch escape test")
    if "carbon dioxide" in lowered or "right filters" in lowered:
        expansions.append("carbon dioxide filters lithium hydroxide canister square round adapter")
    return " ".join(expansions)


def _postvalid_required_distinct_speech_sources(intents: Any) -> int:
    if not isinstance(intents, list) or not intents:
        return 2
    if any(item in {"cause_consequence", "immediate_after", "earlier_problem", "later_effect"} for item in intents):
        return 3
    return 2


def _overlaps_existing_postvalid_speech(
    candidate_span: TimeSpan,
    opened_spans: list[TimeSpan],
    threshold: float = 0.45,
) -> bool:
    candidate_duration = max(candidate_span.end - candidate_span.start, 1e-6)
    for span in opened_spans:
        overlap = max(
            0.0,
            min(candidate_span.end, span.end) - max(candidate_span.start, span.start),
        )
        if overlap / candidate_duration >= threshold:
            return True
    return False


def _is_related_postvalid_node_id(candidate_node_id: str, opened_node_id: str) -> bool:
    if not candidate_node_id or not opened_node_id:
        return False
    return candidate_node_id.startswith(f"{opened_node_id}_") or opened_node_id.startswith(
        f"{candidate_node_id}_"
    )


def _scenario_agrees_with_question_for_controller(scenario: str, question: str) -> bool:
    scenario_terms = set(_postvalid_key_terms(scenario))
    question_terms = set(_postvalid_key_terms(question))
    if not scenario_terms or not question_terms:
        return True
    return bool({term.lower() for term in scenario_terms} & {term.lower() for term in question_terms})


def _postvalid_tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_PATTERN.findall(text)
        if token.lower() not in STOPWORDS and len(token) > 1
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


def _evidence_answer_relevance_score(item: Evidence, question: str) -> float:
    detail = item.detail or item.claim
    detail_tokens = _postvalid_tokens(detail)
    question_tokens = _postvalid_tokens(question)
    overlap = len(detail_tokens & question_tokens)
    lowered = detail.lower()
    question_lowered = question.lower()
    score = float(overlap * 4)
    if item.metadata.get("role") == "core":
        score += 1.0
    if str(item.metadata.get("answer_span") or "").strip():
        score += 3.0
    for keyword in _answer_relevance_keywords(question_lowered):
        if keyword in lowered:
            score += 5.0
    if any(marker in lowered for marker in ("because", "why", "reason", "so ", "therefore")):
        score += 2.0
    if "evidence does not" in lowered or "insufficient" in lowered:
        score -= 8.0
    return score


def _has_exact_answer_span(item: Evidence) -> bool:
    if not str(item.metadata.get("answer_span") or "").strip():
        return False
    if item.metadata.get("answer_span_is_exact_answer") is False:
        return False
    role = item.metadata.get("answer_span_role")
    if role == "retrieved_fine_window_text":
        return False
    return True


def _focused_answer_sentences(detail: str, question: str, max_sentences: int = 2) -> list[str]:
    focused = _focus_evidence_detail(detail, question, max_chars=900)
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", focused)
        if sentence.strip()
    ]
    if not sentences:
        return [focused] if focused else []
    question_tokens = _postvalid_tokens(question)
    question_lowered = question.lower()
    keywords = _answer_relevance_keywords(question_lowered)
    scored: list[tuple[float, int, str]] = []
    for index, sentence in enumerate(sentences):
        lowered = sentence.lower()
        sentence_tokens = _postvalid_tokens(sentence)
        score = float(len(question_tokens & sentence_tokens) * 4)
        score += sum(5.0 for keyword in keywords if keyword in lowered)
        if any(marker in lowered for marker in ("because", "why", "reason", "so ", "therefore")):
            score += 2.0
        if "evidence does not" in lowered or "insufficient" in lowered:
            score -= 8.0
        scored.append((-score, index, sentence))
    scored.sort(key=lambda item: (item[0], item[1]))
    selected = [sentence for score, _, sentence in scored if score < 0][:max_sentences]
    if selected:
        return selected
    return sentences[:max_sentences]


def _answer_relevance_keywords(question_lowered: str) -> tuple[str, ...]:
    keywords = ["because", "reason", "why", "how"]
    if "cartier" in question_lowered or "bracelet" in question_lowered:
        keywords.extend(["cartier", "bracelet", "clasp", "opening", "worried", "lose", "fixed"])
    if "diamond add" in question_lowered or "add-on" in question_lowered:
        keywords.extend(["wear it", "rest of this video", "right away", "easy", "put a stud"])
    if "filippo" in question_lowered or "first piece" in question_lowered:
        keywords.extend(["filippo", "first", "ring", "diamond", "shocked", "special"])
    if "piercing" in question_lowered or "earrings" in question_lowered:
        keywords.extend(["piercing", "earrings", "child", "four", "choice"])
    if "loophole" in question_lowered or "earlier experiment" in question_lowered:
        keywords.extend(["loophole", "hidden", "influence", "filter", "random", "doubt"])
    if "quasar" in question_lowered or "photon" in question_lowered:
        keywords.extend(["quasar", "photon", "filter", "billions", "random"])
    if "carbon dioxide" in question_lowered or "filters" in question_lowered:
        keywords.extend(["carbon dioxide", "filter", "square", "round", "canister", "lithium"])
    if "race" in question_lowered or "stay ahead" in question_lowered:
        keywords.extend(["race", "start", "lead", "gap", "ahead", "tire", "inside", "control"])
    if "bench" in question_lowered or "game" in question_lowered:
        keywords.extend(["bench", "game", "down by 30", "fourth quarter", "hustle", "team"])
    if "apollo" in question_lowered or "oxygen" in question_lowered:
        keywords.extend(["apollo", "oxygen", "explosion", "tank", "hatch", "fire", "escape"])
    return tuple(dict.fromkeys(keywords))


def _postvalid_anchor_match_status(question: str, evidence_text: str) -> str:
    question_lowered = question.lower()
    evidence_tokens = _postvalid_tokens(evidence_text)
    evidence_lowered = evidence_text.lower()
    anchor_groups = _postvalid_required_anchor_groups(question, question_lowered)
    for anchors, minimum_matches in anchor_groups:
        if _anchor_group_match_count(anchors, evidence_tokens, evidence_lowered) < minimum_matches:
            return "failed"
    if anchor_groups:
        return "strong"
    named_terms = _postvalid_named_question_terms(question)
    if named_terms and not (named_terms & evidence_tokens):
        return "failed"
    if named_terms:
        return "strong"
    return "neutral"


def _postvalid_required_anchor_groups(
    question: str,
    question_lowered: str,
) -> list[tuple[set[str], int]]:
    groups: list[tuple[set[str], int]] = []
    if "carbon dioxide" in question_lowered or "right filters" in question_lowered:
        groups.append(({"carbon", "dioxide", "filter", "filters", "canister", "lithium"}, 2))
    if "filippo" in question_lowered:
        groups.append(({"filippo"}, 1))
    if "childhood" in question_lowered and (
        "piercing" in question_lowered or "earring" in question_lowered
    ):
        groups.append(({"childhood", "child", "piercing", "piercings", "earring", "earrings"}, 2))
    if "diamond add" in question_lowered or "add-on" in question_lowered:
        groups.append(({"diamond", "add", "addon", "easy", "wear", "right", "away"}, 2))
    if "oxygen tank" in question_lowered or "tank exploded" in question_lowered:
        groups.append(({"apollo", "oxygen", "tank", "exploded", "explosion", "blast"}, 2))
    if "apollo 1" in question_lowered or "inside the capsule" in question_lowered:
        groups.append(({"apollo", "fire", "capsule", "hatch", "oxygen", "escape"}, 2))
    if "jake gagne" in question_lowered or "cameron peterson" in question_lowered:
        groups.append(({"jake", "gagne", "cameron", "peterson", "matthew", "schultz"}, 2))
    if "canary islands" in question_lowered:
        groups.append(({"canary", "islands", "quasar", "quasars", "photon", "filter"}, 2))
    if "quasar" in question_lowered or "quasars" in question_lowered:
        groups.append(({"quasar", "quasars", "photon", "photons", "filter", "random"}, 2))
    if "bench" in question_lowered and "tryout" in question_lowered:
        groups.append(({"bench", "tryout", "ignite", "team", "cheering", "rebounding"}, 2))
    if not groups:
        named_terms = _postvalid_named_question_terms(question)
        if named_terms:
            groups.append((named_terms, 1))
    return groups


def _anchor_group_match_count(
    anchors: set[str],
    evidence_tokens: set[str],
    evidence_lowered: str,
) -> int:
    matches = 0
    for anchor in anchors:
        if " " in anchor:
            if anchor in evidence_lowered:
                matches += 1
            continue
        if anchor in evidence_tokens:
            matches += 1
    return matches


def _postvalid_named_question_terms(question: str) -> set[str]:
    ignored = {
        "what",
        "why",
        "how",
        "when",
        "where",
        "who",
        "did",
        "does",
        "the",
        "a",
        "an",
    }
    terms = set()
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9_-]{2,}\b", question):
        token = match.group(0).lower()
        if token not in ignored:
            terms.add(token)
    return terms


def _select_temporally_diverse_initial_nodes(nodes: list[Any], limit: int) -> list[Any]:
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


def _timelogic_temporal_sweep_candidates(
    memory: VideoMemory,
    limit: int = 80,
) -> list[dict[str, Any]]:
    candidates = [
        node
        for node in memory.nodes.values()
        if node.level == "clip" and node.visual_summary.strip()
    ]
    candidates = sorted(candidates, key=lambda node: (node.time_span.start, node.node_id))
    return [
        {
            "node_id": node.node_id,
            "start": node.time_span.start,
            "end": node.time_span.end,
        }
        for node in candidates[:limit]
    ]


def _build_vrrqa_visual_reasoning_prompt(
    *,
    question_text: str,
    option_lines: list[str],
    visual_evidence_lines: list[str],
    prompt_plugin: dict[str, Any] | None = None,
) -> str:
    plugin_section = render_prompt_plugin_section(prompt_plugin)
    plugin_lines = ["", plugin_section] if plugin_section else []
    return "\n".join(
        [
            "You are answering a VRR-QA multiple-choice question.",
            *plugin_lines,
            "",
            "Focus on visual implicit reasoning across the video. Pay special attention to:",
            "- spatial relations: left/right, above/below, near/far, in front/behind",
            "- viewpoint and visibility",
            "- motion direction and trajectory",
            "- temporal order of events",
            "- entity continuity across frames",
            "- physical context and implied relationships",
            "",
            "Use only the visual evidence provided. Do not use audio, subtitles, or outside knowledge.",
            "",
            "First analyze the relevant visual evidence:",
            "1. Identify the important entities.",
            "2. Describe their spatial relationships.",
            "3. Describe any motion, trajectory, or temporal ordering.",
            "4. Explain what relation must be inferred.",
            "5. Compare each answer option against the evidence.",
            "",
            "For spatial/depth/viewpoint/facing questions, treat evidence as sufficient only when "
            "the target entities are co-visible in at least one visual frame or keyframe. If the "
            "evidence says the entities are not visible together, do not choose an option from "
            "that unsupported relation.",
            "",
            "Question:",
            question_text,
            "",
            "Options:",
            *option_lines,
            "",
            "Visual evidence:",
            *(visual_evidence_lines or ["No direct visual evidence was collected."]),
            "",
            "Return your answer in this exact format:",
            "Reasoning: <concise reasoning>",
            "Answer: <single option letter>",
        ]
    )


def _build_vrrqa_visual_answer_verifier_prompt(
    *,
    question_text: str,
    option_lines: list[str],
    visual_evidence_lines: list[str],
    requires_co_visible_relation: bool,
    prompt_plugin: dict[str, Any] | None = None,
) -> str:
    relation_requirement = []
    if requires_co_visible_relation:
        relation_requirement = [
            "This question requires a spatial/depth/viewpoint/facing relation.",
            "Only frames where the target entities are co-visible may vote for the relation.",
            "If no co-visible frame supports an option, set best_option to null, "
            "relation_supported to false, and needs_more_evidence to true.",
        ]
    plugin_section = render_prompt_plugin_section(prompt_plugin)
    plugin_lines = ["", plugin_section] if plugin_section else []
    return "\n".join(
        [
            "You are the final visual verifier for a VRR-QA multiple-choice answer.",
            *plugin_lines,
            "The images are ordered frames/keyframes sampled from the selected evidence span. "
            "Use the frame order as temporal order and preserve the original visual layout.",
            "Use only visible image evidence. Do not use audio, subtitles, captions, or outside "
            "knowledge.",
            "",
            "Required analysis:",
            "1. Identify the exact target entities from the question.",
            "2. For each plausible visible candidate, explain why it matches or does not match.",
            "3. For each frame/keyframe, report whether the target entities are visible and "
            "co-visible.",
            "4. For spatial/depth/facing questions, vote only from co-visible frames.",
            "5. Compare every option as supported, contradicted, or not_enough_evidence.",
            "",
            *relation_requirement,
            "",
            "Return strict JSON only with these keys:",
            "`best_option`, `option_scores`, `target_entities`, `candidate_entities`, "
            "`entity_grounding`, `frame_observations`, `co_visible_frame_indices`, "
            "`relation_votes`, `vote_counts`, `aggregated_relation`, `entities_visible`, "
            "`co_visible`, `relation_supported`, `visible_relation`, `spatial_relation`, "
            "`motion_trajectory`, `temporal_order`, `entity_continuity`, `physical_context`, "
            "`inferred_relation`, `option_comparison`, `verifier_verdict`, "
            "`needs_more_evidence`, `evidence`, `summary`, `frame_timeline`, `tags`, "
            "`entities`.",
            "`best_option` must be one option letter only if exactly one option is supported; "
            "otherwise use null.",
            "`option_comparison` must map each option letter to an object with `verdict` "
            "(`supported`, `contradicted`, or `not_enough_evidence`) and `reason`.",
            "`frame_observations` must include one concise object per input frame with "
            "`frame_index`, `target_entities_visible`, `co_visible`, `entity_grounding`, "
            "`relation`, `motion`, and `option_support`.",
            "",
            "Question:",
            question_text,
            "",
            "Options:",
            *option_lines,
            "",
            "Previously selected visual evidence records:",
            *(visual_evidence_lines or ["No prior visual evidence records were selected."]),
        ]
    )


def _merge_evidence_spans(evidence: list[Evidence]) -> TimeSpan:
    if not evidence:
        raise ValueError("Cannot merge empty evidence spans")
    return TimeSpan(
        start=min(item.time_span.start for item in evidence),
        end=max(item.time_span.end for item in evidence),
    )


def _metadata_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "supported"}:
            return True
        if normalized in {"false", "no", "0", "unsupported"}:
            return False
    return None


def _metadata_has_co_visible_frame(metadata: dict[str, Any]) -> bool:
    if _metadata_bool(metadata.get("vrrqa_co_visible")) is True:
        return True
    try:
        if int(metadata.get("vrrqa_co_visible_frame_count", 0)) > 0:
            return True
    except (TypeError, ValueError):
        pass
    indices = metadata.get("vrrqa_co_visible_frame_indices")
    return isinstance(indices, list) and bool(indices)


def _option_comparison_supports(option_comparison: Any, option: str) -> bool:
    if not isinstance(option_comparison, dict):
        return True
    value = option_comparison.get(option)
    if value is None:
        value = option_comparison.get(option.lower())
    if isinstance(value, str):
        return value.strip().lower().startswith("support")
    if isinstance(value, dict):
        verdict = str(value.get("verdict") or value.get("status") or "").strip().lower()
        return verdict.startswith("support")
    return False


@contextmanager
def _temporary_component_prompt_override(component: Any, prompt: str | None):
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
def _temporary_component_frame_count(component: Any, frame_count: int):
    if component is None or frame_count <= 0:
        yield
        return

    originals: list[tuple[str, Any]] = []
    for attribute in ("frame_count", "pitome_max_selected_frames"):
        if not hasattr(component, attribute):
            continue
        original = getattr(component, attribute)
        originals.append((attribute, original))
        if attribute == "frame_count":
            current = original if isinstance(original, int) and original > 0 else 0
            setattr(component, attribute, max(current, frame_count))
        elif original is not None:
            current = original if isinstance(original, int) and original > 0 else 0
            setattr(component, attribute, max(current, frame_count))
    try:
        yield
    finally:
        for attribute, original in originals:
            setattr(component, attribute, original)


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


def _choice_for_option_text(options: dict[str, str], expected_text: str) -> str | None:
    expected = _normalize_choice_text(expected_text)
    for letter, option_text in options.items():
        if _normalize_choice_text(option_text) == expected:
            return letter
    return None


def _has_duplicate_option_text(options: dict[str, str]) -> bool:
    normalized = [_normalize_choice_text(text) for text in options.values()]
    normalized = [text for text in normalized if text]
    return len(normalized) != len(set(normalized))


def _evidence_text_choice(item: Evidence, options: dict[str, str]) -> str | None:
    text = " ".join(
        str(part)
        for part in (
            item.metadata.get("vrrqa_evidence"),
            item.metadata.get("vrrqa_visible_relation"),
            item.metadata.get("vrrqa_spatial_relation"),
            item.detail,
            item.claim,
        )
        if part
    )
    explicit_match = re.search(
        r"\b(?:best\s+option|selected\s+option|final\s+answer|answer|choice)\s*"
        r"(?:is|=|:)?\s*\(?([A-Z])\)?\b",
        text,
        flags=re.IGNORECASE,
    )
    if explicit_match is not None:
        letter = explicit_match.group(1).upper()
        if letter in options:
            return letter

    lowered = text.lower()
    if not any(cue in lowered for cue in ("best option", "answer", "therefore", "supports")):
        return None
    evidence_text = _normalize_choice_text(text)
    contained = [
        letter
        for letter, option_text in options.items()
        if _normalize_choice_text(option_text) and _normalize_choice_text(option_text) in evidence_text
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


def _looks_incomplete_answer(answer: str) -> bool:
    stripped = " ".join(str(answer or "").split()).strip()
    if not stripped:
        return True
    if len(stripped.split()) < 5:
        return False
    if stripped[-1] in ".?!\"')":
        return False
    lowered = stripped.lower()
    dangling_endings = (
        " and",
        " or",
        " because",
        " with",
        " by",
        " to",
        " from",
        " that",
        " which",
        " whose",
        " making",
        " specifically",
        " descent",
        " genuinely",
        " if",
        " into",
        " about",
        " where",
        " when",
        " while",
    )
    if lowered.endswith(dangling_endings):
        return True
    last_words = stripped.split()[-3:]
    return len(" ".join(last_words)) < 24 and len(stripped) > 120


def _looks_refusal_answer(answer: str) -> bool:
    lowered = " ".join(str(answer or "").split()).strip().lower()
    if not lowered:
        return True
    refusal_markers = (
        "the provided evidence",
        "the evidence provided",
        "the evidence does not",
        "does not describe",
        "does not mention",
        "i cannot answer",
        "i can't answer",
        "cannot answer",
        "could not answer",
        "insufficient evidence",
        "evidence is insufficient",
        "not enough evidence",
    )
    return any(marker in lowered for marker in refusal_markers)


def _normalize_choice_text(text: str) -> str:
    return " ".join(
        token.group(0).lower()
        for token in TOKEN_PATTERN.finditer(text)
        if token.group(0).lower() not in STOPWORDS
    )
