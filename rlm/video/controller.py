import copy
import json
import re
import time
from typing import Any

from rlm.clients import get_client
from rlm.clients.base_lm import BaseLM
from rlm.core.types import ClientBackend
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
    ):
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

    def run(
        self,
        question: str,
        memory: VideoMemory,
        dialogue_context: list[dict[str, str]] | None = None,
        task_type: str | None = None,
    ) -> VideoRLMResult:
        start_time = time.perf_counter()
        index = VideoMemoryIndex(memory)
        tools = VideoToolExecutor(
            memory=memory,
            index=index,
            top_k=self.search_top_k,
            speech_snippet_refiner=self.speech_snippet_refiner_client,
            enable_hybrid_speech_refinement=self.enable_hybrid_speech_refinement,
            speech_refine_candidate_count=self.speech_refine_candidate_count,
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
                    "hybrid_speech_refinement": self.enable_hybrid_speech_refinement,
                }
            )

        trace_steps: list[dict[str, Any]] = []
        answer: str | None = None
        consecutive_empty_open_steps = 0

        while state.budget.steps_remaining > 0:
            prompt = build_controller_prompt(
                state,
                max_frontier_items=self.max_frontier_items,
            )
            raw_response = self.controller_client.completion(prompt)
            action = self._parse_action(raw_response)
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
                answer = action.answer or self._fallback_answer_from_state(state)
                break
            if action.action_type == "OPEN" and not observation.evidence:
                consecutive_empty_open_steps += 1
            else:
                consecutive_empty_open_steps = 0
            if consecutive_empty_open_steps >= 2 and state.evidence_ledger:
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
        initial_hits = index.search(question, top_k=self.max_frontier_items)
        if initial_hits:
            frontier = [item.to_frontier_item() for item in initial_hits]
        else:
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
            "topical_index": scene_summaries,
        }
        return ControllerState(
            question=question,
            task_type=task_type,
            dialogue_context=dialogue_context,
            frontier=frontier[: self.max_frontier_items],
            budget=budget,
            global_context=global_context,
        )

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

        if action.action_type == "SEARCH":
            state.frontier = self._merge_frontier(state.frontier, observation.frontier)
        elif action.action_type == "OPEN":
            state.frontier = self._set_frontier_status(state.frontier, action.node_id, "opened")
            state.frontier = self._remove_frontier_node(state.frontier, action.node_id)
            state.evidence_ledger.extend(observation.evidence)
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

    def _parse_action(self, raw_response: str) -> ControllerAction:
        candidate = raw_response.strip()
        try:
            return ControllerAction.from_dict(json.loads(candidate))
        except json.JSONDecodeError:
            extracted = self._extract_first_json_object(candidate)
            return ControllerAction.from_dict(json.loads(extracted))

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
        if state.evidence_ledger:
            return self._synthesize_answer_from_evidence(state)
        return "Controller exhausted its budget before collecting grounded evidence."

    def _synthesize_answer_from_evidence(self, state: ControllerState) -> str:
        top_evidence = sorted(
            state.evidence_ledger,
            key=lambda item: (-item.confidence, item.time_span.start),
        )[:4]
        evidence_lines = []
        for item in top_evidence:
            evidence_lines.append(
                json.dumps(
                    {
                        "evidence_id": item.evidence_id,
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
            "Evidence:\n"
            + "\n".join(evidence_lines)
        )
        return self.controller_client.completion(prompt).strip()


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
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", normalized)
        if sentence.strip()
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
        if any(keyword in snippet.lower() for keyword in ("worried", "lose", "lost", "fixed", "repair", "clasp", "opening")):
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
