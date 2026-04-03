"""
Minimal VideoRLM example using prepared artifacts and a mock controller.

This keeps the example runnable without requiring a real ASR/VL stack.
Replace the prepared artifacts and MockLM with your real adapters/models as
the next step.
"""

import json

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary
from rlm.video import (
    PreparedVideoArtifacts,
    SpeechSpan,
    TimeSpan,
    VideoMemoryBuilder,
    VideoRLM,
    VideoRLMLogger,
    VisualSummarySpan,
)


class ExampleMockLM(BaseLM):
    def __init__(self, responses: list[str]):
        super().__init__(model_name="example-mock-controller")
        self.responses = list(responses)
        self.call_count = 0

    def completion(self, prompt):
        self.call_count += 1
        return self.responses.pop(0)

    async def acompletion(self, prompt):
        return self.completion(prompt)

    def get_usage_summary(self) -> UsageSummary:
        return UsageSummary(
            model_usage_summaries={
                self.model_name: ModelUsageSummary(
                    total_calls=self.call_count,
                    total_input_tokens=self.call_count * 10,
                    total_output_tokens=self.call_count * 10,
                )
            }
        )

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(total_calls=1, total_input_tokens=10, total_output_tokens=10)

artifacts = PreparedVideoArtifacts(
    video_id="meeting",
    duration_seconds=90.0,
    speech_spans=[
        SpeechSpan(
            text="We are changing the plan after reviewing the numbers.",
            time_span=TimeSpan(5.0, 12.0),
        ),
        SpeechSpan(
            text="The team approves the updated schedule and everyone agrees.",
            time_span=TimeSpan(40.0, 55.0),
        ),
    ],
    visual_summaries=[
        VisualSummarySpan(
            summary="A slide shows the updated schedule and approval status.",
            time_span=TimeSpan(30.0, 60.0),
            granularity="scene",
            tags=["schedule", "approval"],
        )
    ],
    metadata={"source_video_path": "meeting.mp4"},
)

builder = VideoMemoryBuilder(
    scene_duration_seconds=30.0,
    segment_duration_seconds=15.0,
    clip_duration_seconds=5.0,
)
memory = builder.build_from_artifacts(artifacts)

controller = ExampleMockLM(
    responses=[
        json.dumps(
            {
                "action_type": "OPEN",
                "node_id": "meeting_scene_001",
                "modality": "speech",
                "evidence_ids": [],
                "query": None,
                "answer": None,
                "rationale": "Inspect the strongest speech candidate first.",
            }
        ),
        json.dumps(
            {
                "action_type": "STOP",
                "node_id": None,
                "modality": None,
                "evidence_ids": ["evidence_00001"],
                "query": None,
                "answer": "The plan changes early in the meeting when the speaker says they are changing it after reviewing the numbers.",
                "rationale": "The answer is directly supported by the collected speech evidence.",
            }
        ),
    ],
)

logger = VideoRLMLogger()
runner = VideoRLM(controller_client=controller, logger=logger, max_steps=4)
result = runner.run("When does the plan change?", memory, task_type="retrieval")

print("Answer:", result.answer)
print("Steps:", len(result.trace))
trace = logger.get_trace()
if trace:
    print("Logged step count:", len(trace["steps"]))
