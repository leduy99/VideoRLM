from pathlib import Path

from rlm.core.types import ModelUsageSummary, UsageSummary
from rlm.video import result_to_training_examples, save_training_examples
from rlm.video.types import BudgetState, ControllerState, VideoRLMResult


def test_result_to_training_examples(tmp_path: Path):
    state = ControllerState(
        question="When does the plan change?",
        budget=BudgetState(steps_used=2, steps_remaining=2),
        global_context={"video_id": "meeting"},
    )
    result = VideoRLMResult(
        answer="The plan changes early.",
        state=state,
        trace=[
            {
                "step_index": 1,
                "state": {"question": "When does the plan change?"},
                "action": {"action_type": "SEARCH", "query": "plan", "modality": "speech"},
                "observation": {"kind": "search", "summary": "found nodes"},
                "next_state": {"frontier": [{"node_id": "scene_001"}]},
                "raw_model_response": "{}",
            }
        ],
        usage_summary=UsageSummary(
            model_usage_summaries={
                "mock": ModelUsageSummary(
                    total_calls=1,
                    total_input_tokens=10,
                    total_output_tokens=10,
                )
            }
        ),
        execution_time=0.1,
    )

    examples = result_to_training_examples(result)
    assert len(examples) == 1
    assert examples[0]["video_id"] == "meeting"
    assert examples[0]["gold_action"]["action_type"] == "SEARCH"

    output_path = tmp_path / "examples.jsonl"
    save_training_examples(examples, output_path)
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
