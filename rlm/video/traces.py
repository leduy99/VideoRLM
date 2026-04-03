import json
from pathlib import Path
from typing import Any

from rlm.video.types import VideoRLMResult


def trace_to_training_examples(
    trace: list[dict[str, Any]],
    video_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Convert runtime trace steps into a simple action-supervision dataset format.

    Each example follows the shape:
    `(state_t, gold_action_t, observation_t, state_t+1)`.
    """

    examples: list[dict[str, Any]] = []
    for item in trace:
        example = {
            "state": item["state"],
            "gold_action": item["action"],
            "observation": item["observation"],
            "next_state": item["next_state"],
        }
        if video_id is not None:
            example["video_id"] = video_id
        examples.append(example)
    return examples


def result_to_training_examples(result: VideoRLMResult) -> list[dict[str, Any]]:
    video_id = result.state.global_context.get("video_id")
    return trace_to_training_examples(result.trace, video_id=video_id)


def save_training_examples(examples: list[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in examples:
            json.dump(item, handle)
            handle.write("\n")
