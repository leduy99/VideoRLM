from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from rlm.clients.base_lm import BaseLM
from rlm.clients.transformers_local import TransformersClient

TASK_CATEGORIES = {
    "Core Perception Tasks": [
        "entity_recognition",
        "event_understanding",
        "temporal_reasoning",
        "audio_understanding",
    ],
    "Reasoning Tasks": [
        "causal_reasoning",
        "quantitative_reasoning",
        "compositional_reasoning",
        "comparative_analysis",
    ],
    "Information Tasks": [
        "information_retrieval",
        "summarization",
        "instruction_extraction",
        "sentiment_analysis",
    ],
    "Multimodal Tasks": [
        "multimodal_synthesis",
        "cross_modal_verification",
        "audio_visual_alignment",
    ],
}

TASK_REMAP = {"motion_analysis": "compositional_reasoning"}

_BOOLEAN_PATTERN = re.compile(r'"criteria_met"\s*:\s*(true|false)', re.IGNORECASE)


@dataclass
class LongShOTOfficialEvalConfig:
    predictions_path: Path
    eval_path: Path
    score_path: Path
    summary_path: Path
    judge_model_name: str
    judge_model_path: str | None = None
    judge_device: str = "cuda:0"
    torch_dtype: str = "bfloat16"
    attn_implementation: str | None = None
    max_new_tokens: int = 96
    sample_limit: int | None = None


@dataclass
class LongShOTOfficialEvalResult:
    evaluated_samples: int
    evaluated_turns: int
    evaluated_criteria: int
    task_accuracies: dict[str, float]
    task_counts: dict[str, int]
    category_averages: dict[str, float]
    overall_accuracy: float


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_official_criterion_prompt(
    *,
    ground_truth_response: str,
    model_response: str,
    criterion_description: str,
) -> str:
    return f"""You are an expert evaluator specializing in video content analysis and multimodal understanding.

Your task is to evaluate the Model Response against the **single evaluation criterion** provided, using the Ground Truth Response as a reference.

Ground Truth Response:
{ground_truth_response}

Model Response:
{model_response}

Evaluation Criterion:
{criterion_description}

Instructions:
- Evaluate ONLY the provided criterion in this assessment.
- Compare the Model Response to the Ground Truth Response and determine if the criterion is satisfied.
- If the Model Response satisfies the criterion, set "criteria_met" to true; otherwise, set it to false.
- Focus on video content understanding, temporal relationships, and multimodal analysis. /no_think

Return strict JSON exactly like {{"criteria_met": true}} or {{"criteria_met": false}}."""


def parse_criteria_met(response_text: str) -> bool:
    response_text = response_text.strip()
    try:
        parsed = json.loads(response_text)
        if isinstance(parsed, dict) and isinstance(parsed.get("criteria_met"), bool):
            return parsed["criteria_met"]
    except json.JSONDecodeError:
        pass

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if fenced_match:
        try:
            parsed = json.loads(fenced_match.group(1))
            if isinstance(parsed.get("criteria_met"), bool):
                return parsed["criteria_met"]
        except json.JSONDecodeError:
            pass

    bool_match = _BOOLEAN_PATTERN.search(response_text)
    if bool_match:
        return bool_match.group(1).lower() == "true"

    lowered = response_text.lower()
    if "criteria_met" not in lowered:
        raise ValueError(f"Judge response missing criteria_met field: {response_text[:200]}")
    if "true" in lowered:
        return True
    if "false" in lowered:
        return False
    raise ValueError(f"Could not parse judge response: {response_text[:200]}")


def build_local_judge(
    *,
    model_name: str,
    model_path: str | None = None,
    device: str = "cuda:0",
    torch_dtype: str = "bfloat16",
    attn_implementation: str | None = None,
    max_new_tokens: int = 96,
) -> BaseLM:
    return TransformersClient(
        model_name=model_name,
        model_path=model_path,
        device=device,
        device_map=device,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )


def evaluate_predictions_official_style(
    config: LongShOTOfficialEvalConfig,
    *,
    judge: BaseLM | None = None,
) -> LongShOTOfficialEvalResult:
    predictions = load_jsonl(config.predictions_path)
    if config.sample_limit is not None:
        predictions = predictions[: config.sample_limit]

    judge_client = judge or build_local_judge(
        model_name=config.judge_model_name,
        model_path=config.judge_model_path,
        device=config.judge_device,
        torch_dtype=config.torch_dtype,
        attn_implementation=config.attn_implementation,
        max_new_tokens=config.max_new_tokens,
    )

    completed_ids = set()
    if config.eval_path.exists():
        completed_ids = {row.get("sample_id") for row in load_jsonl(config.eval_path)}

    evaluated_samples = 0
    evaluated_turns = 0
    evaluated_criteria = 0

    remaining_samples = [sample for sample in predictions if sample.get("sample_id") not in completed_ids]

    for index, sample in enumerate(remaining_samples, start=1):
        sample_id = sample.get("sample_id")
        print(
            f"[official-eval] {index}/{len(remaining_samples)} "
            f"sample_id={sample_id} task={sample.get('task')}"
        )

        for turn in sample.get("conversations", []):
            if turn.get("role") != "assistant" or "candidate_response" not in turn:
                continue
            evaluated_turns += 1
            for criterion in turn.get("criteria", []):
                prompt = build_official_criterion_prompt(
                    ground_truth_response=str(turn.get("content", "")),
                    model_response=str(turn.get("candidate_response", "")),
                    criterion_description=str(criterion.get("description", "")),
                )
                completion = ""
                last_error: Exception | None = None
                for _attempt in range(3):
                    try:
                        completion = judge_client.completion(prompt)
                        criterion["criteria_met"] = parse_criteria_met(completion)
                        last_error = None
                        break
                    except Exception as exc:
                        last_error = exc
                if last_error is not None:
                    criterion["criteria_met"] = None
                    criterion["evaluation_error"] = str(last_error)
                    criterion["evaluation_raw"] = completion[:1000]
                criterion["evaluation_model"] = config.judge_model_name
                evaluated_criteria += 1

        append_jsonl(config.eval_path, sample)
        evaluated_samples += 1

    summary = calculate_official_scores(load_jsonl(config.eval_path))
    write_score_report(
        score_path=config.score_path,
        model_name=config.predictions_path.stem,
        summary=summary,
    )
    write_summary_json(
        summary_path=config.summary_path,
        summary=summary,
        extra={
            "predictions_path": str(config.predictions_path),
            "eval_path": str(config.eval_path),
            "judge_model_name": config.judge_model_name,
            "judge_model_path": config.judge_model_path,
            "judge_device": config.judge_device,
            "evaluated_samples_this_run": evaluated_samples,
            "evaluated_turns_this_run": evaluated_turns,
            "evaluated_criteria_this_run": evaluated_criteria,
        },
    )
    return LongShOTOfficialEvalResult(
        evaluated_samples=evaluated_samples,
        evaluated_turns=evaluated_turns,
        evaluated_criteria=evaluated_criteria,
        task_accuracies=summary["task_accuracies"],
        task_counts=summary["task_counts"],
        category_averages=summary["category_averages"],
        overall_accuracy=summary["overall_accuracy"],
    )


def calculate_official_scores(eval_results: list[dict[str, Any]]) -> dict[str, Any]:
    task_performance: dict[str, dict[str, float | int]] = {}

    for result in eval_results:
        task_type = TASK_REMAP.get(result.get("task", "unknown_task"), result.get("task", "unknown_task"))
        performance = task_performance.setdefault(
            task_type,
            {"score_obtained": 0.0, "score_total": 0.0, "count": 0},
        )
        obtained_score = 0.0
        max_score = 0.0

        for turn in result.get("conversations", []):
            if turn.get("role") != "assistant":
                continue
            for criterion in turn.get("criteria", []):
                weight = float(criterion.get("weight", 0))
                if criterion.get("criteria_met") and not criterion.get("is_penalty"):
                    obtained_score += weight
                if weight > 0:
                    max_score += weight

        performance["score_obtained"] += obtained_score
        performance["score_total"] += max_score
        performance["count"] += 1

    task_accuracies: dict[str, float] = {}
    task_counts: dict[str, int] = {}
    category_averages: dict[str, float] = {}

    for task_type, performance in task_performance.items():
        task_counts[task_type] = int(performance["count"])
        if performance["score_total"] > 0:
            task_accuracies[task_type] = float(performance["score_obtained"]) / float(
                performance["score_total"]
            )
        else:
            task_accuracies[task_type] = 0.0

    for category, tasks in TASK_CATEGORIES.items():
        values = [task_accuracies[task] for task in tasks if task in task_accuracies]
        if values:
            category_averages[category] = sum(values) / len(values)

    overall_accuracy = (
        sum(category_averages.values()) / len(category_averages) if category_averages else 0.0
    )
    return {
        "task_accuracies": task_accuracies,
        "task_counts": task_counts,
        "category_averages": category_averages,
        "overall_accuracy": overall_accuracy,
    }


def write_score_report(*, score_path: Path, model_name: str, summary: dict[str, Any]) -> None:
    score_path.parent.mkdir(parents=True, exist_ok=True)
    with score_path.open("w", encoding="utf-8") as handle:
        handle.write("=" * 60 + "\n")
        handle.write("  LongShOT Bench Evaluation Results\n")
        handle.write("=" * 60 + "\n\n")
        handle.write(f"  Model:  {model_name}\n")
        handle.write(f"  Date:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for category, tasks in TASK_CATEGORIES.items():
            handle.write("-" * 60 + "\n")
            handle.write(f"  {category}\n")
            handle.write("-" * 60 + "\n")
            values: list[float] = []
            count_total = 0
            for task in tasks:
                if task not in summary["task_accuracies"]:
                    continue
                accuracy = summary["task_accuracies"][task]
                count = summary["task_counts"].get(task, 0)
                handle.write(f"    {task:<30} {count:>4}  {accuracy * 100:6.2f}%\n")
                values.append(accuracy)
                count_total += count
            if values:
                handle.write(
                    f"    {'Category Average':<30} {count_total:>4}  "
                    f"{(sum(values) / len(values)) * 100:6.2f}%\n"
                )
            handle.write("\n")

        total_samples = sum(summary["task_counts"].values())
        handle.write("=" * 60 + "\n")
        handle.write(
            f"  OVERALL ACCURACY:             {total_samples:>4}  "
            f"{summary['overall_accuracy'] * 100:6.2f}%\n"
        )
        handle.write("=" * 60 + "\n")


def write_summary_json(
    *,
    summary_path: Path,
    summary: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    payload = dict(summary)
    if extra:
        payload.update(extra)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
