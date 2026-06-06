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
    "Agentic Tasks": [],
}

TASK_REMAP = {"motion_analysis": "compositional_reasoning"}
AGENTIC_TASK_KEYWORDS = (
    "agent",
    "api",
    "browser",
    "computer",
    "gui",
    "navigation",
    "planning",
    "search",
    "tool",
)
OTHER_TASK_CATEGORY = "Other Tasks"

_BOOLEAN_PATTERN = re.compile(r'"criteria_met"\s*:\s*(true|false)', re.IGNORECASE)
ANSWER_ONLY_CRITERION_DESCRIPTION = (
    "The model's final answer is semantically correct for the user question when compared "
    "with the ground truth answer."
)
OFFICIAL_LONGSHOT_JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator specializing in video content analysis and multimodal understanding.

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

- Focus on video content understanding, temporal relationships, and multimodal analysis. /no_think"""


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
    user_question: str | None = None,
    ground_truth_response: str,
    model_response: str,
    criterion_description: str,
) -> str:
    del user_question
    return OFFICIAL_LONGSHOT_JUDGE_PROMPT_TEMPLATE.format(
        ground_truth_response=ground_truth_response,
        model_response=model_response,
        criterion_description=criterion_description,
    )


def build_answer_only_prompt(
    *,
    user_question: str | None = None,
    ground_truth_response: str,
    model_response: str,
) -> str:
    question_block = f"\nUser Question:\n{user_question}\n" if user_question else ""
    return f"""You are an expert evaluator specializing in video question answering.

Judge only whether the Model Final Answer is semantically correct.
Use the Ground Truth Answer as the reference and the User Question to resolve ambiguity.
{question_block}

Ground Truth Answer:
{ground_truth_response}

Model Final Answer:
{model_response}

Instructions:
- Evaluate only final-answer correctness.
- Ignore whether the model used the right tool, followed a tool-call format, cited
  evidence, or satisfied process-specific criteria.
- Accept concise answers when they directly satisfy the user question, including
  equivalent numeric values, boolean values, symbols, code expressions, and short
  answer spans contained in or implied by the ground-truth answer.
- Accept paraphrases and answers with harmless extra wording.
- Mark false if the final answer contradicts, omits, or materially weakens the
  ground-truth answer. /no_think

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


def judge_boolean_prompt(
    *,
    judge_client: BaseLM,
    prompt: str,
) -> tuple[bool | None, str | None, str]:
    completion = ""
    last_error: Exception | None = None
    for _attempt in range(3):
        try:
            completion = judge_client.completion(prompt)
            return parse_criteria_met(completion), None, completion
        except Exception as exc:
            last_error = exc
    assert last_error is not None
    return None, str(last_error), completion[:1000]


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
        existing_rows = load_jsonl(config.eval_path)
        incompatible_ids = [
            row.get("sample_id")
            for row in existing_rows
            if row.get("evaluation_mode") == "answer_only"
        ]
        if incompatible_ids:
            raise ValueError(
                "Existing eval JSONL is answer-only. Use a different --eval-output "
                f"or remove the old file first: {config.eval_path}"
            )
        completed_ids = {row.get("sample_id") for row in existing_rows}

    evaluated_samples = 0
    evaluated_turns = 0
    evaluated_criteria = 0

    remaining_samples = [
        sample for sample in predictions if sample.get("sample_id") not in completed_ids
    ]

    for index, sample in enumerate(remaining_samples, start=1):
        sample_id = sample.get("sample_id")
        print(
            f"[official-eval] {index}/{len(remaining_samples)} "
            f"sample_id={sample_id} task={sample.get('task')}"
        )
        sample["evaluation_mode"] = "official_criteria"

        current_user_question = ""
        for turn in sample.get("conversations", []):
            if turn.get("role") == "user":
                current_user_question = str(turn.get("content", ""))
                continue
            if turn.get("role") != "assistant" or "candidate_response" not in turn:
                continue
            evaluated_turns += 1
            for criterion in turn.get("criteria", []):
                prompt = build_official_criterion_prompt(
                    user_question=current_user_question,
                    ground_truth_response=str(turn.get("content", "")),
                    model_response=str(turn.get("candidate_response", "")),
                    criterion_description=str(criterion.get("description", "")),
                )
                criteria_met, evaluation_error, completion = judge_boolean_prompt(
                    judge_client=judge_client,
                    prompt=prompt,
                )
                criterion["criteria_met"] = criteria_met
                if evaluation_error is not None:
                    criterion["evaluation_error"] = evaluation_error
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


def evaluate_predictions_answer_only(
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
        existing_rows = load_jsonl(config.eval_path)
        incompatible_ids = [
            row.get("sample_id")
            for row in existing_rows
            if row.get("evaluation_mode") != "answer_only"
        ]
        if incompatible_ids:
            raise ValueError(
                "Existing eval JSONL is not answer-only. Use a different --eval-output "
                f"or remove the old file first: {config.eval_path}"
            )
        completed_ids = {row.get("sample_id") for row in existing_rows}

    evaluated_samples = 0
    evaluated_turns = 0
    evaluated_criteria = 0

    remaining_samples = [
        sample for sample in predictions if sample.get("sample_id") not in completed_ids
    ]

    for index, sample in enumerate(remaining_samples, start=1):
        sample_id = sample.get("sample_id")
        print(
            f"[answer-only-eval] {index}/{len(remaining_samples)} "
            f"sample_id={sample_id} task={sample.get('task')}"
        )
        sample["evaluation_mode"] = "answer_only"

        current_user_question = ""
        for turn in sample.get("conversations", []):
            if turn.get("role") == "user":
                current_user_question = str(turn.get("content", ""))
                continue
            if turn.get("role") != "assistant":
                continue
            if "candidate_response" not in turn:
                turn["criteria"] = []
                continue

            evaluated_turns += 1
            ignored_criteria_count = len(turn.get("criteria", []))
            criterion: dict[str, Any] = {
                "name": "answer_correctness",
                "description": ANSWER_ONLY_CRITERION_DESCRIPTION,
                "is_penalty": False,
                "weight": 1.0,
                "evaluation_mode": "answer_only",
            }
            prompt = build_answer_only_prompt(
                user_question=current_user_question,
                ground_truth_response=str(turn.get("content", "")),
                model_response=str(turn.get("candidate_response", "")),
            )
            criteria_met, evaluation_error, completion = judge_boolean_prompt(
                judge_client=judge_client,
                prompt=prompt,
            )
            criterion["criteria_met"] = criteria_met
            if evaluation_error is not None:
                criterion["evaluation_error"] = evaluation_error
                criterion["evaluation_raw"] = completion[:1000]
            criterion["evaluation_model"] = config.judge_model_name
            turn["answer_only_ignored_criteria_count"] = ignored_criteria_count
            turn["criteria"] = [criterion]
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
            "evaluation_mode": "answer_only",
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
    task_categories: dict[str, str] = {}

    for result in eval_results:
        task_type = normalize_task_name(result.get("task", "unknown_task"))
        task_categories[task_type] = task_category(task_type)
        performance = task_performance.setdefault(
            task_type,
            {"score_obtained": 0.0, "score_total": 0.0, "count": 0},
        )
        criteria: list[dict[str, Any]] = []

        for turn in result.get("conversations", []):
            if turn.get("role") != "assistant":
                continue
            criteria.extend(turn.get("criteria", []))

        performance["score_obtained"] += normalized_criteria_score(criteria)
        performance["score_total"] += 1.0
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

    category_tasks: dict[str, list[str]] = {}
    for task_type in sorted(task_accuracies):
        category_tasks.setdefault(
            task_categories.get(task_type, OTHER_TASK_CATEGORY),
            [],
        ).append(task_type)

    ordered_categories = list(TASK_CATEGORIES)
    ordered_categories.extend(
        category for category in sorted(category_tasks) if category not in TASK_CATEGORIES
    )
    for category in ordered_categories:
        tasks = category_tasks.get(category, [])
        values = [task_accuracies[task] for task in tasks]
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
        "task_categories": task_categories,
        "category_tasks": category_tasks,
    }


def normalize_task_name(task_type: Any) -> str:
    task = str(task_type or "unknown_task")
    return TASK_REMAP.get(task, task)


def task_category(task_type: str) -> str:
    for category, tasks in TASK_CATEGORIES.items():
        if task_type in tasks:
            return category
    lowered = task_type.lower()
    if any(keyword in lowered for keyword in AGENTIC_TASK_KEYWORDS):
        return "Agentic Tasks"
    return OTHER_TASK_CATEGORY


def criterion_score_components(criteria: list[dict[str, Any]]) -> tuple[float, float]:
    obtained_score = 0.0
    max_score = 0.0
    for criterion in criteria:
        weight = float(criterion.get("weight", 0.0))
        weight_magnitude = abs(weight)
        if weight_magnitude == 0:
            continue

        is_penalty = bool(criterion.get("is_penalty")) or weight < 0
        criteria_met = criterion.get("criteria_met") is True
        if is_penalty:
            if criteria_met:
                obtained_score -= weight_magnitude
            continue

        max_score += weight_magnitude
        if criteria_met:
            obtained_score += weight_magnitude
    return obtained_score, max_score


def normalized_criteria_score(criteria: list[dict[str, Any]]) -> float:
    obtained_score, max_score = criterion_score_components(criteria)
    if max_score <= 0:
        return 0.0
    return max(0.0, min(1.0, obtained_score / max_score))


def write_score_report(*, score_path: Path, model_name: str, summary: dict[str, Any]) -> None:
    score_path.parent.mkdir(parents=True, exist_ok=True)
    with score_path.open("w", encoding="utf-8") as handle:
        handle.write("=" * 60 + "\n")
        handle.write("  LongShOT Bench Evaluation Results\n")
        handle.write("=" * 60 + "\n\n")
        handle.write(f"  Model:  {model_name}\n")
        handle.write(f"  Date:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        category_tasks = summary.get("category_tasks", {})
        ordered_categories = list(TASK_CATEGORIES)
        ordered_categories.extend(
            category for category in sorted(category_tasks) if category not in TASK_CATEGORIES
        )
        for category in ordered_categories:
            tasks = category_tasks.get(category, [])
            if not tasks:
                continue
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
