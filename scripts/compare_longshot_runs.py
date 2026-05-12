#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlm.video.longshot_official_eval import (  # noqa: E402
    LongShOTOfficialEvalConfig,
    calculate_official_scores,
    evaluate_predictions_official_style,
    load_jsonl,
)


def main() -> int:
    args = parse_args()
    run_a = resolve_predictions_path(args.run_a)
    run_b = resolve_predictions_path(args.run_b)
    label_a = args.label_a or default_label(args.run_a)
    label_b = args.label_b or default_label(args.run_b)

    eval_root = Path(args.eval_root)
    eval_root.mkdir(parents=True, exist_ok=True)

    summary_a = evaluate_or_load_summary(
        label=label_a,
        predictions_path=run_a,
        eval_root=eval_root,
        args=args,
    )
    summary_b = evaluate_or_load_summary(
        label=label_b,
        predictions_path=run_b,
        eval_root=eval_root,
        args=args,
    )

    comparison = build_comparison(
        label_a=label_a,
        label_b=label_b,
        predictions_a=run_a,
        predictions_b=run_b,
        summary_a=summary_a,
        summary_b=summary_b,
    )
    write_outputs(eval_root, comparison)
    print_report(comparison)
    print(f"\nWrote comparison files to {eval_root}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate and compare two LongShOTBench prediction runs using the "
            "repo's official-style weighted rubric metric."
        )
    )
    parser.add_argument("--run-a", required=True, help="Run folder or results.jsonl for baseline")
    parser.add_argument("--run-b", required=True, help="Run folder or results.jsonl for comparison")
    parser.add_argument("--label-a")
    parser.add_argument("--label-b")
    parser.add_argument("--eval-root", default="output/longshot_eval_compare")
    parser.add_argument("--judge-repo", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--judge-model-path")
    parser.add_argument("--judge-device", default="mps")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--attn-implementation")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument(
        "--reuse-existing-eval",
        action="store_true",
        help="Do not call a judge if eval.jsonl already exists for a run.",
    )
    return parser.parse_args()


def resolve_predictions_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_dir():
        path = path / "results.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Could not find predictions file: {path}")
    return path


def default_label(path_text: str) -> str:
    path = Path(path_text)
    return path.name if path.is_dir() else path.parent.name or path.stem


def evaluate_or_load_summary(
    *,
    label: str,
    predictions_path: Path,
    eval_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    run_dir = eval_root / sanitize_label(label)
    eval_path = run_dir / "eval.jsonl"
    score_path = run_dir / "score.txt"
    summary_path = run_dir / "summary.json"

    if args.reuse_existing_eval and eval_path.exists():
        summary = calculate_official_scores(load_jsonl(eval_path))
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    config = LongShOTOfficialEvalConfig(
        predictions_path=predictions_path,
        eval_path=eval_path,
        score_path=score_path,
        summary_path=summary_path,
        judge_model_name=args.judge_repo,
        judge_model_path=args.judge_model_path,
        judge_device=args.judge_device,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        max_new_tokens=args.max_new_tokens,
        sample_limit=args.sample_limit,
    )
    evaluate_predictions_official_style(config)
    return json.loads(summary_path.read_text(encoding="utf-8"))


def build_comparison(
    *,
    label_a: str,
    label_b: str,
    predictions_a: Path,
    predictions_b: Path,
    summary_a: dict[str, Any],
    summary_b: dict[str, Any],
) -> dict[str, Any]:
    return {
        "label_a": label_a,
        "label_b": label_b,
        "predictions_a": str(predictions_a),
        "predictions_b": str(predictions_b),
        "overall": row(
            summary_a.get("overall_accuracy", 0.0),
            summary_b.get("overall_accuracy", 0.0),
        ),
        "categories": compare_mapping(
            summary_a.get("category_averages", {}),
            summary_b.get("category_averages", {}),
        ),
        "tasks": compare_mapping(
            summary_a.get("task_accuracies", {}),
            summary_b.get("task_accuracies", {}),
            counts_a=summary_a.get("task_counts", {}),
            counts_b=summary_b.get("task_counts", {}),
        ),
        "summary_a": summary_a,
        "summary_b": summary_b,
    }


def compare_mapping(
    values_a: dict[str, float],
    values_b: dict[str, float],
    *,
    counts_a: dict[str, int] | None = None,
    counts_b: dict[str, int] | None = None,
) -> dict[str, dict[str, Any]]:
    output = {}
    for key in sorted(set(values_a) | set(values_b)):
        output[key] = row(
            float(values_a.get(key, 0.0)),
            float(values_b.get(key, 0.0)),
            count_a=int((counts_a or {}).get(key, 0)),
            count_b=int((counts_b or {}).get(key, 0)),
        )
    return output


def row(value_a: float, value_b: float, *, count_a: int | None = None, count_b: int | None = None):
    payload: dict[str, Any] = {
        "run_a": value_a,
        "run_b": value_b,
        "delta_b_minus_a": value_b - value_a,
    }
    if count_a is not None:
        payload["count_a"] = count_a
    if count_b is not None:
        payload["count_b"] = count_b
    return payload


def write_outputs(eval_root: Path, comparison: dict[str, Any]) -> None:
    (eval_root / "comparison.json").write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8",
    )
    (eval_root / "comparison.md").write_text(
        render_markdown(comparison),
        encoding="utf-8",
    )


def print_report(comparison: dict[str, Any]) -> None:
    print(render_markdown(comparison))


def render_markdown(comparison: dict[str, Any]) -> str:
    label_a = comparison["label_a"]
    label_b = comparison["label_b"]
    lines = [
        "# LongShOT Official-Style Comparison",
        "",
        f"- Run A: `{label_a}`",
        f"- Run B: `{label_b}`",
        "",
        "## Overall",
        "",
        metric_table({"overall_accuracy": comparison["overall"]}, label_a, label_b),
        "",
        "## Categories",
        "",
        metric_table(comparison["categories"], label_a, label_b),
        "",
        "## Tasks",
        "",
        metric_table(comparison["tasks"], label_a, label_b, include_counts=True),
        "",
    ]
    return "\n".join(lines)


def metric_table(
    rows: dict[str, dict[str, Any]],
    label_a: str,
    label_b: str,
    *,
    include_counts: bool = False,
) -> str:
    header = ["Metric", label_a, label_b, "Delta"]
    if include_counts:
        header.extend(["N A", "N B"])
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for name, values in rows.items():
        cells = [
            name,
            format_percent(values["run_a"]),
            format_percent(values["run_b"]),
            format_delta(values["delta_b_minus_a"]),
        ]
        if include_counts:
            cells.extend([str(values.get("count_a", 0)), str(values.get("count_b", 0))])
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_delta(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.2f} pp"


def sanitize_label(label: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip())
    return normalized or "run"


if __name__ == "__main__":
    raise SystemExit(main())
