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
    run_dir_a = resolve_run_dir(args.run_a, run_a)
    run_dir_b = resolve_run_dir(args.run_b, run_b)
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
        resources_a=collect_resource_summary(run_dir_a, run_a),
        resources_b=collect_resource_summary(run_dir_b, run_b),
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


def resolve_run_dir(path_text: str, predictions_path: Path) -> Path:
    path = Path(path_text)
    if path.is_dir():
        return path
    return predictions_path.parent


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

    existing_eval_matches = eval_path.exists() and eval_matches_predictions(
        eval_path=eval_path,
        predictions_path=predictions_path,
        sample_limit=args.sample_limit,
    )
    if args.reuse_existing_eval and existing_eval_matches:
        summary = calculate_official_scores(load_jsonl(eval_path))
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    if eval_path.exists():
        reason = "does not match current predictions" if args.reuse_existing_eval else "fresh eval requested"
        print(f"[compare] removing cached eval for {label}: {reason}")
        for path in (eval_path, score_path, summary_path):
            if path.exists():
                path.unlink()

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


def eval_matches_predictions(
    *,
    eval_path: Path,
    predictions_path: Path,
    sample_limit: int | None,
) -> bool:
    eval_rows = load_jsonl(eval_path)
    prediction_rows = load_jsonl(predictions_path)
    if sample_limit is not None:
        prediction_rows = prediction_rows[:sample_limit]
    if len(eval_rows) != len(prediction_rows):
        return False
    eval_by_id = {str(row.get("sample_id")): candidate_responses(row) for row in eval_rows}
    prediction_by_id = {
        str(row.get("sample_id")): candidate_responses(row) for row in prediction_rows
    }
    return eval_by_id == prediction_by_id


def candidate_responses(row: dict[str, Any]) -> list[str]:
    return [
        str(turn.get("candidate_response", ""))
        for turn in row.get("conversations", [])
        if turn.get("role") == "assistant" and "candidate_response" in turn
    ]


def build_comparison(
    *,
    label_a: str,
    label_b: str,
    predictions_a: Path,
    predictions_b: Path,
    summary_a: dict[str, Any],
    summary_b: dict[str, Any],
    resources_a: dict[str, Any],
    resources_b: dict[str, Any],
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
        "resources": compare_resources(resources_a, resources_b),
        "resources_a": resources_a,
        "resources_b": resources_b,
        "summary_a": summary_a,
        "summary_b": summary_b,
    }


def collect_resource_summary(run_dir: Path, predictions_path: Path) -> dict[str, Any]:
    result_rows = load_jsonl(predictions_path)
    trace_paths = sorted((run_dir / "traces").glob("*.json"))
    memory_paths = sorted((run_dir / "memories").glob("*.json"))
    artifact_dirs = [path for path in sorted((run_dir / "artifacts").glob("*")) if path.is_dir()]

    trace_stats = collect_trace_stats(trace_paths)
    artifact_stats = collect_artifact_stats(artifact_dirs)
    memory_stats = collect_memory_stats(memory_paths)
    size_stats = {
        "run_dir_bytes": directory_size(run_dir),
        "artifacts_bytes": directory_size(run_dir / "artifacts"),
        "memories_bytes": directory_size(run_dir / "memories"),
        "traces_bytes": directory_size(run_dir / "traces"),
        "results_bytes": predictions_path.stat().st_size if predictions_path.exists() else 0,
    }
    return {
        "run_dir": str(run_dir),
        "predictions_path": str(predictions_path),
        "samples": len(result_rows),
        "assistant_turns": count_candidate_turns(result_rows),
        **trace_stats,
        **artifact_stats,
        **memory_stats,
        **size_stats,
    }


def collect_trace_stats(trace_paths: list[Path]) -> dict[str, Any]:
    total_execution_time = 0.0
    total_steps = 0
    total_tool_calls = 0
    total_controller_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0
    action_counts: dict[str, int] = {}
    opened_clips = 0

    for trace_path in trace_paths:
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
        total_execution_time += float(payload.get("execution_time", 0.0))
        trace = payload.get("trace", [])
        total_steps += len(trace)
        budget = payload.get("state", {}).get("budget", {})
        total_tool_calls += int(budget.get("tool_calls_used", 0))
        opened_clips += int(budget.get("clips_opened", 0))
        for step in trace:
            action_type = step.get("action", {}).get("action_type", "unknown")
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
        usage = payload.get("usage_summary", {}).get("model_usage_summaries", {})
        for model_usage in usage.values():
            total_controller_calls += int(model_usage.get("total_calls", 0))
            total_input_tokens += int(model_usage.get("total_input_tokens", 0))
            total_output_tokens += int(model_usage.get("total_output_tokens", 0))

    return {
        "trace_files": len(trace_paths),
        "controller_execution_time_seconds": total_execution_time,
        "controller_steps": total_steps,
        "tool_calls": total_tool_calls,
        "controller_lm_calls": total_controller_calls,
        "controller_input_tokens": total_input_tokens,
        "controller_output_tokens": total_output_tokens,
        "opened_clips": opened_clips,
        "action_counts": action_counts,
    }


def collect_artifact_stats(artifact_dirs: list[Path]) -> dict[str, Any]:
    speech_spans = 0
    visual_summaries = 0
    ocr_spans = 0
    audio_events = 0
    visual_summary_chars = 0
    visual_granularity_counts: dict[str, int] = {}
    visual_duration_seconds = 0.0
    metadata_modes: dict[str, int] = {}

    for artifact_dir in artifact_dirs:
        manifest_path = artifact_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            counts = manifest.get("counts", {})
            speech_spans += int(counts.get("speech_spans", 0))
            visual_summaries += int(counts.get("visual_summaries", 0))
            ocr_spans += int(counts.get("ocr_spans", 0))
            audio_events += int(counts.get("audio_events", 0))
            mode = manifest.get("metadata", {}).get("visual_span_mode")
            if mode:
                metadata_modes[mode] = metadata_modes.get(mode, 0) + 1
        for item in read_jsonl_if_exists(artifact_dir / "visual.jsonl"):
            summary = str(item.get("summary", ""))
            visual_summary_chars += len(summary)
            granularity = str(item.get("granularity", "unknown"))
            visual_granularity_counts[granularity] = visual_granularity_counts.get(granularity, 0) + 1
            span = item.get("time_span", {})
            visual_duration_seconds += float(span.get("end", 0.0)) - float(span.get("start", 0.0))

    return {
        "artifact_video_count": len(artifact_dirs),
        "speech_spans": speech_spans,
        "vlm_calls": visual_summaries,
        "visual_summaries": visual_summaries,
        "ocr_spans": ocr_spans,
        "audio_events": audio_events,
        "visual_summary_chars": visual_summary_chars,
        "visual_granularity_counts": visual_granularity_counts,
        "visual_duration_seconds": visual_duration_seconds,
        "visual_span_modes": metadata_modes,
    }


def collect_memory_stats(memory_paths: list[Path]) -> dict[str, Any]:
    node_count = 0
    visual_memory_chars = 0
    memory_modes: dict[str, int] = {}
    level_counts: dict[str, int] = {}

    for memory_path in memory_paths:
        payload = json.loads(memory_path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata", {})
        mode = metadata.get("visual_span_mode")
        if mode:
            memory_modes[mode] = memory_modes.get(mode, 0) + 1
        for node in payload.get("nodes", {}).values():
            node_count += 1
            level = str(node.get("level", "unknown"))
            level_counts[level] = level_counts.get(level, 0) + 1
            visual_memory_chars += len(str(node.get("visual_summary") or ""))

    return {
        "memory_files": len(memory_paths),
        "memory_nodes": node_count,
        "memory_visual_chars": visual_memory_chars,
        "memory_level_counts": level_counts,
        "memory_visual_span_modes": memory_modes,
    }


def compare_resources(resources_a: dict[str, Any], resources_b: dict[str, Any]) -> dict[str, Any]:
    numeric_keys = [
        "samples",
        "assistant_turns",
        "trace_files",
        "artifact_video_count",
        "vlm_calls",
        "speech_spans",
        "visual_summaries",
        "visual_summary_chars",
        "visual_duration_seconds",
        "memory_files",
        "memory_nodes",
        "memory_visual_chars",
        "controller_execution_time_seconds",
        "controller_steps",
        "tool_calls",
        "controller_lm_calls",
        "controller_input_tokens",
        "controller_output_tokens",
        "opened_clips",
        "run_dir_bytes",
        "artifacts_bytes",
        "memories_bytes",
        "traces_bytes",
        "results_bytes",
    ]
    return {
        key: resource_row(float(resources_a.get(key, 0)), float(resources_b.get(key, 0)))
        for key in numeric_keys
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


def resource_row(value_a: float, value_b: float) -> dict[str, float]:
    payload = row(value_a, value_b)
    payload["ratio_b_over_a"] = value_b / value_a if value_a else 0.0
    return payload


def count_candidate_turns(result_rows: list[dict[str, Any]]) -> int:
    count = 0
    for row_item in result_rows:
        for turn in row_item.get("conversations", []):
            if turn.get("role") == "assistant" and "candidate_response" in turn:
                count += 1
    return count


def read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return load_jsonl(path)


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


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
        "## Resource Usage",
        "",
        resource_table(comparison["resources"], label_a, label_b),
        "",
        "## Resource Details",
        "",
        details_block(label_a, comparison["resources_a"]),
        "",
        details_block(label_b, comparison["resources_b"]),
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


def resource_table(
    rows: dict[str, dict[str, Any]],
    label_a: str,
    label_b: str,
) -> str:
    display_rows = [
        ("samples", "Samples", "int"),
        ("assistant_turns", "Assistant Turns", "int"),
        ("vlm_calls", "VLM Calls / Visual Summaries", "int"),
        ("visual_duration_seconds", "Visual Seconds Processed", "seconds"),
        ("visual_summary_chars", "Visual Artifact Chars", "int"),
        ("memory_visual_chars", "Memory Visual Chars", "int"),
        ("run_dir_bytes", "Run Folder Size", "bytes"),
        ("artifacts_bytes", "Artifacts Size", "bytes"),
        ("memories_bytes", "Memories Size", "bytes"),
        ("traces_bytes", "Traces Size", "bytes"),
        ("controller_execution_time_seconds", "Controller Trace Time", "seconds"),
        ("controller_steps", "Controller Steps", "int"),
        ("tool_calls", "Tool Calls", "int"),
        ("controller_lm_calls", "Controller LM Calls", "int"),
        ("controller_input_tokens", "Controller Input Tokens", "int"),
        ("controller_output_tokens", "Controller Output Tokens", "int"),
        ("opened_clips", "Opened Clips", "int"),
    ]
    lines = [
        f"| Metric | {label_a} | {label_b} | Delta | Ratio B/A |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key, label, value_type in display_rows:
        values = rows.get(key, resource_row(0.0, 0.0))
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    format_resource(values["run_a"], value_type),
                    format_resource(values["run_b"], value_type),
                    format_resource_delta(values["delta_b_minus_a"], value_type),
                    f"{values['ratio_b_over_a']:.2f}x",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def details_block(label: str, resources: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"### {label}",
            "",
            f"- Visual span modes: `{resources.get('visual_span_modes', {})}`",
            f"- Visual granularity counts: `{resources.get('visual_granularity_counts', {})}`",
            f"- Memory visual span modes: `{resources.get('memory_visual_span_modes', {})}`",
            f"- Memory level counts: `{resources.get('memory_level_counts', {})}`",
            f"- Action counts: `{resources.get('action_counts', {})}`",
        ]
    )


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_delta(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.2f} pp"


def format_resource(value: float, value_type: str) -> str:
    if value_type == "bytes":
        return format_bytes(value)
    if value_type == "seconds":
        return f"{value:.2f}s"
    return str(int(value))


def format_resource_delta(value: float, value_type: str) -> str:
    sign = "+" if value >= 0 else ""
    if value_type == "bytes":
        return f"{sign}{format_bytes(value)}"
    if value_type == "seconds":
        return f"{sign}{value:.2f}s"
    return f"{sign}{int(value)}"


def format_bytes(value: float) -> str:
    sign = "-" if value < 0 else ""
    absolute = abs(value)
    units = ["B", "KB", "MB", "GB"]
    unit_index = 0
    while absolute >= 1024 and unit_index < len(units) - 1:
        absolute /= 1024
        unit_index += 1
    if unit_index == 0:
        return f"{sign}{absolute:.0f} {units[unit_index]}"
    return f"{sign}{absolute:.1f} {units[unit_index]}"


def sanitize_label(label: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip())
    return normalized or "run"


if __name__ == "__main__":
    raise SystemExit(main())
