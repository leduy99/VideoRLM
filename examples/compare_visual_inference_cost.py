from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class CostConfig:
    image_tokens_per_frame: float | None
    prompt_tokens_per_call: float
    input_price_per_million_tokens: float | None
    fixed_cost_per_call: float


@dataclass
class MethodReport:
    name: str
    span_policy: str
    span_count: int
    vlm_calls: int
    frames_sent_to_vlm: int
    dense_candidate_frames: int
    effective_vlm_fps: float
    effective_candidate_fps: float
    estimated_input_tokens: float | None
    estimated_cost: float | None
    seconds_elapsed: float | None = None
    measured_span_count: int | None = None
    selected_frames_per_call: dict[str, float] | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare visual preprocessing inference cost between the current VideoRLM "
            "sampling policy and a larger-span PiToMe-style policy."
        )
    )
    parser.add_argument("--video", help="Input video path. Required for --run-pitome.")
    parser.add_argument(
        "--duration-seconds",
        type=positive_float,
        help="Video duration. If omitted, --video is probed with ffprobe.",
    )
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--frame-width", type=positive_int, default=768)

    parser.add_argument("--original-scene-duration-seconds", type=positive_float, default=180.0)
    parser.add_argument("--original-clip-duration-seconds", type=positive_float, default=15.0)
    parser.add_argument("--original-frame-count", type=positive_int, default=3)

    parser.add_argument("--pitome-clip-duration-seconds", type=positive_float, default=60.0)
    parser.add_argument("--pitome-dense-frame-rate", type=positive_float, default=1.0)
    parser.add_argument("--pitome-uniform-frame-count", type=positive_int, default=8)
    parser.add_argument(
        "--pitome-estimated-selected-frames",
        type=positive_int,
        default=8,
        help="Frame budget to use for the cheap estimate without running image selection.",
    )
    parser.add_argument("--pitome-protect-ratio", type=bounded_ratio, default=0.15)
    parser.add_argument("--pitome-similarity-threshold", type=bounded_ratio, default=0.8)
    parser.add_argument("--pitome-embedding-size", type=positive_int, default=16)
    parser.add_argument(
        "--pitome-max-selected-frames",
        type=positive_int,
        help="Optional cap applied only in the measured PiToMe cost report.",
    )
    parser.add_argument(
        "--run-pitome",
        action="store_true",
        help="Actually extract dense frames and run PiToMe selection. No VLM calls are made.",
    )
    parser.add_argument(
        "--max-pitome-spans",
        type=positive_int,
        help="Limit measured PiToMe selection to the first N proposed spans.",
    )
    parser.add_argument(
        "--pitome-output-dir",
        help="Directory for extracted candidate frames. Uses a temporary directory by default.",
    )

    parser.add_argument(
        "--image-tokens-per-frame",
        type=positive_float,
        help="Optional model-specific token estimate for each image sent to the VLM.",
    )
    parser.add_argument("--prompt-tokens-per-call", type=non_negative_float, default=0.0)
    parser.add_argument(
        "--input-price-per-million-tokens",
        type=positive_float,
        help="Optional input token price used with --image-tokens-per-frame.",
    )
    parser.add_argument("--fixed-cost-per-call", type=non_negative_float, default=0.0)
    parser.add_argument("--json-output", help="Optional path to write the full report as JSON.")
    return parser


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive number, got {value}")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative number, got {value}")
    return parsed


def bounded_ratio(value: str) -> float:
    parsed = float(value)
    if parsed < 0 or parsed > 1:
        raise argparse.ArgumentTypeError(f"expected a ratio within [0, 1], got {value}")
    return parsed


def main() -> int:
    args = build_parser().parse_args()
    duration_seconds = resolve_duration(args)
    cost_config = CostConfig(
        image_tokens_per_frame=args.image_tokens_per_frame,
        prompt_tokens_per_call=args.prompt_tokens_per_call,
        input_price_per_million_tokens=args.input_price_per_million_tokens,
        fixed_cost_per_call=args.fixed_cost_per_call,
    )

    original = build_original_report(args, duration_seconds, cost_config)
    pitome_estimate = build_pitome_estimate_report(args, duration_seconds, cost_config)
    report: dict[str, Any] = {
        "video": str(Path(args.video).resolve()) if args.video else None,
        "duration_seconds": duration_seconds,
        "cost_config": asdict(cost_config),
        "original": asdict(original),
        "pitome_estimate": asdict(pitome_estimate),
        "estimate_delta": compare_reports(original, pitome_estimate),
    }

    if args.run_pitome:
        measured = build_pitome_measured_report(args, duration_seconds, cost_config)
        report["pitome_measured"] = asdict(measured)
        report["measured_delta"] = compare_reports(original, measured)

    print_human_summary(report)
    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report to {output_path}")
    return 0


def resolve_duration(args: argparse.Namespace) -> float:
    if args.duration_seconds is not None:
        return args.duration_seconds
    if not args.video:
        raise ValueError("Pass either --duration-seconds or --video.")

    from rlm.video.media import probe_media_duration

    return probe_media_duration(args.video, ffprobe_bin=args.ffprobe_bin)


def build_original_report(
    args: argparse.Namespace,
    duration_seconds: float,
    cost_config: CostConfig,
) -> MethodReport:
    scene_spans = subdivide(duration_seconds, args.original_scene_duration_seconds)
    clip_spans = subdivide(duration_seconds, args.original_clip_duration_seconds)
    span_count = len(scene_spans) + len(clip_spans)
    frames_sent = span_count * args.original_frame_count
    return make_report(
        name="original",
        span_policy=(
            f"{len(scene_spans)} scene spans @ {args.original_scene_duration_seconds:g}s + "
            f"{len(clip_spans)} clip spans @ {args.original_clip_duration_seconds:g}s"
        ),
        span_count=span_count,
        duration_seconds=duration_seconds,
        frames_sent_to_vlm=frames_sent,
        dense_candidate_frames=frames_sent,
        cost_config=cost_config,
    )


def build_pitome_estimate_report(
    args: argparse.Namespace,
    duration_seconds: float,
    cost_config: CostConfig,
) -> MethodReport:
    from rlm.video.media import sample_span_timestamps_by_rate

    spans = subdivide(duration_seconds, args.pitome_clip_duration_seconds)
    dense_candidate_frames = sum(
        len(
            sample_span_timestamps_by_rate(
                span,
                args.pitome_dense_frame_rate,
                min_frames=args.pitome_uniform_frame_count,
            )
        )
        for span in spans
    )
    frames_sent = len(spans) * args.pitome_estimated_selected_frames
    return make_report(
        name="pitome_estimate",
        span_policy=(
            f"{len(spans)} clip spans @ {args.pitome_clip_duration_seconds:g}s, "
            f"{args.pitome_dense_frame_rate:g} dense fps, "
            f"estimate {args.pitome_estimated_selected_frames} selected frames/span"
        ),
        span_count=len(spans),
        duration_seconds=duration_seconds,
        frames_sent_to_vlm=frames_sent,
        dense_candidate_frames=dense_candidate_frames,
        cost_config=cost_config,
    )


def build_pitome_measured_report(
    args: argparse.Namespace,
    duration_seconds: float,
    cost_config: CostConfig,
) -> MethodReport:
    if not args.video:
        raise ValueError("--run-pitome requires --video.")

    spans = subdivide(duration_seconds, args.pitome_clip_duration_seconds)
    measured_spans = spans[: args.max_pitome_spans] if args.max_pitome_spans else spans
    selected_counts: list[int] = []
    dense_counts: list[int] = []
    started_at = time.perf_counter()

    if args.pitome_output_dir:
        output_root = Path(args.pitome_output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        for index, span in enumerate(measured_spans, start=1):
            run_pitome_for_span(args, span, output_root / f"span_{index:04d}", selected_counts, dense_counts)
    else:
        with tempfile.TemporaryDirectory(prefix="videorlm_pitome_cost_") as temp_dir:
            output_root = Path(temp_dir)
            for index, span in enumerate(measured_spans, start=1):
                run_pitome_for_span(
                    args,
                    span,
                    output_root / f"span_{index:04d}",
                    selected_counts,
                    dense_counts,
                )

    seconds_elapsed = time.perf_counter() - started_at
    if not selected_counts:
        raise ValueError("No PiToMe spans were measured.")

    scale = len(spans) / len(measured_spans)
    frames_sent = round(sum(selected_counts) * scale)
    dense_candidate_frames = round(sum(dense_counts) * scale)
    span_policy_prefix = (
        f"{len(measured_spans)} measured of {len(spans)}"
        if len(measured_spans) != len(spans)
        else f"{len(spans)} measured"
    )
    report = make_report(
        name="pitome_measured",
        span_policy=(
            f"{span_policy_prefix} clip spans @ {args.pitome_clip_duration_seconds:g}s, "
            f"{args.pitome_dense_frame_rate:g} dense fps"
        ),
        span_count=len(spans),
        duration_seconds=duration_seconds,
        frames_sent_to_vlm=frames_sent,
        dense_candidate_frames=dense_candidate_frames,
        cost_config=cost_config,
        seconds_elapsed=seconds_elapsed,
    )
    report.measured_span_count = len(measured_spans)
    report.selected_frames_per_call = summarize_counts(selected_counts)
    return report


def run_pitome_for_span(
    args: argparse.Namespace,
    span: Any,
    output_dir: Path,
    selected_counts: list[int],
    dense_counts: list[int],
) -> None:
    from rlm.video.pitome import select_visual_frames_for_span

    selection = select_visual_frames_for_span(
        media_path=args.video,
        span=span,
        strategy="pitome",
        uniform_frame_count=args.pitome_uniform_frame_count,
        dense_frame_rate=args.pitome_dense_frame_rate,
        ffmpeg_bin=args.ffmpeg_bin,
        width=args.frame_width,
        output_dir=output_dir,
        protect_ratio=args.pitome_protect_ratio,
        similarity_threshold=args.pitome_similarity_threshold,
        embedding_size=args.pitome_embedding_size,
    )
    selected_count = len(selection.timestamps)
    if args.pitome_max_selected_frames is not None:
        selected_count = min(selected_count, args.pitome_max_selected_frames)
    selected_counts.append(selected_count)
    dense_counts.append(selection.dense_frame_count)


def make_report(
    *,
    name: str,
    span_policy: str,
    span_count: int,
    duration_seconds: float,
    frames_sent_to_vlm: int,
    dense_candidate_frames: int,
    cost_config: CostConfig,
    seconds_elapsed: float | None = None,
) -> MethodReport:
    estimated_input_tokens = estimate_input_tokens(
        frames_sent_to_vlm=frames_sent_to_vlm,
        vlm_calls=span_count,
        cost_config=cost_config,
    )
    return MethodReport(
        name=name,
        span_policy=span_policy,
        span_count=span_count,
        vlm_calls=span_count,
        frames_sent_to_vlm=frames_sent_to_vlm,
        dense_candidate_frames=dense_candidate_frames,
        effective_vlm_fps=frames_sent_to_vlm / duration_seconds if duration_seconds else 0.0,
        effective_candidate_fps=(
            dense_candidate_frames / duration_seconds if duration_seconds else 0.0
        ),
        estimated_input_tokens=estimated_input_tokens,
        estimated_cost=estimate_cost(
            estimated_input_tokens=estimated_input_tokens,
            vlm_calls=span_count,
            cost_config=cost_config,
        ),
        seconds_elapsed=seconds_elapsed,
    )


def estimate_input_tokens(
    *,
    frames_sent_to_vlm: int,
    vlm_calls: int,
    cost_config: CostConfig,
) -> float | None:
    if cost_config.image_tokens_per_frame is None:
        return None
    return (
        frames_sent_to_vlm * cost_config.image_tokens_per_frame
        + vlm_calls * cost_config.prompt_tokens_per_call
    )


def estimate_cost(
    *,
    estimated_input_tokens: float | None,
    vlm_calls: int,
    cost_config: CostConfig,
) -> float | None:
    fixed_cost = vlm_calls * cost_config.fixed_cost_per_call
    if estimated_input_tokens is None or cost_config.input_price_per_million_tokens is None:
        return fixed_cost if fixed_cost else None
    token_cost = estimated_input_tokens * cost_config.input_price_per_million_tokens / 1_000_000
    return token_cost + fixed_cost


def subdivide(duration_seconds: float, window_seconds: float) -> list[Any]:
    from rlm.video.types import TimeSpan

    spans: list[Any] = []
    cursor = 0.0
    while cursor < duration_seconds:
        next_end = min(duration_seconds, cursor + window_seconds)
        spans.append(TimeSpan(cursor, next_end))
        cursor = next_end
    return spans


def summarize_counts(values: list[int]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": statistics.fmean(values),
        "median": float(statistics.median(values)),
    }


def compare_reports(baseline: MethodReport, candidate: MethodReport) -> dict[str, float | None]:
    return {
        "vlm_call_reduction_pct": percent_reduction(baseline.vlm_calls, candidate.vlm_calls),
        "frame_reduction_pct": percent_reduction(
            baseline.frames_sent_to_vlm,
            candidate.frames_sent_to_vlm,
        ),
        "dense_candidate_frame_delta_pct": percent_reduction(
            baseline.dense_candidate_frames,
            candidate.dense_candidate_frames,
        ),
        "estimated_cost_reduction_pct": percent_reduction(
            baseline.estimated_cost,
            candidate.estimated_cost,
        ),
    }


def percent_reduction(baseline: float | None, candidate: float | None) -> float | None:
    if baseline is None or candidate is None or baseline == 0:
        return None
    return (baseline - candidate) * 100.0 / baseline


def print_human_summary(report: dict[str, Any]) -> None:
    print(f"Duration: {report['duration_seconds']:.2f}s")
    print_method("Original", report["original"])
    print_method("PiToMe estimate", report["pitome_estimate"])
    print_delta("Estimate delta vs original", report["estimate_delta"])
    if "pitome_measured" in report:
        print_method("PiToMe measured", report["pitome_measured"])
        print_delta("Measured delta vs original", report["measured_delta"])


def print_method(label: str, report: dict[str, Any]) -> None:
    print(f"\n{label}")
    print(f"  policy: {report['span_policy']}")
    print(f"  VLM calls: {report['vlm_calls']}")
    print(f"  frames sent to VLM: {report['frames_sent_to_vlm']}")
    print(f"  dense candidate frames: {report['dense_candidate_frames']}")
    print(f"  effective VLM FPS: {report['effective_vlm_fps']:.4f}")
    print(f"  effective candidate FPS: {report['effective_candidate_fps']:.4f}")
    if report["estimated_input_tokens"] is not None:
        print(f"  estimated input tokens: {report['estimated_input_tokens']:.0f}")
    if report["estimated_cost"] is not None:
        print(f"  estimated cost: {report['estimated_cost']:.6f}")
    if report["seconds_elapsed"] is not None:
        print(f"  local selection time: {report['seconds_elapsed']:.2f}s")
    if report["selected_frames_per_call"] is not None:
        print(f"  selected frames/call: {report['selected_frames_per_call']}")


def print_delta(label: str, delta: dict[str, float | None]) -> None:
    print(f"\n{label}")
    for key, value in delta.items():
        value_text = "n/a" if value is None else f"{value:.2f}%"
        print(f"  {key}: {value_text}")


if __name__ == "__main__":
    raise SystemExit(main())
