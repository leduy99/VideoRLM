from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MethodName = Literal["original", "pitome"]
SpanKind = Literal["scene", "clip"]


@dataclass
class SpanInputReport:
    method: MethodName
    span_kind: SpanKind
    span_index: int
    start: float
    end: float
    duration: float
    vlm_frame_paths: list[str]
    vlm_timestamps: list[float]
    candidate_frame_count: int
    candidate_frame_paths: list[str] = field(default_factory=list)
    selected_frame_count: int | None = None
    selection_metadata: dict[str, Any] = field(default_factory=dict)
    image_metadata: list[dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float = 0.0


@dataclass
class MethodSummary:
    method: MethodName
    span_count: int
    vlm_call_count: int
    frames_ready_for_vlm: int
    candidate_frame_count: int
    effective_vlm_fps: float
    effective_candidate_fps: float
    elapsed_seconds: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the visual preprocessing inputs that would be fed to a VLM. "
            "This runs frame extraction/selection only; it does not call a VLM."
        )
    )
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output-dir", required=True, help="Directory for frames and manifest")
    parser.add_argument(
        "--duration-seconds",
        type=positive_float,
        help="Video duration. If omitted, ffprobe is used.",
    )
    parser.add_argument("--method", choices=["original", "pitome", "both"], default="both")
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--frame-width", type=positive_int, default=768)
    parser.add_argument(
        "--validate-images",
        action="store_true",
        help="Open each VLM-ready frame with Pillow and record image dimensions.",
    )
    parser.add_argument(
        "--max-spans",
        type=positive_int,
        help="Optional cap per method for quick smoke tests.",
    )

    parser.add_argument("--original-scene-duration-seconds", type=positive_float, default=180.0)
    parser.add_argument("--original-clip-duration-seconds", type=positive_float, default=15.0)
    parser.add_argument("--original-frame-count", type=positive_int, default=3)
    parser.add_argument(
        "--original-skip-scenes",
        action="store_true",
        help="Only materialize original clip spans, skipping scene-level duplicate VLM inputs.",
    )

    parser.add_argument("--pitome-clip-duration-seconds", type=positive_float, default=60.0)
    parser.add_argument("--pitome-dense-frame-rate", type=positive_float, default=1.0)
    parser.add_argument("--pitome-min-frame-count", type=positive_int, default=8)
    parser.add_argument("--pitome-protect-ratio", type=bounded_ratio, default=0.15)
    parser.add_argument("--pitome-similarity-threshold", type=bounded_ratio, default=0.8)
    parser.add_argument("--pitome-embedding-size", type=positive_int, default=16)
    parser.add_argument(
        "--pitome-max-selected-frames",
        type=positive_int,
        help="Cap selected PiToMe frames copied into the VLM-ready directory.",
    )
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


def bounded_ratio(value: str) -> float:
    parsed = float(value)
    if parsed < 0 or parsed > 1:
        raise argparse.ArgumentTypeError(f"expected a ratio within [0, 1], got {value}")
    return parsed


def main() -> int:
    args = build_parser().parse_args()
    video_path = Path(args.video)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    duration_seconds = resolve_duration(args)

    reports: dict[str, list[SpanInputReport]] = {}
    summaries: dict[str, MethodSummary] = {}
    if args.method in {"original", "both"}:
        reports["original"] = materialize_original_inputs(args, video_path, output_dir, duration_seconds)
        summaries["original"] = summarize_method("original", reports["original"], duration_seconds)
    if args.method in {"pitome", "both"}:
        reports["pitome"] = materialize_pitome_inputs(args, video_path, output_dir, duration_seconds)
        summaries["pitome"] = summarize_method("pitome", reports["pitome"], duration_seconds)

    manifest = {
        "video": str(video_path.resolve()),
        "duration_seconds": duration_seconds,
        "output_dir": str(output_dir.resolve()),
        "config": vars(args),
        "summaries": {name: asdict(summary) for name, summary in summaries.items()},
        "spans": {
            name: [asdict(report) for report in span_reports]
            for name, span_reports in reports.items()
        },
    }
    manifest_path = output_dir / "pre_vlm_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print_summary(manifest_path, summaries)
    return 0


def resolve_duration(args: argparse.Namespace) -> float:
    if args.duration_seconds is not None:
        return args.duration_seconds

    from rlm.video.media import probe_media_duration

    return probe_media_duration(args.video, ffprobe_bin=args.ffprobe_bin)


def materialize_original_inputs(
    args: argparse.Namespace,
    video_path: Path,
    output_dir: Path,
    duration_seconds: float,
) -> list[SpanInputReport]:
    from rlm.video.media import extract_frames_for_span, sample_span_timestamps

    span_specs: list[tuple[SpanKind, int, Any]] = []
    if not args.original_skip_scenes:
        span_specs.extend(
            ("scene", index, span)
            for index, span in enumerate(
                subdivide(duration_seconds, args.original_scene_duration_seconds),
                start=1,
            )
        )
    span_specs.extend(
        ("clip", index, span)
        for index, span in enumerate(
            subdivide(duration_seconds, args.original_clip_duration_seconds),
            start=1,
        )
    )
    if args.max_spans is not None:
        span_specs = span_specs[: args.max_spans]

    reports: list[SpanInputReport] = []
    method_dir = output_dir / "original"
    method_dir.mkdir(parents=True, exist_ok=True)
    for span_kind, span_index, span in span_specs:
        started_at = time.perf_counter()
        span_dir = method_dir / f"{span_kind}_{span_index:04d}"
        recreate_directory(span_dir)
        vlm_dir = span_dir / "vlm_frames"
        timestamps = sample_span_timestamps(span, args.original_frame_count)
        frame_paths = extract_frames_for_span(
            media_path=video_path,
            span=span,
            frame_count=args.original_frame_count,
            ffmpeg_bin=args.ffmpeg_bin,
            width=args.frame_width,
            output_dir=vlm_dir,
        )
        reports.append(
            SpanInputReport(
                method="original",
                span_kind=span_kind,
                span_index=span_index,
                start=span.start,
                end=span.end,
                duration=span.duration,
                vlm_frame_paths=[str(path) for path in frame_paths],
                vlm_timestamps=timestamps,
                candidate_frame_count=len(frame_paths),
                candidate_frame_paths=[str(path) for path in frame_paths],
                selected_frame_count=len(frame_paths),
                image_metadata=read_image_metadata(frame_paths) if args.validate_images else [],
                elapsed_seconds=time.perf_counter() - started_at,
            )
        )
    return reports


def materialize_pitome_inputs(
    args: argparse.Namespace,
    video_path: Path,
    output_dir: Path,
    duration_seconds: float,
) -> list[SpanInputReport]:
    from rlm.video.pitome import select_visual_frames_for_span

    spans = list(enumerate(subdivide(duration_seconds, args.pitome_clip_duration_seconds), start=1))
    if args.max_spans is not None:
        spans = spans[: args.max_spans]

    reports: list[SpanInputReport] = []
    method_dir = output_dir / "pitome"
    method_dir.mkdir(parents=True, exist_ok=True)
    for span_index, span in spans:
        started_at = time.perf_counter()
        span_dir = method_dir / f"clip_{span_index:04d}"
        recreate_directory(span_dir)
        candidate_dir = span_dir / "candidate_frames"
        selected_dir = span_dir / "vlm_frames"
        selection = select_visual_frames_for_span(
            media_path=video_path,
            span=span,
            strategy="pitome",
            uniform_frame_count=args.pitome_min_frame_count,
            dense_frame_rate=args.pitome_dense_frame_rate,
            ffmpeg_bin=args.ffmpeg_bin,
            width=args.frame_width,
            output_dir=candidate_dir,
            protect_ratio=args.pitome_protect_ratio,
            similarity_threshold=args.pitome_similarity_threshold,
            embedding_size=args.pitome_embedding_size,
        )
        selected_paths = selection.frame_paths
        selected_timestamps = selection.timestamps
        if args.pitome_max_selected_frames is not None:
            selected_paths = selected_paths[: args.pitome_max_selected_frames]
            selected_timestamps = selected_timestamps[: args.pitome_max_selected_frames]

        copied_paths = copy_selected_frames(selected_paths, selected_timestamps, selected_dir)
        reports.append(
            SpanInputReport(
                method="pitome",
                span_kind="clip",
                span_index=span_index,
                start=span.start,
                end=span.end,
                duration=span.duration,
                vlm_frame_paths=[str(path) for path in copied_paths],
                vlm_timestamps=selected_timestamps,
                candidate_frame_count=selection.dense_frame_count,
                candidate_frame_paths=[str(path) for path in sorted(candidate_dir.glob("*.jpg"))],
                selected_frame_count=len(copied_paths),
                selection_metadata=selection.to_metadata(),
                image_metadata=read_image_metadata(copied_paths) if args.validate_images else [],
                elapsed_seconds=time.perf_counter() - started_at,
            )
        )
    return reports


def subdivide(duration_seconds: float, window_seconds: float) -> list[Any]:
    from rlm.video.types import TimeSpan

    spans: list[Any] = []
    cursor = 0.0
    while cursor < duration_seconds:
        next_end = min(duration_seconds, cursor + window_seconds)
        spans.append(TimeSpan(cursor, next_end))
        cursor = next_end
    return spans


def copy_selected_frames(
    frame_paths: list[Path],
    timestamps: list[float],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied_paths: list[Path] = []
    for index, (frame_path, timestamp) in enumerate(zip(frame_paths, timestamps, strict=True), start=1):
        target = output_dir / f"selected_{index:03d}_{timestamp:.3f}s{frame_path.suffix}"
        shutil.copy2(frame_path, target)
        copied_paths.append(target)
    return copied_paths


def recreate_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def read_image_metadata(frame_paths: list[Path]) -> list[dict[str, Any]]:
    from PIL import Image

    metadata: list[dict[str, Any]] = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as image:
            metadata.append(
                {
                    "path": str(frame_path),
                    "width": image.width,
                    "height": image.height,
                    "mode": image.mode,
                }
            )
    return metadata


def summarize_method(
    method: MethodName,
    reports: list[SpanInputReport],
    duration_seconds: float,
) -> MethodSummary:
    frames_ready = sum(len(report.vlm_frame_paths) for report in reports)
    candidates = sum(report.candidate_frame_count for report in reports)
    elapsed = sum(report.elapsed_seconds for report in reports)
    return MethodSummary(
        method=method,
        span_count=len(reports),
        vlm_call_count=len(reports),
        frames_ready_for_vlm=frames_ready,
        candidate_frame_count=candidates,
        effective_vlm_fps=frames_ready / duration_seconds,
        effective_candidate_fps=candidates / duration_seconds,
        elapsed_seconds=elapsed,
    )


def print_summary(manifest_path: Path, summaries: dict[str, MethodSummary]) -> None:
    for summary in summaries.values():
        print(f"\n{summary.method}")
        print(f"  spans / VLM calls: {summary.vlm_call_count}")
        print(f"  frames ready for VLM: {summary.frames_ready_for_vlm}")
        print(f"  candidate frames processed: {summary.candidate_frame_count}")
        print(f"  effective VLM FPS: {summary.effective_vlm_fps:.4f}")
        print(f"  effective candidate FPS: {summary.effective_candidate_fps:.4f}")
        print(f"  preprocessing time: {summary.elapsed_seconds:.2f}s")
    print(f"\nSaved pre-VLM manifest to {manifest_path}")


if __name__ == "__main__":
    raise SystemExit(main())
