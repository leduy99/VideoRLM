import argparse
import json
import shutil
from pathlib import Path

from rlm.video.pitome import select_visual_frames_for_span
from rlm.video.types import TimeSpan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare uniform frame selection versus PiToMe selection on one video span."
    )
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--start", required=True, type=float, help="Span start time in seconds")
    parser.add_argument("--end", required=True, type=float, help="Span end time in seconds")
    parser.add_argument("--output-dir", required=True, help="Directory to save the comparison outputs")
    parser.add_argument("--uniform-frame-count", type=int, default=3)
    parser.add_argument("--dense-frame-rate", type=float, default=1.0)
    parser.add_argument("--pitome-protect-ratio", type=float, default=0.15)
    parser.add_argument("--pitome-similarity-threshold", type=float, default=0.8)
    parser.add_argument("--pitome-embedding-size", type=int, default=16)
    parser.add_argument("--frame-width", type=int, default=768)
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    return parser


def _copy_selected_frames(frame_paths: list[Path], timestamps: list[float], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied_paths: list[str] = []
    for index, (frame_path, timestamp) in enumerate(zip(frame_paths, timestamps, strict=True), start=1):
        target = output_dir / f"{index:03d}_{timestamp:.2f}s{frame_path.suffix}"
        shutil.copy2(frame_path, target)
        copied_paths.append(str(target))
    return copied_paths


def main() -> int:
    args = build_parser().parse_args()
    span = TimeSpan(args.start, args.end)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    uniform = select_visual_frames_for_span(
        media_path=args.video,
        span=span,
        strategy="uniform",
        uniform_frame_count=args.uniform_frame_count,
        ffmpeg_bin=args.ffmpeg_bin,
        width=args.frame_width,
        output_dir=output_dir / "uniform_raw",
    )
    pitome = select_visual_frames_for_span(
        media_path=args.video,
        span=span,
        strategy="pitome",
        uniform_frame_count=args.uniform_frame_count,
        dense_frame_rate=args.dense_frame_rate,
        ffmpeg_bin=args.ffmpeg_bin,
        width=args.frame_width,
        output_dir=output_dir / "pitome_raw",
        protect_ratio=args.pitome_protect_ratio,
        similarity_threshold=args.pitome_similarity_threshold,
        embedding_size=args.pitome_embedding_size,
    )

    uniform_saved = _copy_selected_frames(
        uniform.frame_paths,
        uniform.timestamps,
        output_dir / "uniform_selected",
    )
    pitome_saved = _copy_selected_frames(
        pitome.frame_paths,
        pitome.timestamps,
        output_dir / "pitome_selected",
    )

    report = {
        "video": str(Path(args.video).resolve()),
        "span": span.to_dict(),
        "uniform": {
            "timestamps": uniform.timestamps,
            "selected_frame_paths": uniform_saved,
            "metadata": uniform.to_metadata(),
        },
        "pitome": {
            "timestamps": pitome.timestamps,
            "selected_frame_paths": pitome_saved,
            "metadata": pitome.to_metadata(),
        },
    }
    report_path = output_dir / "comparison.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Uniform selected {len(uniform.timestamps)} frames: {uniform.timestamps}")
    print(
        "PiToMe selected "
        f"{len(pitome.timestamps)} of {pitome.dense_frame_count} dense frames: {pitome.timestamps}"
    )
    print(f"Saved comparison report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
