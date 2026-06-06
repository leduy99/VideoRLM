#!/usr/bin/env python3
"""Export representative VRR-QA validation examples for presentations."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


REPRESENTATIVE_QUESTION_IDS = [
    "8699b2d8-ff9e-4664-80ef-7a074ea518f1",
    "de46595f-a9c3-4ee8-9cc2-6cbbdb08e004",
    "7af4ffda-2a7e-4fa1-a1b7-372006445505",
    "d05dcedf-fbb9-44ea-9510-93abf53f9117",
    "95fb073c-9638-4d4d-9343-d431ec3336f6",
    "a8a9141b-e3b7-46b8-b3b8-18391cf0b64c",
    "23359178-4e5d-4c3a-aada-4058c6f0ebea",
    "1bbf1092-91b2-4e3e-8065-a84a99712a3d",
    "b9796db0-cad2-40c3-85f8-c72ec52747c0",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a small presentation bundle from the VRR-QA val set."
    )
    parser.add_argument(
        "--annotations",
        default="data/val_set/ImplicitQAv0.1.2.jsonl",
        type=Path,
        help="VRR-QA validation JSONL.",
    )
    parser.add_argument(
        "--video-dir",
        default="data/val_set/videos",
        type=Path,
        help="Directory with validation videos.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/val_representative_samples",
        type=Path,
        help="Directory to write the sample bundle.",
    )
    parser.add_argument(
        "--clip-padding",
        default=0.75,
        type=float,
        help="Seconds added before and after the annotated segment.",
    )
    parser.add_argument(
        "--no-clips",
        action="store_true",
        help="Only export metadata and thumbnails.",
    )
    args = parser.parse_args()

    rows = load_rows(args.annotations)
    by_question_id = {str(row["question_id"]): row for row in rows}
    samples = [
        sample_from_row(by_question_id[question_id], args.video_dir)
        for question_id in REPRESENTATIVE_QUESTION_IDS
    ]

    output_dir = args.output_dir
    assets_dir = output_dir / "assets"
    clips_dir = output_dir / "clips"
    assets_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    for index, sample in enumerate(samples, start=1):
        prefix = f"{index:02d}_cat{sample['category_id']}_{sample['question_id']}"
        frame_paths = export_frames(sample, assets_dir, prefix)
        strip_path = assets_dir / f"{prefix}_strip.jpg"
        make_strip(frame_paths, strip_path)
        sample["thumbnail_strip"] = str(strip_path)

        if not args.no_clips:
            clip_path = clips_dir / f"{prefix}.mp4"
            export_clip(sample, clip_path, args.clip_padding)
            sample["clip"] = str(clip_path)

    write_jsonl(samples, output_dir / "samples.jsonl")
    write_markdown(samples, output_dir / "README.md")
    print(f"Wrote {len(samples)} samples to {output_dir}")


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def sample_from_row(row: dict[str, Any], video_dir: Path) -> dict[str, Any]:
    video_path = video_dir / f"{row['video_id']}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    start = float(row["question_start_time"])
    stop = float(row["question_stop_time"])
    if stop <= start:
        raise ValueError(f"Invalid segment for {row['question_id']}: {start}..{stop}")

    return {
        "video_id": row["video_id"],
        "video_path": str(video_path),
        "video_url": row["video_url"],
        "question_id": row["question_id"],
        "category_id": row["category_id"],
        "category": row["category"],
        "question": row["question_text"],
        "options": row["options"],
        "answer_choice": row["answer_choice"],
        "answer_text": row["answer_text"],
        "start_time": round(start, 3),
        "stop_time": round(stop, 3),
        "duration": round(stop - start, 3),
    }


def export_frames(
    sample: dict[str, Any], assets_dir: Path, prefix: str
) -> list[Path]:
    start = float(sample["start_time"])
    stop = float(sample["stop_time"])
    midpoint = (start + stop) / 2
    timestamps = [
        ("start", start),
        ("mid", midpoint),
        ("end", max(start, stop - 0.25)),
    ]
    frame_paths = []
    for label, timestamp in timestamps:
        frame_path = assets_dir / f"{prefix}_{label}.jpg"
        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                str(sample["video_path"]),
                "-frames:v",
                "1",
                "-vf",
                "scale=426:-2",
                "-q:v",
                "2",
                str(frame_path),
            ]
        )
        frame_paths.append(frame_path)
    return frame_paths


def make_strip(frame_paths: list[Path], output_path: Path) -> None:
    frames = [Image.open(path).convert("RGB") for path in frame_paths]
    width = sum(image.width for image in frames)
    height = max(image.height for image in frames)
    label_height = 34
    strip = Image.new("RGB", (width, height + label_height), "white")
    draw = ImageDraw.Draw(strip)
    x = 0
    labels = ["start", "middle", "end"]
    for label, image in zip(labels, frames, strict=True):
        strip.paste(image, (x, label_height))
        draw.text((x + 8, 9), label, fill=(20, 20, 20))
        x += image.width
    strip.save(output_path, quality=92)


def export_clip(sample: dict[str, Any], output_path: Path, padding: float) -> None:
    start = max(0.0, float(sample["start_time"]) - padding)
    duration = float(sample["duration"]) + (padding * 2)
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(sample["video_path"]),
            "-t",
            f"{duration:.3f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )


def write_jsonl(samples: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")


def write_markdown(samples: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# VRR-QA Validation Representative Samples",
        "",
        "Balanced examples from the validation split, one per reasoning category.",
        "",
    ]
    for index, sample in enumerate(samples, start=1):
        options = ", ".join(
            f"{letter}. {text}" for letter, text in sample["options"].items()
        )
        lines.extend(
            [
                f"## {index}. {sample['category']}",
                "",
                f"- Question ID: `{sample['question_id']}`",
                f"- Video ID: `{sample['video_id']}`",
                (
                    f"- Segment: {sample['start_time']:.2f}s to "
                    f"{sample['stop_time']:.2f}s ({sample['duration']:.1f}s)"
                ),
                f"- Question: {sample['question']}",
                f"- Options: {options}",
                f"- Answer: {sample['answer_choice']}. {sample['answer_text']}",
                f"- Thumbnail strip: `{sample['thumbnail_strip']}`",
                f"- Clip: `{sample.get('clip', '')}`",
                f"- Source video: `{sample['video_path']}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
