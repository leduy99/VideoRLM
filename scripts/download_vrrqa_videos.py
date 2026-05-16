#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from rlm.video.vrrqa import (
    VRRQA_ANNOTATION_FILENAME,
    VRRQA_DATASET_PATH,
    VRRQA_SPLIT,
    VRRQAVideoResolver,
    ensure_vrrqa_annotations,
    load_vrrqa_samples,
    unique_vrrqa_videos,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download VRR-QA/ImplicitQA annotations and source videos."
    )
    parser.add_argument("--annotations", default=f"data/vrrqa/{VRRQA_ANNOTATION_FILENAME}")
    parser.add_argument("--dataset-path", default=VRRQA_DATASET_PATH)
    parser.add_argument("--split", default=VRRQA_SPLIT)
    parser.add_argument("--video-dir", default="data/vrrqa/videos")
    parser.add_argument("--manifest", default="data/vrrqa/download_manifest.jsonl")
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--yt-dlp-bin", default="yt-dlp")
    parser.add_argument("--cookies-from-browser")
    parser.add_argument("--skip-failed", action="store_true")
    parser.add_argument(
        "--no-annotation-download",
        action="store_true",
        help="Require the annotation JSONL to already exist.",
    )
    parser.add_argument(
        "--extra-ytdlp-arg",
        action="append",
        default=[],
        help="Extra argument passed through to yt-dlp. Repeat for multiple args.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    annotation_path = Path(args.annotations)
    if not args.no_annotation_download:
        ensure_vrrqa_annotations(annotation_path, dataset_path=args.dataset_path)
    samples = load_vrrqa_samples(
        annotation_path=annotation_path if annotation_path.exists() else None,
        dataset_path=args.dataset_path,
        split=args.split,
        sample_limit=args.sample_limit,
        video_ids=args.video_id,
    )
    videos = unique_vrrqa_videos(samples)
    resolver = VRRQAVideoResolver(
        args.video_dir,
        download_missing=True,
        yt_dlp_bin=args.yt_dlp_bin,
        cookies_from_browser=args.cookies_from_browser,
        extra_ytdlp_args=args.extra_ytdlp_arg,
    )
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    failed = 0
    with manifest_path.open("a", encoding="utf-8") as manifest:
        for index, item in enumerate(videos, start=1):
            video_id = item["video_id"]
            video_url = item["video_url"]
            try:
                cached = resolver.find(video_id)
                if cached is None:
                    print(f"[VRR-QA] {index}/{len(videos)} downloading {video_id}", flush=True)
                    cached = resolver.download(video_id, video_url)
                else:
                    print(f"[VRR-QA] {index}/{len(videos)} cached {video_id}", flush=True)
                record = {
                    "video_id": video_id,
                    "video_url": video_url,
                    "path": str(cached),
                    "status": "ok",
                }
                completed += 1
            except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as exc:
                record = {
                    "video_id": video_id,
                    "video_url": video_url,
                    "path": None,
                    "status": "failed",
                    "error": str(exc),
                }
                failed += 1
                print(f"[VRR-QA] failed {video_id}: {exc}", flush=True)
                if not args.skip_failed:
                    manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                    raise
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(
        f"[VRR-QA] download complete ok={completed} failed={failed} manifest={manifest_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
