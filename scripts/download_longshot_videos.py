#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlm.video.longshot import LongShOTVideoResolver, load_longshot_samples  # noqa: E402


def main() -> int:
    args = parse_args()
    samples = load_longshot_samples(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        split=args.split,
        sample_limit=args.sample_limit,
        sample_ids=args.sample_id,
        video_ids=args.video_id,
        task_filters=args.task_filter,
    )
    video_ids = sorted({str(sample["video_id"]) for sample in samples})
    if args.video_limit is not None:
        video_ids = video_ids[: args.video_limit]

    resolver = LongShOTVideoResolver(
        args.video_dir,
        download_missing=True,
        yt_dlp_bin=args.yt_dlp_bin,
        cookies_from_browser=args.cookies_from_browser,
        extra_ytdlp_args=args.yt_dlp_arg,
    )
    print(
        f"[download-longshot] dataset={args.dataset_path}/{args.dataset_name} "
        f"split={args.split} samples={len(samples)} unique_videos={len(video_ids)}"
    )
    failed: list[str] = []
    for index, video_id in enumerate(video_ids, start=1):
        print(f"[download-longshot] {index}/{len(video_ids)} video_id={video_id}", flush=True)
        try:
            path = resolver.resolve(video_id)
        except Exception as exc:
            print(f"[download-longshot] failed video_id={video_id}: {exc}", flush=True)
            failed.append(video_id)
            if not args.skip_failed:
                raise
            continue
        print(f"[download-longshot] ready {path}", flush=True)
    if failed:
        print(f"[download-longshot] failed_count={len(failed)} failed_video_ids={failed}", flush=True)
        return 1 if not args.skip_failed else 0
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download LongShOTBench videos without running inference.")
    parser.add_argument("--dataset-path", default="MBZUAI/longshot-bench")
    parser.add_argument("--dataset-name", default="postvalid_tools_v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--video-dir", default="data/longshotbench/videos")
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--video-limit", type=int)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--task-filter", action="append", default=[])
    parser.add_argument("--yt-dlp-bin", default="yt-dlp")
    parser.add_argument("--cookies-from-browser")
    parser.add_argument(
        "--yt-dlp-arg",
        action="append",
        default=[],
        help=(
            "Extra argument passed through to yt-dlp. Repeat for multiple args, "
            "for example --yt-dlp-arg=--js-runtimes --yt-dlp-arg=deno."
        ),
    )
    parser.add_argument("--skip-failed", action="store_true", help="Continue when a video is unavailable.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
