#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from rlm.video.qwen import QwenLocalVideoStackConfig
from rlm.video.vrrqa import (
    VRRQA_ANNOTATION_FILENAME,
    VRRQA_DATASET_PATH,
    VRRQA_SPLIT,
    VRRQABenchmarkRunner,
    VRRQAVideoResolver,
    load_vrrqa_samples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run visual-only VideoRLM on VRR-QA.")
    parser.add_argument("--annotations", default=f"data/vrrqa/{VRRQA_ANNOTATION_FILENAME}")
    parser.add_argument("--dataset-path", default=VRRQA_DATASET_PATH)
    parser.add_argument("--split", default=VRRQA_SPLIT)
    parser.add_argument("--output", required=True)
    parser.add_argument("--video-dir", default="data/vrrqa/videos")
    parser.add_argument("--segment-dir", default="data/vrrqa/segments")
    parser.add_argument("--artifacts-dir")
    parser.add_argument("--memory-dir")
    parser.add_argument("--trace-dir")
    parser.add_argument("--strategy", choices=["original", "lazy-pitome"], default="original")
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--question-id", action="append", default=[])
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--skip-unavailable-videos", action="store_true")
    parser.add_argument("--yt-dlp-bin", default="yt-dlp")
    parser.add_argument("--cookies-from-browser")
    parser.add_argument("--controller-repo", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--visual-repo", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--speech-repo", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--controller-device", default="mps")
    parser.add_argument("--visual-device", default="mps")
    parser.add_argument("--speech-device", default="mps")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--attn-implementation")
    parser.add_argument("--frame-count", type=int, default=16)
    parser.add_argument("--frame-width", type=int, default=768)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--search-top-k", type=int, default=5)
    parser.add_argument("--max-frontier-items", type=int, default=8)
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the VRR-QA benchmark progress bar.",
    )
    parser.add_argument("--semantic-frame-embedding-repo")
    parser.add_argument("--semantic-frame-embedding-model-path")
    parser.add_argument("--semantic-frame-embedding-device", default="mps")
    parser.add_argument("--semantic-frame-embedding-torch-dtype", default="float32")
    parser.add_argument("--semantic-frame-embedding-batch-size", type=int, default=8)
    parser.add_argument("--pitome-dense-frame-rate", type=float, default=0.2)
    parser.add_argument("--pitome-min-frame-count", type=int, default=16)
    parser.add_argument("--pitome-embedding-backend", choices=["pixel", "hybrid"], default="hybrid")
    parser.add_argument("--pitome-embedding-size", type=int, default=32)
    parser.add_argument("--pitome-anchor-frame-count", type=int, default=8)
    parser.add_argument("--pitome-max-selected-frames", type=int, default=8)
    parser.add_argument("--pitome-scene-threshold", type=float, default=0.35)
    parser.add_argument("--pitome-max-scene-boundary-frames", type=int, default=6)
    parser.add_argument(
        "--multi-window-memory",
        action="store_true",
        help="Keep regular scene/segment/clip subdivision instead of one node per QA segment.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    annotation_path = Path(args.annotations)
    samples = load_vrrqa_samples(
        annotation_path=annotation_path if annotation_path.exists() else None,
        dataset_path=args.dataset_path,
        split=args.split,
        sample_limit=args.sample_limit,
        question_ids=args.question_id,
        video_ids=args.video_id,
        categories=args.category,
    )
    output_path = Path(args.output)
    output_root = output_path.parent

    semantic_model = args.semantic_frame_embedding_repo if args.strategy == "lazy-pitome" else None
    config = QwenLocalVideoStackConfig.default(
        controller_device=args.controller_device,
        visual_device=args.visual_device,
        speech_device=args.speech_device,
        controller_model=args.controller_repo,
        visual_model=args.visual_repo,
        speech_model=args.speech_repo,
        forced_aligner_model=None,
        semantic_frame_embedding_model=semantic_model,
        semantic_frame_embedding_device=args.semantic_frame_embedding_device,
        semantic_frame_embedding_torch_dtype=args.semantic_frame_embedding_torch_dtype,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
    )
    config.enable_speech_recognition = False
    config.frame_count = args.frame_count
    config.frame_width = args.frame_width
    config.ffmpeg_bin = args.ffmpeg_bin
    config.verbose = args.verbose
    config.semantic_frame_embedding_batch_size = args.semantic_frame_embedding_batch_size
    if config.semantic_frame_embedding is not None and args.semantic_frame_embedding_model_path:
        config.semantic_frame_embedding.model_path = args.semantic_frame_embedding_model_path
    if args.strategy == "lazy-pitome":
        config.use_pitome = True
        config.lazy_visual_refinement = True
        config.search_mode = "graph"
        config.pitome_dense_frame_rate = args.pitome_dense_frame_rate
        config.pitome_min_frame_count = args.pitome_min_frame_count
        config.pitome_embedding_backend = args.pitome_embedding_backend
        config.pitome_embedding_size = args.pitome_embedding_size
        config.pitome_anchor_frame_count = args.pitome_anchor_frame_count
        config.pitome_max_selected_frames = args.pitome_max_selected_frames
        config.pitome_scene_threshold = args.pitome_scene_threshold
        config.pitome_max_scene_boundary_frames = args.pitome_max_scene_boundary_frames
    else:
        config.use_pitome = False
        config.search_mode = "lexical"

    bundle = config.build_bundle(
        max_steps=args.max_steps,
        search_top_k=args.search_top_k,
        max_frontier_items=args.max_frontier_items,
    )
    bundle.memory_builder.speech_recognizer = None
    bundle.memory_builder.audio_extractor = None
    bundle.memory_builder.visual_span_mode = "clip"
    bundle.memory_builder.aggregate_child_visual_summaries = False
    bundle.memory_builder.parent_visual_summary_mode = "full"
    if hasattr(bundle.visual_summarizer, "summary_granularity"):
        bundle.visual_summarizer.summary_granularity = "clip"
    if bundle.visual_refiner is not None and hasattr(bundle.visual_refiner, "summary_granularity"):
        bundle.visual_refiner.summary_granularity = "clip"

    runner = VRRQABenchmarkRunner(
        video_rlm=bundle.controller,
        memory_builder=bundle.memory_builder,
        video_resolver=VRRQAVideoResolver(
            args.video_dir,
            download_missing=args.download_missing,
            yt_dlp_bin=args.yt_dlp_bin,
            cookies_from_browser=args.cookies_from_browser,
        ),
        segment_dir=args.segment_dir,
        artifact_cache_dir=args.artifacts_dir or output_root / "artifacts",
        memory_cache_dir=args.memory_dir or output_root / "memories",
        trace_dir=args.trace_dir or output_root / "traces",
        ffmpeg_bin=args.ffmpeg_bin,
        verbose=args.verbose,
        show_progress=not args.no_progress,
        skip_unavailable_videos=args.skip_unavailable_videos,
        single_window_memory=not args.multi_window_memory,
    )
    runner.run_samples(samples, output_path=output_path)
    print(f"Saved VRR-QA predictions to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
