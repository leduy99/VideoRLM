import argparse
import json
from pathlib import Path

from rlm.video.controller import VideoRLM
from rlm.video.logger import VideoRLMLogger
from rlm.video.longshot import (
    LongShOTBenchmarkRunner,
    LongShOTVideoResolver,
    load_longshot_samples,
)
from rlm.video.longshot_official_eval import (
    LongShOTOfficialEvalConfig,
    evaluate_predictions_official_style,
)
from rlm.video.memory import VideoMemoryBuilder
from rlm.video.qwen import QwenLocalVideoStackConfig, QwenVideoStackConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VideoRLM utility CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare-artifacts",
        help="Run ASR/visual preprocessing and save artifact sidecars.",
    )
    prepare.add_argument("--video", required=True, help="Input video path")
    prepare.add_argument("--duration-seconds", required=True, type=float)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--video-id")
    _add_shared_qwen_endpoint_args(prepare)
    prepare.add_argument("--speech-model", default="Qwen3-ASR-0.6B")
    prepare.add_argument("--visual-model", default="Qwen3-VL-8B")
    prepare.add_argument("--ffmpeg-bin", default="ffmpeg")

    build_memory = subparsers.add_parser(
        "build-memory",
        help="Build a hierarchical memory JSON file from prepared artifacts.",
    )
    build_memory.add_argument("--artifacts", required=True, help="Artifact JSON file or directory")
    build_memory.add_argument("--output", required=True, help="Output memory JSON file")
    build_memory.add_argument("--scene-duration-seconds", type=float, default=180.0)
    build_memory.add_argument("--segment-duration-seconds", type=float, default=45.0)
    build_memory.add_argument("--clip-duration-seconds", type=float, default=15.0)

    ask = subparsers.add_parser(
        "ask",
        help="Run the VideoRLM controller over a built memory file.",
    )
    ask.add_argument("--memory", required=True)
    ask.add_argument("--question", required=True)
    ask.add_argument("--task-type")
    ask.add_argument("--trace-out")
    ask.add_argument("--log-dir")
    ask.add_argument("--max-steps", type=int, default=8)
    ask.add_argument("--search-top-k", type=int, default=5)
    ask.add_argument("--max-frontier-items", type=int, default=8)
    ask.add_argument("--controller-model", default="Qwen3-8B")
    ask.add_argument("--controller-base-url", required=True)
    ask.add_argument("--controller-api-key")

    longshot = subparsers.add_parser(
        "run-longshot",
        help="Run VideoRLM on LongShOTBench samples and emit LongShOT-compatible predictions.",
    )
    longshot.add_argument("--output", required=True, help="Output JSONL file")
    longshot.add_argument("--video-dir", required=True, help="Directory containing benchmark videos")
    longshot.add_argument("--dataset-path", default="MBZUAI/longshot-bench")
    longshot.add_argument("--dataset-name", default="postvalid_v1")
    longshot.add_argument("--split", default="test")
    longshot.add_argument("--sample-limit", type=int)
    longshot.add_argument("--sample-id", action="append", default=[])
    longshot.add_argument("--video-id", action="append", default=[])
    longshot.add_argument("--task-filter", action="append", default=[])
    longshot.add_argument("--download-missing", action="store_true")
    longshot.add_argument("--yt-dlp-bin", default="yt-dlp")
    longshot.add_argument("--cookies-from-browser")
    longshot.add_argument("--artifacts-dir")
    longshot.add_argument("--memory-dir")
    longshot.add_argument("--trace-dir")
    longshot.add_argument("--history-mode", choices=["gold", "candidate"], default="gold")
    longshot.add_argument("--controller-model", default="Qwen3-8B")
    longshot.add_argument("--speech-model", default="Qwen3-ASR-0.6B")
    longshot.add_argument("--visual-model", default="Qwen3-VL-8B")
    longshot.add_argument("--embedding-model")
    longshot.add_argument("--max-steps", type=int, default=8)
    longshot.add_argument("--search-top-k", type=int, default=5)
    longshot.add_argument("--max-frontier-items", type=int, default=8)
    longshot.add_argument("--scene-duration-seconds", type=float, default=180.0)
    longshot.add_argument("--segment-duration-seconds", type=float, default=45.0)
    longshot.add_argument("--clip-duration-seconds", type=float, default=15.0)
    longshot.add_argument("--ffmpeg-bin", default="ffmpeg")
    longshot.add_argument("--log-dir")
    _add_shared_qwen_endpoint_args(longshot)

    download_local = subparsers.add_parser(
        "download-qwen-local-models",
        help="Download the default local Qwen VideoRLM stack from Hugging Face.",
    )
    _add_local_qwen_args(download_local)
    download_local.add_argument("--controller-device", default="cuda:0")
    download_local.add_argument("--visual-device", default="cuda:1")
    download_local.add_argument("--speech-device", default="cuda:2")

    longshot_local = subparsers.add_parser(
        "run-longshot-local",
        help="Run VideoRLM on LongShOTBench using local Hugging Face Qwen checkpoints.",
    )
    longshot_local.add_argument("--output", required=True, help="Output JSONL file")
    longshot_local.add_argument("--video-dir", required=True, help="Directory containing benchmark videos")
    longshot_local.add_argument("--dataset-path", default="MBZUAI/longshot-bench")
    longshot_local.add_argument("--dataset-name", default="postvalid_v1")
    longshot_local.add_argument("--split", default="test")
    longshot_local.add_argument("--sample-limit", type=int)
    longshot_local.add_argument("--sample-id", action="append", default=[])
    longshot_local.add_argument("--video-id", action="append", default=[])
    longshot_local.add_argument("--task-filter", action="append", default=[])
    longshot_local.add_argument("--download-missing", action="store_true")
    longshot_local.add_argument("--yt-dlp-bin", default="yt-dlp")
    longshot_local.add_argument("--cookies-from-browser")
    longshot_local.add_argument("--artifacts-dir")
    longshot_local.add_argument("--memory-dir")
    longshot_local.add_argument("--trace-dir")
    longshot_local.add_argument("--history-mode", choices=["gold", "candidate"], default="gold")
    longshot_local.add_argument("--max-steps", type=int, default=8)
    longshot_local.add_argument("--search-top-k", type=int, default=5)
    longshot_local.add_argument("--max-frontier-items", type=int, default=8)
    longshot_local.add_argument("--scene-duration-seconds", type=float, default=180.0)
    longshot_local.add_argument("--segment-duration-seconds", type=float, default=45.0)
    longshot_local.add_argument("--clip-duration-seconds", type=float, default=15.0)
    longshot_local.add_argument("--ffmpeg-bin", default="ffmpeg")
    longshot_local.add_argument("--log-dir")
    longshot_local.add_argument("--controller-device", default="cuda:0")
    longshot_local.add_argument("--visual-device", default="cuda:1")
    longshot_local.add_argument("--speech-device", default="cuda:2")
    _add_local_qwen_args(longshot_local)

    official_eval = subparsers.add_parser(
        "eval-longshot-official",
        help="Evaluate LongShOT predictions with official-style rubric prompts and scoring.",
    )
    official_eval.add_argument("--predictions", required=True, help="Input predictions JSONL file")
    official_eval.add_argument("--eval-output", required=True, help="Output evaluated JSONL file")
    official_eval.add_argument("--score-output", required=True, help="Human-readable score report")
    official_eval.add_argument("--summary-output", required=True, help="Machine-readable score summary")
    official_eval.add_argument("--judge-repo", default="Qwen/Qwen3-14B")
    official_eval.add_argument("--judge-model-path")
    official_eval.add_argument("--judge-device", default="cuda:0")
    official_eval.add_argument("--torch-dtype", default="bfloat16")
    official_eval.add_argument("--attn-implementation")
    official_eval.add_argument("--max-new-tokens", type=int, default=96)
    official_eval.add_argument("--sample-limit", type=int)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "prepare-artifacts":
        return _cmd_prepare_artifacts(args)
    if args.command == "build-memory":
        return _cmd_build_memory(args)
    if args.command == "ask":
        return _cmd_ask(args)
    if args.command == "run-longshot":
        return _cmd_run_longshot(args)
    if args.command == "download-qwen-local-models":
        return _cmd_download_qwen_local_models(args)
    if args.command == "run-longshot-local":
        return _cmd_run_longshot_local(args)
    if args.command == "eval-longshot-official":
        return _cmd_eval_longshot_official(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


def _cmd_prepare_artifacts(args: argparse.Namespace) -> int:
    bundle = _build_qwen_bundle(args, logger=None)

    artifacts = bundle.memory_builder.prepare_artifacts(
        video_path=args.video,
        duration_seconds=args.duration_seconds,
        video_id=args.video_id,
    )
    output_dir = bundle.memory_builder.save_artifacts_dir(artifacts, args.output_dir)
    print(f"Saved artifacts to {output_dir}")
    return 0


def _cmd_build_memory(args: argparse.Namespace) -> int:
    builder = VideoMemoryBuilder(
        scene_duration_seconds=args.scene_duration_seconds,
        segment_duration_seconds=args.segment_duration_seconds,
        clip_duration_seconds=args.clip_duration_seconds,
    )
    artifacts = _load_artifacts(builder, args.artifacts)
    memory = builder.build_from_artifacts(artifacts)
    builder.save_memory(memory, args.output)
    print(f"Saved memory JSON to {args.output}")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    builder = VideoMemoryBuilder()
    memory = builder.load_memory(args.memory)
    logger = VideoRLMLogger(log_dir=args.log_dir) if args.log_dir else None
    runner = _build_runner(args, logger=logger)
    result = runner.run(args.question, memory, task_type=args.task_type)

    print(result.answer)
    if args.trace_out:
        output_path = Path(args.trace_out)
        output_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        print(f"Saved trace to {output_path}")
    return 0


def _cmd_run_longshot(args: argparse.Namespace) -> int:
    logger = VideoRLMLogger(log_dir=args.log_dir) if args.log_dir else None
    bundle = _build_qwen_bundle(args, logger=logger)
    output_path = Path(args.output)
    artifact_dir = Path(args.artifacts_dir) if args.artifacts_dir else output_path.parent / "artifacts"
    memory_dir = Path(args.memory_dir) if args.memory_dir else output_path.parent / "memories"
    trace_dir = Path(args.trace_dir) if args.trace_dir else None

    runner = LongShOTBenchmarkRunner(
        video_rlm=bundle.controller,
        memory_builder=bundle.memory_builder,
        video_resolver=LongShOTVideoResolver(
            args.video_dir,
            download_missing=args.download_missing,
            yt_dlp_bin=args.yt_dlp_bin,
            cookies_from_browser=args.cookies_from_browser,
        ),
        artifact_cache_dir=artifact_dir,
        memory_cache_dir=memory_dir,
        trace_dir=trace_dir,
        history_mode=args.history_mode,
    )
    samples = load_longshot_samples(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        split=args.split,
        sample_limit=args.sample_limit,
        sample_ids=args.sample_id,
        video_ids=args.video_id,
        task_filters=args.task_filter,
    )
    results = runner.run_samples(samples, output_path=output_path)
    print(f"Saved {len(results)} LongShOT prediction records to {output_path}")
    return 0


def _cmd_download_qwen_local_models(args: argparse.Namespace) -> int:
    config = _build_local_qwen_config(args)
    downloads = config.download_models()
    for name, path in downloads.items():
        print(f"{name}: {path}")
    return 0


def _cmd_run_longshot_local(args: argparse.Namespace) -> int:
    logger = VideoRLMLogger(log_dir=args.log_dir) if args.log_dir else None
    config = _build_local_qwen_config(args)
    bundle = config.build_bundle(
        logger=logger,
        max_steps=args.max_steps,
        search_top_k=args.search_top_k,
        max_frontier_items=args.max_frontier_items,
    )
    output_path = Path(args.output)
    artifact_dir = Path(args.artifacts_dir) if args.artifacts_dir else output_path.parent / "artifacts"
    memory_dir = Path(args.memory_dir) if args.memory_dir else output_path.parent / "memories"
    trace_dir = Path(args.trace_dir) if args.trace_dir else None

    runner = LongShOTBenchmarkRunner(
        video_rlm=bundle.controller,
        memory_builder=bundle.memory_builder,
        video_resolver=LongShOTVideoResolver(
            args.video_dir,
            download_missing=args.download_missing,
            yt_dlp_bin=args.yt_dlp_bin,
            cookies_from_browser=args.cookies_from_browser,
        ),
        artifact_cache_dir=artifact_dir,
        memory_cache_dir=memory_dir,
        trace_dir=trace_dir,
        history_mode=args.history_mode,
    )
    samples = load_longshot_samples(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        split=args.split,
        sample_limit=args.sample_limit,
        sample_ids=args.sample_id,
        video_ids=args.video_id,
        task_filters=args.task_filter,
    )
    results = runner.run_samples(samples, output_path=output_path)
    print(f"Saved {len(results)} LongShOT prediction records to {output_path}")
    return 0


def _cmd_eval_longshot_official(args: argparse.Namespace) -> int:
    config = LongShOTOfficialEvalConfig(
        predictions_path=Path(args.predictions),
        eval_path=Path(args.eval_output),
        score_path=Path(args.score_output),
        summary_path=Path(args.summary_output),
        judge_model_name=args.judge_repo,
        judge_model_path=args.judge_model_path,
        judge_device=args.judge_device,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        max_new_tokens=args.max_new_tokens,
        sample_limit=args.sample_limit,
    )
    result = evaluate_predictions_official_style(config)
    print(
        "Saved official-style eval to "
        f"{config.eval_path} with overall accuracy {result.overall_accuracy * 100:.2f}%"
    )
    return 0


def _build_runner(args: argparse.Namespace, logger: VideoRLMLogger | None = None) -> VideoRLM:
    return VideoRLM(
        controller_backend="openai",
        controller_backend_kwargs={
            "model_name": args.controller_model,
            "base_url": args.controller_base_url,
            "api_key": args.controller_api_key,
        },
        logger=logger,
        max_steps=args.max_steps,
        search_top_k=args.search_top_k,
        max_frontier_items=args.max_frontier_items,
    )


def _load_artifacts(builder: VideoMemoryBuilder, path: str):
    input_path = Path(path)
    if input_path.is_dir():
        return builder.load_artifacts_dir(input_path)
    return builder.load_artifacts(input_path)


def _build_qwen_bundle(args: argparse.Namespace, logger: VideoRLMLogger | None):
    stack = QwenVideoStackConfig.from_shared_endpoint(
        base_url=args.base_url,
        api_key=args.api_key,
        controller_model=getattr(args, "controller_model", "Qwen3-8B"),
        visual_model=getattr(args, "visual_model", "Qwen3-VL-8B"),
        speech_model=getattr(args, "speech_model", "Qwen3-ASR-0.6B"),
        embedding_model=getattr(args, "embedding_model", None),
    )
    stack.ffmpeg_bin = getattr(args, "ffmpeg_bin", "ffmpeg")
    stack.scene_duration_seconds = getattr(args, "scene_duration_seconds", 180.0)
    stack.segment_duration_seconds = getattr(args, "segment_duration_seconds", 45.0)
    stack.clip_duration_seconds = getattr(args, "clip_duration_seconds", 15.0)
    return stack.build_bundle(
        logger=logger,
        max_steps=getattr(args, "max_steps", 8),
        search_top_k=getattr(args, "search_top_k", 5),
        max_frontier_items=getattr(args, "max_frontier_items", 8),
    )


def _build_local_qwen_config(args: argparse.Namespace) -> QwenLocalVideoStackConfig:
    config = QwenLocalVideoStackConfig.default(
        controller_device=args.controller_device,
        visual_device=args.visual_device,
        speech_device=args.speech_device,
        controller_model=args.controller_repo,
        visual_model=args.visual_repo,
        speech_model=args.speech_repo,
        forced_aligner_model=None if args.no_forced_aligner else args.forced_aligner_repo,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
    )
    config.ffmpeg_bin = getattr(args, "ffmpeg_bin", "ffmpeg")
    config.scene_duration_seconds = getattr(args, "scene_duration_seconds", 180.0)
    config.segment_duration_seconds = getattr(args, "segment_duration_seconds", 45.0)
    config.clip_duration_seconds = getattr(args, "clip_duration_seconds", 15.0)
    config.controller_enable_thinking = False
    return config


def _add_shared_qwen_endpoint_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible endpoint base URL")
    parser.add_argument("--api-key", help="API key for the endpoint")


def _add_local_qwen_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--controller-repo", default="Qwen/Qwen3-8B")
    parser.add_argument("--visual-repo", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--speech-repo", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--forced-aligner-repo", default="Qwen/Qwen3-ForcedAligner-0.6B")
    parser.add_argument("--no-forced-aligner", action="store_true")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attn-implementation")


if __name__ == "__main__":
    raise SystemExit(main())
