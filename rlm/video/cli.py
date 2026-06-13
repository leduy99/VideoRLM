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
    evaluate_predictions_answer_only,
    evaluate_predictions_official_style,
)
from rlm.video.memory import VideoMemoryBuilder
from rlm.video.qwen import (
    OpenAICompatibleModelConfig,
    QwenLocalVideoStackConfig,
    QwenVideoStackConfig,
)


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
    prepare.add_argument("--verbose", action="store_true", help="Print live progress to stdout")
    _add_visual_preprocessing_args(prepare)

    build_memory = subparsers.add_parser(
        "build-memory",
        help="Build a hierarchical memory JSON file from prepared artifacts.",
    )
    build_memory.add_argument("--artifacts", required=True, help="Artifact JSON file or directory")
    build_memory.add_argument("--output", required=True, help="Output memory JSON file")
    build_memory.add_argument("--scene-duration-seconds", type=float, default=180.0)
    build_memory.add_argument("--segment-duration-seconds", type=float, default=45.0)
    build_memory.add_argument("--clip-duration-seconds", type=float, default=15.0)
    _add_fine_speech_window_args(build_memory)
    build_memory.add_argument(
        "--use-pitome",
        action="store_true",
        help="Build memory assuming clip-only PiToMe visual summaries.",
    )
    build_memory.add_argument(
        "--parent-visual-summary-mode",
        choices=["auto", "full", "compact"],
        default="auto",
        help=(
            "How scene/segment nodes store child visual summaries. "
            "auto uses compact parent rollups for PiToMe and full summaries otherwise."
        ),
    )
    build_memory.add_argument(
        "--verbose", action="store_true", help="Print live progress to stdout"
    )

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
    ask.add_argument("--search-mode", choices=["lexical", "graph"], default="lexical")
    ask.add_argument("--max-frontier-items", type=int, default=8)
    ask.add_argument("--controller-model", default="Qwen3-8B")
    ask.add_argument("--controller-base-url", required=True)
    ask.add_argument("--controller-api-key")
    ask.add_argument("--verbose", action="store_true", help="Print live progress to stdout")

    longshot = subparsers.add_parser(
        "run-longshot",
        help="Run VideoRLM on LongShOTBench samples and emit LongShOT-compatible predictions.",
    )
    longshot.add_argument("--output", required=True, help="Output JSONL file")
    longshot.add_argument(
        "--video-dir", required=True, help="Directory containing benchmark videos"
    )
    longshot.add_argument("--dataset-path", default="MBZUAI/longshot-bench")
    longshot.add_argument(
        "--dataset-name",
        default="postvalid_v2",
        help=(
            "Optional Hugging Face dataset config name. The current "
            "MBZUAI/longshot-bench benchmark config is postvalid_v2."
        ),
    )
    longshot.add_argument(
        "--longshot-context-dataset-name",
        default="postvalid_v1",
        help=(
            "Internal LongShot variant name passed to VideoRLM prompts/routing. "
            "This is separate from the Hugging Face config name."
        ),
    )
    longshot.add_argument("--split", default="test")
    longshot.add_argument("--sample-limit", type=int)
    longshot.add_argument(
        "--sample-start-index",
        type=int,
        help="1-based inclusive sample index after filtering/sorting.",
    )
    longshot.add_argument(
        "--sample-end-index",
        type=int,
        help="1-based inclusive sample index after filtering/sorting.",
    )
    longshot.add_argument("--sample-id", action="append", default=[])
    longshot.add_argument("--video-id", action="append", default=[])
    longshot.add_argument("--task-filter", action="append", default=[])
    longshot.add_argument("--download-missing", action="store_true")
    longshot.add_argument(
        "--skip-unavailable-videos",
        action="store_true",
        help="Skip benchmark samples whose video cannot be found or downloaded.",
    )
    longshot.add_argument(
        "--memory-cache-only",
        action="store_true",
        help="Skip samples whose memory JSON is not already cached; never build video memory.",
    )
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
    longshot.add_argument("--verbose", action="store_true", help="Print live progress to stdout")
    longshot.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the LongShOT benchmark progress bar.",
    )
    _add_visual_preprocessing_args(longshot)
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
    longshot_local.add_argument(
        "--video-dir", required=True, help="Directory containing benchmark videos"
    )
    longshot_local.add_argument("--dataset-path", default="MBZUAI/longshot-bench")
    longshot_local.add_argument(
        "--dataset-name",
        default="postvalid_v2",
        help=(
            "Optional Hugging Face dataset config name. The current "
            "MBZUAI/longshot-bench benchmark config is postvalid_v2."
        ),
    )
    longshot_local.add_argument(
        "--longshot-context-dataset-name",
        default="postvalid_v1",
        help=(
            "Internal LongShot variant name passed to VideoRLM prompts/routing. "
            "This is separate from the Hugging Face config name."
        ),
    )
    longshot_local.add_argument("--split", default="test")
    longshot_local.add_argument("--sample-limit", type=int)
    longshot_local.add_argument(
        "--sample-start-index",
        type=int,
        help="1-based inclusive sample index after filtering/sorting.",
    )
    longshot_local.add_argument(
        "--sample-end-index",
        type=int,
        help="1-based inclusive sample index after filtering/sorting.",
    )
    longshot_local.add_argument("--sample-id", action="append", default=[])
    longshot_local.add_argument("--video-id", action="append", default=[])
    longshot_local.add_argument("--task-filter", action="append", default=[])
    longshot_local.add_argument("--download-missing", action="store_true")
    longshot_local.add_argument(
        "--skip-unavailable-videos",
        action="store_true",
        help="Skip benchmark samples whose video cannot be found or downloaded.",
    )
    longshot_local.add_argument(
        "--memory-cache-only",
        action="store_true",
        help="Skip samples whose memory JSON is not already cached; never build video memory.",
    )
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
    longshot_local.add_argument(
        "--verbose", action="store_true", help="Print live progress to stdout"
    )
    longshot_local.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the LongShOT benchmark progress bar.",
    )
    longshot_local.add_argument("--controller-device", default="cuda:0")
    longshot_local.add_argument("--visual-device", default="cuda:1")
    longshot_local.add_argument("--speech-device", default="cuda:2")
    _add_visual_preprocessing_args(longshot_local)
    _add_local_qwen_args(longshot_local)

    official_eval = subparsers.add_parser(
        "eval-longshot-official",
        help="Evaluate LongShOT predictions with official-style rubric prompts and scoring.",
    )
    official_eval.add_argument("--predictions", required=True, help="Input predictions JSONL file")
    official_eval.add_argument("--eval-output", required=True, help="Output evaluated JSONL file")
    official_eval.add_argument("--score-output", required=True, help="Human-readable score report")
    official_eval.add_argument(
        "--summary-output", required=True, help="Machine-readable score summary"
    )
    official_eval.add_argument("--judge-repo", default="Qwen/Qwen3-14B")
    official_eval.add_argument("--judge-model-path")
    official_eval.add_argument("--judge-device", default="cuda:0")
    official_eval.add_argument("--torch-dtype", default="bfloat16")
    official_eval.add_argument("--attn-implementation")
    official_eval.add_argument("--max-new-tokens", type=int, default=96)
    official_eval.add_argument("--sample-limit", type=int)
    official_eval.add_argument(
        "--answer-only",
        action="store_true",
        help="Judge only final-answer correctness instead of every official rubric criterion.",
    )

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
        enable_fine_speech_windows=getattr(args, "enable_fine_speech_windows", False),
        fine_speech_window_seconds=getattr(args, "fine_speech_window_seconds", 15.0),
        fine_speech_window_stride_seconds=getattr(
            args,
            "fine_speech_window_stride_seconds",
            5.0,
        ),
        visual_span_mode="clip" if args.use_pitome else "scene_and_clip",
        aggregate_child_visual_summaries=args.use_pitome,
        parent_visual_summary_mode=_resolve_parent_visual_summary_mode(args),
        verbose=args.verbose,
    )
    artifacts = _load_artifacts(builder, args.artifacts)
    memory = builder.build_from_artifacts(artifacts)
    builder.save_memory(memory, args.output)
    print(f"Saved memory JSON to {args.output}")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    builder = VideoMemoryBuilder()
    memory = builder.load_memory(args.memory)
    logger = _build_logger(args)
    runner = _build_runner(args, logger=logger)
    result = runner.run(args.question, memory, task_type=args.task_type)

    print(result.answer)
    if args.trace_out:
        output_path = Path(args.trace_out)
        output_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        print(f"Saved trace to {output_path}")
    return 0


def _cmd_run_longshot(args: argparse.Namespace) -> int:
    logger = _build_logger(args)
    bundle = _build_qwen_bundle(args, logger=logger)
    output_path = Path(args.output)
    artifact_dir = (
        Path(args.artifacts_dir) if args.artifacts_dir else output_path.parent / "artifacts"
    )
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
        dataset_name=args.dataset_name,
        context_dataset_name=args.longshot_context_dataset_name,
        history_mode=args.history_mode,
        verbose=args.verbose,
        show_progress=not args.no_progress,
        skip_unavailable_videos=args.skip_unavailable_videos,
        memory_cache_only=args.memory_cache_only,
    )
    samples = load_longshot_samples(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        split=args.split,
        sample_limit=args.sample_limit,
        sample_start_index=args.sample_start_index,
        sample_end_index=args.sample_end_index,
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
    logger = _build_logger(args)
    config = _build_local_qwen_config(args)
    bundle = config.build_bundle(
        logger=logger,
        max_steps=args.max_steps,
        search_top_k=args.search_top_k,
        max_frontier_items=args.max_frontier_items,
    )
    output_path = Path(args.output)
    artifact_dir = (
        Path(args.artifacts_dir) if args.artifacts_dir else output_path.parent / "artifacts"
    )
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
        dataset_name=args.dataset_name,
        context_dataset_name=args.longshot_context_dataset_name,
        history_mode=args.history_mode,
        verbose=args.verbose,
        show_progress=not args.no_progress,
        skip_unavailable_videos=args.skip_unavailable_videos,
        memory_cache_only=args.memory_cache_only,
    )
    samples = load_longshot_samples(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        split=args.split,
        sample_limit=args.sample_limit,
        sample_start_index=args.sample_start_index,
        sample_end_index=args.sample_end_index,
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
    if args.answer_only:
        result = evaluate_predictions_answer_only(config)
        mode = "answer-only"
    else:
        result = evaluate_predictions_official_style(config)
        mode = "official-style"
    print(
        f"Saved {mode} eval to "
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
        search_mode=args.search_mode,
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
    stack.frame_count = getattr(args, "frame_count", 3)
    stack.frame_width = getattr(args, "frame_width", 768)
    stack.scene_duration_seconds = getattr(args, "scene_duration_seconds", 180.0)
    stack.segment_duration_seconds = getattr(args, "segment_duration_seconds", 45.0)
    stack.clip_duration_seconds = getattr(args, "clip_duration_seconds", 15.0)
    stack.enable_fine_speech_windows = getattr(args, "enable_fine_speech_windows", False)
    stack.fine_speech_window_seconds = getattr(args, "fine_speech_window_seconds", 15.0)
    stack.fine_speech_window_stride_seconds = getattr(
        args,
        "fine_speech_window_stride_seconds",
        5.0,
    )
    stack.verbose = getattr(args, "verbose", False)
    stack.enable_refinement_frontier = not getattr(
        args,
        "disable_refinement_frontier",
        False,
    )
    stack.enable_dynamic_evidence_retrieval = not getattr(
        args,
        "disable_dynamic_evidence_retrieval",
        False,
    )
    _apply_visual_preprocessing_args(stack, args)
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
        semantic_frame_embedding_model=getattr(args, "semantic_frame_embedding_repo", None),
        semantic_frame_embedding_device=getattr(
            args,
            "semantic_frame_embedding_device",
            "cpu",
        ),
        semantic_frame_embedding_torch_dtype=getattr(
            args,
            "semantic_frame_embedding_torch_dtype",
            "float32",
        ),
        speech_embedding_model=getattr(args, "speech_embedding_repo", None),
        speech_embedding_device=getattr(args, "speech_embedding_device", "cpu"),
        video_window_embedding_model=(
            getattr(args, "video_window_reranker_repo", None)
            if getattr(args, "enable_video_window_reranking", False)
            else None
        ),
        video_window_embedding_device=getattr(
            args,
            "video_window_reranker_device",
            "cuda:0",
        ),
        video_window_embedding_torch_dtype=getattr(
            args,
            "video_window_reranker_torch_dtype",
            "float32",
        ),
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
    )
    controller_model_path = getattr(args, "controller_model_path", None)
    if controller_model_path:
        config.controller.model_path = controller_model_path
    controller_max_new_tokens = getattr(args, "controller_max_new_tokens", None)
    if controller_max_new_tokens is not None:
        config.controller.max_new_tokens = controller_max_new_tokens
    speech_model_path = getattr(args, "speech_model_path", None)
    if speech_model_path:
        config.speech.model_path = speech_model_path
    if getattr(args, "controller_trust_remote_code", False):
        config.controller.trust_remote_code = True
    controller_api_base_url = getattr(args, "controller_api_base_url", None)
    controller_api_model = getattr(args, "controller_api_model", None)
    if controller_api_base_url or controller_api_model:
        if not controller_api_base_url or not controller_api_model:
            raise ValueError(
                "--controller-api-base-url and --controller-api-model must be set together"
            )
        completion_kwargs = {}
        controller_api_max_tokens = getattr(args, "controller_api_max_tokens", None)
        if controller_api_max_tokens is not None:
            completion_kwargs["max_tokens"] = controller_api_max_tokens
        extra_client_kwargs = (
            {"completion_kwargs": completion_kwargs} if completion_kwargs else None
        )
        config.api_controller = OpenAICompatibleModelConfig(
            model_name=controller_api_model,
            base_url=controller_api_base_url,
            api_key=getattr(args, "controller_api_key", None),
            timeout=getattr(args, "controller_api_timeout", 300.0),
            extra_client_kwargs=extra_client_kwargs,
        )
    semantic_model_path = getattr(args, "semantic_frame_embedding_model_path", None)
    if config.semantic_frame_embedding is not None and semantic_model_path:
        config.semantic_frame_embedding.model_path = semantic_model_path
    speech_embedding_model_path = getattr(args, "speech_embedding_model_path", None)
    if config.speech_embedding is not None and speech_embedding_model_path:
        config.speech_embedding.model_path = speech_embedding_model_path
    video_window_model_path = getattr(args, "video_window_reranker_model_path", None)
    if config.video_window_embedding is not None and video_window_model_path:
        config.video_window_embedding.model_path = video_window_model_path
    config.semantic_frame_embedding_batch_size = getattr(
        args,
        "semantic_frame_embedding_batch_size",
        8,
    )
    config.enable_video_window_reranking = getattr(
        args,
        "enable_video_window_reranking",
        False,
    )
    config.video_window_rerank_candidate_count = getattr(
        args,
        "video_window_rerank_candidate_count",
        24,
    )
    config.video_window_rerank_weight = getattr(args, "video_window_rerank_weight", 0.75)
    config.video_window_rerank_window_seconds = getattr(
        args,
        "video_window_rerank_window_seconds",
        None,
    )
    config.video_window_rerank_min_score = getattr(args, "video_window_rerank_min_score", None)
    config.video_window_embedding_frame_count = getattr(
        args,
        "video_window_reranker_frame_count",
        8,
    )
    config.video_window_embedding_frame_size = getattr(
        args,
        "video_window_reranker_frame_size",
        224,
    )
    config.ffmpeg_bin = getattr(args, "ffmpeg_bin", "ffmpeg")
    config.frame_count = getattr(args, "frame_count", 3)
    config.frame_width = getattr(args, "frame_width", 768)
    config.scene_duration_seconds = getattr(args, "scene_duration_seconds", 180.0)
    config.segment_duration_seconds = getattr(args, "segment_duration_seconds", 45.0)
    config.clip_duration_seconds = getattr(args, "clip_duration_seconds", 15.0)
    config.speech_chunk_duration_seconds = getattr(args, "speech_chunk_duration_seconds", 60.0)
    config.speech_asr_chunk_batch_size = getattr(args, "speech_asr_chunk_batch_size", 1)
    config.speech.max_new_tokens = getattr(args, "speech_max_new_tokens", 512)
    config.enable_speech_recognition = not getattr(args, "no_speech_recognition", False)
    config.speech_backend = getattr(args, "speech_backend", "qwen")
    config.faster_whisper_model = getattr(args, "faster_whisper_model", "small")
    config.faster_whisper_device = getattr(args, "faster_whisper_device", "cpu")
    config.faster_whisper_compute_type = getattr(
        args,
        "faster_whisper_compute_type",
        "default",
    )
    config.lazy_speech_refinement = getattr(args, "lazy_speech_refinement", False)
    config.enable_targeted_asr_refinement = getattr(
        args,
        "enable_targeted_asr_refinement",
        False,
    )
    config.force_eager_speech_recognition = getattr(
        args,
        "eager_speech_recognition",
        False,
    )
    if config.lazy_speech_refinement and config.force_eager_speech_recognition:
        raise ValueError(
            "--lazy-speech-refinement and --eager-speech-recognition cannot both be set"
        )
    config.lazy_visual_refinement = getattr(args, "lazy_visual_refinement", False)
    config.offload_components_after_use = getattr(
        args,
        "offload_components_after_use",
        False,
    )
    config.enable_dynamic_evidence_retrieval = not getattr(
        args,
        "disable_dynamic_evidence_retrieval",
        False,
    )
    config.enable_refinement_frontier = not getattr(
        args,
        "disable_refinement_frontier",
        False,
    )
    config.enable_paddle_ocr = getattr(args, "enable_paddle_ocr", False)
    config.paddle_ocr_lang = getattr(args, "paddle_ocr_lang", "en")
    config.paddle_ocr_version = getattr(args, "paddle_ocr_version", "PP-OCRv5")
    config.paddle_ocr_device = getattr(args, "paddle_ocr_device", None)
    config.paddle_ocr_window_seconds = getattr(args, "paddle_ocr_window_seconds", 45.0)
    config.paddle_ocr_frame_count = getattr(args, "paddle_ocr_frame_count", 6)
    config.paddle_ocr_frame_width = getattr(args, "paddle_ocr_frame_width", 960)
    config.paddle_ocr_min_confidence = getattr(args, "paddle_ocr_min_confidence", 0.35)
    config.paddle_ocr_enable_mkldnn = getattr(args, "paddle_ocr_enable_mkldnn", False)
    config.paddle_ocr_cache_dir = getattr(args, "paddle_ocr_cache_dir", None)
    config.paddle_ocr_frame_extraction_strategy = getattr(
        args,
        "paddle_ocr_frame_extraction_strategy",
        "seek",
    )
    config.paddle_ocr_frame_extraction_workers = getattr(
        args,
        "paddle_ocr_frame_extraction_workers",
        1,
    )
    config.paddle_ocr_text_detection_model_name = getattr(
        args,
        "paddle_ocr_text_detection_model_name",
        None,
    )
    config.paddle_ocr_text_recognition_model_name = getattr(
        args,
        "paddle_ocr_text_recognition_model_name",
        None,
    )
    config.paddle_ocr_text_recognition_batch_size = getattr(
        args,
        "paddle_ocr_text_recognition_batch_size",
        None,
    )
    config.enable_controller_evidence_classifier = getattr(
        args,
        "enable_controller_evidence_classifier",
        False,
    )
    config.controller_enable_thinking = False
    config.verbose = getattr(args, "verbose", False)
    _apply_visual_preprocessing_args(config, args)
    _apply_official_video_strategy(config)
    return config


def _build_logger(args: argparse.Namespace) -> VideoRLMLogger | None:
    verbose = getattr(args, "verbose", False)
    log_dir = getattr(args, "log_dir", None)
    if not verbose and not log_dir:
        return None
    return VideoRLMLogger(log_dir=log_dir, console=verbose)


def _add_shared_qwen_endpoint_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible endpoint base URL")
    parser.add_argument("--api-key", help="API key for the endpoint")


def _add_visual_preprocessing_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--frame-count", type=int, default=3)
    parser.add_argument("--frame-width", type=int, default=768)
    parser.add_argument(
        "--use-pitome",
        action="store_true",
        help=(
            "Use the official lazy PiToMe strategy: cheap PiToMe/SigLIP visual indexing, "
            "lazy ASR indexing, graph search, and on-demand QwenVL/QwenASR refinement."
        ),
    )
    parser.add_argument("--pitome-dense-frame-rate", type=float, default=1.0)
    parser.add_argument("--pitome-min-frame-count", type=int)
    parser.add_argument("--pitome-protect-ratio", type=float, default=0.15)
    parser.add_argument("--pitome-similarity-threshold", type=float, default=0.8)
    parser.add_argument("--pitome-embedding-size", type=int, default=16)
    parser.add_argument("--pitome-embedding-backend", choices=["pixel", "hybrid"], default="pixel")
    parser.add_argument(
        "--pitome-frame-width",
        type=int,
        help="Optional frame extraction width for the cheap PiToMe index pass.",
    )
    parser.add_argument(
        "--pitome-embedding-device",
        help="Optional Torch device for PiToMe embedding and similarity math, for example cuda:0.",
    )
    parser.add_argument(
        "--pitome-frame-extraction-strategy",
        choices=["auto", "batch", "seek", "sequence"],
        default="auto",
        help="FFmpeg strategy for extracting cheap PiToMe frames.",
    )
    parser.add_argument(
        "--pitome-frame-extraction-workers",
        type=int,
        default=1,
        help="Parallel ffmpeg workers for the seek-based PiToMe frame extraction path.",
    )
    parser.add_argument("--pitome-anchor-frame-count", type=int, default=0)
    parser.add_argument("--pitome-max-selected-frames", type=int)
    parser.add_argument(
        "--pitome-scene-threshold",
        type=float,
        default=0.35,
        help="FFmpeg scene-change threshold for adding real scene-boundary frames.",
    )
    parser.add_argument(
        "--pitome-max-scene-boundary-frames",
        type=int,
        default=6,
        help="Maximum detected scene-boundary frames to add inside each PiToMe span.",
    )
    parser.add_argument(
        "--pitome-scene-sample-rate",
        type=float,
        default=1.0,
        help="Frame rate sampled before FFmpeg scene-boundary detection; use 0 to disable sampling.",
    )
    parser.add_argument(
        "--pitome-scene-keyframes-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use decoder keyframes only for faster PiToMe scene-boundary detection.",
    )
    parser.add_argument(
        "--pitome-edge-boundary-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add cheap start/end boundary frames to each PiToMe span.",
    )
    parser.add_argument(
        "--visual-index-batch-size",
        type=int,
        default=1,
        help="Number of lazy visual-index spans to batch for semantic frame embedding.",
    )
    parser.add_argument(
        "--visual-index-workers",
        type=int,
        default=1,
        help="Parallel workers for lazy visual-index frame selection.",
    )
    parser.add_argument(
        "--parent-visual-summary-mode",
        choices=["auto", "full", "compact"],
        default="auto",
        help=(
            "How scene/segment nodes store child visual summaries. "
            "auto uses compact parent rollups for PiToMe and full summaries otherwise."
        ),
    )
    parser.add_argument(
        "--search-mode",
        choices=["auto", "lexical", "graph"],
        default="auto",
        help=(
            "Search backend. auto uses graph search for the lazy PiToMe strategy and "
            "lexical search for original VideoRLM."
        ),
    )
    _add_fine_speech_window_args(parser)


def _add_fine_speech_window_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--enable-fine-speech-windows",
        action="store_true",
        help=(
            "Build overlapping fine ASR retrieval windows while keeping scene/segment/clip "
            "nodes as context."
        ),
    )
    parser.add_argument(
        "--fine-speech-window-seconds",
        type=float,
        default=15.0,
        help="Duration of each fine ASR retrieval window.",
    )
    parser.add_argument(
        "--fine-speech-window-stride-seconds",
        type=float,
        default=5.0,
        help="Stride between fine ASR retrieval windows.",
    )


def _apply_visual_preprocessing_args(config, args: argparse.Namespace) -> None:
    config.use_pitome = getattr(args, "use_pitome", False)
    config.pitome_dense_frame_rate = getattr(args, "pitome_dense_frame_rate", 1.0)
    config.pitome_min_frame_count = getattr(args, "pitome_min_frame_count", None)
    config.pitome_protect_ratio = getattr(args, "pitome_protect_ratio", 0.15)
    config.pitome_similarity_threshold = getattr(args, "pitome_similarity_threshold", 0.8)
    config.pitome_embedding_size = getattr(args, "pitome_embedding_size", 16)
    config.pitome_embedding_backend = getattr(args, "pitome_embedding_backend", "pixel")
    config.pitome_embedding_device = getattr(args, "pitome_embedding_device", None)
    config.pitome_frame_width = getattr(args, "pitome_frame_width", None)
    config.pitome_frame_extraction_strategy = getattr(
        args,
        "pitome_frame_extraction_strategy",
        "auto",
    )
    config.pitome_frame_extraction_workers = getattr(args, "pitome_frame_extraction_workers", 1)
    config.pitome_anchor_frame_count = getattr(args, "pitome_anchor_frame_count", 0)
    config.pitome_max_selected_frames = getattr(args, "pitome_max_selected_frames", None)
    config.pitome_scene_threshold = getattr(args, "pitome_scene_threshold", 0.35)
    config.pitome_max_scene_boundary_frames = getattr(
        args,
        "pitome_max_scene_boundary_frames",
        6,
    )
    scene_sample_rate = getattr(args, "pitome_scene_sample_rate", 1.0)
    config.pitome_scene_sample_rate = None if scene_sample_rate == 0 else scene_sample_rate
    config.pitome_scene_keyframes_only = getattr(args, "pitome_scene_keyframes_only", True)
    config.pitome_edge_boundary_frames = getattr(args, "pitome_edge_boundary_frames", True)
    config.visual_index_batch_size = getattr(args, "visual_index_batch_size", 1)
    config.visual_index_workers = getattr(args, "visual_index_workers", 1)
    config.enable_fine_speech_windows = getattr(args, "enable_fine_speech_windows", False)
    config.fine_speech_window_seconds = getattr(args, "fine_speech_window_seconds", 15.0)
    config.fine_speech_window_stride_seconds = getattr(
        args,
        "fine_speech_window_stride_seconds",
        5.0,
    )
    parent_mode = getattr(args, "parent_visual_summary_mode", "auto")
    config.parent_visual_summary_mode = None if parent_mode == "auto" else parent_mode
    search_mode = getattr(args, "search_mode", "auto")
    config.search_mode = None if search_mode == "auto" else search_mode


def _apply_official_video_strategy(config) -> None:
    lazy_visual_requested = (
        bool(getattr(config, "use_pitome", False))
        or bool(getattr(config, "lazy_visual_refinement", False))
    )
    lazy_speech_requested = bool(getattr(config, "lazy_speech_refinement", False))
    if not lazy_visual_requested and not lazy_speech_requested:
        return
    config.use_pitome = True
    config.lazy_visual_refinement = True
    if (
        getattr(config, "enable_speech_recognition", True)
        and not getattr(config, "force_eager_speech_recognition", False)
    ):
        config.lazy_speech_refinement = True


def _resolve_parent_visual_summary_mode(args: argparse.Namespace) -> str:
    parent_mode = getattr(args, "parent_visual_summary_mode", "auto")
    if parent_mode != "auto":
        return parent_mode
    return "compact" if getattr(args, "use_pitome", False) else "full"


def _add_local_qwen_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--controller-repo", default="Qwen/Qwen3-8B")
    parser.add_argument("--controller-model-path")
    parser.add_argument("--controller-max-new-tokens", type=int)
    parser.add_argument("--controller-trust-remote-code", action="store_true")
    parser.add_argument(
        "--controller-api-base-url",
        help="Use an OpenAI-compatible API controller instead of the local controller.",
    )
    parser.add_argument("--controller-api-key")
    parser.add_argument("--controller-api-model")
    parser.add_argument("--controller-api-max-tokens", type=int)
    parser.add_argument("--controller-api-timeout", type=float, default=300.0)
    parser.add_argument("--visual-repo", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--speech-repo", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument(
        "--speech-model-path",
        help="Optional local path for the ASR checkpoint. Defaults to output/models/<speech-repo>.",
    )
    parser.add_argument("--forced-aligner-repo", default="Qwen/Qwen3-ForcedAligner-0.6B")
    parser.add_argument("--no-forced-aligner", action="store_true")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attn-implementation")
    parser.add_argument(
        "--no-speech-recognition",
        action="store_true",
        help="Skip local ASR during memory construction for faster visual-only runs.",
    )
    parser.add_argument(
        "--speech-chunk-duration-seconds",
        type=float,
        default=60.0,
        help="Local Qwen ASR audio chunk duration in seconds.",
    )
    parser.add_argument(
        "--speech-max-new-tokens",
        type=int,
        default=512,
        help="Maximum generated tokens per local Qwen ASR chunk.",
    )
    parser.add_argument(
        "--speech-asr-chunk-batch-size",
        type=int,
        default=1,
        help=(
            "Number of local Qwen ASR chunks to transcribe in one batched model call. "
            "Use 1 if batching is unsupported or uses too much GPU memory."
        ),
    )
    parser.add_argument(
        "--speech-backend",
        choices=["qwen", "faster-whisper"],
        default="qwen",
        help="Local ASR backend. Use --no-speech-recognition to disable ASR entirely.",
    )
    parser.add_argument("--faster-whisper-model", default="small")
    parser.add_argument("--faster-whisper-device", default="cpu")
    parser.add_argument("--faster-whisper-compute-type", default="default")
    parser.add_argument(
        "--lazy-speech-refinement",
        action="store_true",
        help=(
            "Compatibility alias for the official lazy PiToMe strategy. "
            "Build timestamp-only lazy ASR memory, then run local ASR only when "
            "a retrieved speech node is opened."
        ),
    )
    parser.add_argument(
        "--eager-speech-recognition",
        "--no-lazy-speech-refinement",
        dest="eager_speech_recognition",
        action="store_true",
        help=(
            "Build full ASR transcripts during memory construction even when PiToMe "
            "or lazy visual refinement is enabled. Use this for speech-heavy LongShot "
            "postvalid_v1 tasks where retrieval needs searchable transcript text."
        ),
    )
    parser.add_argument(
        "--enable-targeted-asr-refinement",
        action="store_true",
        help=(
            "Run a second local ASR pass on selected speech windows when the opened "
            "coarse transcript is weak for temporal/explanatory LongShot questions."
        ),
    )
    parser.add_argument(
        "--lazy-visual-refinement",
        action="store_true",
        help=(
            "Compatibility alias for the official lazy PiToMe strategy. "
            "Build cheap PiToMe+embedding visual memory, then run local QwenVL only when "
            "a retrieved visual node is opened."
        ),
    )
    parser.add_argument(
        "--enable-paddle-ocr",
        action="store_true",
        help="Run lightweight PaddleOCR over sampled video frames and index exact screen text.",
    )
    parser.add_argument("--paddle-ocr-lang", default="en")
    parser.add_argument("--paddle-ocr-version", default="PP-OCRv5")
    parser.add_argument("--paddle-ocr-device", default=None)
    parser.add_argument("--paddle-ocr-window-seconds", type=float, default=45.0)
    parser.add_argument("--paddle-ocr-frame-count", type=int, default=6)
    parser.add_argument("--paddle-ocr-frame-width", type=int, default=960)
    parser.add_argument("--paddle-ocr-min-confidence", type=float, default=0.35)
    parser.add_argument(
        "--paddle-ocr-enable-mkldnn",
        action="store_true",
        help="Enable Paddle MKLDNN/oneDNN for OCR. Disabled by default due runtime crashes.",
    )
    parser.add_argument("--paddle-ocr-cache-dir")
    parser.add_argument(
        "--paddle-ocr-frame-extraction-strategy",
        choices=["auto", "batch", "seek", "sequence"],
        default="seek",
    )
    parser.add_argument("--paddle-ocr-frame-extraction-workers", type=int, default=1)
    parser.add_argument("--paddle-ocr-text-detection-model-name")
    parser.add_argument("--paddle-ocr-text-recognition-model-name")
    parser.add_argument("--paddle-ocr-text-recognition-batch-size", type=int)
    parser.add_argument(
        "--enable-controller-evidence-classifier",
        action="store_true",
        help=(
            "Use the controller LLM to classify opened evidence as "
            "core/support/background/noise before filling evidence slots."
        ),
    )
    parser.add_argument(
        "--semantic-frame-embedding-repo",
        help=(
            "Optional local image-text embedding model for PiToMe selected frames, "
            "for example openai/clip-vit-base-patch32."
        ),
    )
    parser.add_argument("--semantic-frame-embedding-model-path")
    parser.add_argument("--semantic-frame-embedding-device", default="cpu")
    parser.add_argument("--semantic-frame-embedding-torch-dtype", default="float32")
    parser.add_argument("--semantic-frame-embedding-batch-size", type=int, default=8)
    parser.add_argument(
        "--speech-embedding-repo",
        help=(
            "Optional sentence-transformers model for speech/ASR dense retrieval, "
            "for example sentence-transformers/all-MiniLM-L6-v2."
        ),
    )
    parser.add_argument("--speech-embedding-model-path")
    parser.add_argument("--speech-embedding-device", default="cpu")
    parser.add_argument(
        "--enable-video-window-reranking",
        action="store_true",
        help=(
            "Enable stage-2 video-window reranking after SigLIP/semantic first-stage "
            "retrieval."
        ),
    )
    parser.add_argument(
        "--video-window-reranker-repo",
        default="OpenGVLab/InternVideo2-Stage2_6B",
        help="Local InternVideo stage-2 embedding model used for window reranking.",
    )
    parser.add_argument("--video-window-reranker-model-path")
    parser.add_argument("--video-window-reranker-device", default="cuda:0")
    parser.add_argument("--video-window-reranker-torch-dtype", default="float32")
    parser.add_argument("--video-window-reranker-frame-count", type=int, default=8)
    parser.add_argument("--video-window-reranker-frame-size", type=int, default=224)
    parser.add_argument("--video-window-rerank-candidate-count", type=int, default=24)
    parser.add_argument("--video-window-rerank-weight", type=float, default=0.75)
    parser.add_argument("--video-window-rerank-window-seconds", type=float)
    parser.add_argument("--video-window-rerank-min-score", type=float)
    parser.add_argument(
        "--offload-components-after-use",
        action="store_true",
        help=(
            "Unload idle local models after preprocessing phases, stage-2 reranking, "
            "and each sample run to reduce peak GPU memory."
        ),
    )
    parser.add_argument(
        "--disable-dynamic-evidence-retrieval",
        action="store_true",
        help=(
            "Disable the LongShot multi-target dynamic-programming retrieval planner. "
            "By default complex LongShot questions retrieve an ordered evidence chain."
        ),
    )
    parser.add_argument(
        "--disable-refinement-frontier",
        action="store_true",
        help=(
            "Disable OPEN-time refinement frontier expansion so controller search stays "
            "on the selected retrieval chain instead of chasing child nodes."
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
