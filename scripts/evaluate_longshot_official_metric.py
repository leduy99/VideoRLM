#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlm.video.longshot_official_eval import (  # noqa: E402
    LongShOTOfficialEvalConfig,
    calculate_official_scores,
    evaluate_predictions_answer_only,
    evaluate_predictions_official_style,
    load_jsonl,
    write_score_report,
    write_summary_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate LongShOT predictions with criterion-level judge calls, or rescore an "
            "existing judged JSONL using the LongShOT weighted rubric metric."
        )
    )
    parser.add_argument("--predictions", help="Input LongShOT prediction JSONL.")
    parser.add_argument("--eval-output", required=True, help="Judged/evaluated JSONL path.")
    parser.add_argument("--score-output", required=True, help="Human-readable metric report.")
    parser.add_argument("--summary-output", required=True, help="Machine-readable metric JSON.")
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Skip judge calls and calculate metrics from --eval-output.",
    )
    parser.add_argument(
        "--answer-only",
        action="store_true",
        help=(
            "Judge only final-answer semantic correctness. This ignores official "
            "tool/process criteria and writes one answer_correctness criterion per answer."
        ),
    )
    parser.add_argument("--model-name", help="Model name shown in the score report.")
    parser.add_argument("--judge-repo", default="Qwen/Qwen3-14B")
    parser.add_argument("--judge-model-path")
    parser.add_argument("--judge-device", default="cuda:0")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attn-implementation")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--sample-limit", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    eval_path = Path(args.eval_output)
    score_path = Path(args.score_output)
    summary_path = Path(args.summary_output)

    if args.score_only:
        if not eval_path.exists():
            raise FileNotFoundError(f"Cannot score missing eval JSONL: {eval_path}")
        eval_results = load_jsonl(eval_path)
        if args.sample_limit is not None:
            eval_results = eval_results[: args.sample_limit]
        if args.answer_only:
            incompatible_ids = [
                row.get("sample_id")
                for row in eval_results
                if row.get("evaluation_mode") != "answer_only"
            ]
            if incompatible_ids:
                raise ValueError(
                    "--answer-only --score-only requires an answer-only eval JSONL. "
                    f"Got incompatible rows in {eval_path}."
                )
        summary = calculate_official_scores(eval_results)
        model_name = args.model_name or eval_path.stem
        write_score_report(score_path=score_path, model_name=model_name, summary=summary)
        write_summary_json(
            summary_path=summary_path,
            summary=summary,
            extra={
                "eval_path": str(eval_path),
                "score_only": True,
                "evaluation_mode": (
                    "answer_only" if args.answer_only else "official_criteria"
                ),
                "sample_limit": args.sample_limit,
            },
        )
        print(
            "Saved LongShOT metric report to "
            f"{score_path} with overall accuracy {summary['overall_accuracy'] * 100:.2f}%"
        )
        return 0

    if not args.predictions:
        raise ValueError("--predictions is required unless --score-only is set.")

    predictions_path = Path(args.predictions)
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
    if args.answer_only:
        result = evaluate_predictions_answer_only(config)
        mode = "answer-only"
    else:
        result = evaluate_predictions_official_style(config)
        mode = "official-style"
    print(
        f"Saved LongShOT {mode} eval to "
        f"{eval_path} with overall accuracy {result.overall_accuracy * 100:.2f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
