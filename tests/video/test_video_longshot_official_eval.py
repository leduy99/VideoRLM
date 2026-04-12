import json
from pathlib import Path

import rlm.video.cli as video_cli
from rlm.video.longshot_official_eval import (
    LongShOTOfficialEvalConfig,
    build_official_criterion_prompt,
    calculate_official_scores,
    parse_criteria_met,
)


def test_parse_criteria_met_handles_direct_json():
    assert parse_criteria_met('{"criteria_met": true}') is True
    assert parse_criteria_met('{"criteria_met": false}') is False


def test_parse_criteria_met_handles_fenced_json():
    assert parse_criteria_met("```json\n{\"criteria_met\": true}\n```") is True


def test_build_official_criterion_prompt_mentions_all_inputs():
    prompt = build_official_criterion_prompt(
        ground_truth_response="ground truth",
        model_response="candidate",
        criterion_description="must mention timing",
    )
    assert "ground truth" in prompt
    assert "candidate" in prompt
    assert "must mention timing" in prompt
    assert "criteria_met" in prompt


def test_calculate_official_scores_matches_weighted_rubric():
    eval_results = [
        {
            "task": "event_understanding",
            "conversations": [
                {
                    "role": "assistant",
                    "criteria": [
                        {"weight": 2, "criteria_met": True, "is_penalty": False},
                        {"weight": 1, "criteria_met": False, "is_penalty": False},
                    ],
                }
            ],
        },
        {
            "task": "motion_analysis",
            "conversations": [
                {
                    "role": "assistant",
                    "criteria": [
                        {"weight": 1, "criteria_met": True, "is_penalty": False},
                        {"weight": -1, "criteria_met": True, "is_penalty": True},
                    ],
                }
            ],
        },
    ]

    summary = calculate_official_scores(eval_results)

    assert summary["task_accuracies"]["event_understanding"] == 2 / 3
    assert summary["task_accuracies"]["compositional_reasoning"] == 1.0
    assert summary["task_counts"]["event_understanding"] == 1
    assert summary["task_counts"]["compositional_reasoning"] == 1


def test_cli_eval_longshot_official(monkeypatch, tmp_path: Path, capsys):
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps(
            {
                "sample_id": "sample_1",
                "task": "entity_recognition",
                "conversations": [
                    {
                        "role": "assistant",
                        "content": "gold",
                        "candidate_response": "pred",
                        "criteria": [
                            {
                                "description": "must identify entity",
                                "weight": 1,
                                "is_penalty": False,
                            }
                        ],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_eval(config: LongShOTOfficialEvalConfig):
        assert config.predictions_path == predictions_path
        config.eval_path.write_text("{}", encoding="utf-8")
        config.score_path.write_text("score", encoding="utf-8")
        config.summary_path.write_text("{}", encoding="utf-8")

        class Result:
            overall_accuracy = 0.5

        return Result()

    monkeypatch.setattr(video_cli, "evaluate_predictions_official_style", fake_eval)

    exit_code = video_cli.main(
        [
            "eval-longshot-official",
            "--predictions",
            str(predictions_path),
            "--eval-output",
            str(tmp_path / "eval.jsonl"),
            "--score-output",
            str(tmp_path / "score.txt"),
            "--summary-output",
            str(tmp_path / "summary.json"),
        ]
    )

    assert exit_code == 0
    assert "overall accuracy 50.00%" in capsys.readouterr().out
