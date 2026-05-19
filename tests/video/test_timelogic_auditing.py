import json
from pathlib import Path

import pytest

from rlm.video.timelogic_auditing import (
    build_audit_samples,
    mix_submission_by_category,
    parse_timelogic_category,
)
from scripts.timelogic.merge_submission_shards import build_submission
from scripts.timelogic.split_dataset import _video_balanced_split


def test_parse_timelogic_category_examples():
    assert (
        parse_timelogic_category("Which event always occurs before opening the door?")
        == "always_before"
    )
    assert (
        parse_timelogic_category("What happens immediately after placing the cup down?")
        == "immediate_next"
    )
    assert parse_timelogic_category("What did the person do until poured egg?") == "until_since"
    assert parse_timelogic_category("Does closing a refrigerator imply holding some food ?") == "implies"
    assert (
        parse_timelogic_category("Is it true that person sitting in a bed always co-occur with holding a book ?")
        == "always_cooccur"
    )
    assert parse_timelogic_category("What is the order of actions, first then second?") == "ordering"


def test_risk_increases_with_repeated_open(tmp_path: Path):
    trace = {
        "answer": "A",
        "state": {
            "question": "Which action occurs before adding salt?",
            "evidence_board": {"missing_required_slots": []},
            "evidence_ledger": [],
        },
        "trace": [
            {
                "action": {
                    "action_type": "OPEN",
                    "node_id": "clip_1",
                    "modality": "visual",
                    "target_slot": "main_claim",
                },
                "observation": {"kind": "open", "evidence": []},
            },
            {
                "action": {
                    "action_type": "OPEN",
                    "node_id": "clip_1",
                    "modality": "visual",
                    "target_slot": "main_claim",
                },
                "observation": {"kind": "open", "evidence": []},
            },
            {
                "action": {
                    "action_type": "OPEN",
                    "node_id": "clip_1",
                    "modality": "visual",
                    "target_slot": "main_claim",
                },
                "observation": {"kind": "open", "evidence": []},
            },
        ],
    }
    trace_path = tmp_path / "sample_1.json"
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    rows = build_audit_samples(
        [
            {
                "question_id": "1",
                "question": "Which action occurs before adding salt?",
                "answer_choice": "A",
                "trace_path": str(trace_path),
            }
        ]
    )
    assert rows[0]["repeated_open_count"] == 2
    assert rows[0]["risk_score"] >= 0.3
    assert rows[0]["likely_failure_stage"] == "open"


def test_make_submission_mix_replaces_only_target_category():
    base_rows = [
        {
            "question_id": "1",
            "question": "Which action happens before opening the fridge?",
            "answer_choice": "A",
        },
        {
            "question_id": "2",
            "question": "Does closing a refrigerator imply holding some food ?",
            "answer_choice": "No",
        },
    ]
    replacement_rows = {
        "before_after": [
            {
                "question_id": "1",
                "question": "Which action happens before opening the fridge?",
                "answer_choice": "C",
            }
        ]
    }
    mixed = mix_submission_by_category(base_rows=base_rows, replacements=replacement_rows)
    assert mixed[0]["answer_choice"] == "C"
    assert mixed[1]["answer_choice"] == "No"


def test_invalid_answer_is_flagged(tmp_path: Path):
    trace = {
        "answer": "maybe",
        "state": {
            "question": "Does the person hold a book?",
            "evidence_board": {"missing_required_slots": []},
            "evidence_ledger": [],
        },
        "trace": [],
    }
    trace_path = tmp_path / "sample_2.json"
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    rows = build_audit_samples(
        [
            {
                "question_id": "2",
                "question": "Does the person hold a book?",
                "answer": "maybe",
                "trace_path": str(trace_path),
            }
        ]
    )
    assert rows[0]["invalid_answer"] is True
    assert rows[0]["likely_failure_stage"] == "format"


def test_split_dataset_keeps_same_video_together():
    rows = [
        {"question_id": "1", "video_id": "a.mp4"},
        {"question_id": "2", "video_id": "a.mp4"},
        {"question_id": "3", "video_id": "b.mp4"},
        {"question_id": "4", "video_id": "c.mp4"},
    ]
    shards = _video_balanced_split(rows, 2)
    locations = {}
    for shard_index, shard in enumerate(shards):
        for row in shard:
            locations.setdefault(row["video_id"], set()).add(shard_index)
    assert all(len(indices) == 1 for indices in locations.values())
    assert sorted(sum((shard for shard in shards), []), key=lambda row: row["question_id"]) == rows


def test_merge_submission_shards_fallback_and_strict_validation():
    dataset = [
        {"question_id": "1", "mode": "mc"},
        {"question_id": "2", "mode": "boolean"},
        {"question_id": "3", "mode": "mc"},
    ]
    predictions = [
        {"question_id": "1", "answer_choice": "B"},
        {"question_id": "2", "answer_choice": "maybe"},
    ]
    rows, summary = build_submission(dataset, predictions)
    assert rows == [
        {"question_id": "1", "answer_choice": "B"},
        {"question_id": "2", "answer_choice": "No"},
        {"question_id": "3", "answer_choice": "A"},
    ]
    assert summary["fallback_qids"] == ["2", "3"]

    with pytest.raises(ValueError, match="Missing or invalid answer"):
        build_submission(dataset, predictions, strict=True)
