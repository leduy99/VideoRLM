from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HIGH_RISK_THRESHOLD = 0.5
VALID_ANSWERS = {"A", "B", "C", "D", "Yes", "No"}


@dataclass
class AuditSample:
    qid: str
    video_id: str | None
    question: str | None
    options: Any
    pred_answer: str | None
    raw_record: dict[str, Any]
    trace_path: str | None
    trace: dict[str, Any] | None
    trace_missing: bool


def parse_timelogic_category(question: str, options: Any = None) -> str:
    text = _normalize_text(" ".join(part for part in [question, _flatten_options(options)] if part))
    if not text:
        return "unknown"
    if any(phrase in text for phrase in ("always co occur", "always cooccur", "always overlap")):
        return "always_cooccur"
    if any(
        phrase in text
        for phrase in (
            "always occurs immediately before",
            "always occurs immediately after",
            "always immediately before",
            "always immediately after",
            "always right after",
            "always directly after",
            "always next",
        )
    ):
        return "always_next"
    if any(
        phrase in text
        for phrase in (
            "always occurs before",
            "always occurs after",
            "always before",
            "always after",
        )
    ):
        return "always_before"
    if any(
        phrase in text
        for phrase in (
            "immediately after",
            "right after",
            "directly after",
            "immediately before",
            "right before",
            "directly before",
        )
    ):
        return "immediate_next"
    if any(token in text for token in (" until ", " since ", " up to ", " leading up to ")):
        return "until_since"
    if any(token in text for token in (" imply ", " implies ", " imply that ", " if and only if ")):
        return "implies"
    if any(
        token in text
        for token in (
            "co occur",
            "cooccur",
            "simultaneously",
            "at the same time",
            "overlap",
            "disjoint",
            "together",
        )
    ):
        return "cooccur_disjoint"
    if any(
        token in text
        for token in (
            "order",
            "ordered",
            "ordering",
            "sequence",
            "first",
            "second",
            "third",
            "fourth",
        )
    ):
        return "ordering"
    if any(token in text for token in (" next ", " then ", " followed by ", " which in turn ")):
        return "next"
    if any(token in text for token in (" before ", " after ", " earlier ", " later ")):
        return "before_after"
    if any(token in text for token in (" eventually ", " at some point ", " sometime ", " finally ")):
        return "eventual"
    if "always" in text:
        return "always"
    return "unknown"


def load_prediction_records(path: str | Path) -> tuple[list[dict[str, Any]], str]:
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with input_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows, "jsonl"
    if suffix == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload, "json"
        if isinstance(payload, dict):
            for key in ("predictions", "rows", "records", "data"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value, "json"
        raise ValueError(f"Unsupported JSON payload shape in {input_path}")
    raise ValueError(f"Unsupported prediction file format: {input_path}")


def write_records(path: str | Path, rows: list[dict[str, Any]], fmt: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jsonl":
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def load_trace_index(trace_source: str | Path | None) -> dict[str, Path]:
    if trace_source is None:
        return {}
    source = Path(trace_source)
    if not source.exists():
        raise ValueError(f"Trace source does not exist: {source}")
    if source.is_file():
        qid = _qid_from_trace_path(source)
        if qid is None:
            raise ValueError(f"Could not infer qid from trace file name: {source}")
        return {qid: source}
    index: dict[str, Path] = {}
    for path in sorted(source.rglob("sample_*.json")):
        qid = _qid_from_trace_path(path)
        if qid is not None:
            index[qid] = path
    return index


def build_audit_samples(
    prediction_rows: list[dict[str, Any]],
    *,
    trace_source: str | Path | None = None,
    direct_baseline_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    trace_index = load_trace_index(trace_source)
    direct_answers = {
        qid: _extract_prediction_answer(row)
        for row in (direct_baseline_rows or [])
        if (qid := _extract_qid(row)) is not None
    }
    results = []
    for row in prediction_rows:
        sample = _build_audit_sample(row, trace_index)
        audit_row = _audit_single_sample(sample, direct_answers.get(sample.qid))
        results.append(audit_row)
    return sorted(results, key=lambda item: (-item["risk_score"], item["qid"]))


def write_audit_outputs(rows: list[dict[str, Any]], out_dir: str | Path) -> None:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suspect_path = output_dir / "suspect_predictions.csv"
    category_path = output_dir / "category_report.csv"
    summary_path = output_dir / "audit_summary.json"

    suspect_fields = [
        "qid",
        "video_id",
        "category",
        "pred_answer",
        "direct_answer",
        "risk_score",
        "risk_flags",
        "likely_failure_stage",
        "option_margin",
        "repeated_open_count",
        "no_evidence_open_count",
        "core_evidence_count",
        "missing_slots",
        "question",
    ]
    _write_csv(suspect_path, suspect_fields, rows)
    category_rows = build_category_report(rows)
    _write_csv(
        category_path,
        [
            "category",
            "count",
            "avg_risk",
            "high_risk_rate",
            "invalid_answer_rate",
            "solver_disagreement_rate",
            "repeated_open_rate",
            "no_evidence_rate",
            "avg_option_margin",
            "answer_distribution",
        ],
        category_rows,
    )
    summary_path.write_text(json.dumps(build_audit_summary(rows), ensure_ascii=False, indent=2), encoding="utf-8")


def build_category_report(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    report_rows = []
    for category, items in sorted(grouped.items()):
        margins = [item["option_margin"] for item in items if item["option_margin"] is not None]
        answers = Counter(item["pred_answer"] or "" for item in items)
        report_rows.append(
            {
                "category": category,
                "count": len(items),
                "avg_risk": round(sum(item["risk_score"] for item in items) / len(items), 4),
                "high_risk_rate": round(sum(item["risk_score"] >= HIGH_RISK_THRESHOLD for item in items) / len(items), 4),
                "invalid_answer_rate": round(sum(bool(item["invalid_answer"]) for item in items) / len(items), 4),
                "solver_disagreement_rate": round(sum(bool(item["solver_disagreement"]) for item in items) / len(items), 4),
                "repeated_open_rate": round(sum(item["repeated_open_count"] > 0 for item in items) / len(items), 4),
                "no_evidence_rate": round(sum(item["no_evidence_open_count"] > 0 for item in items) / len(items), 4),
                "avg_option_margin": round(sum(margins) / len(margins), 4) if margins else None,
                "answer_distribution": json.dumps(dict(sorted(answers.items())), ensure_ascii=False),
            }
        )
    return report_rows


def build_audit_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answer_distribution = Counter(row["pred_answer"] or "" for row in rows)
    compared = [row for row in rows if row["direct_answer"]]
    agreement = None
    if compared:
        agreement = round(
            sum(row["pred_answer"] == row["direct_answer"] for row in compared) / len(compared),
            4,
        )
    return {
        "total_samples": len(rows),
        "average_risk": round(sum(row["risk_score"] for row in rows) / max(len(rows), 1), 4),
        "high_risk_count": sum(row["risk_score"] >= HIGH_RISK_THRESHOLD for row in rows),
        "category_counts": dict(sorted(Counter(row["category"] for row in rows).items())),
        "top_20_highest_risk_qids": [row["qid"] for row in rows[:20]],
        "answer_distribution": dict(sorted(answer_distribution.items())),
        "direct_vs_videorlm_agreement": agreement,
        "compared_sample_count": len(compared),
    }


def mix_submission_by_category(
    *,
    base_rows: list[dict[str, Any]],
    replacements: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    replacement_answers = {
        category: {
            qid: _require_valid_answer(_extract_prediction_answer(row), qid=qid, source=category)
            for row in rows
            if (qid := _extract_qid(row)) is not None
        }
        for category, rows in replacements.items()
    }
    mixed_rows: list[dict[str, Any]] = []
    seen_qids: set[str] = set()
    for row in base_rows:
        qid = _extract_qid(row)
        if qid is None:
            raise ValueError(f"Missing qid/question_id/id in base row: {row}")
        if qid in seen_qids:
            raise ValueError(f"Duplicate qid in base rows: {qid}")
        seen_qids.add(qid)
        question = _extract_question(row)
        options = row.get("options")
        if not question:
            raise ValueError(
                f"Base row {qid} is missing question text; category replacement cannot be determined."
            )
        category = parse_timelogic_category(question, options)
        answer = _require_valid_answer(_extract_prediction_answer(row), qid=qid, source="base")
        if category in replacement_answers:
            if qid not in replacement_answers[category]:
                raise ValueError(f"Replacement file for category {category} is missing qid {qid}")
            answer = replacement_answers[category][qid]
        mixed_rows.append(_replace_answer_fields(row, answer))
    return mixed_rows


def _build_audit_sample(row: dict[str, Any], trace_index: dict[str, Path]) -> AuditSample:
    qid = _extract_qid(row)
    if qid is None:
        raise ValueError(f"Missing qid/question_id/id in row: {row}")
    trace_path = _extract_trace_path(row, trace_index)
    trace = None
    trace_missing = False
    if trace_path is not None:
        path = Path(trace_path)
        if path.exists():
            trace = json.loads(path.read_text(encoding="utf-8"))
        else:
            trace_missing = True
    else:
        trace_missing = True
    question = _extract_question(row) or _extract_question_from_trace(trace)
    video_id = row.get("video_id") or _extract_video_id_from_trace(trace)
    return AuditSample(
        qid=qid,
        video_id=video_id,
        question=question,
        options=row.get("options"),
        pred_answer=_extract_prediction_answer(row),
        raw_record=row,
        trace_path=trace_path,
        trace=trace,
        trace_missing=trace_missing,
    )


def _audit_single_sample(sample: AuditSample, direct_answer: str | None) -> dict[str, Any]:
    trace = sample.trace or {}
    trace_steps = trace.get("trace", []) if isinstance(trace, dict) else []
    actions = [step.get("action", {}) for step in trace_steps if isinstance(step, dict)]
    observations = [step.get("observation", {}) for step in trace_steps if isinstance(step, dict)]
    search_actions = [action for action in actions if action.get("action_type") == "SEARCH"]
    open_actions = [action for action in actions if action.get("action_type") == "OPEN"]
    repeated_open_count = _repeated_count(
        (
            action.get("node_id"),
            action.get("modality"),
            action.get("target_slot"),
        )
        for action in open_actions
    )
    repeated_search_count = _repeated_count(
        (
            _normalize_text(action.get("query") or ""),
            action.get("modality"),
            action.get("target_slot"),
        )
        for action in search_actions
    )
    unique_opened_nodes = len({action.get("node_id") for action in open_actions if action.get("node_id")})
    no_evidence_open_count = sum(
        1
        for observation in observations
        if observation.get("kind") == "open" and not observation.get("evidence")
    )

    final_state = trace.get("state", {}) if isinstance(trace, dict) else {}
    evidence_ledger = final_state.get("evidence_ledger", []) if isinstance(final_state, dict) else []
    core_evidence_count = sum(
        1 for item in evidence_ledger if item.get("metadata", {}).get("role") == "core"
    )
    support_evidence_count = sum(
        1 for item in evidence_ledger if item.get("metadata", {}).get("role") == "support"
    )
    missing_slots = (
        final_state.get("evidence_board", {}).get("missing_required_slots", [])
        if isinstance(final_state, dict)
        else []
    )
    option_scores = _extract_option_scores(sample.raw_record, trace)
    option_margin = _option_margin(option_scores)
    pred_answer = sample.pred_answer
    invalid_answer = pred_answer not in VALID_ANSWERS if pred_answer is not None else True
    category = parse_timelogic_category(sample.question or "", sample.options)
    solver_disagreement = bool(direct_answer) and direct_answer != pred_answer
    temporal_violations = extract_temporal_invariant_violations(trace)

    risk_flags: list[str] = []
    risk = 0.0
    if sample.trace_missing:
        risk += 0.25
        risk_flags.append("trace_missing")
    if category == "unknown":
        risk += 0.15
        risk_flags.append("category_unknown")
    if invalid_answer:
        risk += 0.50
        risk_flags.append("invalid_answer")
    if repeated_open_count > 1:
        risk += 0.15
        risk_flags.append("repeated_open")
    if repeated_search_count > 1:
        risk += 0.10
        risk_flags.append("repeated_search")
    if no_evidence_open_count > 0:
        risk += 0.15
        risk_flags.append("no_evidence_open")
    if trace and core_evidence_count == 0:
        risk += 0.20
        risk_flags.append("zero_core_evidence")
    if missing_slots:
        risk += 0.25
        risk_flags.append("missing_required_slots")
    if option_margin is not None and option_margin < 0.08:
        risk += 0.20
        risk_flags.append("low_option_margin")
    if solver_disagreement:
        risk += 0.15
        risk_flags.append("solver_disagreement")
    if temporal_violations:
        risk += 0.25
        risk_flags.append("temporal_invariant_violation")
    risk = round(min(risk, 1.0), 4)

    return {
        "qid": sample.qid,
        "question_id": sample.qid,
        "video_id": sample.video_id,
        "category": category,
        "question": sample.question,
        "pred_answer": pred_answer,
        "direct_answer": direct_answer,
        "search_count": len(search_actions),
        "open_count": len(open_actions),
        "repeated_open_count": repeated_open_count,
        "repeated_search_count": repeated_search_count,
        "unique_opened_nodes": unique_opened_nodes,
        "no_evidence_open_count": no_evidence_open_count,
        "core_evidence_count": core_evidence_count if trace else None,
        "support_evidence_count": support_evidence_count if trace else None,
        "missing_slots": ",".join(missing_slots),
        "option_scores": json.dumps(option_scores, ensure_ascii=False) if option_scores is not None else None,
        "option_margin": option_margin,
        "direct_baseline_answer": direct_answer,
        "solver_disagreement": solver_disagreement,
        "invalid_answer": invalid_answer,
        "trace_missing": sample.trace_missing,
        "risk_score": risk,
        "risk_flags": "|".join(risk_flags),
        "likely_failure_stage": _likely_failure_stage(
            invalid_answer=invalid_answer,
            trace_missing=sample.trace_missing,
            repeated_open_count=repeated_open_count,
            repeated_search_count=repeated_search_count,
            no_evidence_open_count=no_evidence_open_count,
            core_evidence_count=core_evidence_count if trace else None,
            missing_slots=missing_slots,
            option_margin=option_margin,
            solver_disagreement=solver_disagreement,
            temporal_violations=temporal_violations,
        ),
        "temporal_invariant_violations": temporal_violations,
    }


def extract_temporal_invariant_violations(trace: dict[str, Any] | None) -> list[str]:
    if not trace:
        return []
    relations = _extract_temporal_relations(trace)
    if not relations:
        return []
    by_kind: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for relation in relations:
        kind = relation.get("kind")
        args = relation.get("args", ())
        if kind:
            by_kind[kind].add(tuple(args))
    violations: list[str] = []
    for lhs, rhs in by_kind.get("before", set()):
        if (rhs, lhs) not in by_kind.get("after", set()):
            violations.append(f"before_missing_after:{lhs}:{rhs}")
    for lhs, rhs in by_kind.get("immediately_before", set()):
        if (lhs, rhs) not in by_kind.get("before", set()):
            violations.append(f"immediate_without_before:{lhs}:{rhs}")
    for lhs, rhs in by_kind.get("cooccur", set()):
        if (lhs, rhs) in by_kind.get("disjoint", set()) or (rhs, lhs) in by_kind.get("disjoint", set()):
            violations.append(f"cooccur_disjoint_conflict:{lhs}:{rhs}")
    for order in by_kind.get("order", set()):
        if len(order) < 3:
            continue
        for first, second in zip(order, order[1:], strict=False):
            if (first, second) not in by_kind.get("before", set()):
                violations.append(f"order_missing_before:{first}:{second}")
    return sorted(set(violations))


def _extract_temporal_relations(trace: dict[str, Any]) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            maybe = _normalize_relation(value)
            if maybe is not None:
                relations.append(maybe)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(trace)
    return relations


def _normalize_relation(payload: dict[str, Any]) -> dict[str, Any] | None:
    relation = payload.get("relation") or payload.get("type") or payload.get("operator")
    if not isinstance(relation, str):
        return None
    lowered = relation.strip().lower()
    lhs = payload.get("lhs") or payload.get("left") or payload.get("source") or payload.get("a")
    rhs = payload.get("rhs") or payload.get("right") or payload.get("target") or payload.get("b")
    if lowered in {"before", "after", "cooccur", "disjoint"} and lhs and rhs:
        return {"kind": lowered, "args": (str(lhs), str(rhs))}
    if lowered in {"immediately_before", "immediate_before"} and lhs and rhs:
        return {"kind": "immediately_before", "args": (str(lhs), str(rhs))}
    if lowered in {"order", "ordering", "sequence"}:
        items = payload.get("items") or payload.get("events") or payload.get("sequence")
        if isinstance(items, list) and len(items) >= 2:
            return {"kind": "order", "args": tuple(str(item) for item in items)}
    return None


def _extract_qid(row: dict[str, Any]) -> str | None:
    for key in ("qid", "question_id", "id"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return None


def _extract_prediction_answer(row: dict[str, Any]) -> str | None:
    for key in (
        "answer_choice",
        "prediction",
        "normalized_prediction",
        "pred_answer",
        "answer",
        "raw_answer",
    ):
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _extract_question(row: dict[str, Any]) -> str | None:
    for key in ("question", "formatted_question", "prompt"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_trace_path(row: dict[str, Any], trace_index: dict[str, Path]) -> str | None:
    explicit = row.get("trace_path")
    if explicit:
        return str(explicit)
    qid = _extract_qid(row)
    if qid and qid in trace_index:
        return str(trace_index[qid])
    return None


def _extract_question_from_trace(trace: dict[str, Any] | None) -> str | None:
    if not trace:
        return None
    state = trace.get("state")
    if isinstance(state, dict):
        question = state.get("question")
        if isinstance(question, str) and question.strip():
            return question.strip()
    return None


def _extract_video_id_from_trace(trace: dict[str, Any] | None) -> str | None:
    if not trace:
        return None
    state = trace.get("state")
    if not isinstance(state, dict):
        return None
    global_context = state.get("global_context", {})
    if isinstance(global_context, dict):
        value = global_context.get("video_id")
        if value:
            return str(value)
    return None


def _extract_option_scores(*payloads: Any) -> dict[str, float] | None:
    candidate = _find_option_scores(payloads)
    if not isinstance(candidate, dict):
        return None
    scores: dict[str, float] = {}
    for key, value in candidate.items():
        try:
            scores[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return scores or None


def _find_option_scores(payloads: Any) -> Any:
    keys = {"option_scores", "choice_scores", "answer_scores", "scores"}

    def visit(value: Any) -> Any:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in keys and isinstance(child, dict):
                    if all(str(option) in {"A", "B", "C", "D"} for option in child):
                        return child
                found = visit(child)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = visit(child)
                if found is not None:
                    return found
        return None

    return visit(list(payloads))


def _option_margin(option_scores: dict[str, float] | None) -> float | None:
    if not option_scores or len(option_scores) < 2:
        return None
    values = sorted(option_scores.values(), reverse=True)
    return round(values[0] - values[1], 4)


def _likely_failure_stage(
    *,
    invalid_answer: bool,
    trace_missing: bool,
    repeated_open_count: int,
    repeated_search_count: int,
    no_evidence_open_count: int,
    core_evidence_count: int | None,
    missing_slots: list[str],
    option_margin: float | None,
    solver_disagreement: bool,
    temporal_violations: list[str],
) -> str:
    if invalid_answer:
        return "format"
    if trace_missing:
        return "parser"
    if temporal_violations:
        return "temporal_eval"
    if option_margin is not None and option_margin < 0.08:
        return "answer_selection"
    if solver_disagreement:
        return "answer_selection"
    if no_evidence_open_count > 0 or repeated_open_count > 1:
        return "open"
    if repeated_search_count > 1 or core_evidence_count == 0 or missing_slots:
        return "retrieval"
    return "unknown"


def _replace_answer_fields(row: dict[str, Any], answer: str) -> dict[str, Any]:
    updated = dict(row)
    if "answer_choice" in updated:
        updated["answer_choice"] = answer
    if "normalized_prediction" in updated:
        updated["normalized_prediction"] = answer
    if "prediction" in updated:
        updated["prediction"] = answer
    if "pred_answer" in updated:
        updated["pred_answer"] = answer
    if "answer" in updated and not isinstance(updated.get("trace"), dict):
        updated["answer"] = answer
    if not any(key in updated for key in ("answer_choice", "prediction", "pred_answer", "answer")):
        updated["answer_choice"] = answer
    return updated


def _require_valid_answer(answer: str | None, *, qid: str, source: str) -> str:
    if answer not in VALID_ANSWERS:
        raise ValueError(f"Invalid answer for qid {qid} in {source}: {answer}")
    return answer


def _qid_from_trace_path(path: Path) -> str | None:
    match = re.search(r"sample_(\d+)\.json$", path.name)
    if not match:
        return None
    return match.group(1)


def _repeated_count(items) -> int:
    seen: set[Any] = set()
    repeats = 0
    for item in items:
        if not item or item in seen:
            if item:
                repeats += 1
            continue
        seen.add(item)
    return repeats


def _flatten_options(options: Any) -> str:
    if options is None:
        return ""
    if isinstance(options, str):
        return options
    if isinstance(options, dict):
        return " ".join(f"{key} {value}" for key, value in options.items())
    if isinstance(options, list):
        return " ".join(str(item) for item in options)
    return str(options)


def _normalize_text(text: str) -> str:
    normalized = text.lower().replace("-", " ").replace("_", " ")
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    return " " + " ".join(normalized.split()) + " "


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
