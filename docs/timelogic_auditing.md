# TimeLogic Auditing

## Why Audit Without Ground Truth

For the TimeLogic challenge, we often only have:
- our own submission file,
- raw prediction records,
- traces,
- and the EvalAI leaderboard score.

That means we need a way to estimate failure risk before we know the official label.

The audit tools added here help answer questions like:
- Which predictions look suspicious even without labels?
- Which temporal categories are likely hurting the score?
- Are we failing in retrieval, `OPEN`, temporal reasoning, or answer formatting?
- Should we replace one category with a different baseline and test the delta on EvalAI?

The goal is not to change the solver. The goal is to inspect solver behavior from the outside.

## Files

- Script: `scripts/timelogic/audit_traces.py`
- Script: `scripts/timelogic/make_submission_mix.py`
- Parser and audit module: `rlm/video/timelogic_auditing.py`

## Categories

`parse_timelogic_category(question, options=None) -> str` maps each sample into one of:

- `eventual`
- `always`
- `before_after`
- `next`
- `immediate_next`
- `cooccur_disjoint`
- `until_since`
- `implies`
- `ordering`
- `always_before`
- `always_next`
- `always_cooccur`
- `unknown`

This is rule-based and intentionally lightweight. It is meant for grouping and replacement experiments, not for solving the benchmark.

## Run The Audit

Basic usage:

```bash
python scripts/timelogic/audit_traces.py \
  --predictions output/timelogic/validation_full_visual_only/predictions.jsonl \
  --traces output/timelogic/validation_full_visual_only/traces
```

With a direct baseline file for disagreement analysis:

```bash
python scripts/timelogic/audit_traces.py \
  --predictions output/timelogic/validation_full_visual_only/predictions.jsonl \
  --traces output/timelogic/validation_full_visual_only/traces \
  --direct-baseline output/timelogic/direct_qwen3vl_predictions.jsonl \
  --out-dir output/timelogic/audit_videorlm_vs_direct
```

You can also run it on a shard:

```bash
python scripts/timelogic/audit_traces.py \
  --predictions output/timelogic/validation_full_visual_only_resplit/shard_a_run/predictions.jsonl \
  --traces output/timelogic/validation_full_visual_only_resplit/shard_a_run/traces
```

## Audit Outputs

The script writes three files:

### `suspect_predictions.csv`

Sorted by descending `risk_score`.

Useful columns:
- `category`
- `pred_answer`
- `direct_answer`
- `risk_score`
- `risk_flags`
- `likely_failure_stage`
- `option_margin`
- `repeated_open_count`
- `no_evidence_open_count`
- `core_evidence_count`
- `missing_slots`

This is the best file for manual spot checks.

### `category_report.csv`

Grouped by category with:
- `count`
- `avg_risk`
- `high_risk_rate`
- `invalid_answer_rate`
- `solver_disagreement_rate`
- `repeated_open_rate`
- `no_evidence_rate`
- `avg_option_margin`
- `answer_distribution`

This is the best file for deciding which category to replace first.

### `audit_summary.json`

Global overview:
- total sample count
- average risk
- high-risk count
- category counts
- top-20 risky qids
- answer distribution
- direct-vs-VideoRLM agreement

## How To Read `risk_score`

`risk_score` is a heuristic score in `[0, 1]`.

It goes up when we see patterns that usually correlate with wrong answers:
- trace missing
- unknown category
- invalid answer format
- repeated `OPEN`
- repeated `SEARCH`
- `OPEN` with no evidence
- zero core evidence
- missing required slots
- low option margin
- disagreement with a direct baseline
- temporal invariant violations

Suggested reading:
- `0.00 - 0.24`: low concern
- `0.25 - 0.49`: moderate concern
- `>= 0.50`: high concern

It is not a probability of error. It is a debugging priority score.

## How To Read `likely_failure_stage`

The audit assigns a coarse failure stage:
- `format`
- `parser`
- `retrieval`
- `open`
- `temporal_eval`
- `answer_selection`
- `unknown`

Use it as a triage hint:
- many `format` cases: fix output coercion
- many `retrieval` cases: inspect queries/frontier
- many `open` cases: inspect evidence extraction and repeated opens
- many `temporal_eval` cases: inspect relation logic / invariants
- many `answer_selection` cases: compare against direct baselines or option scores

## Create A Submission Mix

This tool starts from one base prediction file and replaces selected categories from other prediction files.

Example:

```bash
python scripts/timelogic/make_submission_mix.py \
  --base output/timelogic/videorlm_predictions.jsonl \
  --replace-category before_after=output/timelogic/direct_qwen3vl_predictions.jsonl \
  --replace-category implies=output/timelogic/direct_qwen3vl_predictions.jsonl \
  --out output/timelogic/mixed_before_after_implies.jsonl
```

Rules:
- base rows must contain `qid/question_id/id`
- base rows must contain question text so category can be parsed
- answers must stay valid (`A/B/C/D/Yes/No`)
- if a replaced category is missing a qid, the script fails loudly

## Suggested Workflow

1. Run a direct baseline.
2. Run the VideoRLM baseline.
3. Audit both outputs with `audit_traces.py`.
4. Identify one category with high risk or high disagreement.
5. Build a one-category replacement submission with `make_submission_mix.py`.
6. Submit that mix to EvalAI.
7. Compare leaderboard delta.
8. Repeat category-by-category instead of changing everything at once.

This keeps debugging disciplined when the official score is the only real supervision signal.
