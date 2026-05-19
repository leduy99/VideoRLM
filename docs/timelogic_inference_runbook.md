# TimeLogic Inference Runbook

This runbook is the handoff checklist for rerunning TimeLogic validation inference on a
new server. It keeps TimeLogic as a VideoRLM task mode: build video memory, run the
evidence loop, write traces, then export the official submission JSON.

## GPU Rule

On the H200 machine, all CUDA work must run through SLURM. Do not run CUDA Python
directly from the login shell, because it can be auto-killed.

Use `/share_X/users/$USER` for data and outputs instead of `/home`.

## Environment

From the repo root:

```bash
conda env create -f environment.yml
conda activate videorlm
python -m pip install -e .
```

Quick CPU-only sanity checks:

```bash
python scripts/timelogic/run_inference.py --help
python scripts/timelogic/split_dataset.py --help
python scripts/timelogic/merge_submission_shards.py --help
python scripts/timelogic/audit_traces.py --help
```

## Data

Download the validation metadata and videos:

```bash
mkdir -p output/timelogic

curl -L \
  -o output/timelogic/timelogic_val_data.json \
  https://raw.githubusercontent.com/Swetha5/TimeLogic/challenge/data/val/timelogic_val_data.json

curl -L \
  -o output/timelogic/val_videos.zip \
  https://www.crcv.ucf.edu/cvpr2026-vidllms-workshop/challenge/data/timelogicqa/val_videos.zip

unzip -q output/timelogic/val_videos.zip -d output/timelogic
```

The extracted video directory used in our runs was:

```text
output/timelogic/combined_2k_videos
```

If the archive extracts to a different folder name, pass that folder through
`--videos-dir` or the `VIDEOS_DIR` SLURM variable.

Known missing videos from the validation zip observed in this workspace:

```text
ct_1jqTfi145xQ.mp4
ct_78c2HuuwmVA.mp4
ct_7MWzU--xApU.mp4
ct_BtDvFEFiQ5k.mp4
ct_GbgRRMMJHTU.mp4
ct_I-9uVsmWoEU.mp4
ct_L0MVdMNihGI.mp4
ct_LKd2oIsM3uE.mp4
ct_MEYXUyEXd88.mp4
ct_Mlscv4JxrfU.mp4
ct_Sm-Er9tMi8g.mp4
ct_Uj0WzaLGg3Y.mp4
ct_VEjQ3lIZIb4.mp4
ct_XhZnEq3mJy4.mp4
ct_sBJJ0Cj0GG4.mp4
ct_ygv6jXn59t8.mp4
ct_yyIOce1XvpY.mp4
```

Those samples cannot be visually inferred unless the missing videos are recovered.
The merge script will produce fallback labels for missing/invalid predictions unless
`--strict` is used.

## Smoke Test

Request an interactive debug GPU and run a tiny visual-only inference:

```bash
srun --partition=debug --gres=gpu:1 --cpus-per-task=8 --mem=128G --time=02:00:00 --pty bash

conda activate videorlm
python scripts/timelogic/run_inference.py \
  --dataset-json output/timelogic/timelogic_val_data.json \
  --videos-dir output/timelogic/combined_2k_videos \
  --output-dir output/timelogic/smoke_visual_only \
  --sample-id 1 \
  --sample-id 2 \
  --visual-only \
  --no-forced-aligner
```

Expected files:

```text
output/timelogic/smoke_visual_only/predictions.jsonl
output/timelogic/smoke_visual_only/submission.json
output/timelogic/smoke_visual_only/traces/sample_*.json
output/timelogic/smoke_visual_only/summary.json
```

Exit the debug allocation when done.

## Full Validation With Two Shards

Split by video so related questions stay together:

```bash
python scripts/timelogic/split_dataset.py \
  --dataset-json output/timelogic/timelogic_val_data.json \
  --num-shards 2 \
  --out-dir output/timelogic/shards_2
```

Submit two one-GPU jobs:

```bash
DATASET_JSON=output/timelogic/shards_2/shard_0_dataset.json \
VIDEOS_DIR=output/timelogic/combined_2k_videos \
OUTPUT_DIR=output/timelogic/validation_shard_0 \
CONDA_ENV=videorlm \
VISUAL_ONLY=1 \
NO_FORCED_ALIGNER=1 \
sbatch scripts/timelogic/slurm_run_shard.sbatch

DATASET_JSON=output/timelogic/shards_2/shard_1_dataset.json \
VIDEOS_DIR=output/timelogic/combined_2k_videos \
OUTPUT_DIR=output/timelogic/validation_shard_1 \
CONDA_ENV=videorlm \
VISUAL_ONLY=1 \
NO_FORCED_ALIGNER=1 \
sbatch scripts/timelogic/slurm_run_shard.sbatch
```

Monitor:

```bash
squeue -u "$USER"
tail -f logs/timelogic_vrlm_<JOBID>.out
```

The runner uses `--resume`, so resubmitting the same shard output directory skips
completed rows with valid `predictions.jsonl` records.

## Merge Official Submission

After both shard jobs finish:

```bash
python scripts/timelogic/merge_submission_shards.py \
  --dataset-json output/timelogic/timelogic_val_data.json \
  --predictions output/timelogic/validation_shard_0/predictions.jsonl \
  --predictions output/timelogic/validation_shard_1/predictions.jsonl \
  --out output/timelogic/final_submission.json
```

The submission file is a JSON list in the official format:

```json
[
  {"question_id": "1", "answer_choice": "D"},
  {"question_id": "2", "answer_choice": "Yes"}
]
```

Check the merge summary:

```text
output/timelogic/final_submission_merge_summary.json
```

Important fields:

- `fallback_count`: number of missing or invalid predictions replaced by safe defaults.
- `missing_qids`: question IDs not found in the shard prediction files.
- `invalid_qids`: question IDs with malformed labels.
- `answer_distribution`: quick sanity check for label collapse.

## Audit The Run

Run the no-groundtruth audit after merging or on each shard:

```bash
python scripts/timelogic/audit_traces.py \
  --predictions output/timelogic/validation_shard_0/predictions.jsonl \
  --traces output/timelogic/validation_shard_0/traces \
  --out-dir output/timelogic/validation_shard_0/audit
```

Read:

```text
suspect_predictions.csv
category_report.csv
audit_summary.json
```

Use the audit workflow in `docs/timelogic_auditing.md` to compare VideoRLM against
a direct baseline and create category-level replacement submissions.

## Current TimeLogic Mode

The inference runner calls:

```python
bundle.controller.run(question, memory, task_type="timelogic_temporal_reasoning")
```

That task type biases the question spec toward visual temporal evidence instead of
speech. For TimeLogic validation, keep `--visual-only` unless you are explicitly
testing a multimodal ablation.
