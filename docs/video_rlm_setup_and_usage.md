# VideoRLM Setup And Usage

This document is the practical setup guide for the `VideoRLM` part of the repo.

It is intentionally focused on the workflow that was actually verified on a fresh Conda environment, not just on the code structure.

## Recommended Install Path

Use the repo-root [environment.yml](/share_4/users/duy/project/rlm/environment.yml).

This is the easiest one-shot install for the full VideoRLM stack because it includes:

- Python `3.12`
- `ffmpeg`
- editable install of the repo itself
- local model runtime dependencies such as `torch`, `transformers`, `qwen-asr`
- LongShOT-related dependencies such as `datasets`, `huggingface-hub`, `yt-dlp`
- test and lint tools used in this repo

## Verified Clean Install

The following flow was verified on `2026-04-12` using a brand-new Conda env named `videorlm-bootstrap-test`.

### 1. Create the environment

From the repo root:

```bash
conda env create -f environment.yml
```

If you want to create a differently named environment for testing:

```bash
conda env create -n videorlm-bootstrap-test -f environment.yml
```

If the environment already exists and you want to sync it with the file again:

```bash
conda env update -f environment.yml --prune
```

### 2. Activate it

```bash
conda activate videorlm
```

Or, if you overrode the environment name:

```bash
conda activate videorlm-bootstrap-test
```

### 3. Verify the install

These commands were run successfully in the clean test environment:

```bash
videorlm --help
videorlm run-longshot-local --help
python examples/video_rlm_example.py
python -m pytest tests/video tests/test_imports.py -q
```

Observed verification result from the clean environment:

- `videorlm --help` worked
- `videorlm run-longshot-local --help` worked
- `python examples/video_rlm_example.py` ran successfully
- `python -m pytest tests/video tests/test_imports.py -q` passed with `89 passed, 3 skipped`

## What The Install Gives You

After installation, you should have:

- the Python package importable as `rlm`
- the VideoRLM CLI available as `videorlm`
- `ffmpeg` available in the environment
- GPU-enabled `torch` available for local Qwen-based runs

Quick sanity check:

```bash
python -c "import rlm, torch; print(rlm.VideoRLM.__name__); print(torch.__version__); print(torch.cuda.is_available())"
```

## Main Ways To Use The Repo

There are three practical ways to use VideoRLM in this repo.

## 1. Smoke Test The Core Loop

This is the fastest way to confirm the repo is working end-to-end without downloading real checkpoints.

```bash
python examples/video_rlm_example.py
```

This uses a mock controller and prepared in-memory artifacts, so it is ideal for:

- installation sanity checks
- quick onboarding
- checking that logging and controller transitions still work

## 2. Use An OpenAI-Compatible Serving Endpoint

This mode is the best choice if you already have `Qwen3-8B`, `Qwen3-VL-8B`, and `Qwen3-ASR-0.6B` served behind a single OpenAI-compatible endpoint.

Typical flow:

### Step A. Prepare artifacts

```bash
videorlm prepare-artifacts \
  --video /path/to/video.mp4 \
  --duration-seconds 600 \
  --output-dir output/my_run/artifacts \
  --video-id demo_video \
  --base-url http://localhost:8000/v1 \
  --api-key dummy
```

What this does:

- runs speech recognition
- runs visual summarization
- saves artifact sidecars to disk

### Step B. Build memory

```bash
videorlm build-memory \
  --artifacts output/my_run/artifacts \
  --output output/my_run/memory.json
```

What this does:

- converts raw artifact sidecars into hierarchical `video -> scene -> segment -> clip` memory

### Step C. Ask a question

```bash
videorlm ask \
  --memory output/my_run/memory.json \
  --question "When does the speaker change the plan?" \
  --controller-base-url http://localhost:8000/v1 \
  --controller-api-key dummy \
  --trace-out output/my_run/trace.json \
  --log-dir output/my_run/logs
```

What this gives you:

- final answer in stdout
- a saved trace JSON if `--trace-out` is provided
- per-step logs if `--log-dir` is provided

## 3. Run On LongShOTBench

There are two benchmark entrypoints.

### Option A. Endpoint-based benchmark run

```bash
videorlm run-longshot \
  --output output/benchmarks/longshot/predictions.jsonl \
  --video-dir /path/to/longshot/videos \
  --base-url http://localhost:8000/v1 \
  --api-key dummy \
  --trace-dir output/benchmarks/longshot/traces \
  --memory-dir output/benchmarks/longshot/memories \
  --artifacts-dir output/benchmarks/longshot/artifacts
```

### Option B. Local Hugging Face checkpoints

First download the checkpoints:

```bash
videorlm download-qwen-local-models
```

Then run the benchmark locally:

```bash
videorlm run-longshot-local \
  --output output/benchmarks/longshot_local/predictions.jsonl \
  --video-dir /path/to/longshot/videos \
  --trace-dir output/benchmarks/longshot_local/traces \
  --memory-dir output/benchmarks/longshot_local/memories \
  --artifacts-dir output/benchmarks/longshot_local/artifacts
```

Useful flags:

- `--sample-limit 2`
- `--sample-id sample_6168`
- `--video-id 1evyOuQz-jM`
- `--task-filter event_understanding`
- `--download-missing`

## Recommended First Commands

If you are opening this repo for the first time, this is the order I recommend:

1. Create the env from `environment.yml`
2. Run `videorlm --help`
3. Run `python examples/video_rlm_example.py`
4. Run `python -m pytest tests/video tests/test_imports.py -q`
5. Only then move to `prepare-artifacts`, `build-memory`, and `ask`

## Where Output Goes

This repo intentionally keeps generated runtime output under [`output/`](/share_4/users/duy/project/rlm/output), which is gitignored.

Common locations:

- benchmark runs: `output/<run_name>/`
- traces: `output/<run_name>/traces/`
- memories: `output/<run_name>/memories/`
- artifacts: `output/<run_name>/artifacts/`
- logs: `output/<run_name>/logs/`

## Important Notes

### `environment.yml` is the canonical full-stack install

The base `pyproject.toml` still reflects the core `RLM` library.

For VideoRLM work, the environment file is the canonical install path because VideoRLM needs additional packages that are not part of the smallest core install.

### Creating the env does not download model checkpoints

The environment installs the software stack.

It does not automatically download:

- `Qwen/Qwen3-8B`
- `Qwen/Qwen3-VL-8B-Instruct`
- `Qwen/Qwen3-ASR-0.6B`
- `Qwen/Qwen3-ForcedAligner-0.6B`

Checkpoint download happens later through:

```bash
videorlm download-qwen-local-models
```

### The local benchmark path is heavier than the smoke path

The clean install was verified successfully, but local model mode still implies:

- large model downloads
- GPU memory requirements
- much slower first-run latency than the smoke example

If you just want to verify the repo is healthy, use:

```bash
python examples/video_rlm_example.py
```

first.
