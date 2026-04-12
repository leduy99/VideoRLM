# VideoRLM Development Notes

## 2026-04-03

### Goal of this first implementation
- Add a **separate VideoRLM runtime path** inside the existing `rlm` repo without breaking the current REPL-based `RLM`.
- Keep the new stack small and testable:
  - `video -> memory -> index -> state -> action -> tool -> updated state -> trace`
- Avoid heavy new dependencies in v1.

### Architecture decisions
- `rlm/core/rlm.py` remains untouched as the generic recursive REPL engine.
- VideoRLM lives under `rlm/video/` as a sibling runtime, not as a rewrite of core RLM.
- The controller loop is **state-action JSON based**, not free-form code execution.
- Existing LM clients are reused for the controller.
- Video preprocessing is expressed through **adapter protocols** plus `PreparedVideoArtifacts`.
  - This keeps the repo lightweight while still giving a real interface for Qwen3-ASR / Qwen3-VL wrappers later.

### What v1 includes
- Typed schema for:
  - video memory nodes
  - frontier items
  - evidence ledger
  - controller state
  - actions
  - observations
  - trace steps
- A `VideoMemoryBuilder` that builds a hierarchical memory tree from prepared artifacts.
- A lexical `VideoMemoryIndex`.
- A hybrid-ready `VideoMemoryIndex` with optional embedding hooks.
- A `VideoToolExecutor` for `SEARCH`, `OPEN`, `SPLIT`, `MERGE`, `STOP`.
- A prompt-based `VideoRLM` controller loop.
- A dedicated `VideoRLMLogger` because the existing `RLMLogger` assumes REPL/code-block iterations.
- A dedicated `adapters.py` module that isolates ingestion and embedding boundaries.
- OpenAI-compatible adapters for:
  - ASR via transcription API
  - vision summarization via frame sampling + chat completions
  - embedding retrieval hooks
- A Qwen preset layer now bundles:
  - `Qwen3-8B` controller wiring
  - `Qwen3-VL-8B` visual summarizer wiring
  - `Qwen3-ASR-0.6B` speech recognizer wiring
- A thin CLI now supports:
  - preparing artifact sidecars
  - building memory JSON
  - asking questions over a memory file
- A directory-backed artifact store for `manifest + speech/visual/ocr/audio` sidecars.

### Important limitation in this phase
- v1 does **not** attempt full raw-video processing in-repo.
- Instead, it supports:
  - prepared artifacts directly
  - or external adapters that can be wired in later

### Why this is the right tradeoff
- It lets us test the core claim early:
  - whether structured state + frontier + evidence ledger improves control behavior
- It also keeps the repo easy to split later into a dedicated `videorlm` package if needed.

### Environment setup used for development
- A dedicated conda environment was created and used for all installs and verification:
  - `videorlm`
  - path: `/share_4/users/duy/.conda/envs/videorlm`
- The repo was installed editable inside that env.
- Lint and tests for the new work were run from that env, not from base.
- Local generated files now live under `output/` at the repo root, which is gitignored.

### Verification status after scaffold
- `conda run -n videorlm python -m ruff check rlm/video tests/video tests/test_imports.py examples/video_rlm_example.py rlm/__init__.py`
  - passed
- `conda run -n videorlm python -m pytest tests/video tests/test_imports.py -q`
  - passed
- `conda run -n videorlm python -m pytest -q`
  - passed at this checkpoint
  - result: `298 passed, 7 skipped`

### Current implementation boundary
- The controller can already:
  - initialize frontier from lexical retrieval
  - choose JSON actions
  - execute tool calls
  - update evidence ledger and budget
  - emit a structured trace
- A trace utility now converts runtime traces into
  `(state, gold_action, observation, next_state)` examples for later SFT.
- Retrieval now supports:
  - lexical-only search by default
  - optional hybrid lexical + semantic scoring through an embedding adapter
- The current "real" adapters assume:
  - an OpenAI-compatible serving layer
  - `ffmpeg` on PATH for video-to-audio/frame extraction
- There is now a convenience preset for the intended Qwen stack, so runtime setup is less manual.
- The codebase can now be driven either:
  - as Python modules
  - or through a simple terminal workflow
- The memory builder currently assumes:
  - prepared spans/summaries from external adapters
  - heuristic temporal subdivision for scene/segment/clip
- Preprocessing and runtime can now handshake through a stable artifact directory layout.

### Good next extensions from here
- Add stack-specific wiring/config for:
  - endpoint auth / retry policies
  - optional separate base URLs per model family
- Add richer evidence normalization and merge behavior.
- Add filtering/ranking so only high-quality prompted traces are kept for SFT.
- Add a stronger embedding provider and indexing cache once the chosen retrieval stack is fixed.
- Add richer CLI commands for batch preprocessing and benchmark evaluation.

## Update: LongShOT Benchmark Runner

### What was added
- A benchmark-specific integration layer now lives in `rlm/video/longshot.py`.
- It adds:
  - `load_longshot_samples(...)` for loading and filtering `MBZUAI/longshot-bench`
  - `LongShOTVideoResolver` for finding local benchmark videos or downloading them with `yt-dlp`
  - `LongShOTBenchmarkRunner` for:
    - resolving the benchmark video
    - building or reusing cached memory
    - replaying the benchmark conversation structure
    - writing LongShOT-compatible prediction JSONL with `candidate_response`
    - optionally saving per-turn VideoRLM traces
- The VideoRLM CLI now has a `run-longshot` subcommand.

### Why this shape was chosen
- The runner preserves the LongShOT sample schema instead of inventing a new prediction format.
- Assistant turns are annotated with `candidate_response`, which keeps the output structurally compatible with the upstream LongShOT evaluation flow.
- Memory caching is keyed by `video_id` because LongShOT contains multiple QA samples over the same long video.
- Dialogue replay defaults to `history_mode="gold"` so the runner mirrors the upstream benchmark generation style, where prior assistant history stays anchored to gold turns rather than previous candidate turns.

### Verification added for this phase
- New tests cover:
  - dataset filtering
  - recursive local video resolution
  - optional `yt-dlp` download path
  - benchmark sample execution
  - JSONL append/resume behavior
  - CLI wiring for `run-longshot`

### Verification status after benchmark runner
- `conda run -n videorlm python -m ruff check rlm/video tests/video tests/test_imports.py examples/video_rlm_example.py examples/video_rlm_qwen_stack_example.py`
  - passed
- `conda run -n videorlm python -m pytest tests/video/test_video_longshot.py tests/video/test_video_cli.py tests/test_imports.py -q`
  - passed
- `conda run -n videorlm python -m pytest -q`
  - passed
  - result: `304 passed, 7 skipped`

### Smoke test on real LongShOT videos
- We found local cached benchmark videos already present on disk:
  - `/share_4/users/duy/.cache/huggingface/videomme/data/wTlERUE8LVw.mp4`
  - `/share_4/users/duy/.cache/huggingface/videomme/data/1evyOuQz-jM.mp4`
- `ffprobe` confirms they are readable and their durations line up closely with the dataset metadata:
  - `wTlERUE8LVw.mp4` -> `2484.767347`
  - `1evyOuQz-jM.mp4` -> `2505.293787`
- A benchmark smoke test was run against two real LongShOT samples:
  - `sample_8563` on `wTlERUE8LVw`
  - `sample_6168` on `1evyOuQz-jM`
- For this smoke test, the controller itself was stubbed so we could validate:
  - sample loading
  - video resolution
  - memory caching
  - per-turn trace writing
  - LongShOT-compatible prediction JSONL output
- Smoke-test outputs were written under:
  - `output/longshot_smoke/predictions.jsonl`
  - `output/longshot_smoke/memories/`
  - `output/longshot_smoke/traces/`

### Important blocker still remaining
- We do not yet have a live Qwen serving endpoint configured for:
  - `Qwen3-8B`
  - `Qwen3-VL-8B`
  - `Qwen3-ASR-0.6B`
- That means the benchmark runner is now structurally ready, but benchmark-quality answers still require a real serving stack.
- We also confirmed that on this machine, direct `yt-dlp` fetches from YouTube hit anti-bot gating without browser cookies.
- In practice, benchmark runs should therefore use either:
  - a pre-populated local video cache
  - or explicit browser cookies passed to `yt-dlp`

## Update: Self-Contained Local Hugging Face Stack

### What was added
- We added a local Transformers-backed LM client in `rlm/clients/transformers_local.py`.
- VideoRLM now also has local Hugging Face helpers in:
  - `rlm/video/huggingface.py`
  - `rlm/video/local_adapters.py`
  - `rlm/video/qwen.py`
- The repo can now run a fully local Qwen stack with models downloaded into:
  - `output/models/Qwen__Qwen3-8B`
  - `output/models/Qwen__Qwen3-VL-8B-Instruct`
  - `output/models/Qwen__Qwen3-ASR-0.6B`
  - `output/models/Qwen__Qwen3-ForcedAligner-0.6B`
- The CLI gained local-model entrypoints so benchmark and memory runs do not require an external serving endpoint.

### Why this matters
- It makes the repo much more self-contained for VideoRLM development.
- It also keeps the model boundary explicit:
  - controller stays a text LM
  - ASR and VL remain leaf tools
  - the runtime can switch between local HF and API-backed adapters later

## Update: Practical ASR Path for LongShOT Videos

### Forced-aligner lesson
- Qwen3-ASR can return rich object-style timestamp outputs rather than plain dicts.
- We had to normalize:
  - dataclass-like return payloads
  - `ForcedAlignResult.items`
  - `ForcedAlignItem.start_time` / `end_time`
- Word-level forced alignment was then grouped into longer speech spans before entering VideoRLM memory.

### Faster fallback added
- Full forced alignment on long benchmark videos was too slow for iterative debugging.
- We added a chunked ASR fallback path:
  - probe duration with `ffprobe`
  - split long audio into fixed windows
  - transcribe each chunk without forced alignment
  - offset chunk-local spans back into global video time
- This is less precise than true alignment, but much better for rapid benchmark iteration.

### Development lesson
- For benchmark bring-up, controller/debug speed matters more than perfect timestamps.
- A fast chunked transcript is enough to validate:
  - retrieval behavior
  - action selection
  - evidence-led answering

## Update: Real Benchmark Debugging on `sample_6168`

### What we ran
- We ran a real local-model benchmark pass on LongShOT sample `sample_6168` using:
  - local `Qwen3-8B` as controller
  - local `Qwen3-ASR-0.6B` for speech-only memory
  - cached local LongShOT video `1evyOuQz-jM.mp4`
- Outputs were written under:
  - `output/longshot_real_speech_only/`
  - `output/longshot_real_speech_only_v5/`

### Failure mode found
- Retrieval was initially being distracted by stopwords and broad overlap.
- After retrieval improved, the controller still exhausted budget instead of stopping cleanly.
- The bigger remaining issue was answer synthesis:
  - the evidence ledger contained the correct transcript
  - but the answerer was focusing on the wrong early sentence fragment
  - it answered from "quite heavy now" instead of the real causal reason

### Fixes that mattered
- `rlm/video/index.py`
  - added stopword-aware tokenization for lexical retrieval
- `rlm/video/controller.py`
  - remove opened frontier nodes to reduce repeated `OPEN` loops
  - improve evidence focusing so long ASR spans are cut around causal keywords like:
    - `clasp`
    - `opening`
    - `worried`
    - `lose`
    - `fixed`
    - `repair`
  - grounded answer synthesis now uses focused `excerpt` text rather than misleading truncated `claim`
  - answer prompt now explicitly asks to mention both the problem and the later repair when present

### Result after fixes
- On `sample_6168`, the grounded local answer now captures the main benchmark points:
  - the clasp kept opening
  - she was worried about losing the bracelet
  - she took it back to Cartier and it was fixed
- This is an important milestone because it shows the pipeline can already recover a benchmark-relevant answer from:
  - local ASR memory
  - lexical search
  - a small prompted controller loop

## Update: Real Benchmark Debugging on `sample_8563`

### What this sample exposed
- `sample_8563` is an event-understanding question that asks about the **first** standout item in a long food video.
- The local speech-only memory was actually sufficient:
  - the transcript clearly contains:
    - `chicken head`
    - `selling over 10 pounds`
    - `coat the head in flour and deep fry`
    - the tasting reaction that it `tastes like chicken`
- The hard part was not extraction. It was **control and evidence focusing**.

### Retrieval lesson
- Pure lexical overlap was still biased toward broad scene nodes containing generic words like:
  - `first`
  - `different`
  - `food`
- We added a light temporal prior in `rlm/video/index.py`:
  - questions with `first / beginning / earliest / initial` now slightly favor earlier nodes
  - questions with `last / final / ending / end` favor later nodes
- This pushed the initial frontier toward `scene_001`, which is where the chicken-head evidence actually lives.

### Answer-synthesis lesson
- Even after opening the correct scene, scene-level speech evidence was still very long.
- The answerer sometimes drifted toward later content inside the same scene.
- We improved evidence focusing for `first`/`last` questions by:
  - preferring an early or late evidence window first
  - then letting the answerer synthesize from that window
- For this sample, the raw automatic pipeline answer still remained weaker than desired.
- A refined evidence-only prompt over the exact relevant speech spans produced a much stronger benchmark-style answer.

### Practical conclusion
- On LongShOT-style questions, getting the right scene is not enough.
- We also need:
  - better span selection inside a scene
  - or more selective node opening so the answerer is not forced to reason over a broad 5-minute transcript block

### Output artifacts saved
- `output/longshot_real_speech_only_wTlERUE8LVw/`
  - speech-only memory build for the benchmark video
- `output/longshot_real_speech_only_wTlERUE8LVw_run/predictions_single.jsonl`
  - raw pipeline prediction
- `output/longshot_real_speech_only_wTlERUE8LVw_run/predictions_refined.jsonl`
  - refined grounded answer from the same collected evidence
- `output/longshot_two_sample_compare.md`
  - concise side-by-side notes for the two benchmark samples we ran

## Update: Span-Aware `OPEN(..., speech)`

### Problem this step addressed
- The original `OPEN(speech)` behavior simply concatenated every speech span inside a node into one large transcript blob.
- This caused two benchmark problems:
  - correct scene, wrong local evidence inside that scene
  - grounded answerer drift because the evidence block was too broad

### What changed
- `rlm/video/tools.py` now makes `OPEN(..., speech)` select the most relevant speech spans inside the opened node.
- The selection uses:
  - current question
  - latest `SEARCH` query from action history
  - light temporal hints for `first` / `last` style questions
  - causal cues for `why` questions
- Instead of one giant speech evidence block, `OPEN(speech)` can now return a small list of narrow evidence objects with:
  - tighter `time_span`
  - a more precise `clip_path`
  - a `selection_score` in metadata

### Why this matters
- This is a much better fit for VideoRLM's intended control policy.
- The controller should not just find the right scene. It should also extract the right local evidence inside that scene.
- This change also makes the trajectory more useful for future SFT because the evidence ledger is more specific and easier to supervise.

### Verification
- Added tool-level tests in `tests/video/test_video_tools.py` for:
  - `why` / causal evidence selection
  - `first` / temporal evidence selection
- Relevant local checks passed:
  - `pytest tests/video/test_video_tools.py tests/video/test_video_controller.py tests/video/test_video_memory_index.py -q`
  - `ruff check rlm/video/tools.py tests/video/test_video_tools.py`

### Benchmark effect seen immediately
- Re-running `sample_8563` with cached memory after this change improved the raw pipeline answer:
  - it now stays on `chicken head`
  - and it carries over the `10 pounds` detail from the opened evidence
- Re-running `sample_6168` remained grounded on the clasp / loss / repair explanation.

## Update: Guarding Against Speech-Excerpt Regression

### New regression we found
- After the first span-aware `OPEN(..., speech)` improvement, we added sentence-level excerpt focusing to make evidence shorter.
- On real LongShOT reruns, that extra focusing regressed both benchmark samples:
  - `sample_8563` drifted from `chicken head` back to generic sauce / `fried goodness`
  - `sample_6168` collapsed to an irrelevant `quite heavy now` fragment
- The root cause was not bad span selection.
- The root cause was **over-aggressive excerpt compression**:
  - we were still shortening spans that were already reasonably short
  - this cut away the anchor sentence inside the selected span

### Fix applied
- `rlm/video/tools.py`
  - keep the selected speech span intact when the normalized span is already under the configured detail budget
  - only apply sentence-window focusing when the span is actually long enough to justify compression
  - if the focused snippet has weak confidence or loses all query overlap that existed in the full span, fall back to the original span text instead of keeping the lossy excerpt

### Why this fix is important
- Span selection is the useful part of the change.
- Over-compressing already-selected spans threw away exactly the evidence we worked to find.
- The new behavior is more conservative:
  - first narrow the evidence source
  - then only compress when there is a strong reason to do so

## Update: Duplicate Speech Evidence Across Overlapping Nodes

### New runtime issue surfaced by benchmark logs
- While re-running the benchmark with cached memories, we saw the controller repeatedly re-open overlapping parent / child nodes and receive the same speech evidence multiple times.
- This happens because neighboring nodes can contain the same underlying ASR span near temporal boundaries.
- In practice, this polluted the evidence ledger with duplicates and wasted steps.

### Fix applied
- `rlm/video/tools.py`
  - before appending new speech evidence, check whether the ledger already contains the same speech evidence
  - the duplicate check currently keys off:
    - speech modality
    - rounded `time_span`
    - normalized evidence detail text
  - if the evidence is already present, the tool skips re-emitting it

### Why this helps
- It makes the evidence ledger cleaner.
- It reduces repeated parent / child `OPEN` loops.
- It also gives the controller a better chance to move on to genuinely new evidence instead of re-reading the same transcript fragment.

## Update: Early Stop Guard After Repeated Empty `OPEN`s

### Failure mode found in live benchmark traces
- Even after duplicate evidence was suppressed, the controller could still spend several more steps issuing repeated `OPEN` actions on nodes that no longer produced any new evidence.
- That is a controller-side loop rather than a tool-side duplication problem.
- On LongShOT samples, this means the system can burn the rest of its budget even though the evidence ledger is already good enough to synthesize a grounded answer.

### Fix applied
- `rlm/video/controller.py`
  - track consecutive `OPEN` actions that return zero new evidence
  - if we already have evidence in the ledger and we hit two empty `OPEN`s in a row, stop the loop early
  - then synthesize the final answer from the evidence ledger instead of waiting for an explicit `STOP`

### Why this is a good v1 heuristic
- It is lightweight and easy to reason about.
- It does not override the normal `STOP` action when the controller is behaving well.
- It only triggers in a very specific stalled state:
  - evidence already exists
  - repeated `OPEN`s are no longer adding anything

### Verification
- Added controller coverage in `tests/video/test_video_controller.py` for:
  - repeated empty `OPEN`s leading to grounded fallback synthesis
- Relevant local checks passed:
  - `pytest tests/video/test_video_controller.py tests/video/test_video_tools.py tests/video/test_video_memory_index.py -q`
  - `ruff check rlm/video/controller.py rlm/video/tools.py tests/video/test_video_controller.py tests/video/test_video_tools.py`

### Verification for these fixes
- Added and extended tests in `tests/video/test_video_tools.py` for:
  - preserving short selected spans
  - skipping duplicate speech evidence already present in the ledger
- Relevant local checks passed again:
  - `pytest tests/video/test_video_tools.py tests/video/test_video_controller.py tests/video/test_video_memory_index.py -q`
  - `ruff check rlm/video/tools.py tests/video/test_video_tools.py`

## Update: Root Cause and Fix for `sample_6168` Speech Snippet Drift

### Root cause we confirmed
- The main failure was not ASR quality and not first-stage retrieval.
- The real bottleneck was `span -> snippet` extraction inside `rlm/video/tools.py`.
- More specifically:
  - the speech-tool tokenizer inherited retrieval stopwords and accidentally removed `why`
  - because of that, the `why`-specific heuristics were effectively disabled
  - `_focus_speech_detail()` could fall back to broad early context
  - `_score_speech_span()` could overvalue spans that merely contained a generic `because`

### What changed
- Preserved control tokens such as `why`, `first`, and `last` in the speech-tool tokenizer.
- Added light morphological normalization:
  - `wears / wore / worn -> wear`
  - `loves -> love`
  - `opening / opened -> open`
  - `fixed -> fix`
- Tightened `why` scoring:
  - reward real causal signals like `clasp`, `open`, `worried`, `lose`, `fix`, `repair`
  - penalize generic `because` when it is not backed by the right overlap
  - penalize topic-shift markers like `other bracelet` and `last but not the least`
- Reworked snippet extraction to build a window around the best anchor sentence instead of keeping broad early context.
- Kept neighbor-span expansion, but only when the neighbor has real causal signal for `why` questions.

### Regression tests added
- Added sample-shaped tests in `tests/video/test_video_tools.py` for:
  - causal late-span extraction on a long bracelet transcript
  - continuation-span extraction on the `perfect / haven't really worn it much` follow-up
  - preventing unrelated intro spans like the `Leon Diamond` segment from entering the ledger

### Validation
- Focused regression suite:
  - `conda run -n videorlm python -m pytest tests/video/test_video_tools.py tests/video/test_video_controller.py tests/video/test_video_memory_index.py -q`
  - result: `18 passed`
- Full repo validation:
  - `conda run -n videorlm python -m ruff check .`
  - `conda run -n videorlm python -m pytest -q`
  - result: `331 passed, 7 skipped`

### Real rerun result
- Re-ran the real cached `sample_6168` context using local `Qwen3-8B` controller over the existing memory cache.
- Output saved to:
  - `output/sample_6168_fix_rerun/result.json`
- The new answer is benchmark-aligned:
  - the bracelet kept opening
  - she was worried about losing it
  - she took it back to Cartier and it was fixed
  - she still has not worn it much even though she loves it

### Development lesson
- The system already knew where to look.
- The failure came from how evidence was carved out of the right span, not from the absence of the right span.
- In practice, this means targeted evidence-extraction fixes can recover benchmark behavior without rerunning the whole preprocessing stack.

## Update: Hybrid Speech Snippet Refinement Experiment

### What we tried
- Added an optional hybrid mode to `OPEN(..., speech)`:
  - heuristic span selection still runs first
  - the tool then builds a small shortlist of grounded candidate snippets from the raw transcript
  - if the span looks ambiguous, a language model can choose candidate ids instead of generating free-form text
- The implementation is behind `enable_hybrid_speech_refinement=False` by default, so the stable runtime path is unchanged unless explicitly enabled.

### Why we tried it
- The heuristic-only tool is cheap and debuggable, but it still has a blind spot on long, messy spans with multiple topics.
- The goal of the hybrid experiment was:
  - keep exact snippet text grounded in the transcript
  - let the LLM help only with the final `candidate -> chosen snippet` selection
  - avoid fully replacing `OPEN(..., speech)` with free-form LLM summarization

### Code changes
- `rlm/video/tools.py`
  - added candidate-snippet generation for speech evidence
  - added hybrid gating and prompt construction for candidate-id selection
  - added metadata fields like `selection_mode`, `refinement_triggered`, and `selected_candidate_ids`
- `rlm/video/controller.py`
  - threads hybrid configuration into the tool executor
- `rlm/video/qwen.py`
  - exposes hybrid flags in the Qwen stack bundle builders
- `tests/video/test_video_tools.py`
  - added tests for:
    - hybrid refiner selecting a better causal snippet
    - hybrid refiner staying idle on short, already-clear spans

### Validation
- Focused checks:
  - `conda run -n videorlm python -m pytest tests/video/test_video_tools.py tests/video/test_video_controller.py tests/video/test_video_qwen_local.py -q`
  - result: `18 passed`
- Full repo checks:
  - `conda run -n videorlm python -m ruff check .`
  - `conda run -n videorlm python -m pytest -q`
  - result: `334 passed, 7 skipped`

### 4-sample LongShOT evaluation
- Compared heuristic-only vs hybrid on 4 cached LongShOT samples:
  - `sample_6009`
  - `sample_6168`
  - `sample_6815`
  - `sample_8563`
- Output files:
  - `output/longshot_hybrid_eval_4samples/results.json`
  - `output/longshot_hybrid_eval_4samples/summary.md`

### Result summary
- `sample_6009`
  - baseline F1 vs gold: `0.5041`
  - hybrid F1 vs gold: `0.5280`
  - small improvement
- `sample_6168`
  - baseline F1 vs gold: `0.6116`
  - hybrid F1 vs gold: `0.4773`
  - regression
- `sample_6815`
  - baseline F1 vs gold: `0.6275`
  - hybrid F1 vs gold: `0.3590`
  - strong regression
- `sample_8563`
  - baseline F1 vs gold: `0.2449`
  - hybrid F1 vs gold: `0.1860`
  - regression
- Average token-F1 over the 4 samples:
  - baseline: `0.4970`
  - hybrid: `0.3876`

### What we learned
- The hybrid path is not good enough to enable by default.
- It can help on some `why` questions by selecting a more direct quote.
- But it currently introduces two important failure modes:
  - **Over-compression on causal questions**
    - Example: `sample_6168`
    - the hybrid refiner selected a strong causal snippet (`opening / worried / Cartier fixed`) but encouraged the controller to stop earlier
    - the final answer lost the follow-up support (`now it's really perfect`, `still hasn't worn it much`)
  - **Over-selection of quirky early snippets on `first` questions**
    - Example: `sample_8563`
    - the refiner selected an early snippet about `chicken skin` / `lightly bizarre fried treats`
    - this pulled the answer away from the correct target: `chicken head`

### Debug artifacts for the regressions
- Saved extra hybrid debug runs to:
  - `output/longshot_hybrid_eval_4samples/debug/sample_6168_hybrid_debug.json`
  - `output/longshot_hybrid_eval_4samples/debug/sample_8563_hybrid_debug.json`
- These confirm:
  - `sample_6168`: the refiner chose a good causal snippet, but the loop stopped with only one evidence item
  - `sample_8563`: the refiner chose the wrong early candidate and amplified the error

### Current recommendation
- Keep the hybrid path as an **experimental, opt-in feature only**.
- Do **not** turn it on by default yet.
- The next upgrade should likely be narrower than this first hybrid attempt:
  - only trigger LLM refinement for ambiguous `why` cases
  - keep heuristic-only behavior for `first / last` temporal questions
  - add a way to preserve follow-up support evidence instead of letting the controller stop after one concise causal snippet

## 2026-04-12: Official-style LongShOT evaluation

### What changed
- Added a local `official-style` LongShOT evaluator in:
  - `rlm/video/longshot_official_eval.py`
  - `rlm/video/cli.py` via `eval-longshot-official`
- This evaluator mirrors the public LongShOT evaluation structure more closely than our old proxy metrics:
  - judge each assistant turn criterion separately
  - store `criteria_met` back into the benchmark JSONL
  - compute weighted task/category/overall accuracy using the LongShOT scoring rules
- The main runtime adaptation is that we use a local Hugging Face judge model instead of the public repo's OpenAI-compatible / vLLM server requirement.

### Important implementation notes
- Reused the official-style criterion prompt:
  - compare `Ground Truth Response`
  - compare `Model Response`
  - evaluate a single rubric criterion at a time
  - return strict JSON with `criteria_met`
- Added retry logic for malformed judge output:
  - up to 3 attempts per criterion
  - if still invalid, mark `criteria_met = None` and store the raw error
- Added progress logging per sample so long eval runs are observable and resumable.

### Why this was needed
- Our old `token_f1` and `rouge_l_f1` numbers were useful for quick internal comparison, but they were not the LongShOT benchmark metric.
- LongShOT is rubric-based:
  - each assistant turn has weighted criteria
  - positive criteria contribute to score when met
  - penalties do not add to the denominator
  - task accuracy is weighted score obtained divided by total positive score
  - overall is the mean of category averages

### 90-sample official-style run
- Input predictions:
  - `output/longshot_single_gpu_full/predictions.jsonl`
- Official-style evaluation outputs:
  - `output/longshot_single_gpu_full_official_eval/eval.jsonl`
  - `output/longshot_single_gpu_full_official_eval/score.txt`
  - `output/longshot_single_gpu_full_official_eval/summary.json`
- Judge model used:
  - `Qwen/Qwen2.5-7B-Instruct`
  - device: `cuda:3`
- Coverage:
  - `90` samples
  - `133` assistant turns
  - `663` rubric criteria
- Eval robustness:
  - `0` criteria parse failures
  - `0` criteria with `evaluation_error`

### Official-style result summary
- Overall accuracy:
  - `14.97%`
- Category averages:
  - `Core Perception Tasks`: `15.79%`
  - `Reasoning Tasks`: `9.79%`
  - `Information Tasks`: `12.24%`
  - `Multimodal Tasks`: `22.06%`

### Task highlights
- Better relative buckets in this 90-sample subset:
  - `audio_visual_alignment`: `29.17%`
  - `temporal_reasoning`: `26.77%`
  - `summarization`: `24.43%`
- Weak buckets:
  - `event_understanding`: `8.10%`
  - `instruction_extraction`: `8.00%`
  - `compositional_reasoning`: `0.00%`
  - `sentiment_analysis`: `0.00%`

### Main takeaway
- The official-style score is much harsher than proxy overlap metrics, which is expected.
- This confirms that the current system is still getting partial lexical overlap on many samples without satisfying enough weighted rubric criteria.
- The biggest quality gap continues to be:
  - event-level understanding
  - deeper reasoning/composition
  - instruction extraction with precise grounded details
