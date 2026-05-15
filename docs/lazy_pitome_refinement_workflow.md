# Lazy PiToMe Retrieval with On-Demand Visual and Speech Refinement

This document describes the official lazy PiToMe VideoRLM path driven by:

```bash
bash scripts/run_longshot_lazy_pitome_refinement_local.sh
```

VideoRLM now has two supported long-video strategies:

```text
original     full VideoRLM preprocessing with regular visual and speech summaries
lazy-pitome  cheap PiToMe/SigLIP indexing plus on-demand QwenVL/QwenASR refinement
```

The goal of the lazy PiToMe strategy is to stop running QwenVL and ASR over every long video span during memory
construction. Instead, preprocessing builds cheap visual and speech timestamp indexes, then
the heavy local models run only on the few windows that retrieval decides are useful for the
current question.

## High-Level Sketch

```text
LongShOT sample
  |
  v
Resolve local video
  |
  v
Build or load video memory
  |
  +--> Lazy ASR timestamp indexing
        |
        +--> split the video into speech windows
        +--> save lazy speech spans without running the ASR model
  |
  +--> Lazy visual indexing per clip window
        |
        +--> sample dense frames cheaply
        +--> select PiToMe representatives
        +--> add edge frames and detected scene-boundary frames
        +--> compute lightweight PiToMe vectors
        +--> compute SigLIP/CLIP image embeddings
        +--> fuse SigLIP/CLIP vectors into stored PiToMe frame embeddings
        +--> save timestamps + embeddings + lazy metadata
  |
  v
Question-time VideoRLM loop
  |
  +--> classify question/evidence slot modality
  +--> graph search over text, speech, temporal edges, PiToMe vectors, semantic frame embeddings
  +--> diversify top windows so they do not all come from one moment
  +--> OPEN selected node
        |
        +--> if visual node is lazy, run QwenVL only for that node's time window
        +--> update node with refined summary
        +--> if speech node is lazy, extract that window's audio and run ASR only there
        +--> update node with refined transcript spans
  |
  v
Final answer from retrieved evidence only
```

## Stage 1: Script Configuration

The runner is:

```text
scripts/run_longshot_lazy_pitome_refinement_local.sh
```

Important defaults:

```text
OUTPUT_ROOT=output/longshot_lazy_pitome_refinement_local_faiss
DATASET_NAME=postvalid_v1
SAMPLE_LIMIT=20
CONTROLLER_REPO=Qwen/Qwen3-0.6B
VISUAL_REPO=Qwen/Qwen3-VL-2B-Instruct
SKIP_SPEECH_RECOGNITION=0
SPEECH_CHUNK_DURATION_SECONDS=120
CLIP_DURATION_SECONDS=480
PITOME_DENSE_FRAME_RATE=0.2
PITOME_MAX_SELECTED_FRAMES=8
SEMANTIC_FRAME_EMBEDDING_REPO=google/siglip-base-patch16-224
PITOME_SCENE_THRESHOLD=0.35
PITOME_MAX_SCENE_BOUNDARY_FRAMES=6
```

The script always enables:

```text
--use-pitome
--search-mode graph
--lazy-visual-refinement
--lazy-speech-refinement  # unless SKIP_SPEECH_RECOGNITION=1
```

This means memory construction uses the lazy visual indexer and lazy ASR indexer, while
question-time `OPEN` calls can invoke QwenVL or ASR on demand.

## Stage 2: Bundle Construction

The local stack is wired in:

```text
rlm/video/qwen.py
```

When `use_pitome=True` or either lazy refinement flag is set:

```text
memory_builder.visual_summarizer = LazyPiToMeVisualIndexer(...)
memory_builder.speech_recognizer = LazySpeechRecognizer(...)
controller.visual_refiner = LocalQwenVisualSummarizer(...)
controller.speech_refiner = LocalQwenASRSpeechRecognizer(...) or FasterWhisperSpeechRecognizer(...)
controller.search_mode = "graph"
controller.image_text_embedding_provider = LocalImageTextEmbeddingProvider(...)
```

So there are two separate visual components:

```text
LazyPiToMeVisualIndexer
  Used during preprocessing.
  Cheap. No QwenVL call.

LocalQwenVisualSummarizer
  Used only during visual OPEN on selected lazy nodes.
  Expensive. Calls QwenVL.
```

When `lazy_speech_refinement=True` and speech recognition is enabled:

```text
memory_builder.speech_recognizer = LazySpeechRecognizer(...)
controller.speech_refiner = LocalQwenASRSpeechRecognizer(...) or FasterWhisperSpeechRecognizer(...)
```

So there are also two separate speech components:

```text
LazySpeechRecognizer
  Used during preprocessing.
  Cheap. Builds timestamp-only placeholder spans.

LocalQwenASRSpeechRecognizer / FasterWhisperSpeechRecognizer
  Used only during speech OPEN on selected lazy nodes.
  Expensive. Runs ASR on one extracted audio window.
```

## Stage 3: Preprocess Once Per Video

Memory construction starts in:

```text
rlm/video/memory.py
```

For each video, `VideoMemoryBuilder.prepare_artifacts()` does:

```text
1. Plan visual spans from the video duration.
2. Build lazy ASR timestamp spans if lazy speech refinement is enabled.
3. Run lazy visual summarization on clip windows.
4. Save prepared artifacts.
5. Build hierarchical memory nodes.
```

With this runner, visual spans are clip windows of `CLIP_DURATION_SECONDS`, default `480`.
That means a 45-minute video becomes roughly 6 visual windows instead of dozens of
60-second QwenVL summaries.

## Stage 4: Lazy Visual Indexing

Lazy visual indexing lives in:

```text
rlm/video/local_adapters.py
```

`LazyPiToMeVisualIndexer.summarize()` processes each visual window:

```text
1. Sample dense frames at PITOME_DENSE_FRAME_RATE.
2. Build cheap PiToMe frame embeddings.
3. Select representative frames with PiToMe.
4. Add boundary frames near the start and end of the window.
5. Run FFmpeg scene-change detection and add representative scene-boundary frames.
6. Limit selected frames by temporal coverage.
7. Compute semantic image embeddings for selected frames.
8. Fuse compact PiToMe vectors with compact SigLIP/CLIP vectors.
9. Store timestamps, fused PiToMe vectors, semantic vectors, and lazy flags.
```

The stored metadata includes values like:

```text
selected_frame_timestamps
boundary_frame_timestamps
scene_boundary_frame_timestamps
pitome_frame_embeddings
pitome_frame_embedding_semantic_fusion = true
semantic_frame_embeddings
visual_summary_mode = lazy_pitome_index
on_demand_visual_refinement = true
```

The summary text is intentionally generic:

```text
PiToMe visual index for 0.00-480.00 with N representative frames.
Open this node visually to run QwenVL refinement.
```

This is important: preprocessing does not ask QwenVL what is in every window.
It only builds a searchable visual index.

## Stage 4b: Lazy ASR Indexing

Lazy ASR indexing also lives in:

```text
rlm/video/local_adapters.py
```

`LazySpeechRecognizer.recognize()` does not load or call Qwen ASR. It only:

```text
1. Probes the video duration.
2. Splits the video into SPEECH_CHUNK_DURATION_SECONDS windows.
3. Stores one placeholder SpeechSpan per window.
4. Marks each span with language = "lazy_asr".
```

The placeholder text is intentionally generic:

```text
Lazy ASR index for 0.00-120.00. Open this node as speech to run ASR refinement.
```

This gives the memory graph speech-aware time windows without paying the full ASR cost upfront.

## Stage 5: Memory Graph Construction

The memory builder turns artifacts into hierarchical nodes:

```text
video
  scene
    segment
      clip
```

For PiToMe lazy mode, the clip nodes carry the useful visual metadata:

```text
visual_summary: generic lazy index text
metadata: selected timestamps + embeddings + refinement flag
```

Scene and segment nodes can still roll up child clip details compactly, but the real visual
payload for this mode is in the clip metadata.

## Stage 6: Question-Time Modality Routing

When a question comes in, VideoRLM builds a question spec in:

```text
rlm/video/evidence_pipeline.py
```

It infers whether the evidence should likely be:

```text
speech
visual
ocr
audio
cross_modal
```

Examples:

```text
"what did she say" -> speech
"what is shown on screen" -> visual/ocr
"what sound is heard" -> audio/speech
"why did the subject stop moving" -> often visual
```

The selected modality controls which search path and which `OPEN` tool the controller should use.

## Stage 7: Graph Search Over Lazy Visual Memory

Search is handled by:

```text
rlm/video/index.py
rlm/video/evidence_pipeline.py
```

Graph search combines multiple signals:

```text
VLM summary lexical score
PiToMe frame vector similarity
SigLIP/CLIP text-image similarity
OCR/code atom overlap
nearby speech transcript overlap
temporal edges between parent, child, and sibling nodes
```

For lazy visual nodes, the VLM summary lexical score is weak because the summary is generic.
The stronger signals are:

```text
semantic_frame_embeddings
pitome_frame_embeddings
scene_boundary_frame_timestamps
nearby speech text
temporal graph edges
```

The current implementation stores embeddings in the memory/artifact JSON and performs graph/vector
search through `VideoMemoryIndex`. When `faiss` is installed, `VideoMemoryIndex` builds an in-memory
FAISS `IndexFlatIP` over normalized semantic frame embeddings and uses it for query-to-frame lookup
and semantic frame-neighbor expansion. If `faiss` is not installed, it falls back to the older
in-process scan over cached metadata.

To activate the FAISS path in the local conda environment:

```bash
conda install -n videorlm -c conda-forge faiss-cpu
```

## Stage 8: Diversity and Coverage Selection

After search returns candidate hits, `search_v2()` applies temporal diversity:

```text
1. Sort hits by score.
2. Prefer clip-level hits over broader overlapping scene/segment hits when the score is close.
3. Avoid filling the top-k list with multiple windows from the same moment.
4. Backfill with deferred hits if there are not enough diverse candidates.
```

This helps select top windows like:

```text
clip around 0:00-8:00
clip around 15:00-23:00
clip around 32:00-40:00
```

instead of:

```text
scene 0:00-8:00
segment 0:00-8:00
clip 0:00-8:00
```

## Stage 9: On-Demand Heavy Visual Refinement

When the controller opens a visual node, `VideoToolExecutor` checks:

```text
metadata["on_demand_visual_refinement"] == true
```

If true, it calls:

```text
LocalQwenVisualSummarizer.summarize(source_video_path, [node.time_span])
```

This runs QwenVL only on that selected window, using the current visual summarizer settings.
After QwenVL returns, the node is updated:

```text
node.visual_summary = refined QwenVL summary
node.tags += refined tags
node.entities += refined entities
metadata["visual_summary_mode"] = "on_demand_refined"
metadata["visual_refinement"] = "qwenvl_on_demand"
metadata["on_demand_visual_refinement"] = false
```

So if the same node is opened again later in the same run, it does not rerun QwenVL.

## Stage 10: On-Demand Speech Refinement

When the controller opens a speech node, `VideoToolExecutor` checks whether that node contains
lazy ASR spans:

```text
span.language == "lazy_asr"
```

If true, it:

```text
1. Extracts only that node's audio window from the source video.
2. Calls the configured full ASR recognizer on that temporary audio file.
3. Offsets returned ASR timestamps back into the original video timeline.
4. Replaces the placeholder speech spans with real transcript spans.
5. Marks metadata["speech_summary_mode"] = "on_demand_refined".
```

So if the same speech node is opened again later in the same run, it does not rerun ASR.

## Stage 11: Evidence Construction

The opened visual node becomes an evidence item:

```text
claim: Visual evidence: ...
detail: refined QwenVL summary
time_span: node time span
source_node_id: opened node id
metadata: refinement information + slot information
```

Speech evidence, OCR evidence, and audio evidence are added through their own `OPEN` modes.
For lazy speech nodes, the evidence is created after on-demand ASR has replaced the placeholder
with real transcript spans. The final answer prompt receives only the retrieved evidence ledger,
not the whole video memory.

## Stage 12: Final Answer

The final answer is generated by the controller using the evidence collected by tool calls.
If enough core/support evidence exists, it answers from that evidence. If required slots are still
missing, it abstains or reports missing evidence.

The intended final flow is:

```text
Search cheaply.
Open a few high-value windows.
Run QwenVL only where needed.
Answer from the refined evidence.
```

## Why This Should Be Faster

The original local visual path does:

```text
QwenVL over every planned visual span before any question is asked.
```

The lazy PiToMe path does:

```text
Cheap frame extraction + embeddings over every planned visual span.
QwenVL only for selected visual windows at question time.
```

The original speech path does:

```text
ASR over every speech chunk before any question is asked.
```

The lazy speech path does:

```text
Timestamp-only placeholders over every speech chunk.
ASR only for selected speech windows at question time.
```

This saves time when most windows are irrelevant to a question.

## Main Tradeoffs

Benefits:

```text
Lower preprocessing cost.
Lower QwenVL call count.
Lower ASR and visual inference cost on long videos.
Better routing when semantic frame embeddings match the question.
```

Risks:

```text
If retrieval misses the right window, QwenVL never sees it.
If retrieval misses the right speech window, ASR never transcribes it.
Very large clip windows can still be expensive to refine.
Weak semantic embeddings can hurt visual recall.
Lazy ASR placeholders are weaker for lexical speech search than full transcripts.
```

Practical knobs:

```text
PITOME_DENSE_FRAME_RATE
CLIP_DURATION_SECONDS
PITOME_MAX_SELECTED_FRAMES
PITOME_SCENE_THRESHOLD
PITOME_MAX_SCENE_BOUNDARY_FRAMES
SEMANTIC_FRAME_EMBEDDING_REPO
SKIP_SPEECH_RECOGNITION
SPEECH_CHUNK_DURATION_SECONDS
SPEECH_BACKEND
SAMPLE_LIMIT
```

For quick testing, use:

```bash
SAMPLE_LIMIT=5 bash scripts/run_longshot_lazy_pitome_refinement_local.sh
```

To disable speech entirely:

```bash
SKIP_SPEECH_RECOGNITION=1 bash scripts/run_longshot_lazy_pitome_refinement_local.sh
```

The old eager PiToMe preprocessing path is no longer an official mode. The compatibility
script `scripts/run_longshot_pitome_local.sh` delegates to this lazy PiToMe runner.
