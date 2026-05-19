# Qwen3-8B vs Qwen3.5-9B on the Same 90 LongShOT Samples

This note compares two controller runs on the same 90-sample LongShOT subset:

- Baseline controller:
  - `Qwen/Qwen3-8B`
  - predictions: `output/longshot_single_gpu_full/predictions.jsonl`
  - official-style eval: `output/longshot_single_gpu_full_official_eval/summary.json`
- New controller:
  - `Qwen/Qwen3.5-9B`
  - predictions: `output/longshot_single_gpu_full_qwen35_9b/predictions.jsonl`
  - official-style eval: `output/longshot_single_gpu_full_qwen35_9b_official_eval/summary.json`

Both runs use:

- the same 90 samples
- the same 8 videos
- the same memory cache for those videos
- the same official-style evaluator
- the same judge model:
  - `Qwen/Qwen2.5-7B-Instruct`

## Bottom line

`Qwen3.5-9B` was more difficult to stabilize as a controller and produced a worse benchmark result than `Qwen3-8B` in this system configuration.

Official-style overall accuracy:

- `Qwen3-8B`: `14.97%`
- `Qwen3.5-9B`: `8.00%`
- delta: `-6.97` absolute points

This means the stronger base model did **not** translate into a stronger VideoRLM controller under the current runtime constraints.

## Why this is not surprising

A stronger base model is not automatically a stronger controller.

The controller in VideoRLM is not asked to write a free-form answer directly. It is asked to:

- read a structured state
- choose a valid next action
- output strict JSON
- keep reasoning grounded in evidence
- stop at the right moment

That makes controller quality a function of:

- model capability
- prompt fit
- action schema fit
- parser robustness
- tool behavior
- stopping behavior

So the relevant comparison is not:

- “Which model is stronger in the abstract?”

It is:

- “Which model works better as a strict planner/controller in this exact loop?”

For this experiment, `Qwen3-8B` worked better.

## Important caveat

This was **not** a perfectly pure model-only swap.

To make `Qwen3.5-9B` stable enough to finish the run, we had to add controller-specific mitigations:

- increase controller `max_new_tokens`
- tighten the controller system prompt
- make the action history more compact
- explicitly discourage long rationales

These changes were necessary because `Qwen3.5-9B` repeatedly produced overly long JSON actions that got truncated and crashed parsing.

So the practical comparison is:

- baseline:
  - `Qwen3-8B` with the older controller prompt/runtime
- new run:
  - `Qwen3.5-9B` with a more aggressively constrained controller prompt/runtime

This caveat matters because the stabilization changes themselves likely affected answer style and stopping behavior.

## Score comparison

### Overall

- `Qwen3-8B`: `14.97%`
- `Qwen3.5-9B`: `8.00%`

### Category averages

- Core Perception Tasks:
  - `15.79% -> 11.13%`
  - delta: `-4.66`
- Reasoning Tasks:
  - `9.79% -> 4.44%`
  - delta: `-5.34`
- Information Tasks:
  - `12.24% -> 2.10%`
  - delta: `-10.15`
- Multimodal Tasks:
  - `22.06% -> 14.33%`
  - delta: `-7.74`

### Task deltas

- Improved:
  - `event_understanding`: `8.10% -> 12.43%`

- Flat:
  - `compositional_reasoning`: `0.00% -> 0.00%`
  - `sentiment_analysis`: `0.00% -> 0.00%`

- Worse:
  - `temporal_reasoning`: `26.77% -> 14.91%`
  - `information_retrieval`: `16.55% -> 8.39%`
  - `multimodal_synthesis`: `14.96% -> 5.74%`
  - `causal_reasoning`: `18.18% -> 10.39%`
  - `summarization`: `24.43% -> 0.00%`
  - `instruction_extraction`: `8.00% -> 0.00%`
  - `audio_understanding`: `9.09% -> 0.00%`
  - `audio_visual_alignment`: `29.17% -> 22.92%`
  - `comparative_analysis`: `11.18% -> 2.94%`
  - `entity_recognition`: `19.19% -> 17.17%`

## Runtime behavior differences

### Average steps per turn

- `Qwen3-8B`: `5.75`
- `Qwen3.5-9B`: `5.19`

### Average answer length

- `Qwen3-8B`:
  - average answer tokens: `50.15`
  - median answer tokens: `49`
- `Qwen3.5-9B`:
  - average answer tokens: `34.91`
  - median answer tokens: `21`

### Average execution time per turn

- `Qwen3-8B`: `31.11s`
- `Qwen3.5-9B`: `59.47s`

## Main interpretation

The new controller is:

- taking fewer steps
- producing much shorter answers
- and still scoring worse

That pattern strongly suggests:

- earlier stopping
- less evidence collection
- less rubric-complete answers
- more under-specification

This is more informative than simply saying “the stronger model got worse.”

It says:

- the new controller likely became **more compressed and more confident**
- but that confidence was not matched by rubric completeness

## Most likely reasons the score dropped

### 1. The model is over-compressed by the stabilization prompt

To stop `Qwen3.5-9B` from crashing the action parser, we added stronger prompt instructions:

- be terse
- do not explain thinking
- keep rationale short
- compact action history

Those changes improved controller stability, but they also likely pushed the model toward:

- shorter answers
- fewer elaborations
- fewer follow-up details

That aligns with the observed answer-length drop:

- median answer length fell from `49` tokens to `21`

Official LongShOT scoring is rubric-heavy, so answers that are shorter but less complete will usually score much worse.

### 2. The controller appears to stop earlier with less evidence

`Qwen3.5-9B` used fewer steps on average:

- `5.75 -> 5.19`

That is not automatically good.

In this benchmark, many turns require:

- one main point
- plus one or two supporting details
- plus sometimes a causal explanation or temporal clarification

If the controller stops after capturing only the main gist, it can still sound reasonable while failing the rubric.

This is exactly the sort of behavior that hurts:

- summarization
- instruction extraction
- temporal reasoning
- multimodal synthesis

### 3. The stronger model seems to generalize or abstract too aggressively

In several examples, `Qwen3.5-9B` produces an answer that is:

- semantically plausible
- concise
- but missing the benchmark-specific details that the rubric expects

This is a classic “sounds smart but scores worse” failure mode.

The benchmark does not reward elegance. It rewards:

- explicit detail coverage
- grounded specificity
- mention of the exact essential points

### 4. The current controller prompt was implicitly better matched to Qwen3-8B

The prompt and schema were originally stabilized around `Qwen3-8B` behavior.

`Qwen3.5-9B` brought different behavior:

- more verbose raw JSON actions
- stronger tendency to narrate reasoning
- more pressure on parser robustness

Once we constrained it hard enough to make it stable, we likely over-corrected and made it too terse for the benchmark.

So this result should not be read as:

- “Qwen3.5-9B is worse than Qwen3-8B in general”

It should be read as:

- “Qwen3.5-9B is worse than Qwen3-8B in this controller setup as currently tuned”

## Concrete examples

### Example 1: `sample_6095`

Ground truth expects all of these ideas:

- quasar light traveled for billions of years
- this makes hidden influence extremely unlikely
- this closes a major loophole
- the filter choices are truly random and not correlated with the particles

Baseline `Qwen3-8B` answer:

- still imperfect
- but includes more of the missing benchmark structure

`Qwen3.5-9B` answer:

- shorter
- keeps the “billions of years” part
- but drops the stronger loophole/randomness explanation

This is a good example of:

- less hallucination
- but also less rubric coverage

### Example 2: `sample_6528`

Ground truth:

- several reasons for doing both squats
- feeling “dead”
- normally choosing one or the other
- pendulum squat as favorite for quads
- still wanting barbell squat for strength work

Baseline answer:

- not perfect
- but at least includes the “do both” motivation

`Qwen3.5-9B` answer:

- “She decided to do both because she was feeling dead.”

This is too compressed.

It captures one local detail but drops the structure of the answer.

That is exactly the type of response official-style rubric scoring punishes.

### Example 3: `sample_6097`

Ground truth:

- she had not done a workout split video in a long time
- viewers wanted to see her current routine
- she is in a bulk phase
- it was highly requested

`Qwen3.5-9B` answer:

- “She decided to do a full week of workouts because she stated 'this week's full week of workouts all done'.”

This is not just incomplete.
It is anchored to the wrong local phrase.

So here the problem is not only brevity.
It is also weaker evidence selection or weaker answer synthesis from the selected evidence.

### Example 4: `sample_6536`

Ground truth requires a structured explanation of workout ordering:

- compound first
- then unilateral
- then isolation
- plus why that order makes sense

Baseline answer:

- gets a fair amount of the structure

`Qwen3.5-9B` answer:

- collapses into vague high-level phrasing
- loses the full instructional sequence

That helps explain why `instruction_extraction` and `summarization` both collapsed.

## Why event understanding improved slightly

`event_understanding` is the only bucket that improved materially.

My best explanation is:

- the shorter, more decisive controller/answer style sometimes helps for direct event questions
- when the answer can be supported by one short event claim, the compressed style may avoid over-talking

But this advantage did not generalize to tasks that demand:

- multiple supporting details
- temporal sequencing
- instructional structure
- richer multimodal synthesis

## What this result suggests about next steps

### 1. Do not switch the default controller to Qwen3.5-9B yet

At the moment, `Qwen3-8B` remains the better controller setting for this repo.

### 2. If we want Qwen3.5-9B to win, we probably need a different controller prompt

The current “make it terse so the parser survives” approach likely hurts answer completeness too much.

The right next iteration is probably:

- keep action JSON compact
- but decouple the action schema from long answer text
- especially avoid long free-text `answer` and `rationale` inside controller actions

### 3. Separate planner stability from final answer richness

Right now the controller action and the eventual answer are too entangled.

Better design:

- controller outputs a compact structural action
- answer synthesis remains free enough to be detailed

That would let us:

- keep parsing stable
- without forcing the whole system to become terse

### 4. Revisit STOP behavior

Because `Qwen3.5-9B` uses fewer steps, we should explicitly test whether it is stopping earlier on partial evidence.

Likely fixes:

- stricter STOP gate
- reward or heuristic for evidence sufficiency
- require support for at least one main claim plus one key detail before stopping

## Final conclusion

The result does **not** mean `Qwen3.5-9B` is a weaker base model.

It means:

- in the current VideoRLM controller design,
- with strict JSON action output,
- with the current prompt and stop behavior,
- and after adding stability constraints,

`Qwen3.5-9B` became:

- more concise
- more brittle as a planner
- less rubric-complete as an answerer

That combination reduced the final LongShOT official-style score from `14.97%` to `8.00%`.

The most likely practical explanation is:

`Qwen3.5-9B` is not worse at reasoning in general; it is worse matched to the current controller interface and prompt regime than `Qwen3-8B`.
