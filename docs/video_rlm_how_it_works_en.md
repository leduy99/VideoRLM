# How VideoRLM Works

## The Goal of VideoRLM

VideoRLM is designed to answer questions over long videos without trying to "read the whole video at once."

Instead of stuffing the entire transcript, frames, OCR, and audio into one prompt, VideoRLM behaves more like an agent with a plan:

1. Turn the video into structured memory.
2. Identify the most suspicious or relevant region.
3. Open that region with the right modality.
4. Collect evidence.
5. Stop when there is enough support to answer.

The core idea is:

`VideoRLM does not try to memorize the whole video. It learns how to decide what to inspect next.`

## From a User's Point of View

If a user provides:

- a long video
- a question

then VideoRLM tries to answer by:

1. locating the right part of the video
2. reading the right kind of information
3. extracting evidence
4. synthesizing a grounded answer

The final objective is not only to be correct, but also to:

- answer correctly
- make the answer traceable to video evidence
- reduce cost compared with reading the entire video

## A Simple Diagram

```text
raw video
  -> memory builder
  -> structured video memory
  -> search index
  -> controller state
  -> controller action
  -> tool execution
  -> new evidence + updated state
  -> final grounded answer
```

## 1. The Video Becomes "Memory"

Instead of treating the video as just an `.mp4` file, VideoRLM turns it into an external memory structure.

That memory is organized as a temporal tree:

```text
video -> scene -> segment -> clip
```

Each node in the tree may contain:

- `time_span`
- `speech_spans`
- `visual_summary`
- `ocr_spans`
- `audio_events`
- `clip_path`
- `tags/entities`

Why this matters:

- the video becomes a database that can be queried
- the controller does not need to see the full raw video at once

In the repo, this mainly lives in:

- [memory.py](/share_4/users/duy/project/rlm/rlm/video/memory.py)
- [types.py](/share_4/users/duy/project/rlm/rlm/video/types.py)
- [artifact_store.py](/share_4/users/duy/project/rlm/rlm/video/artifact_store.py)

## 2. The System Builds an Index for Fast Search

Once memory exists, the system builds a retrieval layer to answer:

`If we need relevant information, which node should we inspect first?`

In the current v1, `SEARCH` mainly relies on:

- lexical overlap between the query and transcript
- lexical overlap between the query and visual summary or OCR
- a light temporal prior for question types such as `first` or `last`

This creates the `frontier`, which is the list of nodes currently worth investigating.

In the repo, this lives in:

- [index.py](/share_4/users/duy/project/rlm/rlm/video/index.py)

## 3. State Is a "Snapshot" of Reasoning

The controller does not look directly at the whole video.
It looks at `state`.

State is a structured snapshot of the system at one decision point.

It includes the most important pieces:

- `question`
- `dialogue_context`
- `frontier`
- `evidence_ledger`
- `action_history`
- `budget`
- `global_context`

A short way to understand it:

- `frontier` = what looks suspicious right now
- `evidence_ledger` = what has already been confirmed
- `budget` = how many steps remain

In the repo, this lives in:

- [types.py](/share_4/users/duy/project/rlm/rlm/video/types.py)

## 4. The Controller Does Not Answer Immediately

Instead of generating the final answer right away, the controller produces a JSON action.

In the current v1, the action space is intentionally small:

- `SEARCH(query, modality)`
- `OPEN(node_id, modality)`
- `SPLIT(node_id)`
- `MERGE(evidence_ids)`
- `STOP(answer, evidence_ids)`

Meaning:

- `SEARCH`: find relevant nodes
- `OPEN`: inspect a node in more detail using the chosen modality
- `SPLIT`: break a large node into smaller child nodes
- `MERGE`: combine several evidence items into one logical bundle
- `STOP`: stop and answer

In the repo, this lives in:

- [controller.py](/share_4/users/duy/project/rlm/rlm/video/controller.py)
- [prompts.py](/share_4/users/duy/project/rlm/rlm/video/prompts.py)

## 5. What `SEARCH` Does

`SEARCH` is the step that asks:

`Where should we look next?`

It takes:

- the current question
- an optional preferred modality, such as `speech`

It returns:

- a ranked list of `frontier items`
- each item has:
  - `node_id`
  - `time_span`
  - `score`
  - `why_candidate`

Example:

- if a scene transcript contains terms such as `plan`, `change`, and `schedule`
- that scene will rank highly for a question about a plan change

Important:

- `SEARCH` does not deeply read the evidence yet
- it only ranks candidate nodes

## 6. What `OPEN` Does

`OPEN` is the step that asks:

`Inside that node, what specific content supports the answer?`

If `SEARCH` answers:

`which scene should we inspect?`

then `OPEN` answers:

`what inside that scene actually matters?`

Depending on the modality, `OPEN` behaves differently:

- `OPEN(..., speech)` reads transcript spans inside the node
- `OPEN(..., visual)` reads a visual summary or a clip
- `OPEN(..., ocr)` reads on-screen text
- `OPEN(..., audio)` reads audio events

For `speech`, the current system tries to:

1. gather the `speech spans` in the node
2. score which spans are most relevant to the question
3. select a small number of the best spans
4. convert them into `Evidence`

Each `Evidence` item usually contains:

- `claim`
- `modality`
- `time_span`
- `source_node_id`
- `detail`
- `confidence`

In the repo, this lives in:

- [tools.py](/share_4/users/duy/project/rlm/rlm/video/tools.py)

## 7. What the Evidence Ledger Is For

This is where the system accumulates evidence.

It helps prevent the model from:

- reading the same thing over and over
- answering based on vague intuition
- losing track after opening too many nodes

Ideally, the final answer should be synthesized from this ledger.

Another way to say it:

- `state.frontier` tells the system what to inspect next
- `state.evidence_ledger` tells the system what it already knows with support

## 8. How the System Stops

There are two main ways:

1. The controller explicitly chooses `STOP`
2. If the budget is nearly exhausted or the loop stalls, the system falls back to answer synthesis from the evidence ledger

In the current v1, the final answer is synthesized from the evidence that has already been collected.

In the repo, this lives in:

- [controller.py](/share_4/users/duy/project/rlm/rlm/video/controller.py)

## 9. Why VideoRLM Is Different from One-Shot Prompting

One-shot prompting usually looks like:

1. summarize the whole video
2. put that summary into a model
3. ask for an answer

The problems with one-shot:

- important details are easy to lose
- very long context can become noisy
- hallucinations become more likely
- evidence is harder to trace

VideoRLM is different because:

- it searches step by step
- it opens specific nodes
- it accumulates evidence
- it produces a trace that can be debugged

That is why it is a better fit for long videos.

## 10. What v1 Already Does Well, and What It Still Misses

### Working reasonably well

- building structured memory
- node-level retrieval
- a clear state-action loop
- traces and logs for debugging
- a LongShOT benchmark runner

### Still weak

- `OPEN(..., speech)` can still extract the wrong snippet inside a long span
- the controller can still loop too long before stopping
- visual and cross-modal grounding are still limited
- full recursion is not yet implemented in the stronger long-term form

## 11. The Simplest Possible Summary

If you had to explain VideoRLM in three lines:

1. The video is converted into structured memory.
2. The model does not answer immediately; it decides what to inspect next.
3. The final answer is synthesized from evidence collected during that search process.

## 12. If You Are an End User

You can think of VideoRLM as a video-watching assistant:

- it does not have perfect memory
- it does not read everything at once
- it keeps asking itself:
  - which segment matters?
  - should I read transcript or visuals?
  - do I already have enough evidence?

If it gets those three questions right, it can answer long-video questions better than a single huge prompt.

## 13. If You Are a Developer

Here is a quick concept-to-code map:

- memory:
  - [memory.py](/share_4/users/duy/project/rlm/rlm/video/memory.py)
- state and types:
  - [types.py](/share_4/users/duy/project/rlm/rlm/video/types.py)
- retrieval:
  - [index.py](/share_4/users/duy/project/rlm/rlm/video/index.py)
- tools:
  - [tools.py](/share_4/users/duy/project/rlm/rlm/video/tools.py)
- controller loop:
  - [controller.py](/share_4/users/duy/project/rlm/rlm/video/controller.py)
- benchmark:
  - [longshot.py](/share_4/users/duy/project/rlm/rlm/video/longshot.py)
- CLI:
  - [cli.py](/share_4/users/duy/project/rlm/rlm/video/cli.py)

## 14. Closing Line

`VideoRLM does not try to cram the whole video into the model. It builds an external memory, then lets the model decide what to inspect next.`
