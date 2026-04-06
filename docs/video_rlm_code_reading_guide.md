# VideoRLM Code Reading Guide

Tài liệu này dành cho lúc bạn muốn đọc code VideoRLM theo đúng luồng chạy thật, thay vì mở file ngẫu nhiên rồi bị lạc.

Mục tiêu của guide này là trả lời 4 câu:

1. Benchmark run hiện tại bắt đầu từ đâu?
2. Từ benchmark runner, code call sang những component nào?
3. Memory, state, action, tool, evidence đi qua nhau như thế nào?
4. Nếu muốn debug hoặc sửa logic, nên mở file nào trước?

## 1. Đọc theo thứ tự nào là hợp lý nhất?

Nếu chỉ có 15-30 phút, hãy đọc theo đúng thứ tự này:

1. [`output/longshot_single_gpu_full/run_benchmark.py`](/share_4/users/duy/project/rlm/output/longshot_single_gpu_full/run_benchmark.py)
2. [`rlm/video/longshot.py`](/share_4/users/duy/project/rlm/rlm/video/longshot.py)
3. [`rlm/video/qwen.py`](/share_4/users/duy/project/rlm/rlm/video/qwen.py)
4. [`rlm/video/controller.py`](/share_4/users/duy/project/rlm/rlm/video/controller.py)
5. [`rlm/video/tools.py`](/share_4/users/duy/project/rlm/rlm/video/tools.py)
6. [`rlm/video/index.py`](/share_4/users/duy/project/rlm/rlm/video/index.py)
7. [`rlm/video/memory.py`](/share_4/users/duy/project/rlm/rlm/video/memory.py)
8. [`rlm/video/types.py`](/share_4/users/duy/project/rlm/rlm/video/types.py)

Nếu đọc theo thứ tự này, bạn sẽ thấy:

- benchmark runner gọi vào LongShOT runner như thế nào
- LongShOT runner gọi controller ra sao
- controller dùng state/action/tool loop thế nào
- tool lấy evidence từ memory ra sao
- memory được build từ ASR/VL artifact như thế nào

## 2. Điểm bắt đầu của benchmark run hiện tại

Benchmark đang chạy bây giờ bắt đầu từ:

- [`output/longshot_single_gpu_full/run_benchmark.py`](/share_4/users/duy/project/rlm/output/longshot_single_gpu_full/run_benchmark.py)

Đây không phải là code lõi của repo, mà là runner mỏng được tạo để chạy full benchmark trên 1 GPU.

Nó làm 5 việc chính:

1. load toàn bộ samples LongShOT từ file Arrow local
2. lọc các sample có video local sẵn
3. build `QwenLocalVideoStackConfig`
4. tạo `LongShOTBenchmarkRunner`
5. loop qua từng sample và append kết quả vào `predictions.jsonl`

Vì vậy, nếu bạn đang nhìn vào:

- [`output/longshot_single_gpu_full/run.log`](/share_4/users/duy/project/rlm/output/longshot_single_gpu_full/run.log)

thì file code đầu tiên nên đối chiếu là chính runner này.

## 3. Call chain tổng quát

Đây là call chain ngắn gọn của benchmark run:

```text
run_benchmark.py
  -> LongShOTBenchmarkRunner.run_sample(...)
    -> video_resolver.resolve(video_id)
    -> _load_or_build_memory(...)
      -> VideoMemoryBuilder.prepare_artifacts(...)
      -> VideoMemoryBuilder.build_from_artifacts(...)
    -> VideoRLM.run(question, memory, dialogue_context, task_type)
      -> VideoMemoryIndex(...)
      -> VideoToolExecutor(...)
      -> _build_initial_state(...)
      -> while budget:
           build_controller_prompt(...)
           controller_client.completion(...)
           _parse_action(...)
           tools.execute(...)
           _apply_observation(...)
      -> VideoRLMResult
    -> write candidate_response into LongShOT sample format
    -> append one line to predictions.jsonl
```

Đọc được chain này là bạn đã nắm khoảng 80% workflow của hệ.

## 4. Benchmark layer: LongShOT chạy qua framework thế nào?

File quan trọng nhất ở tầng benchmark là:

- [`rlm/video/longshot.py`](/share_4/users/duy/project/rlm/rlm/video/longshot.py)

Bạn nên tập trung vào 3 phần:

### `load_longshot_samples(...)`

Hàm này load dataset từ Hugging Face rồi lọc theo:

- `sample_ids`
- `video_ids`
- `task_filters`
- `sample_limit`

Nó chỉ là lớp load dữ liệu benchmark.

### `LongShOTVideoResolver`

Class này chịu trách nhiệm:

- tìm video local theo `video_id`
- nếu cho phép thì gọi `yt-dlp` để tải thiếu

Khi benchmark fail vì thiếu video, thường lỗi sẽ nằm ở đây chứ chưa vào VideoRLM.

### `LongShOTBenchmarkRunner.run_sample(...)`

Đây là hàm quan trọng nhất nếu bạn muốn hiểu “LongShOT sample được bơm vào VideoRLM thế nào”.

Nó làm:

1. resolve ra `video_path`
2. load hoặc build `VideoMemory`
3. duyệt từng turn trong `conversations`
4. chỉ gọi `video_rlm.run(...)` ở các turn `assistant`
5. ghi `candidate_response` vào đúng format benchmark

Điểm đáng chú ý:

- `history_mode="gold"` nghĩa là ở multi-turn, context dùng assistant answer gốc của benchmark
- nếu đổi sang `candidate`, turn sau sẽ nhìn answer do chính model vừa sinh

## 5. Runtime bundle: các model được ráp vào framework thế nào?

File quan trọng:

- [`rlm/video/qwen.py`](/share_4/users/duy/project/rlm/rlm/video/qwen.py)

Đây là nơi nối:

- controller model
- visual model
- ASR model
- forced aligner
- memory builder
- VideoRLM controller loop

### Hai kiểu stack

Trong file này có hai hướng:

- `QwenVideoStackConfig`
  - dùng OpenAI-compatible endpoint
- `QwenLocalVideoStackConfig`
  - dùng local Hugging Face / Transformers

Benchmark hiện tại đang dùng:

- `QwenLocalVideoStackConfig.default(...)`

### Single-GPU run hiện tại map thiết bị ra sao?

Runner hiện tại set:

- `controller_device="cuda:0"`
- `visual_device="cuda:0"`
- `speech_device="cuda:0"`

nhưng process được launch với:

- `CUDA_VISIBLE_DEVICES=1`

Nghĩa là:

- bên trong process chỉ nhìn thấy đúng 1 GPU vật lý
- GPU vật lý đó được rename thành `cuda:0`
- nên cả controller, visual và speech đều cùng sống trên 1 GPU

## 6. Lazy-load: model load khi nào?

Đây là một điểm rất quan trọng để hiểu behavior runtime.

Model **không nhất thiết load hết ngay khi process start**.

Chúng được lazy-load theo first use.

### Controller

File:

- [`rlm/clients/transformers_local.py`](/share_4/users/duy/project/rlm/rlm/clients/transformers_local.py)

Class:

- `TransformersClient`

Lần đầu gọi:

- `completion(...)`

thì `_ensure_loaded()` mới chạy và load:

- tokenizer
- causal LM

### ASR

File:

- [`rlm/video/local_adapters.py`](/share_4/users/duy/project/rlm/rlm/video/local_adapters.py)

Class:

- `LocalQwenASRSpeechRecognizer`

Lần đầu gọi:

- `recognize(...)`

thì `_ensure_loaded()` mới load `Qwen3ASRModel`.

### Visual model

Cũng ở:

- [`rlm/video/local_adapters.py`](/share_4/users/duy/project/rlm/rlm/video/local_adapters.py)

Class:

- `LocalQwenVisualSummarizer`

Lần đầu gọi:

- `summarize(...)`

thì `_ensure_loaded()` mới load:

- `Qwen3VLForConditionalGeneration`
- `AutoProcessor`

Kết luận:

- object được instantiate trước
- model weight chỉ load khi cần
- sau khi load xong thì giữ lại để tái sử dụng cho sample sau

## 7. Memory layer: video được biến thành thứ gì?

File chính:

- [`rlm/video/memory.py`](/share_4/users/duy/project/rlm/rlm/video/memory.py)

### `PreparedVideoArtifacts`

Đây là output gần raw của preprocessing:

- `speech_spans`
- `visual_summaries`
- `ocr_spans`
- `audio_events`

Nói ngắn gọn:

- artifacts là sidecar data sau khi ASR/VL/OCR/audio extractor chạy xong

### `VideoMemoryBuilder.prepare_artifacts(...)`

Hàm này gọi các extractor:

- speech recognizer
- visual summarizer
- OCR extractor
- audio extractor

Nó chưa tạo cây node hoàn chỉnh, mới tạo artifacts.

### `VideoMemoryBuilder.build_from_artifacts(...)`

Đây là chỗ video được dựng thành cây:

```text
video -> scene -> segment -> clip
```

Mỗi node sẽ được gắn:

- `time_span`
- `visual_summary`
- `speech_spans`
- `ocr_spans`
- `audio_events`
- `tags`
- `entities`
- `clip_path`
- `children`

Khi bạn thấy framework gọi `OPEN(node_id, speech)`, thì dữ liệu nó mở chính là những gì đã được gắn vào `VideoNode` ở bước này.

## 8. Data model lõi: state/action/evidence nằm ở đâu?

File quan trọng nhất cho phần schema là:

- [`rlm/video/types.py`](/share_4/users/duy/project/rlm/rlm/video/types.py)

Nếu muốn hiểu abstraction của VideoRLM, hãy đọc file này rất kỹ.

Các kiểu dữ liệu quan trọng nhất là:

### Video representation

- `TimeSpan`
- `SpeechSpan`
- `VisualSummarySpan`
- `OCRSpan`
- `AudioEvent`
- `VideoNode`
- `VideoMemory`

### Controller working state

- `FrontierItem`
- `Evidence`
- `BudgetState`
- `ControllerState`

### Action / trace / result

- `ControllerAction`
- `Observation`
- `TraceStep`
- `VideoRLMResult`

Một cách hiểu nhanh:

- `VideoMemory` là database của video
- `ControllerState` là ảnh chụp reasoning ở một bước
- `ControllerAction` là quyết định của controller
- `Observation` là output sau khi tool chạy
- `TraceStep` là log đầy đủ của một bước chuyển state

## 9. Search layer: framework chọn node ứng viên thế nào?

File:

- [`rlm/video/index.py`](/share_4/users/duy/project/rlm/rlm/video/index.py)

Class chính:

- `VideoMemoryIndex`

Đây là nơi `SEARCH(...)` hoạt động.

### `search(...)`

Hàm này duyệt các node trong memory và chấm điểm từng node.

Nó có thể dùng:

- lexical score
- semantic score nếu có embedding provider
- temporal score cho query kiểu `first`, `last`

### `_score_node(...)`

Mỗi node được chấm trên từng modality:

- `speech`
- `visual`
- `ocr`
- `audio`

Node nào đạt điểm cao nhất thì trở thành `SearchHit`.

### Ý nghĩa thực tế

`SEARCH` không mở clip ngay.

Nó chỉ làm:

- xếp hạng node
- biến node phù hợp thành `FrontierItem`

Nên nếu retrieval sai, thường bạn nên debug ở đây trước khi đụng controller.

## 10. Controller loop: trái tim của framework

File:

- [`rlm/video/controller.py`](/share_4/users/duy/project/rlm/rlm/video/controller.py)

Class:

- `VideoRLM`

Đây là nơi workflow `state -> action -> observation -> next_state` chạy thật.

### `run(...)`

Đây là entry point chính của runtime reasoning.

Nó làm:

1. tạo `VideoMemoryIndex`
2. tạo `VideoToolExecutor`
3. build initial state
4. lặp cho đến khi hết budget hoặc gặp `STOP`

Trong mỗi vòng:

1. build prompt từ state
2. gọi controller LLM
3. parse JSON action
4. thực thi action qua tools
5. apply observation để update state
6. log thành `TraceStep`

### `_build_initial_state(...)`

Hàm này dựng state đầu tiên từ:

- `question`
- `dialogue_context`
- `task_type`
- `topical_index`
- initial frontier từ `index.search(question)`

### `_apply_observation(...)`

Đây là chỗ state mutation thật sự xảy ra.

Ví dụ:

- `SEARCH` cập nhật frontier
- `OPEN` append evidence vào ledger
- `SPLIT` thêm child nodes vào frontier
- `MERGE` thêm merged evidence
- `STOP` đánh dấu evidence nào đã dùng trong final answer

Nếu muốn debug vì sao state đi sai, đây là hàm cần đọc ngay.

## 11. Tool layer: action được thực thi ra sao?

File:

- [`rlm/video/tools.py`](/share_4/users/duy/project/rlm/rlm/video/tools.py)

Class:

- `VideoToolExecutor`

Nó map action sang execution thật:

- `SEARCH`
- `OPEN`
- `SPLIT`
- `MERGE`
- `STOP`

### `execute(...)`

Đây là dispatcher:

- action type nào thì gọi handler tương ứng

### `search(...)`

Gọi thẳng sang `VideoMemoryIndex.search(...)`, rồi đổi `SearchHit` thành `FrontierItem`.

### `open(...)`

Đây là tool quan trọng nhất.

Với:

- `speech`
  - chọn speech spans liên quan
  - cắt snippet
  - tạo `Evidence`
- `visual`
  - lấy `visual_summary`
- `ocr`
  - lấy text OCR
- `audio`
  - lấy labels audio

Với speech, logic thực sự quan trọng nằm ở:

- `_build_speech_evidence(...)`
- `_select_relevant_speech_spans(...)`
- `_focus_speech_detail(...)`

Đây cũng là tầng mà mình đã phải sửa nhiều nhất khi debug LongShOT.

### `split(...)`

Không chạy model.

Nó chỉ:

- lấy children của node hiện tại
- biến chúng thành frontier items mới

### `merge(...)`

Gộp nhiều evidence đã có thành một evidence mới kiểu `cross_modal`.

### `stop(...)`

Không tự suy luận thêm gì nhiều.

Nó chỉ đóng gói:

- answer
- evidence ids được chọn

Việc answer cuối cùng có grounded hay không vẫn phụ thuộc nhiều vào evidence ledger trước đó.

## 12. Prompt layer: controller nhìn state dưới dạng gì?

File:

- [`rlm/video/prompts.py`](/share_4/users/duy/project/rlm/rlm/video/prompts.py)

Controller không nhìn thẳng object Python.

Nó nhìn một prompt được build từ:

- question
- dialogue context
- frontier
- evidence ledger
- budget
- global context

Nên nếu controller chọn action kỳ lạ, phải đọc prompt builder để biết model thật sự đã được thấy gì.

## 13. Logger và trace: muốn debug thì mở gì?

File:

- [`rlm/video/logger.py`](/share_4/users/duy/project/rlm/rlm/video/logger.py)

Và output thực tế ở benchmark run hiện tại:

- [run.log](/share_4/users/duy/project/rlm/output/longshot_single_gpu_full/run.log)
- [progress.json](/share_4/users/duy/project/rlm/output/longshot_single_gpu_full/progress.json)
- [predictions.jsonl](/share_4/users/duy/project/rlm/output/longshot_single_gpu_full/predictions.jsonl)
- [videorlm_2026-04-06_12-42-45_3fdc29b6.jsonl](/share_4/users/duy/project/rlm/output/longshot_single_gpu_full/logs/videorlm_2026-04-06_12-42-45_3fdc29b6.jsonl)

Muốn debug từng mức thì đọc như sau:

### Muốn biết benchmark chạy tới đâu

Mở:

- [run.log](/share_4/users/duy/project/rlm/output/longshot_single_gpu_full/run.log)
- [progress.json](/share_4/users/duy/project/rlm/output/longshot_single_gpu_full/progress.json)

### Muốn biết sample nào đã có answer gì

Mở:

- [predictions.jsonl](/share_4/users/duy/project/rlm/output/longshot_single_gpu_full/predictions.jsonl)

### Muốn biết controller đã nghĩ và hành động thế nào

Mở:

- file trace tương ứng trong [traces](/share_4/users/duy/project/rlm/output/longshot_single_gpu_full/traces)
- hoặc file logger step-by-step trong [logs](/share_4/users/duy/project/rlm/output/longshot_single_gpu_full/logs)

## 14. Nếu muốn sửa bug thì nên vào đâu trước?

Đây là cách định vị bug nhanh nhất:

### Bug kiểu không tìm đúng đoạn video

Đọc:

- [`rlm/video/index.py`](/share_4/users/duy/project/rlm/rlm/video/index.py)

### Bug kiểu tìm đúng node nhưng answer vẫn sai

Đọc:

- [`rlm/video/tools.py`](/share_4/users/duy/project/rlm/rlm/video/tools.py)

Đặc biệt là speech evidence extraction.

### Bug kiểu controller loop vô ích, stop quá sớm hoặc quá muộn

Đọc:

- [`rlm/video/controller.py`](/share_4/users/duy/project/rlm/rlm/video/controller.py)
- [`rlm/video/prompts.py`](/share_4/users/duy/project/rlm/rlm/video/prompts.py)

### Bug kiểu memory build sai hoặc scene/segment/clip không hợp lý

Đọc:

- [`rlm/video/memory.py`](/share_4/users/duy/project/rlm/rlm/video/memory.py)
- [`rlm/video/local_adapters.py`](/share_4/users/duy/project/rlm/rlm/video/local_adapters.py)

## 15. Một cách đọc thực tế để hiểu rất nhanh

Nếu mình phải onboarding một đồng nghiệp mới, mình sẽ bảo họ làm đúng thế này:

1. Mở [run.log](/share_4/users/duy/project/rlm/output/longshot_single_gpu_full/run.log) để thấy runtime đang chạy sample nào.
2. Mở [run_benchmark.py](/share_4/users/duy/project/rlm/output/longshot_single_gpu_full/run_benchmark.py) để biết benchmark loop bắt đầu từ đâu.
3. Mở [`rlm/video/longshot.py`](/share_4/users/duy/project/rlm/rlm/video/longshot.py) để hiểu sample được bơm vào framework thế nào.
4. Mở [`rlm/video/controller.py`](/share_4/users/duy/project/rlm/rlm/video/controller.py) để hiểu state-action loop.
5. Mở trace của một sample bất kỳ trong [traces](/share_4/users/duy/project/rlm/output/longshot_single_gpu_full/traces), rồi đối chiếu action đó với [`rlm/video/tools.py`](/share_4/users/duy/project/rlm/rlm/video/tools.py).
6. Nếu action là `SEARCH`, đối chiếu thêm với [`rlm/video/index.py`](/share_4/users/duy/project/rlm/rlm/video/index.py).
7. Nếu action là `OPEN`, đối chiếu thêm với [`rlm/video/memory.py`](/share_4/users/duy/project/rlm/rlm/video/memory.py) và [`rlm/video/types.py`](/share_4/users/duy/project/rlm/rlm/video/types.py).

Đọc theo đúng vòng này, bạn sẽ hiểu framework nhanh hơn nhiều so với việc đọc schema trước rồi mới đọc runtime.

## 16. Câu chốt để nhớ

Muốn hiểu VideoRLM đang chạy như thế nào, đừng bắt đầu từ model.

Hãy bắt đầu từ:

```text
benchmark runner
  -> sample runner
  -> memory
  -> controller loop
  -> tools
  -> evidence
  -> prediction
```

Model chỉ là một phần trong loop đó.

Phần “framework” thật sự nằm ở cách state, action, observation và evidence nối với nhau.
