# VideoRLM: Technical Details

Tài liệu này giải thích phần kỹ thuật của VideoRLM theo đúng kiến trúc hiện đang có trong repo. Nếu tài liệu [video_rlm_how_it_works.md](/share_4/users/duy/project/rlm/docs/video_rlm_how_it_works.md) là bản giải thích trực quan, thì file này là bản "mở nắp máy" để hiểu từng component hoạt động ra sao.

Nếu bạn muốn đọc code theo đúng call graph của benchmark runtime, xem thêm:
[video_rlm_code_reading_guide.md](/share_4/users/duy/project/rlm/docs/video_rlm_code_reading_guide.md)

## 1. Mục Tiêu Kỹ Thuật Của VideoRLM

VideoRLM v1 không cố train một video model mới. Nó ghép các thành phần sau thành một hệ suy luận có cấu trúc:

- một `external memory` cho video dài
- một `controller` quyết định nên làm gì tiếp theo
- một `tool layer` để đọc bằng chứng theo modality
- một `trace/logging format` để sau này train policy

Ý tưởng cốt lõi là:

`model không cần nhìn toàn bộ video cùng lúc; model chỉ cần nhìn state hiện tại và chọn action kế tiếp.`

## 2. Hai Loại Dữ Liệu Khác Nhau

Trong implementation hiện tại có hai lớp dữ liệu khác nhau, và việc tách chúng ra là rất quan trọng.

### 2.1. Prepared Artifacts

Đây là dữ liệu thu được sau bước tiền xử lý video:

- `speech_spans`
- `visual_summaries`
- `ocr_spans`
- `audio_events`

Chúng được gom trong [`PreparedVideoArtifacts`](/share_4/users/duy/project/rlm/rlm/video/memory.py).

Lớp này là dữ liệu "gần raw", chưa phải working state của controller.

### 2.2. Runtime State

Đây là dữ liệu mà controller thật sự nhìn thấy trong lúc suy luận:

- câu hỏi gì
- đang nghi node nào
- đã có evidence nào
- đã dùng bao nhiêu bước

Lớp này nằm trong [`ControllerState`](/share_4/users/duy/project/rlm/rlm/video/types.py).

Nói ngắn gọn:

- `artifacts` là dữ liệu đầu vào của memory builder
- `state` là dữ liệu đầu vào của controller

## 3. Memory Layer

### 3.1. TimeSpan

Kiểu cơ bản nhất là [`TimeSpan`](/share_4/users/duy/project/rlm/rlm/video/types.py), chứa:

- `start`
- `end`

Nó có các utility quan trọng:

- `duration`
- `overlaps(other)`
- `contains(value)`
- `to_display()`

VideoRLM dùng `TimeSpan` ở gần như mọi nơi: speech spans, OCR spans, node spans, evidence spans, frontier items.

### 3.2. Atomic Spans

Các đơn vị dữ liệu gắn thời gian nhỏ nhất gồm:

- [`SpeechSpan`](/share_4/users/duy/project/rlm/rlm/video/types.py)
- [`VisualSummarySpan`](/share_4/users/duy/project/rlm/rlm/video/types.py)
- [`OCRSpan`](/share_4/users/duy/project/rlm/rlm/video/types.py)
- [`AudioEvent`](/share_4/users/duy/project/rlm/rlm/video/types.py)

Mỗi span đều gắn với một `TimeSpan`. Đây là nền tảng để hệ có thể truy vết ngược từ answer về evidence thật.

### 3.3. VideoNode

Đơn vị chính trong memory là [`VideoNode`](/share_4/users/duy/project/rlm/rlm/video/types.py).

Một node chứa:

- `node_id`
- `level`
- `time_span`
- `visual_summary`
- `speech_spans`
- `ocr_spans`
- `audio_events`
- `tags`
- `entities`
- `clip_path`
- `keyframe_paths`
- `children`
- `parent_id`
- `metadata`
- `uncertainty`

Điểm quan trọng:

- node không phải chỉ là caption
- node giữ luôn các con trỏ về bằng chứng theo nhiều modality
- node biết cha và con của nó, nên `SPLIT` làm việc rất tự nhiên

### 3.4. VideoMemory

Toàn bộ video sau khi build sẽ thành [`VideoMemory`](/share_4/users/duy/project/rlm/rlm/video/types.py).

Nó chứa:

- `video_id`
- `root_id`
- `nodes`
- `metadata`

`VideoMemory` là database toàn cục của một video. Khi controller chạy, nó không sửa memory; nó chỉ truy vấn memory qua `SEARCH`, `OPEN`, `SPLIT`.

## 4. Memory Builder

Memory builder nằm chủ yếu trong [`memory.py`](/share_4/users/duy/project/rlm/rlm/video/memory.py).

### 4.1. Vai Trò

`VideoMemoryBuilder` làm 3 việc:

1. gọi các adapter để lấy artifacts
2. chia timeline thành nhiều mức
3. gom artifacts vào từng node

### 4.2. Cây Thời Gian

Builder hiện chia video thành:

```text
video -> scene -> segment -> clip
```

Mặc định:

- `scene_duration_seconds = 180`
- `segment_duration_seconds = 45`
- `clip_duration_seconds = 15`

Đây là segmentation heuristic, chưa phải scene detection "thật". Mục tiêu v1 là ổn định và dễ debug, không phải segmentation tối ưu.

### 4.3. Cách Node Được Tạo

Với mỗi khoảng thời gian, builder:

- lấy các `visual_summaries` chồng thời gian
- lấy các `speech_spans` overlap
- lấy các `ocr_spans` overlap
- lấy các `audio_events` overlap
- suy ra `tags/entities`
- dựng `clip_path`

Nghĩa là node được tạo bằng cách "project" artifacts vào một interval cố định.

## 5. Index và Retrieval

Lớp retrieval nằm trong [`VideoMemoryIndex`](/share_4/users/duy/project/rlm/rlm/video/index.py).

### 5.1. SearchHit

Kết quả retrieval trước khi đi vào state là [`SearchHit`](/share_4/users/duy/project/rlm/rlm/video/index.py).

Một hit gồm:

- `node_id`
- `time_span`
- `level`
- `score`
- `reason`
- `modality`
- `matched_terms`
- `score_breakdown`

Sau đó `SearchHit` mới được chuyển thành `FrontierItem`.

### 5.2. Cách Chấm Điểm

Hiện tại `SEARCH` dùng ba thành phần điểm:

- `lexical_score`
- `semantic_score`
- `temporal_score`

Trong v1 hiện tại:

- lexical là thành phần chính
- semantic chỉ có khi gắn `embedding_provider`
- temporal dùng để boost các câu hỏi kiểu `first`, `last`

Điều này có nghĩa:

- nếu query và transcript match tốt, node sẽ lên cao
- nếu query mơ hồ hoặc cần paraphrase mạnh, v1 còn yếu

### 5.3. Frontier Không Phải Memory

Đây là chỗ rất hay bị nhầm:

- `memory` = toàn bộ node của video
- `frontier` = working set nhỏ đang đáng điều tra

Controller chỉ nhìn `frontier`, không duyệt toàn bộ `memory` mỗi bước.

## 6. Runtime State

Schema runtime chính nằm ở [`ControllerState`](/share_4/users/duy/project/rlm/rlm/video/types.py).

### 6.1. FrontierItem

[`FrontierItem`](/share_4/users/duy/project/rlm/rlm/video/types.py) là một node đang được nghi ngờ.

Nó chứa:

- `node_id`
- `time_span`
- `level`
- `score`
- `why_candidate`
- `recommended_modalities`
- `status`

`status` hiện dùng các giá trị:

- `unopened`
- `opened`
- `expanded`
- `exhausted`

### 6.2. Evidence

[`Evidence`](/share_4/users/duy/project/rlm/rlm/video/types.py) là đối tượng quan trọng nhất nếu nhìn từ góc độ grounded reasoning.

Nó chứa:

- `evidence_id`
- `claim`
- `modality`
- `time_span`
- `source_node_id`
- `confidence`
- `detail`
- `used_in_final_answer`
- `metadata`

Điểm rất quan trọng:

- `claim` là phiên bản ngắn gọn để answerer dùng nhanh
- `detail` là excerpt hoặc nội dung cụ thể để synthesize answer
- `time_span` giúp truy vết về video

### 6.3. BudgetState

[`BudgetState`](/share_4/users/duy/project/rlm/rlm/video/types.py) giữ phần control:

- `steps_used`
- `steps_remaining`
- `tool_calls_used`
- `max_depth`
- `current_depth`
- `clips_opened`
- `tokens_spent`

V1 chưa dùng recursion thật, nên `max_depth/current_depth` mới là scaffold cho tương lai.

### 6.4. ControllerState

`ControllerState` là ảnh chụp của toàn bộ quá trình suy luận tại một thời điểm.

Nó chứa:

- `question`
- `task_type`
- `dialogue_context`
- `subquestion`
- `frontier`
- `evidence_ledger`
- `action_history`
- `budget`
- `global_context`

Đây chính là object được prompt hóa để controller chọn action tiếp theo.

## 7. Action Protocol

Schema action nằm trong [`ControllerAction`](/share_4/users/duy/project/rlm/rlm/video/types.py).

### 7.1. Action Types

V1 có 5 action:

- `SEARCH`
- `OPEN`
- `SPLIT`
- `MERGE`
- `STOP`

### 7.2. Validation

`ControllerAction.__post_init__()` ép một số invariant:

- `SEARCH` phải có `query` và `modality`
- `OPEN` phải có `node_id` và `modality`
- `SPLIT` phải có `node_id`
- `MERGE` phải có `evidence_ids`

Điều này giúp runtime fail fast nếu model sinh JSON sai schema.

### 7.3. Observation

Mỗi action sau khi thực thi sẽ trả về một [`Observation`](/share_4/users/duy/project/rlm/rlm/video/types.py).

Observation có:

- `kind`
- `summary`
- `frontier`
- `evidence`
- `node_id`
- `metadata`

Luồng của hệ thực ra là:

```text
state_t -> action_t -> observation_t -> state_t+1
```

Đây cũng chính là format tốt cho dữ liệu SFT sau này.

## 8. Tool Layer

Tool executor nằm ở [`VideoToolExecutor`](/share_4/users/duy/project/rlm/rlm/video/tools.py).

### 8.1. SEARCH

`SEARCH` chỉ gọi index, rồi chuyển `SearchHit` thành `FrontierItem`.

Nó chưa đọc bằng chứng sâu, nên có thể hiểu là bước `triage`.

### 8.2. OPEN

`OPEN` là nơi đọc evidence thật.

Với `visual`, `ocr`, `audio`, logic hiện tại khá trực tiếp:

- lấy nội dung liên quan từ node
- tạo một `Evidence`

Với `speech`, logic phức tạp hơn:

1. lấy tất cả `speech_spans` trong node
2. chấm điểm từng span theo question và query gần nhất
3. chọn một số span liên quan nhất
4. cắt excerpt quan trọng từ span
5. bỏ qua evidence trùng
6. tạo `Evidence`

Đây là nơi đã có nhiều cải tiến gần đây, vì benchmark thật cho thấy lỗi chủ yếu nằm ở `speech evidence extraction`.

### 8.3. SPLIT

`SPLIT` không chạy model; nó chỉ mở rộng node hiện tại thành các child nodes đã có sẵn trong memory.

Điều này có nghĩa là:

- `SPLIT` rất rẻ
- nhưng chất lượng `SPLIT` phụ thuộc trực tiếp vào memory tree ban đầu

### 8.4. MERGE

`MERGE` lấy một danh sách `evidence_ids`, ghép claim và detail lại thành một `cross_modal evidence`.

V1 mới dùng logic trung bình confidence khá đơn giản.

### 8.5. STOP

`STOP` đánh dấu evidence nào được dùng cho final answer.

Đây là bước quan trọng nếu sau này muốn học `stop decision` hoặc `evidence sufficiency`.

## 9. Controller Loop

Lõi runtime nằm ở [`VideoRLM.run()`](/share_4/users/duy/project/rlm/rlm/video/controller.py).

### 9.1. Initial State

Khi bắt đầu chạy:

- hệ tạo `VideoMemoryIndex`
- tạo `VideoToolExecutor`
- build initial state bằng `_build_initial_state()`

Initial frontier được lấy bằng search trên chính câu hỏi gốc. Nếu search không trả gì, hệ fallback sang top-level nodes.

### 9.2. Main Loop

Ở mỗi bước:

1. build prompt từ state
2. gọi controller LM
3. parse action JSON
4. thực thi action qua tool layer
5. cập nhật state
6. log trace step

### 9.3. Parse Action

Model đôi khi không trả JSON sạch ngay từ đầu. Vì vậy controller có `_extract_first_json_object()` để tìm object JSON đầu tiên trong raw response.

Đây là một chi tiết nhỏ nhưng rất thực dụng để tăng độ bền của runtime.

### 9.4. State Update

`_apply_observation()` làm các việc:

- tăng `steps_used`
- giảm `steps_remaining`
- tăng `tool_calls_used`
- tăng `clips_opened` nếu action là `OPEN`
- cập nhật `tokens_spent`
- append `action_history`
- cập nhật frontier hoặc evidence ledger tùy action

### 9.5. Early Stop Guard

Hiện tại có một guard hữu ích:

- nếu có 2 lần `OPEN` liên tiếp mà không thu được evidence mới
- và ledger đã có evidence
- hệ sẽ fallback sang answer synthesis

Guard này giúp giảm loop vô ích, nhưng không cứu được nếu evidence đã bị chọn sai từ trước.

## 10. Answer Synthesis

Khi controller không tự `STOP` tốt, hệ vẫn có thể trả lời qua `_fallback_answer_from_state()`.

Logic hiện tại:

- lấy top evidence theo confidence
- rút gọn excerpt bằng `_focus_evidence_detail()`
- gọi cùng controller model như một grounded answerer

Điểm mạnh:

- answer vẫn có cơ hội grounded theo evidence ledger

Điểm yếu:

- nếu ledger nhiễm evidence sai, answer cũng sẽ trôi theo

## 11. Logging và Trace

### 11.1. TraceStep

[`TraceStep`](/share_4/users/duy/project/rlm/rlm/video/types.py) lưu:

- `step_index`
- `state`
- `action`
- `observation`
- `next_state`
- `raw_model_response`

Đây là object quan trọng nhất cho debugging và training.

### 11.2. Logger

[`VideoRLMLogger`](/share_4/users/duy/project/rlm/rlm/video/logger.py) lưu:

- metadata của run
- từng step dưới dạng JSONL

Đây là lý do vì sao hiện tại rất dễ viết các readable notes hoặc phân tích failure theo từng bước.

### 11.3. Training Examples

[`traces.py`](/share_4/users/duy/project/rlm/rlm/video/traces.py) có hàm chuyển trace thành dataset:

- `(state, gold_action, observation, next_state)`

Đây chính là cầu nối từ prompted runtime sang SFT controller.

## 12. Benchmark Integration

LongShOTBench runner nằm ở [`longshot.py`](/share_4/users/duy/project/rlm/rlm/video/longshot.py).

Nó chịu trách nhiệm:

- load sample benchmark
- resolve video path
- build hoặc load memory cache
- replay conversation turns
- ghi prediction, trace, memory ra disk

Nói cách khác:

- `controller.py` là runtime lõi
- `longshot.py` là lớp benchmark orchestration

## 13. Những Điểm Còn Yếu Trong V1

Từ các run benchmark hiện tại, bottleneck lớn nhất chưa nằm ở memory builder mà nằm ở:

- `OPEN(..., speech)` đôi lúc chọn đúng span nhưng cắt sai snippet
- answer synthesis bị kéo theo evidence sai
- retrieval vẫn còn thiên lexical, chưa đủ mạnh cho paraphrase khó

Nói ngắn gọn:

- framework đã chạy đúng luồng
- nhưng chất lượng bằng chứng, nhất là evidence speech, vẫn là nút thắt lớn nhất

## 14. Nên Đọc File Nào Nếu Muốn Sửa Từng Lớp?

- data model:
  - [types.py](/share_4/users/duy/project/rlm/rlm/video/types.py)
- memory builder:
  - [memory.py](/share_4/users/duy/project/rlm/rlm/video/memory.py)
- retrieval:
  - [index.py](/share_4/users/duy/project/rlm/rlm/video/index.py)
- tools:
  - [tools.py](/share_4/users/duy/project/rlm/rlm/video/tools.py)
- controller loop:
  - [controller.py](/share_4/users/duy/project/rlm/rlm/video/controller.py)
- trace export:
  - [traces.py](/share_4/users/duy/project/rlm/rlm/video/traces.py)
- benchmark:
  - [longshot.py](/share_4/users/duy/project/rlm/rlm/video/longshot.py)

## 15. Câu Chốt

Về mặt kỹ thuật, VideoRLM hiện tại là một `stateful controller over external video memory`.

Nó chưa phải một video foundation model mới. Nó là một runtime có cấu trúc, đủ rõ ràng để:

- debug
- benchmark
- log trajectory
- và sau đó train policy tốt hơn từ chính trajectory đó
