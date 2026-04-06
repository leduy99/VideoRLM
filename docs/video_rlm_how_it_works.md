# VideoRLM Hoạt Động Như Thế Nào

Nếu bạn muốn đi sâu vào data model, state machine và runtime loop, xem thêm:
[video_rlm_technical_details.md](/share_4/users/duy/project/rlm/docs/video_rlm_technical_details.md)

Nếu bạn muốn đọc code theo đúng workflow benchmark đang chạy, xem thêm:
[video_rlm_code_reading_guide.md](/share_4/users/duy/project/rlm/docs/video_rlm_code_reading_guide.md)

## Mục Tiêu Của VideoRLM

VideoRLM được thiết kế để trả lời câu hỏi trên video dài mà không cần "đọc hết video một lần".

Thay vì nhồi toàn bộ transcript, frame, OCR và audio vào cùng một prompt, VideoRLM hoạt động giống một agent có kế hoạch:

1. Biến video thành bộ nhớ có cấu trúc.
2. Tìm đoạn nào đáng nghi nhất.
3. Mở đúng đoạn đó bằng đúng modality phù hợp.
4. Gom bằng chứng.
5. Dừng lại khi đã đủ căn cứ để trả lời.

Ý tưởng cốt lõi là:

`VideoRLM không cố "nhớ toàn bộ video". Nó học cách quyết định nên xem gì tiếp theo.`

## Nhìn Từ Góc Độ Người Dùng

Nếu người dùng đưa vào:

- một video dài
- một câu hỏi

thì VideoRLM sẽ cố gắng trả lời theo quy trình:

1. tìm đúng khu vực trong video
2. đọc đúng loại thông tin cần thiết
3. trích xuất bằng chứng
4. tổng hợp thành câu trả lời grounded

Mục tiêu cuối cùng không chỉ là "trả lời đúng", mà còn là:

- trả lời đúng
- có thể truy vết dựa trên đoạn nào của video
- giảm chi phí so với cách đọc cả video

## Biểu Đồ Đơn Giản

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

## 1. Video Được Biến Thành "Bộ Nhớ"

Thay vì coi video là một file `.mp4` thông thường, VideoRLM biến nó thành một external memory.

Bộ nhớ này được tổ chức thành cây thời gian:

```text
video -> scene -> segment -> clip
```

Mỗi node trong cây có thể chứa:

- `time_span`
- `speech_spans`
- `visual_summary`
- `ocr_spans`
- `audio_events`
- `clip_path`
- `tags/entities`

Ý nghĩa của bước này:

- video trở thành một database có thể tra cứu
- controller không phải nhìn toàn bộ raw video cùng lúc

Trong repo, phần này nằm chủ yếu ở:

- [memory.py](/share_4/users/duy/project/rlm/rlm/video/memory.py)
- [types.py](/share_4/users/duy/project/rlm/rlm/video/types.py)
- [artifact_store.py](/share_4/users/duy/project/rlm/rlm/video/artifact_store.py)

## 2. Hệ Tạo Index Để Tìm Nhanh

Sau khi có memory, hệ thống tạo một lớp retrieval để trả lời câu hỏi:

`Nếu cần tìm thông tin liên quan, nên bắt đầu từ node nào?`

Trong v1 hiện tại, `SEARCH` chủ yếu dựa trên:

- lexical overlap giữa query và transcript
- lexical overlap giữa query và visual summary hoặc OCR
- một ít temporal prior cho câu hỏi kiểu `first` hoặc `last`

Điều này tạo ra `frontier`, tức là danh sách các node đang đáng nghi nhất.

Trong repo, phần này nằm ở:

- [index.py](/share_4/users/duy/project/rlm/rlm/video/index.py)

## 3. State Là "Ảnh Chụp" Của Quá Trình Suy Luận

Controller không nhìn trực tiếp toàn bộ video.
Nó nhìn vào `state`.

State là một ảnh chụp có cấu trúc của hệ thống tại một thời điểm.

Nó gồm những phần quan trọng nhất:

- `question`
- `dialogue_context`
- `frontier`
- `evidence_ledger`
- `action_history`
- `budget`
- `global_context`

Có thể hiểu ngắn gọn:

- `frontier` = những đoạn đang nghi
- `evidence_ledger` = những bằng chứng đã nhặt được
- `budget` = còn bao nhiêu bước để đi tiếp

Trong repo, phần này nằm ở:

- [types.py](/share_4/users/duy/project/rlm/rlm/video/types.py)

## 4. Controller Không Trả Lời Ngay, Mà Chọn Action

Thay vì sinh đáp án ngay, controller sẽ trả về một action JSON.

Trong v1, action space nhỏ và dễ debug:

- `SEARCH(query, modality)`
- `OPEN(node_id, modality)`
- `SPLIT(node_id)`
- `MERGE(evidence_ids)`
- `STOP(answer, evidence_ids)`

Ý nghĩa:

- `SEARCH`: tìm node nào liên quan
- `OPEN`: mở sâu node đó bằng modality cần thiết
- `SPLIT`: tách node lớn thành node con
- `MERGE`: gom nhiều evidence thành một cụm logic
- `STOP`: dừng và trả lời

Trong repo, phần này nằm ở:

- [controller.py](/share_4/users/duy/project/rlm/rlm/video/controller.py)
- [prompts.py](/share_4/users/duy/project/rlm/rlm/video/prompts.py)

## 5. `SEARCH` Làm Gì?

`SEARCH` là bước "tìm chỗ nào nên xem tiếp".

Nó nhận:

- câu hỏi hiện tại
- modality mong muốn, ví dụ `speech`

Nó trả về:

- một danh sách `frontier items`
- mỗi item có:
  - `node_id`
  - `time_span`
  - `score`
  - `why_candidate`

Ví dụ:

- một scene có transcript chứa từ `plan`, `change`, `schedule`
- scene đó sẽ được score cao cho câu hỏi về thay đổi kế hoạch

Điểm quan trọng:

- `SEARCH` chưa "đọc sâu" nội dung
- nó chỉ xếp hạng node ứng viên

## 6. `OPEN` Làm Gì?

`OPEN` là bước "mở node ra để đọc bằng chứng thật sự".

Nếu `SEARCH` trả lời:

`hãy xem scene nào`

thì `OPEN` trả lời:

`trong scene đó, nội dung cụ thể nào hỗ trợ câu trả lời?`

Tùy modality, `OPEN` sẽ khác nhau:

- `OPEN(..., speech)` đọc transcript spans trong node
- `OPEN(..., visual)` đọc visual summary hoặc clip
- `OPEN(..., ocr)` đọc text trên màn hình
- `OPEN(..., audio)` đọc audio events

Với `speech`, hệ thống hiện tại cố gắng:

1. lấy các `speech spans` trong node
2. chấm điểm span nào liên quan nhất tới question
3. chọn một vài span tốt nhất
4. biến chúng thành `Evidence`

Mỗi `Evidence` thường gồm:

- `claim`
- `modality`
- `time_span`
- `source_node_id`
- `detail`
- `confidence`

Trong repo, phần này nằm ở:

- [tools.py](/share_4/users/duy/project/rlm/rlm/video/tools.py)

## 7. Evidence Ledger Dùng Để Làm Gì?

Đây là nơi tích lũy bằng chứng.

Nó giúp hệ thống không bị:

- xem lại cùng một thứ liên tục
- trả lời theo cảm tính
- mất dấu vì đã đọc quá nhiều node

Về mặt lý tưởng, answer cuối cùng nên được tổng hợp từ ledger này.

Nói cách khác:

- `state.frontier` cho biết nên điều tra gì tiếp
- `state.evidence_ledger` cho biết mình đã biết chắc điều gì

## 8. Hệ Dừng Lại Như Thế Nào?

Có hai cách:

1. Controller chủ động ra action `STOP`
2. Nếu budget cạn hoặc loop quá nhiều, hệ fallback sang answer synthesis từ evidence ledger

Trong v1, answer cuối cùng được tổng hợp từ bằng chứng đã thu được.

Trong repo, phần này nằm ở:

- [controller.py](/share_4/users/duy/project/rlm/rlm/video/controller.py)

## 9. Vì Sao VideoRLM Khác Cách One-Shot?

One-shot thường là:

1. tóm tắt cả video
2. nhét vào model
3. yêu cầu trả lời

Vấn đề của one-shot:

- dễ mất chi tiết quan trọng
- dễ bị context quá dài
- dễ hallucinate
- khó truy vết bằng chứng

VideoRLM khác ở chỗ:

- nó tìm theo bước
- nó mở đúng node
- nó tích lũy evidence
- nó có trace để debug

Đó là lý do nó hợp với long video hơn.

## 10. V1 Hiện Tại Làm Tốt Gì, Chưa Tốt Gì?

### Đang làm tốt

- build được memory có cấu trúc
- retrieval theo node
- state-action loop rõ ràng
- trace và log để debug
- benchmark runner cho LongShOT

### Chưa tốt

- `OPEN(..., speech)` vẫn có thể cắt nhầm snippet bên trong span dài
- controller vẫn có thể loop quá nhiều trước khi `STOP`
- visual hoặc cross-modal grounding chưa mạnh
- chưa có recursion đầy đủ theo đúng tầm nhìn dài hạn

## 11. Cách Hiểu Đơn Giản Nhất

Nếu phải giải thích VideoRLM bằng ba dòng:

1. Video được biến thành một bộ nhớ có cấu trúc.
2. Model không trả lời ngay, mà quyết định phải xem gì tiếp theo.
3. Câu trả lời cuối được tổng hợp từ bằng chứng đã nhặt được trong quá trình tìm kiếm.

## 12. Nếu Bạn Là Người Dùng Cuối

Bạn có thể nghĩ VideoRLM như một trợ lý xem video:

- nó không có trí nhớ hoàn hảo
- nó không đọc hết mọi thứ cùng lúc
- nó sẽ tự hỏi:
  - cần tìm đoạn nào?
  - cần đọc transcript hay hình ảnh?
  - đã đủ bằng chứng chưa?

Nếu nó làm tốt ba câu hỏi này, thì nó sẽ trả lời video dài tốt hơn một prompt dài thông thường.

## 13. Nếu Bạn Là Developer

Đây là map nhanh từ khái niệm sang code:

- memory:
  - [memory.py](/share_4/users/duy/project/rlm/rlm/video/memory.py)
- state và types:
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

## 14. Câu Chốt

`VideoRLM không cố nhồi cả video vào model. Nó xây một bộ nhớ bên ngoài, rồi để model quyết định nên xem gì tiếp theo.`
