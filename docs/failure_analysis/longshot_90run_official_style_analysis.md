# Phân Tích Official-Style Eval Cho 90 Sample LongShOT

## Mục tiêu của note này

Note này trả lời câu hỏi:

**Vì sao sau khi chấm theo kiểu gần với LongShOT hơn, score của run 90 sample chỉ khoảng `14.97%`?**

Nó không thay thế note cũ [longshot_90run_analysis.md](/share_4/users/duy/project/rlm/docs/failure_analysis/longshot_90run_analysis.md), mà đứng sau note đó:

- note cũ tập trung vào trace, failure taxonomy, và runtime behavior
- note này tập trung vào **rubric score chính thức kiểu LongShOT** và map ngược về các failure mode đã thấy

## Dữ liệu và output được dùng

Run được chấm:

- predictions: [predictions.jsonl](/share_4/users/duy/project/rlm/output/longshot_single_gpu_full/predictions.jsonl)

Output official-style eval:

- evaluated file: [eval.jsonl](/share_4/users/duy/project/rlm/output/longshot_single_gpu_full_official_eval/eval.jsonl)
- score report: [score.txt](/share_4/users/duy/project/rlm/output/longshot_single_gpu_full_official_eval/score.txt)
- machine summary: [summary.json](/share_4/users/duy/project/rlm/output/longshot_single_gpu_full_official_eval/summary.json)

Coverage:

- `90` samples
- `133` assistant turns
- `663` rubric criteria

Judge setup:

- prompt và weighted scoring bám theo public LongShOT eval
- local judge model: `Qwen/Qwen2.5-7B-Instruct`
- device: `cuda:3`

Điểm cần nói rất rõ:

**Đây là official-style eval, chưa phải official stack y hệt public repo của họ.**

Cụ thể:

- prompt chấm từng criterion là cùng kiểu với LongShOT
- cách cộng trọng số task/category/overall là cùng logic với LongShOT
- nhưng judge model không phải server vLLM external của họ, mà là local HF model trong repo này

Nói ngắn gọn:

- so với `token_f1` / `rouge_l_f1` thì đây gần chuẩn benchmark hơn nhiều
- nhưng vẫn nên gọi là `official-style`, không nên gọi quá đà là `official final score`

## Kết quả chính

### Overall

- **overall accuracy: `14.97%`**

### Category averages

- `Core Perception Tasks`: `15.79%`
- `Reasoning Tasks`: `9.79%`
- `Information Tasks`: `12.24%`
- `Multimodal Tasks`: `22.06%`

### Task breakdown

- `audio_visual_alignment`: `29.17%`
- `temporal_reasoning`: `26.77%`
- `summarization`: `24.43%`
- `entity_recognition`: `19.19%`
- `causal_reasoning`: `18.18%`
- `information_retrieval`: `16.55%`
- `multimodal_synthesis`: `14.96%`
- `comparative_analysis`: `11.18%`
- `audio_understanding`: `9.09%`
- `event_understanding`: `8.10%`
- `instruction_extraction`: `8.00%`
- `compositional_reasoning`: `0.00%`
- `sentiment_analysis`: `0.00%`

### Eval robustness

- `0` criteria parse failures
- `0` criteria có `evaluation_error`

Điều này quan trọng vì nó cho thấy:

- score thấp không phải do evaluator vỡ
- score thấp là do chất lượng answer theo rubric thật sự còn yếu

## Vì sao score official-style thấp hơn proxy metric rất nhiều

Trong note cũ, proxy của 90 sample là:

- `mean_token_f1 = 0.3416`
- `mean_rouge_l_f1 = 0.2314`

Nhưng official-style chỉ còn:

- `overall = 14.97%`

Lý do là vì LongShOT không chấm theo “câu trả lời giống reference bao nhiêu”, mà chấm theo:

- có đạt đủ các ý quan trọng không
- có đạt các ý trọng số cao không
- có bỏ sót chi tiết bắt buộc không
- có bịa thêm không

Nói cách khác:

**Hệ hiện tại thường trả lời “nghe hợp lý” hoặc “trúng một phần”, nhưng không đủ để vượt rubric.**

## Dấu hiệu rubric quan trọng nhất

### 1. Hệ chạm được ý tổng quát, nhưng trượt nhiều ở detail và completeness

Tỷ lệ hit theo tên criterion:

- `factual_correctness`: `20.30%`
- `key_details`: `11.28%`
- `completeness`: `10.26%`
- `essential_information`: `15.00%`
- `accuracy`: `12.12%`

Đây là pattern quan trọng nhất của cả run.

Diễn giải:

- hệ đôi khi bắt được ý chính mơ hồ
- nhưng thường bỏ lỡ:
  - nguyên nhân cụ thể
  - chi tiết thời gian / quan hệ / bước thực hiện
  - follow-up evidence
  - các điểm benchmark coi là “must mention”

Đây chính là dạng fail mà proxy text-overlap không phạt đủ mạnh.

### 2. High-priority criteria vẫn hit rất thấp

Tỷ lệ hit theo priority:

- `high_priority`: `16.23%`
- `medium_priority`: `10.81%`

Điều này nói lên rằng hệ không chỉ fail ở phần “bonus detail”.

Nó fail ngay ở những ý benchmark xem là lõi.

### 3. Hallucination không phải vấn đề chính

Penalty criteria bị judge là vi phạm:

- `9 / 132`

Tức khoảng `6.82%`.

Nói thẳng:

- hệ có hallucination, nhưng không phải failure mode thống trị
- vấn đề lớn hơn là:
  - generic answer
  - thiếu chi tiết đúng
  - thiếu completeness

Nói cách khác:

**Hệ hiện tại fail nhiều hơn vì “under-answer” hoặc “partial answer”, chứ không phải vì “confident nonsense” quá nhiều.**

## Mapping ngược về failure modes đã thấy trong 90-run analysis

## 1. Retrieval đúng vùng nhưng evidence không đủ sắc

Đây là failure mode đã thấy rõ ở note cũ, và official-style eval xác nhận nó rất đau.

Triệu chứng ở rubric:

- `factual_correctness` đôi khi đạt
- `key_details` và `completeness` rớt mạnh

Điều đó khớp với runtime hiện tại:

- `SEARCH` nhiều lúc kéo đúng scene / segment
- nhưng `OPEN(..., speech)` hoặc evidence selection chưa cắt được snippet thật đúng tâm
- hoặc controller dừng khi mới có một mảnh bằng chứng khá ổn, nhưng chưa có support evidence kế tiếp

Case tiêu biểu:

- `sample_6095`
- `sample_6392`

Ở các sample kiểu này, model nói được “đại ý đúng”, nhưng vẫn miss các điều kiện benchmark muốn thấy.

### `sample_6095`

- score: `29.63%`
- answer có nhắc “billions of years”
- nhưng miss:
  - tính ngẫu nhiên của quasar light
  - loophole của experiment
  - hidden connection giữa filter choice và particles

Đây là ví dụ rất điển hình của:

- retrieval đúng domain
- answer đúng nửa đầu
- nhưng evidence chain không đủ để đi tới complete benchmark answer

## 2. Generic answer synthesis làm mất chi tiết bắt buộc

Ở nhiều sample, controller / tool layer có một ít evidence liên quan, nhưng answer cuối bị đẩy về dạng:

- generic
- an toàn
- nghe hợp lý
- thiếu chi tiết benchmark hỏi

Case tiêu biểu:

- `sample_6490` (`summarization`)
- `sample_6923` (`instruction_extraction`)

### `sample_6490`

- score: `0.0%`
- answer nói kiểu:
  - strong initial position
  - strategic move
  - superior starting technique
- nhưng benchmark muốn chi tiết rất cụ thể:
  - strong start right off the line
  - sneak around Cameron Beaubier
  - maintain lead after first laps

Ở đây hệ không bịa nhiều.

Nó chỉ không đủ grounded detail để đạt rubric.

### `sample_6923`

- score: `0.0%`
- question là instruction extraction rất procedural
- model trả lời kiểu khái quát về shoebox projector
- nhưng miss toàn bộ step-by-step actions:
  - quay lưng lại phía mặt trời
  - ánh sáng đi qua lỗ
  - hình chiếu hiện ở mặt trong hộp
  - nhìn qua viewing hole

Điều này map rất rõ về failure mode:

- current answer synthesis không ép đủ chặt việc liệt kê hành động / bước / thứ tự

## 3. Event understanding là điểm yếu nặng nhất vì cần nhiều mảnh bằng chứng phối hợp

`event_understanding` là task có nhiều mẫu nhất trong subset này:

- `20` samples
- official-style score chỉ `8.10%`

Đây không phải ngẫu nhiên.

Task này thường cần:

- đúng object / actor
- đúng moment
- đúng relation
- đúng interpretation của interaction

Mà đó lại đúng là nơi VideoRLM v1 đang yếu:

- frontier đúng nhưng chưa đủ fine
- repeat `OPEN` không tăng information gain
- evidence ledger cuối bị collapse
- answer synthesis không giữ được nhiều evidence phối hợp cùng lúc

Case tiêu biểu:

- `sample_6392`: partial, được `50%`
- `sample_5913`, `sample_5914`, `sample_6428`, `sample_6429`: `0%`

Đặc biệt với các sample `0k2ey_okQ4E`, note cũ đã chỉ ra nguy cơ **data/video mismatch**. Official-style eval không bác bỏ điều đó; nó chỉ cho thấy mismatch kiểu này kéo score xuống rất mạnh.

## 4. Multi-turn / follow-up turns cho thấy hệ không ổn định giữa các turn

Một ví dụ đáng chú ý là `sample_6171`.

Ở turn 1:

- system fail gần như toàn bộ criteria về “main challenge”

Ở turn 2:

- lại trả lời khá hơn nhiều về việc dùng quasar light

Điều này cho thấy:

- cùng một sample
- cùng một video
- nhưng controller/evidence pipeline không ổn định giữa các subquestions

Nói cách khác:

**state/history hiện tại chưa đủ giúp hệ carry đúng hypothesis qua các turn.**

## 5. Controller hiện tại vẫn underperform mạnh ở task cần composition thật sự

Hai task tệ nhất về bản chất reasoning là:

- `compositional_reasoning`: `0.00%`
- `comparative_analysis`: `11.18%`

Đây là dấu hiệu rất phù hợp với kiến trúc v1:

- controller có search/open/merge/stop
- nhưng chưa có recursion thật
- chưa có subquestion generation học được
- chưa có explicit plan decomposition tốt

Vì vậy với câu hỏi cần:

- nối nhiều bước
- so nhiều entity / nhiều thời điểm
- tổng hợp nhiều modality

hệ vẫn thường collapse về một answer ngắn, generic, và thiếu structure.

## So sánh “partial correctness” và “benchmark correctness”

Một cách nhìn rất hữu ích là:

- proxy metric đang đo: “model có chạm vào vùng đúng của semantic space không?”
- official-style metric đang đo: “model có thỏa các tiêu chí benchmark cụ thể không?”

Hiện tại VideoRLM v1 ở trạng thái:

- **semantic vicinity**: có
- **rubric completeness**: còn yếu

Nói ngắn gọn:

**Hệ biết mình đang ở gần đáp án, nhưng chưa biết thu đủ bằng chứng và nói ra đáp án theo cách benchmark yêu cầu.**

## Kết luận root cause ở mức hệ thống

Nếu phải gom lại thành vài nguyên nhân gốc, mình sẽ chốt như sau:

### 1. Tool/evidence resolution chưa đủ tốt

- span đúng nhưng snippet sai
- evidence đúng nhưng thiếu continuation evidence
- node đúng nhưng clip child không tăng signal

### 2. Controller stop/search/open policy còn non

- mở lặp
- thiếu novelty guard
- thiếu stop khi đủ
- hoặc dừng quá sớm khi mới có một phần answer

### 3. Answer synthesis còn quá generic

- không ép đủ “must mention”
- không ép structure theo task type
- không giữ được nhiều evidence cụ thể cùng lúc

### 4. Một phần dữ liệu benchmark local có nguy cơ không match source gốc

- đây là multiplier làm score thấp hơn thực lực của planner

## Điều nên sửa trước benchmark tiếp theo

Nếu chỉ sửa vài thứ có khả năng tăng official-style score thật, mình ưu tiên:

1. **Fix data/video binding trước**
- nhất là nhóm `0k2ey_okQ4E`
- nếu input sai thì mọi cải tiến planner đều bị nhiễu

2. **Nâng `OPEN(..., speech)` và evidence continuation**
- mục tiêu là nâng `key_details` và `completeness`
- đây là chỗ có leverage lớn nhất trên rubric

3. **Task-aware answer synthesis**
- `instruction_extraction`: ép step-by-step
- `event_understanding`: ép actor + action + relation + evidence
- `temporal_reasoning`: ép cause/timing wording

4. **Giữ nhiều evidence hơn ở final state**
- đừng collapse ledger quá sớm
- answerer cần thấy nhiều mảnh bằng chứng song song

5. **Thêm novelty/plateau guard cho controller**
- tránh repeat `OPEN`
- buộc chuyển strategy khi information gain thấp

## Câu chốt

Kết quả `14.97%` không nói rằng VideoRLM v1 “không biết gì”.

Nó nói điều cụ thể hơn:

**VideoRLM v1 thường tìm được vùng liên quan và đôi khi chạm được ý đúng, nhưng chưa thu đủ bằng chứng có độ phân giải cao để thỏa rubric của LongShOTBench.**

Đó là một failure mode khó nhưng sửa được, và nó phù hợp với đúng các điểm yếu mà trace-level analysis trước đó đã chỉ ra.
