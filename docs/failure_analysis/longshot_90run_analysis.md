# Phân Tích Failure Chi Tiết Cho 90 Sample LongShOTBench

## Mục tiêu của note này

Note này là postmortem chi tiết cho run benchmark ở:

- `output/longshot_single_gpu_full/predictions.jsonl`
- `output/longshot_single_gpu_full/run.log`
- `output/longshot_single_gpu_full/progress.json`
- `output/longshot_single_gpu_full/traces/`

Mục tiêu không phải chỉ để chê run này thấp điểm, mà để trả lời 4 câu:

1. Hệ đang fail nhiều nhất ở tầng nào: dữ liệu, retrieval, snippet selection, controller loop, hay answer synthesis?
2. Những failure mode nào là cục bộ theo từng sample, và failure mode nào là có tính hệ thống?
3. Những quyết định thiết kế nào của v1 đã dẫn tới các lỗi này?
4. Nếu chỉ được sửa vài thứ trước benchmark tiếp theo, nên sửa gì trước?

## Phạm vi và cách đọc kết quả

Phân tích này dựa trên:

- `90` samples đã chạy xong
- `133` assistant turns được chấm
- metric hiện dùng là `proxy metric`, chưa phải official LongShOT rubric:
  - `mean_token_f1`
  - `mean_rouge_l_f1`

Các con số chính:

- `90 / 1700` samples runnable đã chạy xong trước khi job bị dừng
- `133` assistant turns được chấm
- `mean_token_f1 = 0.3416`
- `mean_rouge_l_f1 = 0.2314`

Điểm cần nhớ:

- các số này đủ tốt để so sánh nội bộ giữa các phiên bản hệ
- nhưng chưa phải score chính thức của benchmark LongShOT

## Executive Summary

Kết luận ngắn gọn nhất của mình là:

**Run này không fail vì một nguyên nhân duy nhất. Nó fail do 3 tầng lỗi chồng lên nhau.**

### Tầng 1: Có dấu hiệu lỗi nguồn dữ liệu đầu vào

Ít nhất một video có dấu hiệu rất mạnh là nội dung local không khớp với sample benchmark đang hỏi.

Ví dụ rõ nhất là `video_id = 0k2ey_okQ4E`:

- benchmark question hỏi về `Ray Dalio`, `microfinance`, `climate solutions`
- nhưng memory local của video này lại chứa nội dung về `Cristiano Ronaldo`, `football`, `sports science`

Điều này khiến một số failure không còn là lỗi planner thuần túy nữa, mà là lỗi dataset/video binding.

### Tầng 2: Khi dữ liệu đúng, hệ vẫn có lỗi controller + tool khá rõ

Các lỗi lặp đi lặp lại nhiều nhất là:

- lặp `OPEN` vào cùng node hoặc cùng vùng gần như không có novelty
- không có `STOP` rõ ràng trong rất nhiều turn
- retrieval/query formulation yếu với câu hỏi trừu tượng như `why`, `main challenge`, `how did X connect to Y`
- answer synthesis cố “nối ý” từ evidence còn nông, dẫn tới answer nghe hợp lý nhưng không đúng trọng tâm benchmark

### Tầng 3: Evidence pipeline hiện tại đang làm mất độ phân giải

Một phát hiện quan trọng là ở `final_state`, `evidence_ledger` luôn bị nén về tối đa `1` object trên toàn bộ `133/133` turns.

Điều đó có nghĩa là:

- hệ có thể thu được nhiều evidence trung gian trong lúc chạy
- nhưng tới state cuối cùng, các evidence này bị collapse quá sớm
- answer synthesis và failure analysis đều mất khả năng nhìn rõ “bằng chứng nào đúng, bằng chứng nào sai”

## Kết quả tổng quan

### Điểm trung bình theo task

Task yếu nhất:

- `compositional_reasoning`: `0.2610`
- `causal_reasoning`: `0.3090`
- `comparative_analysis`: `0.3111`
- `event_understanding`: `0.3163`
- `multimodal_synthesis`: `0.3172`

Task khá hơn:

- `entity_recognition`: `0.3873`
- `instruction_extraction`: `0.3864`
- `audio_understanding`: `0.4468`
- `audio_visual_alignment`: `0.4704`

Diễn giải:

- hệ đang mạnh hơn ở các câu hỏi tương đối cụ thể, có keyword và evidence trực tiếp
- hệ yếu hơn rõ ở các câu hỏi cần “nối cầu” giữa nhiều mảnh thông tin, hoặc cần hiểu ý trừu tượng hơn là chỉ tìm span phù hợp

### Dấu hiệu cấu trúc đáng chú ý trong trace

Trên `133` assistant turns:

- `83` turns không có `STOP` action rõ ràng
- `83` turns có hiện tượng lặp `OPEN` trên cùng node
- `10` turns có ít nhất một `SEARCH` trả `0 hit`
- `133` turns có `final_state.evidence_ledger` bị nén về tối đa `1` object

So sánh sơ bộ:

- turn có `STOP`: mean token F1 `0.3834`
- turn không có `STOP`: mean token F1 `0.3164`

- turn có `zero-hit SEARCH`: mean token F1 `0.2278`
- turn không có `zero-hit SEARCH`: mean token F1 `0.3508`

Điều này cho thấy:

- thiếu `STOP` là một tín hiệu xấu thật
- `zero-hit SEARCH` là failure mode rất nặng
- còn `repeat OPEN` tự nó chưa đủ để kết luận mọi turn đó tệ, nhưng nó là dấu hiệu của controller đang không có cơ chế chống loop đủ tốt

## Failure Taxonomy

## 1. Failure do dữ liệu đầu vào có thể không khớp benchmark

Đây là failure nghiêm trọng nhất, vì nếu đúng thì mọi phân tích planner phía sau đều bị nhiễu.

### Bằng chứng

`video_id = 0k2ey_okQ4E`

Các sample benchmark gắn với video này hỏi về:

- Ray Dalio
- climate solutions
- microfinance
- self-interest and purpose

Nhưng memory local của video lại cho thấy:

- `scene_001`: bóng đá ban đêm
- transcript mở đầu: `Christian Ronaldo is one of the most valuable footballers...`
- visual summary: soccer player, sports lab, interview shot

Điều này xuất hiện ở nhiều sample cùng video:

- `sample_5913`
- `sample_5914`
- `sample_5917`
- `sample_5918`
- `sample_5938`

và tất cả đều fail theo cùng một pattern: system nói “evidence insufficient” vì thật sự video local không chứa nội dung benchmark đang hỏi.

### Kết luận

Với nhóm sample này, failure không nên quy hết cho controller.

Khả năng cao là:

- cách mình tái dùng cache video local cho LongShOT chưa có bước validate nội dung
- một phần video binding đang sai hoặc không cùng source với benchmark gốc

### Hệ quả

Nếu không sửa chỗ này trước:

- benchmark score sẽ bị kéo xuống bởi lỗi nguồn dữ liệu
- team có thể tốn thời gian sửa retrieval/controller trong khi input đã sai ngay từ đầu

## 2. Failure do controller loop bị lặp và không biết dừng đúng lúc

Đây là lỗi hệ thống lớn thứ hai.

### Triệu chứng

Nhiều turn có pattern:

- `SEARCH`
- `OPEN`
- `OPEN`
- `OPEN` lại vào node đã mở hoặc node gần như tương đương
- không có `STOP`
- cuối cùng fallback answer với evidence yếu

### Case tiêu biểu: `sample_6579`

- task: `information_retrieval`
- question: phản ứng của chuột khi mèo dừng lại, và điều đó cho thấy cả hai đang cảm nhận gì
- token F1: `0.1417`
- trace: `output/longshot_single_gpu_full/traces/sample_6579_turn_003.json`

Hệ đã làm gì:

- search các node visual có keyword `cat`, `mouse`
- mở `scene_010`
- mở `scene_005_seg_002_clip_003`
- rồi quay lại mở gần như các node cùng vùng thêm lần nữa
- không tạo được bằng chứng thật sự bám vào khoảnh khắc benchmark hỏi

Sai lầm ở đây:

- controller không có novelty guard đủ mạnh
- không có cơ chế phát hiện “mình đang đọc lại cùng loại evidence”
- không có rule chuyển modality hay chuyển strategy khi evidence vẫn generic

### Điều đáng chú ý

`repeat OPEN` xuất hiện nhiều, nhưng không phải mọi turn lặp node đều tệ.

Vấn đề thật là:

- hệ lặp `OPEN` mà không tăng information gain
- và cũng không dừng khi information gain đã plateau

## 3. Failure do query và snippet selection yếu với câu hỏi trừu tượng

Đây là failure mode rất rõ ở các câu hỏi kiểu:

- `What was the main challenge...?`
- `Why did she start this series?`
- `How did childhood X connect to current Y?`

### Case tiêu biểu: `sample_6171` turn 1

- task: `entity_recognition`
- token F1: `0.1920`
- trace: `output/longshot_single_gpu_full/traces/sample_6171_turn_001.json`

Question:

`What was the main challenge they were trying to solve with their experiment?`

Gold answer cần:

- loophole trong thí nghiệm entanglement
- nỗi lo hidden influence / hidden variable
- filter choices phải thật sự random

Prediction hiện tại:

- “a long-standing dispute and lack of consensus that had persisted for a century”

Phân tích:

- retrieval đưa hệ vào đúng video, đúng chủ đề khoa học
- nhưng snippet được chọn lại là câu mang tính bối cảnh lịch sử chung
- câu này topical đúng, nhưng không phải câu trả lời benchmark cần

Đây là failure điển hình của:

- lexical match đúng chủ đề
- nhưng semantic match sai intent của câu hỏi

### Case tiêu biểu: `sample_6098` turn 1

- task: `multimodal_synthesis`
- token F1: `0.2020`
- question: `What was the reason she started this week-long workout series?`

Gold cần:

- lâu rồi chưa làm workout split video
- viewer requested nhiều
- đang trong bulk phase nên muốn chia sẻ routine mới

Prediction hiện tại chỉ lấy một detail cục bộ:

- “this is the first week I’m implementing this day into...”

Sai ở đâu:

- hệ lấy một câu cục bộ đúng transcript
- nhưng bỏ lỡ câu “lý do tổng quát” mà benchmark hỏi

### Kết luận

Hệ hiện tại xử lý khá ổn các câu concrete:

- ai
- cái gì
- ở đâu
- chi tiết nào xuất hiện

Nhưng yếu hơn nhiều ở câu hỏi đòi:

- causal abstraction
- bridge reasoning
- meta-intent của speaker

## 4. Failure do answer synthesis bắc cầu quá tay từ evidence nông

Một số turn không fail vì retrieval sai hoàn toàn, mà fail vì answerer cố kết nối một câu chuyện lớn từ vài evidence nông.

### Case tiêu biểu: `sample_6017` turn 3

- task: `compositional_reasoning`
- token F1: `0.1954`
- trace: `output/longshot_single_gpu_full/traces/sample_6017_turn_003.json`

Question:

`How did the story about her childhood piercings connect to her current love for wearing so many earrings?`

Gold cần:

- tinh thần tự quyết từ nhỏ
- thích tự định hình ngoại hình
- hiện tại vẫn tiếp tục tinh thần đó qua việc đeo nhiều earrings

Prediction hiện tại lại bắc cầu qua:

- `date piercing from Maria`
- `Adele studs`
- các món trang sức cụ thể

Tức là:

- evidence có chứa chi tiết trang sức hiện tại
- nhưng không có bằng chứng đủ mạnh cho “cầu nối” giữa tuổi thơ và hiện tại
- answerer vẫn tự nối câu chuyện này thay vì báo thiếu bằng chứng

Đây là dạng hallucinated bridge:

- không phải hallucination trắng trợn hoàn toàn
- mà là suy luận quá mức từ evidence còn thiếu

## 5. Failure do không recover được sau `zero-hit SEARCH`

Đây là failure mode rất dễ nhìn vì nó để lại trace rõ.

### Case tiêu biểu: `sample_5918`, `sample_5917`, `sample_5938`, `sample_5914`, `sample_5913`

Các case này cùng một video `0k2ey_okQ4E` và đều có pattern:

- `SEARCH`
- `OPEN`
- rồi liên tục `SEARCH` lại với `hit_count = 0`
- cuối cùng trả về “insufficient information”

Nếu video local đúng benchmark, đây đã là lỗi recovery rồi.
Nhưng ở đây còn nghi có mismatch dữ liệu đầu vào, nên failure bị khuếch đại hơn nữa.

Điều cần rút ra ở tầng framework:

- sau 1 hoặc 2 lần `zero-hit SEARCH`, hệ hiện chưa có policy recovery tốt
- nó chưa biết:
  - rewrite query
  - đổi modality
  - back off lên scene lớn hơn
  - hoặc đánh dấu sample là suspect-input

## 6. Failure do evidence bị merge/collapse quá sớm

Một phát hiện kỹ thuật rất đáng chú ý là:

- trên `133/133` turns, `final_state.evidence_ledger` không giữ được nhiều hơn `1` object

Điều đó không có nghĩa là hệ chưa từng thấy nhiều evidence trong quá trình chạy.
Nó có nghĩa là:

- đến state cuối, evidence bị merge về một bundle duy nhất
- answerer không còn thấy ranh giới rõ giữa:
  - bằng chứng cốt lõi
  - bằng chứng phụ
  - bằng chứng gây nhiễu

Hệ quả:

- answer synthesis dễ bị “một bundle to” dẫn hướng
- failure analysis khó biết cụ thể bằng chứng nào kéo answer đi sai
- sau này nếu train controller/stop head cũng khó hơn vì signal trong ledger bị mất chi tiết

### Điều này giải thích vì sao một số turn nhìn như có nhiều hành động hợp lý nhưng answer vẫn yếu

Bởi vì:

- trace trung gian có thể có nhiều OPEN hợp lý
- nhưng state cuối không bảo toàn được cấu trúc evidence đủ tốt

## 7. Failure do thiếu STOP rõ ràng và phải dựa vào fallback answering

`83/133` turns không có `STOP` action rõ ràng.

Đây là dấu hiệu controller chưa học được:

- khi nào đủ bằng chứng
- khi nào không thể tiến bộ thêm
- khi nào nên kết thúc có kiểm soát

Tác động thấy được:

- turn có `STOP`: mean token F1 `0.3834`
- turn không có `STOP`: mean token F1 `0.3164`

Tức là:

- chỉ riêng việc hệ biết dừng đúng cách đã liên quan khá mạnh tới chất lượng answer

## Những bước sai lầm của chúng ta trong run này

Mình nghĩ có 6 quyết định sai hoặc thiếu chặt chẽ mà run này đã bộc lộ:

## 1. Chạy full benchmark trước khi validate video corpus

Đây là sai lầm nghiêm trọng nhất.

Chúng ta đã:

- dựng được root video runnable
- nhưng chưa có bước sanity check nội dung video so với sample benchmark

Khi điều đó sai, planner không thể cứu được.

## 2. Để controller được phép mở lại cùng node quá dễ

Hệ hiện chưa có penalty hoặc novelty check đủ mạnh cho:

- mở lại đúng node cũ
- mở node con gần như tương đương mà không tăng information gain

## 3. Chưa có recovery policy sau `zero-hit SEARCH`

Khi search ra `0 hit`, system hiện chưa có protocol rõ ràng:

- query rewrite
- modality switch
- widen time span
- suspect data flag

## 4. Quá phụ thuộc lexical retrieval cho câu hỏi abstract

Điều này làm hệ:

- tìm đúng topic
- nhưng không chắc tìm đúng answer-bearing sentence

Nó đặc biệt yếu ở:

- `main challenge`
- `reason`
- `connection`
- `what did he mean`

## 5. Merge evidence quá sớm ở final state

Điều này làm:

- answer synthesis kém grounded hơn
- trace khó phân tích hơn
- training signal sau này nghèo hơn

## 6. Chưa có monitoring để phát hiện anomaly theo batch

Ví dụ:

- cùng video `0k2ey_okQ4E` fail hàng loạt trên nhiều sample khác nhau

Nếu có dashboard đơn giản theo `video_id` và `task`, mình đã phát hiện group anomaly này sớm hơn nhiều, thay vì sau 90 sample.

## Những thứ hệ đang làm đúng

Để công bằng, run này không chỉ có failure.

Một số tín hiệu tích cực:

- với câu hỏi concrete hơn, hệ vẫn vào đúng vùng video khá tốt
- task `audio_understanding` và `audio_visual_alignment` không tệ
- các sample về `068rdc75mHM` cho thấy khi video đúng và question cụ thể, retrieval + speech evidence của hệ vẫn có thể cho answer khá ổn

Ví dụ:

- `sample_6096` turn 3 có `STOP` rõ
- evidence speech khá sát câu hỏi
- answer dù chưa hoàn hảo vẫn đi đúng hướng benchmark

Điều này rất quan trọng vì nó cho thấy:

- framework không hỏng hoàn toàn
- các lỗi còn lại đủ cụ thể để sửa theo từng lớp

## Ưu tiên sửa trước benchmark tiếp theo

## P0. Validate lại video corpus trước khi benchmark tiếp

Phải có một bước tự động hoặc bán tự động để kiểm tra:

- sample benchmark nói về gì
- memory thô của video local nói về gì
- có mismatch nghiêm trọng hay không

Chỉ riêng bước này có thể kéo benchmark lên rất nhiều nếu lỗi dữ liệu đang tồn tại thật.

## P0. Thêm novelty guard và no-progress guard cho controller

Cần chặn:

- mở lại cùng node nếu chưa có thông tin mới
- lặp `OPEN` nhiều lần mà evidence không đổi

Và khi gặp no-progress:

- phải `STOP`
- hoặc đổi strategy thật rõ

## P1. Thêm recovery policy sau `zero-hit SEARCH`

Khi `SEARCH` fail, controller/tool layer cần thử theo thứ tự:

1. rewrite query ngắn hơn
2. đổi modality
3. widen lên scene/segment lớn hơn
4. đánh dấu suspect input nếu vẫn fail

## P1. Cải thiện query understanding cho câu hỏi abstract

Đặc biệt cho các pattern:

- `why`
- `what was the main challenge`
- `how did X connect to Y`
- `what did he mean`

Ở đây có thể cần:

- query rewriting tốt hơn
- hoặc LLM reranking hẹp ở mức candidate snippets

## P1. Giữ evidence ở final state chi tiết hơn

Không nên collapse toàn bộ ledger về 1 bundle quá sớm.

Ít nhất state cuối nên giữ:

- `core evidence`
- `support evidence`
- `discarded/noisy evidence`

để answer synthesis và future training có signal tốt hơn.

## P2. Tạo dashboard eval theo `video_id` và `task`

Chỉ cần một dashboard rất nhẹ để nhìn:

- video nào fail hàng loạt
- task nào tụt mạnh
- trace nào lặp OPEN / zero-hit SEARCH nhiều

Thứ này sẽ giúp phát hiện anomaly sớm hơn rất nhiều trong lần benchmark sau.

## Câu chốt

Nếu phải tóm gọn run này trong một câu:

**90 sample vừa rồi cho thấy VideoRLM v1 chưa thất bại vì “model yếu” đơn thuần; nó thất bại vì một tổ hợp của lỗi dữ liệu đầu vào, controller loop chưa có no-progress recovery, retrieval chưa đủ mạnh cho câu hỏi trừu tượng, và evidence bị collapse quá sớm ở final state.**

Điều tích cực là các lỗi này đủ cụ thể để sửa có thứ tự, và nhiều lỗi nằm ở framework/control layer chứ chưa bắt buộc phải train model mới ngay.

## Update Log

### 2026-04-07

- tạo note đầu tiên cho run `90` sample của `output/longshot_single_gpu_full`
- chốt 3 nhóm nguyên nhân chính:
  - nghi ngờ mismatch dữ liệu đầu vào ở một phần video
  - controller loop lặp và thiếu `STOP`
  - query/snippet/evidence pipeline còn yếu cho câu hỏi trừu tượng
- xác định `0k2ey_okQ4E` là video cần validate lại đầu tiên
- xác định `final_state.evidence_ledger` đang bị collapse quá sớm trên toàn bộ `133` turns đã chấm
