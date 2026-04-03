# LongShOTAgent vs VideoRLM

## Mục tiêu của note này

Note này dùng để làm rõ một cảm giác rất dễ gặp khi đọc paper LongShOT và nhìn vào VideoRLM hiện tại:

- hai hệ **rất giống nhau** ở bề ngoài
- nhưng **không hoàn toàn giống nhau** ở ý tưởng lõi

Kết luận ngắn nhất của mình là:

**LongShOTAgent là một system recipe mạnh cho long-video reasoning trên LongShOTBench. VideoRLM là một controller framework rõ abstraction hơn, được thiết kế để debug, log, và sau này học ra policy tốt hơn.**

## Chúng giống nhau ở đâu

Cả hai hệ đều đi theo cùng một trực giác nền:

- không nên nhét toàn bộ video dài vào một prompt rồi trả lời one-shot
- nên tiền xử lý video trước
- nên dùng tín hiệu đa phương thức như visual, speech, audio
- nên search đúng vùng rồi mới mở sâu
- nên refine dần thay vì trả lời ngay từ đầu

Nếu chỉ nhìn kiến trúc ngoài cùng, cả hai đều thuộc cùng một họ:

- `video -> processed metadata/memory -> search -> inspect evidence -> answer`

Đây là lý do vì sao cảm giác “rất giống nhau” là hoàn toàn hợp lý.

## Khác nhau ở đâu

Khác biệt lớn nhất nằm ở **trọng tâm thiết kế**.

### LongShOTAgent

LongShOTAgent, theo paper và project page, được trình bày như một **agentic system** để giải bài toán long-video reasoning qua:

- preprocessing
- search
- iterative refinement

Trọng tâm của họ là:

- thiết kế benchmark LongShOTBench
- chứng minh bài toán khó như thế nào
- xây một hệ đủ mạnh để làm mốc trên benchmark đó

### VideoRLM

VideoRLM hiện tại không chỉ là một pipeline answer question from video.
Nó cố tình explicit hóa các khái niệm mà sau này ta muốn học thành policy:

- `VideoMemory`
- `Frontier`
- `Evidence ledger`
- `ControllerState`
- `ControllerAction`
- `Observation`
- `TraceStep`

Trọng tâm của VideoRLM là:

- planner/controller nhìn gì ở mỗi bước
- planner/controller được phép làm gì
- mỗi hành động tạo ra observation nào
- evidence nào được tích lũy
- làm sao log trajectory đủ tốt để train tiếp

Nói ngắn gọn:

- **LongShOTAgent nghiêng về system recipe**
- **VideoRLM nghiêng về control abstraction**

## Ưu điểm của LongShOTAgent / LongShOT

### 1. Benchmark-native rất mạnh

LongShOTBench được thiết kế rất đúng bài:

- open-ended Q&A
- single-turn và multi-turn
- vision + speech + audio
- tool use
- custom rubrics

Điều này làm benchmark của họ thực tế hơn rất nhiều so với kiểu multiple-choice hoặc single-score đơn giản.

### 2. Hệ preprocess/data pipeline giàu hơn hiện tại của VideoRLM

Repo public của họ có pipeline preprocess và dataset generation khá đầy đủ:

- video descriptions
- audio descriptions
- alignment
- multimodal understanding
- key events
- metadata generation
- final consolidation

Ở mặt “dựng benchmark và dựng metadata quy mô lớn”, họ đang đi xa hơn VideoRLM hiện tại.

### 3. Có kết quả benchmark rõ ràng

Trên project page, họ report:

- `LongShOTAgent = 44.66`
- `Gemini-2.5-Flash = 52.95`
- open-source baselines phần lớn dưới `30`

Điều này cho họ một mốc rất mạnh khi nói chuyện nghiên cứu: hệ đã được đo trên benchmark chính chủ.

### 4. Bài toán được đóng khung rất thuyết phục

LongShOT nói rất rõ:

- vì sao long multimodal video understanding khó
- vì sao cần benchmark có rubric
- vì sao agentic reasoning quan trọng

Ở mặt “narrative để publish / để thuyết phục cộng đồng”, họ có lợi thế rõ ràng.

## Khuyết điểm của LongShOTAgent / LongShOT

### 1. Public repo nghiêng nhiều về benchmark/data/eval hơn là agent runtime framework

Đây là điểm mình thấy quan trọng nhất.

Repo public hiện tại cho thấy rất rõ:

- preprocess pipeline
- datagen pipeline
- eval/scoring pipeline

Nhưng phần runtime “video-agent” public không lộ ra một abstraction rõ như:

- state
- action
- observation
- evidence ledger
- transition

Trong `eval/utils.py`, mode `video-agent` được gọi qua một endpoint với `video_id`, nhưng public repo không làm lộ đầy đủ server/runtime tương ứng.

Nói công bằng:

- paper mô tả một agentic system
- nhưng public code hiện tại chưa cho thấy trọn vẹn một framework controller-level rõ ràng để tái dùng và train tiếp

### 2. Tool-calling trong public eval code thiên về simulation benchmark-side

Trong `eval/tool_handler.py`, luồng tool calling chủ yếu là:

- model chọn tool
- hệ lấy `expected_tool_calls`
- simulate tool results
- model trả final answer

Cách này tốt cho benchmark evaluation, nhưng nó khác với một tool runtime thật hoạt động trên memory của video.

### 3. Khó nhìn thấy “planner internals” để debug sâu

Vì public artifact của họ tập trung vào preprocess/eval, nên nếu mục tiêu là debug planner:

- nó search sai ở đâu
- open sai đoạn nào
- stop sai lúc nào
- evidence bị drift từ bước nào

thì VideoRLM hiện tại dễ quan sát hơn.

### 4. Từ góc nhìn training policy, abstraction chưa explicit bằng VideoRLM

Nếu mục tiêu tiếp theo là:

- thu trajectory
- SFT controller
- RL cho `SEARCH/OPEN/STOP`

thì cần một state/action interface rất rõ.

LongShOT paper có tinh thần agentic mạnh, nhưng public repo hiện tại chưa explicit hóa lớp interface đó rõ bằng VideoRLM.

## Ưu điểm của VideoRLM hiện tại

### 1. State/action/evidence được mô hình hóa rất rõ

Đây là điểm mạnh nhất của VideoRLM.

Trong repo hiện tại, planner không phải là một khái niệm mơ hồ.
Nó có schema rõ:

- `VideoMemory`
- `FrontierItem`
- `Evidence`
- `BudgetState`
- `ControllerAction`
- `ControllerState`

Điều này giúp:

- debug dễ
- test dễ
- trace dễ
- train tiếp dễ

### 2. Tool runtime là tool thật

`SEARCH`, `OPEN`, `SPLIT`, `MERGE`, `STOP` trong VideoRLM là action runtime thật trên video memory, không phải chỉ là benchmark-side simulation.

Tức là hệ thực sự:

- search trên index
- mở node
- đọc transcript / OCR / visual summary / audio tags
- update evidence ledger

Điều này rất quan trọng nếu mục tiêu là xây một agent có thể chạy ngoài benchmark.

### 3. Failure analysis tốt hơn

VideoRLM hiện tại cho phép mổ lỗi theo đúng controller loop:

- retrieval sai
- snippet extraction sai
- continuation span sai
- stop quá sớm
- lặp vô ích

Các failure mode này đã được phân tích thực chiến trên sample thật, ví dụ:

- `sample_6168`
- `sample_8563`

Khả năng “nhìn thấy hệ sai từ bước nào” là một lợi thế rất lớn.

### 4. Rất hợp để tiến tới learned controller

VideoRLM hiện tại gần như đã có sẵn xương sống cho bước tiếp theo:

- prompt controller trước
- thu trace
- SFT action selection
- học stop decision
- sau đó mới nghĩ tới recursion sâu hơn hoặc RL

Tức là nó rất hợp với roadmap bạn đang nhắm ngay từ đầu.

### 5. Repo self-contained hơn cho runtime benchmark

Hiện tại repo đã có:

- local model adapters
- memory builder
- artifact store
- benchmark runner cho LongShOT
- trace output
- failure analysis docs

Về mặt “một repo để thật sự chạy, debug, rồi cải tiến controller”, VideoRLM đang khá tự chủ.

## Khuyết điểm của VideoRLM hiện tại

### 1. Chưa có score benchmark mạnh như LongShOTAgent

Điểm yếu lớn nhất là điều này.

VideoRLM hiện chưa có một con số full-benchmark đủ mạnh để cạnh tranh trực diện với `44.66` của LongShOTAgent.

Hiện trạng của mình vẫn là:

- chạy sample thật
- debug failure thật
- cải thiện runtime theo từng failure mode

Điều này rất tốt cho research engineering, nhưng chưa phải câu chuyện benchmark mạnh.

### 2. Perception stack hiện còn đơn giản hơn

Memory builder hiện tại vẫn là bản pragmatic:

- segmentation heuristic
- lexical/hybrid retrieval đơn giản
- speech snippet extraction heuristic là chính

So với pipeline preprocess phong phú của LongShOT, VideoRLM còn nhẹ hơn và chưa sâu bằng ở mặt metadata generation.

### 3. Recursion “đúng nghĩa” vẫn chưa fully on

Tên VideoRLM gợi tới recursive controller, nhưng bản hiện tại vẫn chủ yếu là:

- state
- action
- evidence accumulation
- loop control

chứ chưa phải một recursive learned controller hoàn chỉnh.

Nói cách khác:

- framework đã chuẩn bị đường cho recursion
- nhưng bản runtime hiện tại vẫn gần một agent pipeline hơn là một learned recursive policy hoàn chỉnh

### 4. Một số thành phần còn heuristic-heavy

Ví dụ rõ nhất là:

- speech snippet selection
- continuation span handling
- temporal prior cho `first/last`

Điều này giúp debug nhanh, nhưng cũng tạo ra failure mode khi transcript dài và nhiều topic shift.

## Tóm tắt rất ngắn

### LongShOTAgent mạnh ở đâu

- benchmark-native
- preprocess/data/eval rất mạnh
- có reported result rõ
- narrative nghiên cứu tốt

### LongShOTAgent yếu ở đâu

- public runtime abstraction chưa rõ bằng VideoRLM
- tool use public code thiên về evaluation/simulation
- khó thấy planner internals để train/debug sâu

### VideoRLM mạnh ở đâu

- state/action/evidence rất rõ
- tool runtime thật trên memory
- trace và failure analysis rất tốt
- rất hợp cho hướng SFT/RL controller sau này

### VideoRLM yếu ở đâu

- chưa có benchmark score mạnh bằng
- preprocess/perception stack còn pragmatic
- recursion learned policy chưa fully realized
- vẫn còn phụ thuộc khá nhiều vào heuristic ở một số bước

## Kết luận của mình

Nếu mục tiêu là:

- **làm benchmark và có một system recipe mạnh để chứng minh hiệu năng trên LongShOTBench**

thì LongShOT/LongShOTAgent hiện đang mạnh hơn.

Nếu mục tiêu là:

- **xây một framework controller có thể debug, trace, huấn luyện, và tiến dần tới recursive learned control**

thì VideoRLM hiện có lợi thế kiến trúc rõ ràng hơn.

Câu công bằng nhất theo mình là:

**LongShOTAgent và VideoRLM thuộc cùng một họ hệ thống. LongShOTAgent hiện là benchmark-first system recipe mạnh hơn. VideoRLM hiện là controller-first framework rõ abstraction hơn.**

## Điều VideoRLM nên học từ LongShOT tiếp theo

- làm perception/multimodal metadata giàu hơn
- tăng chất lượng search/refinement cho các task multimodal khó
- chạy eval rộng hơn trên LongShOTBench
- đưa ra một con số benchmark đủ rõ để cạnh tranh công bằng

## Điều LongShOTAgent gợi mở nhưng VideoRLM có thể làm tốt hơn

- explicit hóa planner thành `state -> action -> observation -> next_state`
- chuẩn hóa evidence ledger từ đầu
- biến debug traces thành training data
- tiến tới controller policy học được thay vì chỉ dừng ở prompt/system recipe

## Nguồn

- LongShOT project page: `https://mbzuai-oryx.github.io/LongShOT/`
- LongShOT public repo: `https://github.com/mbzuai-oryx/LongShOT`
- VideoRLM local runtime:
  - [`rlm/video/types.py`](/share_4/users/duy/project/rlm/rlm/video/types.py)
  - [`rlm/video/controller.py`](/share_4/users/duy/project/rlm/rlm/video/controller.py)
  - [`rlm/video/tools.py`](/share_4/users/duy/project/rlm/rlm/video/tools.py)
  - [`rlm/video/memory.py`](/share_4/users/duy/project/rlm/rlm/video/memory.py)
  - [`rlm/video/longshot.py`](/share_4/users/duy/project/rlm/rlm/video/longshot.py)
