# Failure Analysis: `sample_8563` with Heuristic-Only Speech Extraction

## Thông tin mẫu

- `sample_id`: `sample_8563`
- `video_id`: `wTlERUE8LVw`
- `task`: `event_understanding`
- `sample_type`: `multi_turn`
- chế độ phân tích: `heuristic-only`

## Câu hỏi benchmark

What was the first thing they tried that made them realize this was going to be different from regular street food?

## Ground Truth

The first thing that really stood out was the chicken head. They brought it out, and the person in the red bandana said, "This is the chicken head. Chicken head, discarded in many parts of the world, while YRC Chicken is selling over 10 pounds of chicken heads each day." Then they showed how it was coated in flour and deep fried, and when they bit into it, the person said, "It mostly tastes like fried goodness. Surprisingly, it tastes like chicken. It tastes chickeny." That moment, seeing something so unexpected and then actually enjoying it, was when it clicked this wasn't just regular street food.

## Answer hiện tại của hệ

The first thing they tried that made them realize this was going to be different from regular street food was the sauce, specifically mentioning vinegar sauce and sweet chili sauce as options, which they noted mostly tasted like "fried goodness."

## Kết luận ngắn

Failure chính của sample này không phải là ASR thiếu nội dung, và cũng không phải retrieval đi nhầm hoàn toàn sang vùng khác.

Failure chính nằm ở `OPEN(..., speech)`:
- hệ mở đúng `scene_001`
- transcript đúng đã có sẵn trong span
- nhưng heuristic chọn sai `snippet`
- rồi answer synthesis bám vào evidence sai đó và trả lời thành `sauce`

## Những gì hệ làm đúng

### 1. Transcript gốc đã có câu trả lời đúng

Trong artifact speech của video:
- `chicken head`
- `selling over 10 pounds`
- `coat the head in flour and deep fry`
- `tastes like chicken`

đều đã xuất hiện trong các span đầu của `scene_001`.

Điều này có nghĩa là:
- ASR không phải thủ phạm chính của failure này
- memory cũng không thiếu evidence đúng

### 2. Retrieval không hỏng hoàn toàn

Khi `SEARCH` chạy với query:

`first thing that made them realize this was going to be different from regular street food`

top hits gồm:
- `wTlERUE8LVw_scene_003` với score `0.925`
- `wTlERUE8LVw_scene_001` với score `0.9217`
- các segment/clip con của `scene_001`

Điểm đáng chú ý:
- `scene_003` đứng đầu là hơi đáng nghi
- nhưng `scene_001` vẫn nằm ngay top đầu
- và câu trả lời thật đúng là nằm trong `scene_001`

Vì vậy, retrieval có nhiễu, nhưng chưa phải lỗi chí mạng.

## Từng bước hệ đã làm gì

### Step 1: `SEARCH`

Controller gọi:

`SEARCH("first thing that made them realize this was going to be different from regular street food", speech)`

Observation trả về top frontier:
- `scene_003`
- `scene_001`
- `scene_001_seg_001`
- `scene_001_seg_001_clip_001`
- `scene_001_seg_001_clip_002`

Comment:
- `scene_003` được điểm cao chủ yếu vì lexical overlap với các từ rất generic như `first`, `different`, `food`, `going`
- `scene_001` vẫn là ứng viên hợp lý vì đây là phần đầu video và thật sự chứa `chicken head`

### Step 2: `OPEN(scene_001, speech)`

Đây là bước quan trọng nhất.

Tool tạo ra 2 evidence items:

1. span `0.0–120.0`
2. span `120.0–240.0`

Comment:
- việc chọn 2 span này là hợp lý
- span đầu chứa phần giới thiệu `chicken head`
- span sau chứa continuation như `fried goodness` và `tastes like chicken`

Vấn đề là `snippet` cắt ra từ 2 span này đều sai trọng tâm.

## Failure 1: span `0.0–120.0` bị cắt thành intro thay vì `chicken head`

### Transcript đúng trong span này

Ngay trong span `0.0–120.0`, transcript có:
- `What about chicken head?`
- `This is the chicken head.`
- `selling over 10 pounds of chicken heads each day`
- `coat the head in flour and deep fry`

### Nhưng snippet mà heuristic giữ lại là gì

Snippet cuối cùng lại là:
- `In this series, we're exploring Manila's street food scene...`
- `But first, we'll make a pit stop...`
- `street food...`

### Vì sao heuristic làm vậy

Sentence scorer trong `rlm/video/tools.py` đang chấm điểm câu dựa nhiều vào lexical overlap với wording của câu hỏi.

Với query này, token sau tokenize chủ yếu là:
- `first`
- `different`
- `food`
- `going`
- `made`
- `realize`
- `regular`
- `street`
- `thing`
- `tried`

Những token này làm cho các câu intro như:
- `But first...`
- `street food...`

được điểm rất cao, dù chúng không nêu ra object đúng.

Trong debug thực tế:
- câu `But first...` có score khoảng `15.77`
- câu `What about chicken head?` khoảng `9.41`
- câu `This is the chicken head.` khoảng `9.27`

Nói ngắn gọn:
- heuristic đang thưởng quá mạnh cho câu “giống wording của question”
- trong khi câu thật sự trả lời đúng lại là câu “giới thiệu object”

## Failure 2: span `120.0–240.0` bị cắt thành `sauce`

### Transcript đúng trong span này

Span `120.0–240.0` có các ý:
- `fried goodness`
- `tastes like chicken`
- continuation của trải nghiệm ăn món vừa được giới thiệu

### Nhưng snippet giữ lại lại bắt đầu bằng:
- `Sauce is here`
- `vinegar sauce`
- `sweet chili sauce`

### Vì sao heuristic làm vậy

Ở `OPEN(..., speech)`, khi đây là `first-query`, span thứ hai bị đặt:

- `prefer_start = True`

Sau đó `_focus_speech_detail()` trả về vài câu đầu span.

Vì đầu span đúng là bắt đầu bằng `Sauce is here`, nên evidence thứ hai bị neo vào phần sauce thay vì phần tasting reaction.

Vấn đề ở đây không phải span sai.
Vấn đề là:
- heuristic “first query => ưu tiên đầu span” đang quá cứng
- nó không hiểu rằng span thứ hai là continuation của món `chicken head`, không phải một object mới cần lấy câu đầu tiên

## Step 3 và Step 4: controller mở child clips nhưng không cứu được

Sau khi có 2 evidence sai trọng tâm, controller mở tiếp:
- `scene_001_seg_001_clip_001`
- `scene_001_seg_001_clip_002`

Nhưng cả hai `OPEN` này đều không thêm evidence mới.

Hệ quả:
- loop không thu thêm bằng chứng hữu ích
- sau hai `OPEN` rỗng liên tiếp, controller rơi vào fallback answer synthesis

## Vì sao answer cuối cùng ra thành `sauce`

Fallback answer synthesis chỉ được phép nhìn evidence ledger.

Ledger lúc đó chứa:
- một snippet intro
- một snippet `sauce + fried goodness`

Nó không còn thấy rõ:
- `This is the chicken head`
- `10 pounds`
- `coat in flour and deep fry`

Vì vậy câu trả lời cuối bị kéo sang:
- `sauce`
- `fried goodness`

thay vì object đúng là:
- `chicken head`

## Root Cause cuối cùng

Root cause chính của `sample_8563` ở mode heuristic-only là:

1. `first-question` sentence scoring quá thiên về lexical overlap với wording của câu hỏi
2. heuristic chưa có bias đủ mạnh cho kiểu câu giới thiệu object như:
   - `What about X?`
   - `This is the X`
   - `Never had yet`
3. `prefer_start` trên span continuation kéo evidence sang phần `sauce`
4. controller fallback sau chuỗi `OPEN` rỗng, nên answer cuối chỉ phản ánh ledger đã bị lệch từ trước

## Đây không phải lỗi gì

### Không phải lỗi ASR chính

Transcript đã chứa đầy đủ các ý benchmark cần.

### Không phải lỗi retrieval hoàn toàn

Hệ đã vào đúng `scene_001`.

### Không phải lỗi “model không hiểu transcript”

Model cuối không hề được thấy snippet đúng, vì heuristic code đã cắt sai trước đó.

## Hướng fix hợp lý nếu vẫn giữ heuristic-only

### 1. Tăng điểm cho câu giới thiệu object đầu tiên

Với câu hỏi kiểu `first`, nên thưởng mạnh hơn cho pattern:
- `What about ...`
- `This is the ...`
- `Never had yet`
- `That's what that is`

### 2. Giảm điểm cho câu intro roadmap

Nên phạt hoặc hạ trọng số cho các câu kiểu:
- `In this series...`
- `But first...`
- `street food scene...`

vì chúng thường chỉ mở bài chứ không trả lời object thực sự.

### 3. Không dùng `prefer_start` quá cứng cho span thứ hai

Nếu span đầu đã có entity anchor rõ như:
- `chicken head`

thì span tiếp theo nên ưu tiên:
- preparation details
- tasting reaction

chứ không phải mặc định lấy đầu span.

### 4. Kết nối entity anchor với continuation span

Nếu span đầu đã xác định object là `chicken head`, span sau nên được cắt theo logic:
- giữ các câu tiếp tục nói về object đó
- như `fried goodness`, `tastes like chicken`
- thay vì bám vào `sauce`

## Kết luận cuối

`sample_8563` là một failure mode rất điển hình:

- hệ biết phải nhìn vào đầu video
- transcript đúng đã có
- nhưng heuristic lại chọn câu “nghe giống câu hỏi” thay vì câu “thật sự nêu ra đáp án”

Nói ngắn gọn:

**hệ tìm đúng vùng, nhưng cắt sai câu.**
