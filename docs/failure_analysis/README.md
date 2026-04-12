# Failure Analysis

Thư mục này dùng để lưu các note phân tích failure mode cụ thể của VideoRLM.

Mục tiêu của các note ở đây:
- tách riêng phần debug ra khỏi overview tài liệu chung
- giúp nhìn lại từng sample benchmark một cách có hệ thống
- ghi rõ hệ đã đúng ở đâu, sai ở đâu, và vì sao

Các note hiện có:
- `longshot_90run_analysis.md`: postmortem chi tiết cho `90` sample đầu tiên của run full benchmark, gồm số liệu tổng hợp, failure taxonomy, case study, và danh sách ưu tiên sửa
- `longshot_90run_official_style_analysis.md`: phân tích vì sao official-style LongShOT eval của cùng run chỉ đạt khoảng `14.97%`, và map score thấp về các failure mode ở runtime
- `sample_8563_heuristic.md`: phân tích failure của `sample_8563` ở chế độ `heuristic-only`
