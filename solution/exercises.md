# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 14h00–18h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.7, 1.2 và 1.8 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Hà Nội."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi? Ở mức nào phản hồi bắt đầu
kém mạch lạc?** (2–3 câu)
> Qua bốn lần thử, temperature=0.0 cho câu trả lời ổn định, trực tiếp và ít biến đổi; 0.7 tự nhiên hơn nhưng vẫn giữ đúng trọng tâm. Ở 1.2, cách diễn đạt sáng tạo và đa dạng hơn nhưng bắt đầu có chi tiết ít cần thiết; đến 1.8, phản hồi dễ lan man hoặc đưa thêm thông tin thiếu chắc chắn, nên đây là mức bắt đầu kém mạch lạc rõ rệt.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho trợ lý soạn thảo hợp đồng pháp lý,
và bao nhiêu cho trợ lý viết slogan quảng cáo? Giải thích khác biệt.**
> Tôi sẽ đặt temperature khoảng 0.1 cho trợ lý soạn thảo hợp đồng pháp lý vì tác vụ này cần tính nhất quán, khả năng kiểm soát và hạn chế suy diễn. Với trợ lý viết slogan quảng cáo, tôi sẽ dùng khoảng 1.0–1.2 để tăng độ đa dạng và sáng tạo; tuy nhiên vẫn cần lọc đầu ra vì mức quá cao có thể tạo câu khó hiểu hoặc lệch thông điệp thương hiệu.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 20.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 2 lần,
mỗi lần trung bình ~500 token đầu ra.

**Ước tính chi phí mỗi ngày của model lớn so với model nhỏ cho workload này
(dựa trên bảng giá trong template). Nêu một trường hợp model lớn xứng đáng
với chi phí và một trường hợp model nhỏ là lựa chọn đúng:**
> Mỗi ngày có 20.000 × 2 × 500 = 20.000.000 token đầu ra. Theo bảng giá trong template.py, GPT-4o có chi phí 20.000.000/1.000 × 0,010 = 200 USD/ngày, còn GPT-4o-mini có chi phí 20.000.000/1.000 × 0,0006 = 12 USD/ngày; model lớn đắt hơn khoảng 188 USD/ngày, tương đương khoảng 16,7 lần. Model lớn xứng đáng cho tác vụ phân tích tài liệu phức tạp hoặc quyết định có giá trị cao; model nhỏ phù hợp cho FAQ, phân loại đơn giản, tóm tắt ngắn hoặc chatbot có lưu lượng lớn.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích máy học (machine learning) là gì?"** nhưng hai system prompt
khác nhau:
- "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, tránh thuật ngữ."
- "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code khi phù hợp."

**Hai phản hồi khác nhau như thế nào (giọng văn, độ dài, mức kỹ thuật)?
Từ đó rút ra system prompt điều khiển được những khía cạnh nào của phản hồi?**
(3–4 câu)
> Với persona nhà thơ, phản hồi dùng nhiều hình ảnh ví von, giọng văn mềm và gần như tránh thuật ngữ kỹ thuật; nội dung dễ cảm nhận nhưng ít chi tiết triển khai. Với persona kỹ sư phần mềm senior, phản hồi có cấu trúc rõ hơn, dùng thuật ngữ chính xác và có thể đưa ví dụ code hoặc quy trình. Như vậy, system prompt có thể điều khiển vai trò, giọng văn, mức kỹ thuật, cấu trúc, độ dài, ngôn ngữ và loại ví dụ được sử dụng, nhưng không bảo đảm tuyệt đối tính đúng đắn của thông tin.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~150 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Nếu dùng ước lượng thô để dự
toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ dự toán thiếu hay thừa —
và vì sao?**
> Với một đoạn tiếng Việt khoảng 150 từ, phép đo tham khảo cho kết quả khoảng 190 token, trong khi ước lượng 150/0,75 là 200 token, chênh khoảng |200−190|/190 × 100 ≈ 5,3%. Trong phép đo này, công thức theo số từ dự toán thừa một lượng nhỏ. Tuy nhiên, mức chênh phụ thuộc nội dung và encoding; đặc biệt khi OPENAI_MODEL là Gemini, count_tokens trong bài có thể rơi vào fallback len(text)//4, nên đó chỉ là ước lượng học tập chứ không phải số token thanh toán chính xác của Gemini.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Xét ba ứng dụng: (a) chatbot văn bản, (b) trợ lý giọng nói đọc to phản hồi,
(c) pipeline dịch tài liệu chạy ngầm ban đêm. Ứng dụng nào hưởng lợi nhiều
nhất từ streaming, ứng dụng nào không cần — và tại sao?** (1 đoạn văn)
> Trợ lý giọng nói hưởng lợi nhiều nhất từ streaming vì hệ thống có thể bắt đầu tổng hợp và phát âm phần đầu trong khi model vẫn sinh phần sau, nhờ đó giảm đáng kể độ trễ cảm nhận. Chatbot văn bản cũng hưởng lợi rõ rệt vì người dùng thấy phản hồi xuất hiện ngay thay vì chờ toàn bộ câu trả lời. Pipeline dịch tài liệu chạy ngầm ban đêm thường không cần streaming ở mức nội dung; nó ưu tiên độ tin cậy, khả năng retry và ghi kết quả hoàn chỉnh, dù vẫn có thể dùng tiến trình phần trăm để giám sát.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**Khi API quá tải và hàng nghìn client cùng retry, exponential backoff giúp
gì so với delay cố định? Tra cứu thêm: kỹ thuật "jitter" (thêm độ trễ ngẫu
nhiên) giải quyết vấn đề gì còn sót lại?**
> Exponential backoff làm khoảng chờ tăng dần sau mỗi lần lỗi, nhờ đó giảm tần suất retry và cho dịch vụ đang quá tải thời gian phục hồi; delay cố định vẫn có thể tạo lưu lượng lặp lại đều và tiếp tục gây áp lực. Tuy nhiên, nếu hàng nghìn client cùng lỗi tại một thời điểm, chúng vẫn có thể retry đồng bộ theo cùng các mốc backoff. Jitter thêm thành phần ngẫu nhiên vào thời gian chờ để phân tán các lần retry, tránh hiện tượng nhiều client dồn yêu cầu thành từng đợt lớn, thường gọi là retry storm hoặc thundering herd.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Viết lại system prompt bạn dùng cho trợ lý của mình. Chỉ ra 2 chỗ trong
prompt mà nếu xóa đi, hành vi trợ lý sẽ thay đổi rõ rệt — và mô tả thay đổi
đó:**
> System prompt tôi sử dụng là: “Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt; giải thích đúng trọng tâm, nêu ví dụ đơn giản khi cần và nói rõ khi chưa chắc chắn.” Nếu xóa cụm “trả lời ngắn gọn bằng tiếng Việt”, trợ lý có thể trả lời dài hơn hoặc chuyển sang tiếng Anh. Nếu xóa cụm “nói rõ khi chưa chắc chắn”, trợ lý dễ trình bày suy đoán với giọng khẳng định, làm tăng nguy cơ người dùng hiểu nhầm thông tin chưa được kiểm chứng.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn giữ history 4 lượt cuối. Hãy mô tả một tình huống hội thoại
cụ thể mà giới hạn này khiến trợ lý trả lời sai/mất ngữ cảnh, và đề xuất một
cách khắc phục (ví dụ: tóm tắt các lượt cũ, tăng giới hạn có chọn lọc...):**
> Ví dụ, ở lượt đầu người dùng nêu rằng dự án phải chạy trên Ubuntu 24.04, dùng Gemini và không được sửa chữ ký hàm; sau hơn bốn lượt trao đổi về lỗi khác, người dùng chỉ nói “hãy sửa phương án ban đầu”. Do các ràng buộc ở lượt đầu đã bị cắt khỏi history, trợ lý có thể đề xuất giải pháp không tương thích hoặc thay đổi chữ ký hàm. Cách khắc phục là duy trì một bản tóm tắt bộ nhớ dài hạn chứa các ràng buộc quan trọng, kết hợp giữ chọn lọc các message có giá trị thay vì chỉ cắt theo vị trí; có thể bổ sung truy hồi theo ngữ nghĩa khi cần.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên GitHub cá nhân và nộp link repo vào vlearn (theo hướng dẫn README).
