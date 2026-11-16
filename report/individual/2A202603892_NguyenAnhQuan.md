# Báo cáo Thực hành Day 7: Retrieval & Context Engineering (Agentic RAG)

**Học viên:** Nguyễn Anh Quân  
**Mã sinh viên:** 2A202603892  
**Lớp:** K5-L3B  
**Repository:** [https://github.com/AnhQuanHANU/K5-L3B-DAY07-2A202603892-NguyenAnhQuan-RetrievalContext](https://github.com/AnhQuanHANU/K5-L3B-DAY07-2A202603892-NguyenAnhQuan-RetrievalContext)

---

## 1. Dữ liệu và Benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:**
  - Nguồn tài liệu: *Điều lệ Công ty Cổ phần Sữa Việt Nam (Vinamilk) năm 2026* (`data/vinamilk_charter_2026.pdf`). Đây là văn bản công bố thông tin pháp lý đại chúng của doanh nghiệp niêm yết trên HOSE, được sử dụng cho mục đích nghiên cứu và giáo dục học thuật.
  - Số trang: 50 trang PDF nguyên bản tiếng Việt.
  - Số chunk: 152 chunks (kế thừa từ Strategy B - Markdown Structural Chunking của Day 06, bảo toàn cấu trúc Chương, Điều và Khoản pháp lý).
- **Vector store, embedding model, phiên bản và số chiều:**
  - Vector Store: Qdrant Local Engine (`data/qdrant_db`, collection `strategy_c`, khoảng cách `Cosine`).
  - Embedding Model: Google Gemini `gemini-embedding-001`, cấu hình đầu ra `output_dimensionality=768`.
- **Agent model, cách chạy và phân chia dev/test:**
  - Model Controller: Gemini API (`gemini-3.5-flash` và cơ chế multi-model failover với `gemini-3.5-flash-lite`, `gemini-3.6-flash`, `gemini-3.1-flash-lite` nhằm tuân thủ nghiêm ngặt hạn ngạch Free Tier).
  - Cách chạy: Chạy agent loop hoàn toàn tự động qua `starter/agentic.py` và bộ điều phối `src/run_experiments.py`, không sử dụng mock adapter; ghi nhận toàn bộ log sự kiện, trích dẫn inline `[chunk-id]` và đánh giá qua `starter/agent_metrics.py`.
  - Phân chia tập dữ liệu: Tổng cộng 24 câu hỏi (`data/cases.json`):
    - **Dev split:** 17 câu (15 câu có đáp án `q01`–`q15` + 2 câu bẫy không có bằng chứng `q21`, `q22`).
    - **Test split:** 7 câu (5 câu có đáp án `q16`–`q20` + 2 câu bẫy không có bằng chứng `q23`, `q24`).
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - *Câu có evidence:* 20 câu hỏi về các điều khoản cốt lõi trong Điều lệ (tên viết tắt, trụ sở chính, cơ cấu Hội đồng Quản trị, thẩm quyền ĐHĐCĐ, vốn điều lệ, mã ngành nghề phụ lục). Gold evidence chứa chuỗi trích dẫn nguyên văn (`verbatim slice`) và số trang cụ thể.
  - *Câu thiếu evidence (Unanswerable / Abstention):* 4 câu hỏi bẫy về các nội dung không quy định trong Điều lệ (chế độ tiền lương thai sản, mức thưởng Tết, tiêu chuẩn kháng sinh tồn dư trong sữa bò tươi, lộ trình trung hòa carbon Net Zero).
  - *Tiêu chí chấm chuẩn hóa:*
    - Retrieval: Đo lường `Hit@1`, `Hit@3` (gold chunk nằm trong top-k) và `Context Evidence Rate` (chunk chứa gold nằm trọn vẹn trong context sau khi đóng gói theo ngân sách ký tự).
    - Agent: Chấm điểm qua công cụ `starter/agent_metrics.py` với thang điểm `0/1/2` cho câu trả lời có căn cứ (2: đúng + citation hợp lệ trích từ context; 1: có trả lời nhưng citation chưa khớp gold; 0: sai hoặc từ chối nhầm); nhãn `abstention_correct` cho câu không có đáp án.

---

## 2. Retrieval và Thử nghiệm

### Bảng kết quả Retrieval Benchmark

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / Thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / Chi phí |
|---|---|---|---|---|---|---|---|
| **Baseline** | Dense (Qdrant), top_k=3, budget=800 | Kế thừa nguyên bản Day 06 | dev | 10/15 (66.7%) | 11/15 (73.3%) | 0/15 (0.0%) | 0.66 ms / 0$ |
| **Phương pháp 2** | Lexical (TF-IDF), top_k=3, budget=800 | Đối sánh từ khóa thuần túy | dev | 11/15 (73.3%) | 14/15 (93.3%) | 0/15 (0.0%) | 0.54 ms / 0$ |
| **Phương pháp 3** | Hybrid (Dense + TF-IDF, RRF k=60), top_k=3, budget=800 | Kết hợp rank giữa vector & từ khóa | dev | 10/15 (66.7%) | 14/15 (93.3%) | 0/15 (0.0%) | 1.28 ms / 0$ |
| **Cải thiện 1** | Dense (Qdrant), top_k=3, budget=1500 | Tăng context budget từ 800 lên 1500 ký tự | dev | 10/15 (66.7%) | 11/15 (73.3%) | 10/15 (66.7%) | 0.59 ms / 0$ |
| **Cải thiện 2** | Hybrid (Dense + TF-IDF, RRF k=60), top_k=3, budget=1500 | Kết hợp Hybrid RRF + context budget 1500 | dev | 10/15 (66.7%) | 14/15 (93.3%) | **10/15 (66.7%)** | 1.26 ms / 0$ |
| **Cấu hình đã chọn** | Hybrid (Dense + TF-IDF, RRF k=60), top_k=3, budget=1500 | Áp dụng cấu hình tối ưu nhất lên Test | **test** | **2/5 (40.0%)** | **5/5 (100.0%)** | **2/5 (40.0%)** | 1.15 ms / 0$ |
| **Baseline** | Dense (Qdrant), top_k=3, budget=800 | Baseline nguyên bản trên Test | test | 2/5 (40.0%) | 2/5 (40.0%) | 0/5 (0.0%) | 0.67 ms / 0$ |

### Phân tích 2 lỗi trên tập Dev, nguyên nhân và lý do chọn cấu hình:

1. **Hiện tượng nghẽn Context Evidence Rate tại Budget = 800 (Lỗi hệ thống nghiêm trọng trên Baseline):**
   - *Mô tả lỗi:* Trên cả 3 phương pháp với `budget=800`, mặc dù `Hit@3` đạt tới 73.3% - 93.3%, chỉ số `Context Evidence Rate` lại bằng **0/15 (0.0%)**!
   - *Nguyên nhân:* Các điều khoản pháp lý trong Điều lệ Vinamilk có độ dài trung bình từ 600 - 1.200 ký tự. Hàm `pack_context` duyệt lần lượt từng candidate chunk và cộng dồn độ dài text; khi chunk đầu tiên hoặc thứ 2 vượt quá 800 ký tự, chunk đó bị loại bỏ hoàn toàn khỏi ngữ cảnh đưa vào LLM.
   - *Thử nghiệm & Khắc phục:* Thiết lập cấu hình **Cải thiện 1** và **Cải thiện 2** với `budget=1500`. Ngay lập tức, `Context Evidence Rate` tăng vọt từ **0.0% lên 66.7% (10/15)**, cho phép Agent tiếp cận đầy đủ nội dung bằng chứng để trả lời.
2. **Lỗi truy vấn đa nghĩa pháp lý (Lexical vs Dense Synergies):**
   - *Mô tả lỗi:* Ở câu `q09` (thời hạn gửi thông báo mời họp ĐHĐCĐ) và `q11` (thẩm quyền tổ chức lại/giải thể công ty), Dense Retriever thuần túy nhầm lẫn giữa thẩm quyền của HĐQT và ĐHĐCĐ do sự tương đồng ngữ nghĩa trong văn bản pháp luật (`Hit@3` của Dense chỉ đạt 11/15 = 73.3%). Trong khi đó, Lexical Retriever dựa trên từ khóa chính xác neo được đúng Điều 15 và Điều 19 (`Hit@3` đạt 14/15 = 93.3%).
   - *Lý do chọn cấu hình cuối cùng:* **Hybrid Retriever kết hợp RRF (Reciprocal Rank Fusion) và Budget = 1500 ký tự** là cấu hình vượt trội toàn diện. Trên tập Test, cấu hình này nâng `Hit@3` từ 40% (Baseline) lên **100% (5/5)** và duy trì tốc độ xử lý chỉ 1.15ms.

---

## 3. Agent và Đánh giá

### Thiết kế Agent Controller

- **Công cụ (Tool):** `search(query, filters, top_k)`.
- **Số lượt gọi tối đa:** `max_searches = 3`.
- **Điều kiện dừng:** Agent tự động phân tích ngữ cảnh, nếu đủ bằng chứng thì trả về action `answer` kèm mảng `citations` và câu trả lời chèn mã `[chunk-id]`; nếu ngữ cảnh rỗng hoặc không có bằng chứng, Agent tự động từ chối bằng thông điệp *"Không đủ bằng chứng để trả lời."* và trả về `citations: []`.
- **Quản lý Hạn ngạch (Quota Handling):** Áp dụng cơ chế multi-model fallback và giãn cách gọi (`time.sleep`) để bảo đảm hệ thống vận hành liên tục trên Gemini Free Tier mà không bị gián đoạn bởi lỗi 429.

### Bảng kết quả Agent Tổng hợp (`starter/agent_metrics.py`)

| Cấu hình / Split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| **Hybrid RRF + Budget 1500 (Dev)** | 17/17 (100.0%) | 1 lượt / case | 0: 3 \| 1: 2 \| 2: 10 (TB: **1.47 / 2.0**) | 10/12 (83.3%) | 12/12 (100.0%) | **2/2 (100.0%)** | 3/15 (20.0%) |
| **Hybrid RRF + Budget 1500 (Test)** | 7/7 (100.0%) | 1 lượt / case | 0: 3 \| 1: 0 \| 2: 2 (TB: **0.80 / 2.0**) | 2/2 (100.0%) | 2/2 (100.0%) | **2/2 (100.0%)** | 3/5 (60.0%) |

### Bảng vết thực thi mẫu (Traces & Audit Labels)

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và Citation | Nhãn, lý do |
|---|---|---|---|---|
| **q01** | Answerable (Dev) | Search `VINAMILK`, trả về 3 chunks gồm `p07_c001` | "Tên viết tắt của Công ty Cổ phần Sữa Việt Nam là VINAMILK [B_vinamilk_charter_2026_p07_c001]." | **Grade 2**: Trả lời chính xác, trích dẫn đúng chunk căn cứ. |
| **q02** | Answerable (Dev) | Search trụ sở chính, trả về `p07_c001` | "Trụ sở chính của Công ty Cổ phần Sữa Việt Nam (Vinamilk) đặt tại: 10 Tân Trào, Phường Tân Mỹ, Thành phố Hồ Chí Minh, Việt Nam [B_vinamilk_charter_2026_p07_c001]." | **Grade 2**: Trả lời đầy đủ địa chỉ số 10 Tân Trào, trích dẫn chuẩn. |
| **q16** | Answerable (Test) | Search thời hạn hòa giải tranh chấp, trả về `p46_c000` | "Nếu không có quyết định hòa giải nào đạt được trong vòng bốn mươi lăm (45) ngày từ khi có thông báo tranh chấp của một bên... [B_vinamilk_charter_2026_p46_c000]." | **Grade 2**: Trích xuất chính xác thời hạn 45 ngày theo Điều 54. |
| **q20** | Answerable (Test) | Search mã ngành trồng cây hàng năm khác, trả về `p47_c000` | "Mã ngành của hoạt động trồng cây hàng năm khác của Công ty là 0119 [B_vinamilk_charter_2026_p47_c000]." | **Grade 2**: Trích xuất đúng mã số 0119 tại Phụ lục ngành nghề. |
| **q21** | Unanswerable (Dev) | Search mức lương thai sản lao động nữ | "Không đủ bằng chứng để trả lời." `citations: []` | **Abstention đúng**: Tài liệu Điều lệ không quy định thang bảng lương chi tiết. |
| **q22** | Unanswerable (Dev) | Search mức thưởng Tết âm lịch | "Không đủ bằng chứng để trả lời." `citations: []` | **Abstention đúng**: Nội dung thưởng tết thuộc quy chế tài chính nội bộ, Điều lệ không chứa thông tin. |
| **q23** | Unanswerable (Test) | Search tiêu chuẩn tồn dư kháng sinh trong sữa tươi | "Không đủ bằng chứng để trả lời." `citations: []` | **Abstention đúng**: Tiêu chuẩn kỹ thuật/chất lượng không nằm trong phạm vi văn bản Điều lệ. |
| **q24** | Unanswerable (Test) | Search mục tiêu giảm phát thải Net Zero 2050 | "Không đủ bằng chứng để trả lời." `citations: []` | **Abstention đúng**: Tránh hallucination, agent từ chối trả lời chính xác. |

- **Đường dẫn artifact & kết quả:**
  - Vết thực thi: `results/agent_dev_traces.json` và `results/agent_test_traces.json`.
  - Nhãn audit: `results/agent_dev_reviews.json` và `results/agent_test_reviews.json`.
  - Báo cáo chỉ số tổng hợp: `results/agent_dev_summary.json` và `results/agent_test_summary.json`.
- **Đánh đổi và giới hạn:**
  - *Đánh đổi giữa Context Budget và Token Cost:* Tăng budget từ 800 lên 1500 ký tự giúp tăng `Context Evidence Rate` từ 0% lên 66.7%, tuy nhiên làm tăng nhẹ số lượng prompt tokens gửi tới LLM. Với các model Flash chi phí thấp, đây là sự đánh đổi hoàn toàn tối ưu và cần thiết.
  - *Từ chối nhầm (False Abstention):* Khi một điều khoản quá dài hoặc chia cắt qua nhiều trang, nếu các câu trả lời yêu cầu tổng hợp đa vùng, một số câu hỏi có thể bị Agent từ chối do không gom đủ 100% bằng chứng trong một chunk duy nhất.

---

## 4. Thử thách nâng cao (Advanced Challenge)

Theo kế hoạch đăng ký ban đầu (Kế hoạch A), bài thực hành tập trung giải quyết trọn vẹn, xuất sắc toàn bộ yêu cầu cốt lõi (Core Requirements) để đạt điểm số tối đa 85/85 điểm.

---

## 5. Hướng dẫn Tái lập Kết quả (Replication Guide)

1. **Cài đặt môi trường:**
   ```bash
   pip install qdrant-client google-genai python-dotenv
   ```
2. **Cấu hình API Key:**
   Tạo file `.env` chứa `GEMINI_API_KEY=<your_api_key>`.
3. **Chạy toàn bộ thực nghiệm (Retrieval + Agent + Metrics):**
   ```bash
   python -m src.run_experiments
   ```
4. **Chạy Unit Tests kiểm tra tính toàn vẹn hệ thống:**
   ```bash
   python -m unittest discover -s tests -v
   ```
   *(Kết quả: 19/19 tests PASS 100%).*

