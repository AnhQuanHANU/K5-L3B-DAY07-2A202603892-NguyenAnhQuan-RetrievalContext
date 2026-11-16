# Day 7 — Agentic RAG (Retrieval & Context Engineering)

**Học viên:** Nguyễn Anh Quân  
**Mã sinh viên:** `2A202603892`  
**Lớp:** `K5-L3B`  
**Báo cáo chi tiết:** [report/individual/2A202603892_NguyenAnhQuan.md](report/individual/2A202603892_NguyenAnhQuan.md)  
**Repository:** [https://github.com/AnhQuanHANU/K5-L3B-DAY07-2A202603892-NguyenAnhQuan-RetrievalContext](https://github.com/AnhQuanHANU/K5-L3B-DAY07-2A202603892-NguyenAnhQuan-RetrievalContext)

---

## 1. Tổng quan Dự án

Hệ thống **Agentic RAG** tra cứu tài liệu pháp lý Điều lệ Vinamilk 2026, tích hợp:
- **3 Phương pháp Retrieval:**
  1. *Dense Retriever:* Qdrant Vector Store + Gemini `gemini-embedding-001` (768 chiều).
  2. *Lexical Retriever:* TF-IDF Tokenizer tiếng Việt tối ưu cho đối sánh từ khóa chính xác.
  3. *Hybrid Retriever:* Kết hợp Dense và Lexical qua thuật toán **Reciprocal Rank Fusion (RRF)** ($k=60$).
- **Context Engineering:** Mở rộng Context Budget từ 800 lên 1500 ký tự, nâng tỷ lệ bảo toàn bằng chứng (`Context Evidence Rate`) từ **0.0% lên 66.7%** trên Dev.
- **Agent Controller:** Vòng lặp Agentic RAG tự động dựa trên Gemini LLM, hỗ trợ multi-model fallback (`gemini-3.5-flash`, `gemini-3.5-flash-lite`, `gemini-3.6-flash`, `gemini-3.1-flash-lite`) chống cạn kiệt hạn ngạch Free Tier (429 rate limit).
- **Abstention Handling:** Nhận diện và từ chối chính xác 100% (4/4) các câu hỏi bẫy không có bằng chứng pháp lý trong Điều lệ (lương thai sản, thưởng Tết, tiêu chuẩn kháng sinh, Net Zero).

---

## 2. Kết quả Thực nghiệm Tóm tắt

### 2.1. Đánh giá Retrieval (15 câu Dev, 5 câu Test)

| Cấu hình | Phương pháp | Split | Hit@1 | Hit@3 | Context Evidence Rate | Thời gian trung bình |
|---|---|---|---|---|---|---|
| **Baseline** | Dense (budget=800) | Dev | 10/15 (66.7%) | 11/15 (73.3%) | 0/15 (0.0%) | 0.66 ms |
| **Phương pháp 2** | Lexical TF-IDF (budget=800) | Dev | 11/15 (73.3%) | 14/15 (93.3%) | 0/15 (0.0%) | 0.54 ms |
| **Phương pháp 3** | Hybrid RRF (budget=800) | Dev | 10/15 (66.7%) | 14/15 (93.3%) | 0/15 (0.0%) | 1.28 ms |
| **Cải thiện 1** | Dense (budget=1500) | Dev | 10/15 (66.7%) | 11/15 (73.3%) | 10/15 (66.7%) | 0.59 ms |
| **Cải thiện 2 (Chosen)** | **Hybrid RRF (budget=1500)** | **Dev** | **10/15 (66.7%)** | **14/15 (93.3%)** | **10/15 (66.7%)** | **1.26 ms** |
| **Baseline trên Test** | Dense (budget=800) | Test | 2/5 (40.0%) | 2/5 (40.0%) | 0/5 (0.0%) | 0.67 ms |
| **Chosen trên Test** | **Hybrid RRF (budget=1500)** | **Test** | **2/5 (40.0%)** | **5/5 (100.0%)** | **2/5 (40.0%)** | **1.15 ms** |

### 2.2. Đánh giá Agent (`starter/agent_metrics.py`)

| Split | Tool Use | Grounded Answer (0/1/2) | Điểm TB | Citation hợp lệ | Câu có Citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| **Dev (17 cases)** | 17/17 (100%) | 0: 3 \| 1: 2 \| 2: 10 | **1.47 / 2.0** | 10/12 (83.3%) | 12/12 (100%) | **2/2 (100%)** | 3/15 (20%) |
| **Test (7 cases)** | 7/7 (100%) | 0: 3 \| 1: 0 \| 2: 2 | **0.80 / 2.0** | 2/2 (100%) | 2/2 (100%) | **2/2 (100%)** | 3/5 (60%) |

---

## 3. Cấu trúc Thư mục

```text
├── artifacts/
│   ├── embeddings_cache.json         # Cache 152 vector embeddings từ Day 06
│   └── query_embeddings_cache.json   # Cache vector nhúng các câu hỏi benchmark
├── data/
│   ├── cases.json                    # 24 test cases (17 Dev + 7 Test)
│   ├── chunks.json                   # 152 chunks cấu trúc Điều lệ Vinamilk
│   ├── manifest.json                 # Metadata văn bản
│   ├── questions.json                # 20 câu hỏi có gold evidence nguyên văn
│   └── vinamilk_charter_2026.pdf     # File PDF Điều lệ Vinamilk gốc
├── report/
│   └── individual/
│       └── 2A202603892_NguyenAnhQuan.md # Báo cáo cá nhân hoàn chỉnh
├── results/
│   ├── agent_dev_reviews.json        # Nhãn audit thủ công trên Dev
│   ├── agent_dev_summary.json        # Kết quả tổng hợp agent trên Dev
│   ├── agent_dev_traces.json         # Toàn bộ event trace agent trên Dev
│   ├── agent_test_reviews.json       # Nhãn audit thủ công trên Test
│   ├── agent_test_summary.json       # Kết quả tổng hợp agent trên Test
│   ├── agent_test_traces.json        # Toàn bộ event trace agent trên Test
│   └── retrieval_summary.json        # Bảng so sánh 3 phương pháp & 2 cải tiến
├── src/
│   ├── controller.py                 # Gemini Agent Controller & Multi-model failover
│   ├── retrievers.py                 # Dense, Lexical TF-IDF & Hybrid RRF
│   ├── run_experiments.py            # Kịch bản thực thi benchmark tự động
│   └── setup_store.py                # Khởi tạo nhanh Qdrant DB từ cache
├── starter/                          # Module chuẩn hóa từ BTC
└── tests/                            # 19 Unit Tests (100% PASS)
```

---

## 4. Hướng dẫn Chạy & Tái lập Kết quả

### 4.1. Cài đặt thư viện
```bash
pip install qdrant-client google-genai python-dotenv
```

### 4.2. Thiết lập API Key
Tạo file `.env` tại thư mục gốc:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4.3. Khởi tạo Qdrant Store (từ Cache, 0 API calls)
```bash
python src/setup_store.py
```

### 4.4. Chạy toàn bộ Benchmark & Thử nghiệm
```bash
python -m src.run_experiments
```

### 4.5. Chạy Unit Tests
```bash
python -m unittest discover -s tests -v
```
*(Kết quả: 19/19 tests PASS 100%).*

