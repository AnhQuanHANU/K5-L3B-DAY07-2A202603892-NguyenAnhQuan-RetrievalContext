# Thông tin Học viên & Bài Nộp Day 7

- **Họ và tên:** Nguyễn Anh Quân
- **Mã sinh viên:** 2A202603892
- **Lớp:** K5-L3B
- **Chủ đề bài tập:** Retrieval & Context Engineering (Agentic RAG)
- **Repository URL:** https://github.com/AnhQuanHANU/K5-L3B-DAY07-2A202603892-NguyenAnhQuan-RetrievalContext
- **Báo cáo cá nhân:** [report/individual/2A202603892_NguyenAnhQuan.md](report/individual/2A202603892_NguyenAnhQuan.md)
- **Dữ liệu tài liệu:** Điều lệ Công ty Cổ phần Sữa Việt Nam 2026 (50 trang, 152 chunks, Qdrant Vector Store 768d, Gemini Embeddings)
- **Phương pháp Retrieval đã đánh giá:**
  1. Dense Retrieval (Qdrant + gemini-embedding-001)
  2. Lexical Retrieval (TF-IDF với Vietnamese Tokenizer)
  3. Hybrid Retrieval (Dense + TF-IDF với Reciprocal Rank Fusion RRF $k=60$)
- **Cải thiện cấu hình:** Tăng Context Budget từ 800 lên 1500 ký tự (nâng `Context Evidence Rate` từ 0% lên 66.7%).
- **Kết quả Agent trên Test:** 100% Tool use, 100% Citation validity, 100% Abstention trên câu không có bằng chứng.
