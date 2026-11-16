"""Minimalist Black & White Chat Interface for Day 07 Agentic RAG."""
import json
import os
import sys
from pathlib import Path
import streamlit as st

# Ensure root path is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.retrievers import DenseRetriever, LexicalRetriever, HybridRetriever
from src.controller import GeminiController
from starter.agentic import run_agent

# Page config: Minimalist, clean
st.set_page_config(
    page_title="Vinamilk Legal Agent",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom Minimalist Black & White Styling
st.markdown(
    """
    <style>
    /* Dark monochrome palette */
    .stApp {
        background-color: #0d0d0d;
        color: #e6e6e6;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header minimalist */
    .chat-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
        border-bottom: 1px solid #222222;
        margin-bottom: 1.5rem;
    }
    .chat-title {
        font-size: 1.4rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        color: #ffffff;
        margin: 0;
    }
    .chat-subtitle {
        font-size: 0.85rem;
        color: #888888;
        margin-top: 0.3rem;
    }
    
    /* Chat message container styling */
    .stChatMessage {
        background-color: transparent !important;
        border-bottom: 1px solid #1a1a1a;
        padding: 1rem 0;
    }
    
    /* Citation tags */
    .citation-badge {
        display: inline-block;
        background: #1f1f1f;
        color: #ffffff;
        border: 1px solid #333333;
        border-radius: 4px;
        padding: 1px 6px;
        font-size: 0.75rem;
        font-family: monospace;
        margin: 0 2px;
    }
    
    /* Status pills */
    .status-answered {
        display: inline-block;
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 12px;
        background-color: #1a2e1a;
        color: #4ade80;
        border: 1px solid #2e4a2e;
        font-weight: 500;
    }
    .status-abstain {
        display: inline-block;
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 12px;
        background-color: #2e1a1a;
        color: #f87171;
        border: 1px solid #4a2e2e;
        font-weight: 500;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #111111;
        border-right: 1px solid #222222;
    }
    
    /* Button styles */
    .stButton > button {
        background-color: #161616;
        color: #cccccc;
        border: 1px solid #2a2a2a;
        border-radius: 6px;
        font-size: 0.82rem;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #222222;
        color: #ffffff;
        border-color: #444444;
    }
    
    /* Chat input box */
    [data-testid="stChatInput"] {
        background-color: #141414;
        border: 1px solid #262626;
        border-radius: 8px;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_agent():
    """Load retrieval indexes and controller once to prevent Qdrant locking."""
    chunks_path = BASE_DIR / "data" / "chunks.json"
    if not chunks_path.exists():
        st.error("Không tìm thấy data/chunks.json. Vui lòng kiểm tra lại dữ liệu.")
        st.stop()
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    dense = DenseRetriever(storage_path=str(BASE_DIR / "data" / "qdrant_db"))
    lexical = LexicalRetriever(chunks)
    hybrid = HybridRetriever(dense, lexical)
    controller = GeminiController()
    return hybrid, controller


# Load resources
with st.spinner("Đang khởi tạo tài nguyên và kết nối Qdrant..."):
    retriever, controller = load_agent()

# Sidebar controls & info
with st.sidebar:
    st.markdown("### ⚙️ Cấu hình Agent")
    budget = st.slider("Context Budget (ký tự)", min_value=500, max_value=2500, value=1500, step=100)
    top_k = st.slider("Top Chunks", min_value=1, max_value=5, value=3, step=1)
    
    st.markdown("---")
    st.markdown("### 💡 Câu hỏi mẫu")
    sample_questions = [
        "Trụ sở chính của Vinamilk đặt ở đâu?",
        "Hội đồng quản trị có bao nhiêu thành viên?",
        "Vốn điều lệ của Công ty là bao nhiêu?",
        "Mức lương thai sản cho lao động nữ là bao nhiêu? (Câu bẫy)",
        "Thưởng Tết nguyên đán bao nhiêu tháng lương? (Câu bẫy)",
    ]
    for q in sample_questions:
        if st.button(q, key=f"btn_{q}"):
            st.session_state["queued_query"] = q

    st.markdown("---")
    if st.button("🗑️ Xóa đoạn hội thoại"):
        st.session_state["messages"] = []
        st.rerun()

# Header
st.markdown(
    """
    <div class="chat-header">
        <h1 class="chat-title">Vinamilk Legal Agentic RAG</h1>
        <p class="chat-subtitle">Tra cứu Điều lệ Vinamilk 2026 · Hybrid RRF · Trích dẫn chuẩn xác · Tự động từ chối</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Xin chào! Tôi là AI Agent tra cứu pháp lý Điều lệ Vinamilk. Bạn có thể hỏi tôi về các quy định, cơ cấu tổ chức, thẩm quyền hoặc vốn điều lệ.",
            "citations": [],
            "status": "answered",
            "events": [],
        }
    ]

# Display message history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            cits_html = " ".join([f"<span class='citation-badge'>{c}</span>" for c in msg["citations"]])
            st.markdown(f"<div style='margin-top: 6px;'><b>Nguồn:</b> {cits_html}</div>", unsafe_allow_html=True)
        if msg.get("status"):
            st_class = "status-answered" if msg["status"] == "answered" else "status-abstain"
            st_text = "Đã trả lời có căn cứ" if msg["status"] == "answered" else "Từ chối (Không đủ bằng chứng)"
            st.markdown(f"<span class='{st_class}'>{st_text}</span>", unsafe_allow_html=True)
        if msg.get("events"):
            with st.expander("🔎 Chi tiết tra cứu (Trace & Context)"):
                for e in msg["events"]:
                    if e.get("event") == "search":
                        st.markdown(f"**Query:** `{e.get('query')}`")
                        st.markdown(f"**Context IDs:** `{e.get('context_ids')}`")
                        st.text_area("Packed Context", e.get("context", ""), height=150, disabled=True, key=f"ctx_{msg['content'][:15]}")

# Handle query from input box or sample buttons
user_query = st.chat_input("Nhập câu hỏi của bạn về Điều lệ Vinamilk...")
if "queued_query" in st.session_state:
    user_query = st.session_state.pop("queued_query")

if user_query:
    # 1. Show user message
    st.session_state["messages"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Run Agent
    with st.chat_message("assistant"):
        with st.spinner("Agent đang tra cứu văn bản và thẩm định bằng chứng..."):
            res = run_agent(
                question=user_query,
                controller=controller,
                retriever=retriever,
                max_searches=3,
                top_k=top_k,
                budget=budget,
            )

        status = res["status"]
        answer = res["answer"]
        citations = res["citations"]
        events = res["events"]

        # Display assistant answer
        st.markdown(answer)

        if citations:
            cits_html = " ".join([f"<span class='citation-badge'>{c}</span>" for c in citations])
            st.markdown(f"<div style='margin-top: 6px;'><b>Nguồn:</b> {cits_html}</div>", unsafe_allow_html=True)

        st_class = "status-answered" if status == "answered" else "status-abstain"
        st_text = "Đã trả lời có căn cứ" if status == "answered" else "Từ chối (Không đủ bằng chứng)"
        st.markdown(f"<span class='{st_class}'>{st_text}</span>", unsafe_allow_html=True)

        if events:
            with st.expander("🔎 Chi tiết tra cứu (Trace & Context)"):
                for e in events:
                    if e.get("event") == "search":
                        st.markdown(f"**Query:** `{e.get('query')}`")
                        st.markdown(f"**Context IDs:** `{e.get('context_ids')}`")
                        st.text_area("Packed Context", e.get("context", ""), height=150, disabled=True)

        # Save to history
        st.session_state["messages"].append({
            "role": "assistant",
            "content": answer,
            "citations": citations,
            "status": status,
            "events": events,
        })
