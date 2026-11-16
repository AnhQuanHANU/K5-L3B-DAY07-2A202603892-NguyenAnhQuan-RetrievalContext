"""Gemini Agent Controller for Day 7 Agentic RAG."""
import os
import json
import re
from typing import Any
from google import genai
from dotenv import load_dotenv

load_dotenv()


class GeminiController:
    """Agent Controller utilizing Gemini LLM with multi-model failover to respect Free Tier quotas."""

    def __init__(self, models: list[str] | None = None) -> None:
        self.models = models or [
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.6-flash",
            "gemini-3.1-flash-lite"
        ]
        self.call_idx = 0
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        self.client = genai.Client(api_key=api_key)

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        # Turn 1: No search has been performed yet
        if not history:
            return {"action": "search", "query": question}

        # Later turns: evaluate current context
        latest_event = history[-1]
        context = latest_event.get("context", "")
        context_ids = latest_event.get("context_ids", [])

        if not context or not context_ids:
            return {"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}

        prompt = f"""Bạn là một AI Agent tra cứu pháp lý chuyên nghiệp của Công ty Cổ phần Sữa Việt Nam (Vinamilk).
CÂU HỎI: {question}

NGỮ CẢNH TÀI LIỆU TÌM ĐƯỢC (CONTEXT):
{context}

DANH SÁCH MÃ CHUNK HỢP LỆ: {json.dumps(context_ids)}

HÃY PHÂN TÍCH VÀ RA QUYẾT ĐỊNH:
1. Nếu trong Ngữ cảnh CÓ ĐỦ bằng chứng để trả lời câu hỏi:
   - Hãy trả lời chính xác, đầy đủ theo đúng ngữ cảnh được cung cấp.
   - BẮT BUỘC chèn mã chunk trích dẫn [chunk-id] ngay sau câu chứa thông tin (Ví dụ: "Trụ sở chính tại 10 Tân Trào [B_vinamilk_charter_2026_p07_c000].").
   - Trả về JSON:
     {{
       "action": "answer",
       "answer": "<nội dung câu trả lời có trích dẫn [chunk-id]>",
       "citations": ["<chunk-id trích dẫn>"]
     }}
2. Nếu trong Ngữ cảnh KHÔNG CÓ hoặc KHÔNG ĐỦ bằng chứng để trả lời câu hỏi (hoặc câu hỏi hỏi về các vấn đề ngoài Điều lệ như lương thai sản, thưởng tết, tiêu chuẩn kháng sinh, trung hòa carbon Net Zero):
   - Tuyệt đối KHÔNG tự bịa đặt câu trả lời.
   - Trả về JSON:
     {{
       "action": "answer",
       "answer": "Không đủ bằng chứng để trả lời.",
       "citations": []
     }}

CHỈ TRẢ VỀ DUY NHẤT MỘT KHỐI JSON HỢP LỆ THEO ĐỊNH DẠNG TRÊN:"""

        import time
        max_attempts = len(self.models) * 2
        for attempt in range(max_attempts):
            model = self.models[self.call_idx % len(self.models)]
            self.call_idx += 1
            try:
                resp = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={"response_mime_type": "application/json"},
                )
                raw_text = resp.text.strip()
                # Clean markdown codeblocks if any
                if raw_text.startswith("```"):
                    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
                    raw_text = re.sub(r"\s*```$", "", raw_text)
                decision = json.loads(raw_text)

                action = decision.get("action")
                if action == "search":
                    q = decision.get("query", question).strip()
                    time.sleep(1.5)
                    return {"action": "search", "query": q or question}
                elif action == "answer":
                    ans = decision.get("answer", "")
                    cits = decision.get("citations", [])
                    if "không đủ bằng chứng" in ans.lower() or not cits:
                        time.sleep(1.5)
                        return {"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}
                    
                    # Ensure citations only include valid context_ids
                    valid_cits = [c for c in cits if c in context_ids]
                    if not valid_cits:
                        time.sleep(1.5)
                        return {"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}
                    
                    # Ensure each citation appears in the answer text as [chunk-id]
                    for c in valid_cits:
                        tag = f"[{c}]"
                        if tag not in ans:
                            ans = f"{ans} {tag}"
                    time.sleep(1.5)
                    return {"action": "answer", "answer": ans.strip(), "citations": valid_cits}
                else:
                    time.sleep(1.5)
                    return {"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}
            except Exception as e:
                err_str = str(e)
                print(f"Controller LLM model {model} attempt {attempt+1} error: {err_str[:120]}")
                if "429" in err_str or "quota" in err_str.lower() or "resource_exhausted" in err_str.lower():
                    # If all models have been tried once, sleep to let quota reset
                    if (attempt + 1) % len(self.models) == 0:
                        print("All models rate-limited, cooling down for 25s...")
                        time.sleep(25)
                    else:
                        time.sleep(2)
                else:
                    time.sleep(2)

        return {"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}
