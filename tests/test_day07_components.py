"""Unit tests for Day 07 custom components: retrievers and controller logic."""
import unittest
from unittest.mock import MagicMock
from src.retrievers import LexicalRetriever, HybridRetriever
from src.controller import GeminiController


class Day07ComponentsTest(unittest.TestCase):
    def setUp(self):
        self.mock_chunks = [
            {"id": "chunk_1", "text": "Trụ sở chính tại số 10 Tân Trào, Quận 7.", "source_id": "doc1", "source": "p.pdf", "page": 1, "metadata": {}},
            {"id": "chunk_2", "text": "Hội đồng quản trị gồm từ 5 đến 11 thành viên.", "source_id": "doc1", "source": "p.pdf", "page": 2, "metadata": {}},
            {"id": "chunk_3", "text": "Vốn điều lệ của công ty được ghi nhận tại Điều lệ.", "source_id": "doc1", "source": "p.pdf", "page": 3, "metadata": {}},
        ]

    def test_lexical_retriever(self):
        retriever = LexicalRetriever(self.mock_chunks)
        results = retriever.search("trụ sở chính", top_k=2)
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "chunk_1")

    def test_hybrid_retriever_rrf(self):
        mock_dense = MagicMock()
        mock_dense.search.return_value = [
            {"id": "chunk_2", "score": 0.9, "text": "Hội đồng", "source_id": "doc1", "source": "p.pdf", "page": 2, "metadata": {}},
            {"id": "chunk_1", "score": 0.8, "text": "Trụ sở", "source_id": "doc1", "source": "p.pdf", "page": 1, "metadata": {}},
        ]
        lexical = LexicalRetriever(self.mock_chunks)
        hybrid = HybridRetriever(mock_dense, lexical)
        
        results = hybrid.search("trụ sở", top_k=2)
        self.assertLessEqual(len(results), 2)
        retrieved_ids = {r["id"] for r in results}
        self.assertIn("chunk_1", retrieved_ids)

    def test_controller_initial_search(self):
        controller = GeminiController.__new__(GeminiController)
        decision = controller.decide("Câu hỏi kiểm tra", history=[])
        self.assertEqual(decision["action"], "search")
        self.assertEqual(decision["query"], "Câu hỏi kiểm tra")

    def test_controller_empty_context_abstains(self):
        controller = GeminiController.__new__(GeminiController)
        history = [{"tool": "search", "context": "", "context_ids": []}]
        decision = controller.decide("Câu hỏi kiểm tra", history=history)
        self.assertEqual(decision["action"], "answer")
        self.assertEqual(decision["citations"], [])


if __name__ == "__main__":
    unittest.main()

