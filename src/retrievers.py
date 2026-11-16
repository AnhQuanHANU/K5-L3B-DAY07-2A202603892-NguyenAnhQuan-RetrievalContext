"""Retrieval adapters for Day 7: Dense (Qdrant), Lexical (TF-IDF), and Hybrid (RRF)."""
import os
import math
import unicodedata
import re
import json
import time
from collections import Counter
from typing import Any
from pathlib import Path
from dotenv import load_dotenv

import qdrant_client
from google import genai
from google.genai import types

load_dotenv()


def tokenize(text: str) -> list[str]:
    """Tokenize Vietnamese text for lexical matching."""
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())


class DenseRetriever:
    """Dense retriever querying Day 06 Qdrant collection with Gemini embeddings."""

    def __init__(
        self,
        storage_path: str = "./data/qdrant_db",
        collection_name: str = "strategy_c",
        model_name: str = "gemini-embedding-001",
        dimensions: int = 768,
    ) -> None:
        self.storage_path = str(storage_path)
        self.collection_name = collection_name
        self.model_name = model_name
        self.dimensions = dimensions
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        self.client = genai.Client(api_key=api_key)
        self.qdrant = qdrant_client.QdrantClient(path=self.storage_path)
        self.cache_path = Path("./artifacts/query_embeddings_cache.json")
        self.cache: dict[str, list[float]] = {}
        if self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                self.cache = {}

    def embed_query(self, query: str) -> list[float]:
        q_norm = unicodedata.normalize("NFC", query.strip())
        if q_norm in self.cache:
            return self.cache[q_norm]

        for attempt in range(5):
            try:
                res = self.client.models.embed_content(
                    model=self.model_name,
                    contents=query,
                    config=types.EmbedContentConfig(output_dimensionality=self.dimensions),
                )
                vec = res.embeddings[0].values
                self.cache[q_norm] = vec
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.cache_path.write_text(
                    json.dumps(self.cache, ensure_ascii=False), encoding="utf-8"
                )
                time.sleep(0.5)
                return vec
            except Exception as e:
                if attempt == 4:
                    raise e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError("Failed to embed query after retries")

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        query_vec = self.embed_query(query)
        
        qdrant_filter = None
        if filters:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            if conditions:
                qdrant_filter = Filter(must=conditions)

        pts = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        ).points

        results: list[dict[str, Any]] = []
        for p in pts:
            payload = p.payload or {}
            results.append({
                "id": payload.get("chunk_id", str(p.id)),
                "text": payload.get("text", ""),
                "source_id": payload.get("document_id", "vinamilk_charter_2026"),
                "source": payload.get("source", "data/vinamilk_charter_2026.pdf"),
                "page": payload.get("page", 0),
                "metadata": {
                    "chapter": payload.get("chapter", ""),
                    "strategy": payload.get("strategy", "dense"),
                },
                "score": float(p.score) if p.score is not None else 0.0,
            })
        return results

    def close(self):
        try:
            self.qdrant.close()
        except Exception:
            pass


class LexicalRetriever:
    """TF-IDF lexical retriever."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        counts = [Counter(tokenize(c["text"])) for c in chunks]
        df = Counter(term for row in counts for term in row)
        self.idf = {
            term: math.log((1 + len(chunks)) / (1 + count)) + 1
            for term, count in df.items()
        }
        self.vectors = [self._vector(row) for row in counts]

    def _vector(self, counts: Counter) -> dict[str, float]:
        weights = {term: count * self.idf.get(term, 0.0) for term, count in counts.items()}
        norm = math.sqrt(sum(v * v for v in weights.values()))
        return {term: value / norm for term, value in weights.items()} if norm else {}

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        q_counts = Counter(tokenize(query))
        q_vec = self._vector(q_counts)
        results: list[dict[str, Any]] = []
        for chunk, vector in zip(self.chunks, self.vectors):
            # Apply filter if specified
            if filters:
                meta = chunk.get("metadata", {})
                match = all(meta.get(k) == v for k, v in filters.items())
                if not match:
                    continue
            score = sum(val * vector.get(term, 0.0) for term, val in q_vec.items())
            if score > 0:
                results.append({**chunk, "score": score, "strategy": "lexical"})
        return sorted(results, key=lambda c: (-c["score"], c["id"]))[:top_k]


class HybridRetriever:
    """Hybrid Retriever combining Dense and Lexical using Reciprocal Rank Fusion (RRF)."""

    def __init__(self, dense: DenseRetriever, lexical: LexicalRetriever, rrf_k: int = 60) -> None:
        self.dense = dense
        self.lexical = lexical
        self.rrf_k = rrf_k

    def search(
        self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        # Fetch more candidates from both retrievers
        candidate_k = max(top_k * 3, 10)
        dense_hits = self.dense.search(query, filters=filters, top_k=candidate_k)
        lexical_hits = self.lexical.search(query, filters=filters, top_k=candidate_k)

        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, dict[str, Any]] = {}

        for rank, hit in enumerate(dense_hits, start=1):
            cid = hit["id"]
            chunk_map[cid] = hit
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)

        for rank, hit in enumerate(lexical_hits, start=1):
            cid = hit["id"]
            if cid not in chunk_map:
                chunk_map[cid] = hit
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)

        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: (-rrf_scores[cid], cid))
        results: list[dict[str, Any]] = []
        for cid in sorted_cids[:top_k]:
            item = dict(chunk_map[cid])
            item["score"] = rrf_scores[cid]
            item["strategy"] = "hybrid_rrf"
            results.append(item)
        return results
