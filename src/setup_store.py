"""Initialize local Qdrant collection using cached chunk embeddings (0 API cost)."""
import json
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

def setup_store(storage_path="./data/qdrant_db", collection_name="strategy_c"):
    base_dir = Path(__file__).resolve().parents[1]
    store_dir = base_dir / storage_path
    chunks_file = base_dir / "data" / "chunks.json"
    cache_file = base_dir / "artifacts" / "embeddings_cache.json"

    chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
    cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}

    client = QdrantClient(path=str(store_dir))
    collections = [c.name for c in client.get_collections().collections]
    if collection_name in collections:
        print(f"Collection '{collection_name}' already exists in {store_dir}.")
        client.close()
        return

    print(f"Creating collection '{collection_name}' with 768d Cosine...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

    points = []
    for idx, c in enumerate(chunks):
        cid = c["id"]
        vec = cache.get(cid)
        if not vec:
            continue
        points.append(
            PointStruct(
                id=idx + 1,
                vector=vec,
                payload={
                    "chunk_id": cid,
                    "text": c["text"],
                    "document_id": c.get("source_id", "vinamilk_charter_2026"),
                    "source": c.get("source", "data/vinamilk_charter_2026.pdf"),
                    "page": c.get("page", 0),
                    "chapter": c.get("metadata", {}).get("chapter", ""),
                    "strategy": "dense",
                },
            )
        )

    client.upsert(collection_name=collection_name, points=points)
    print(f"Successfully populated {len(points)} points into '{collection_name}'.")
    client.close()

if __name__ == "__main__":
    setup_store()
