# src/hybrid_retriever.py
from typing import Optional
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from src.config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION, EMBEDDING_MODEL
from src.db_postgres import fetch_metadata_by_ids

_model: Optional[SentenceTransformer] = None
_client: Optional[QdrantClient] = None

def get_model():
    global _model
    if _model is None:
        print(f"🚀 Loading model {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def get_client() -> QdrantClient:
    global _client
    if _client is None:
        print("🔗 Connecting Qdrant…")
        _client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            prefer_grpc=False,
            timeout=120,
            check_compatibility=False,
        )
    return _client

def hybrid_search(query: str, k: int = 5):
    model = get_model()
    client = get_client()
    vec = model.encode(query, normalize_embeddings=True).tolist()
    hits = client.search(collection_name=QDRANT_COLLECTION, query_vector=vec, limit=k)

    ids = [h.id for h in hits]
    try:
        metas = fetch_metadata_by_ids(ids)
    except Exception:
        metas = {}

    results = []
    for h in hits:
        p = h.payload or {}
        results.append({
            "id": h.id,
            "score": h.score,
            "text": p.get("text") or p.get("text_excerpt", ""),
            "source": metas.get(h.id, {}).get("source", p.get("source")),
            "bmi": metas.get(h.id, {}).get("bmi_category"),
            "gender": metas.get(h.id, {}).get("gender"),
            "goal": metas.get(h.id, {}).get("goal"),
        })
    return results
