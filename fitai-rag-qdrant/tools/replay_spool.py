# tools/replay_spool.py
import os
from qdrant_client import QdrantClient
from src.config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION
from src.net_utils import replay_spool

if __name__ == "__main__":
    print("🔄 Replaying local spool batches to Qdrant...")
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        prefer_grpc=False,
        timeout=300,
        check_compatibility=False
    )
    n = replay_spool(client, QDRANT_COLLECTION)
    print(f"✅ Replayed {n} spool file(s) successfully.")
