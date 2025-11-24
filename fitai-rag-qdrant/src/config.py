# src/config.py
import os
from dotenv import load_dotenv
load_dotenv()

# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "fitai_dataset")

# Postgres (Neon)
POSTGRES_URL = os.getenv("POSTGRES_URL")

# Redis (nếu bạn dùng cache sau này)
# Redis (ưu tiên REDIS_URL)
REDIS_URL = os.getenv("REDIS_URL")

# Models
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")

# RAG params
TOP_K = int(os.getenv("TOP_K", "6"))
CTX_BUDGET = int(os.getenv("CTX_BUDGET", "2200"))

# Enable advanced system
ENABLE_AUTO_QUERY_EXPANSION = True
ENABLE_PROFILE_ENGINE = True
ENABLE_SELF_CORRECT = True

# CSV paths
CSV_PATHS = [
    os.getenv("CSV_GYM"),
    os.getenv("CSV_PROGRAM_SUMMARY"),
    os.getenv("CSV_PROGRAMS_DETAILED"),
    os.getenv("CSV_FITNESS_DATASET"),
]
