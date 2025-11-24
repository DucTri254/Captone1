# src/utils.py
import hashlib
import socket
import redis
import certifi
from typing import Optional
from urllib.parse import urlparse, urlunparse
from src.config import REDIS_URL

def md5_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

_redis_client: Optional[redis.Redis] = None

def _force_rediss(url: str) -> str:
    try:
        u = urlparse(url)
        if u.scheme == "redis":
            u = u._replace(scheme="rediss")
            return urlunparse(u)
    except Exception:
        pass
    return url

def connect_redis() -> Optional[redis.Redis]:
    """
    Kết nối Redis theo đúng URL trong .env:
    - Nếu URL là redis://  -> non-TLS
    - Nếu URL là rediss:// -> TLS
    Không tự ép chuyển scheme để tránh sai với endpoint của bạn.
    """
    global _redis_client
    if _redis_client is not None:
        return _redis_client

    if not REDIS_URL:
        print("⚠️ Redis URL not found in .env")
        return None

    try:
        u = urlparse(REDIS_URL)
        # Tạo client đúng như URL cung cấp (from_url tự cấu hình ssl khi là rediss://)
        client = redis.from_url(
            REDIS_URL,
            decode_responses=True,
            # Không set ssl/ssl_ca_certs ở đây — from_url sẽ tự xử lý theo scheme
            socket_timeout=5,
            socket_connect_timeout=5,
        )
        client.ping()
        print(f"✅ Connected to Redis via URL ({u.scheme})")
        _redis_client = client
        return _redis_client
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return None

def cache_get(key: str):
    r = connect_redis()
    if not r: return None
    try: return r.get(key)
    except Exception: return None

def cache_set(key: str, value: str, ttl: int = 3600):
    r = connect_redis()
    if not r: return
    try: r.setex(key, ttl, value)
    except Exception as e: print(f"⚠️ Cache set failed: {e}")

def split_text(text: str, chunk_size: int = 480, overlap: int = 100):
    if not text: return []
    s = str(text).strip()
    if not s: return []
    chunks, step = [], max(1, chunk_size - overlap)
    for i in range(0, len(s), step):
        chunks.append(s[i:i+chunk_size])
    return chunks
