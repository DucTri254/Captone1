# src/net_utils.py
import os
import time
import json
import socket
import uuid
from pathlib import Path
from urllib.parse import urlparse
from typing import List, Dict, Any

# Thư mục lưu spool (có thể đổi bằng .env SPOOL_DIR)
SPOOL_DIR = Path(os.getenv("SPOOL_DIR", "spool_qdrant")).resolve()
SPOOL_DIR.mkdir(parents=True, exist_ok=True)

def dns_warmup(qdrant_url: str, tries: int = 5, delay: float = 1.0) -> None:
    """
    Kiểm tra DNS sớm để phát hiện mạng yếu.
    Nếu DNS không resolve được sau 'tries' lần → ném lỗi RuntimeError.
    """
    host = urlparse(qdrant_url).hostname
    if not host:
        return
    last_err = None
    for _ in range(tries):
        try:
            socket.getaddrinfo(host, 443)
            return
        except Exception as e:
            last_err = e
            time.sleep(delay)
            delay = min(delay * 1.8, 8.0)
    raise RuntimeError(f"DNS cannot resolve {host}: {last_err}")

def spool_points(collection: str, points: List[Dict[str, Any]]) -> Path:
    """
    Ghi batch points ra file .ndjson để replay sau:
    Mỗi dòng: {"collection":..., "points":[{id,vector,payload}, ...]}
    """
    fname = SPOOL_DIR / f"{int(time.time())}-{uuid.uuid4().hex}.ndjson"
    with fname.open("w", encoding="utf-8") as f:
        rec = {"collection": collection, "points": points}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return fname

def replay_spool(qdrant_client, collection: str) -> int:
    """
    Đẩy lại toàn bộ file spool (*.ndjson) còn tồn trong thư mục lên Qdrant.
    Trả về số file đã replay thành công.
    """
    ok = 0
    files = sorted(SPOOL_DIR.glob("*.ndjson"))
    for fp in files:
        try:
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    pts = rec.get("points", [])
                    if pts:
                        step = 256
                        for i in range(0, len(pts), step):
                            qdrant_client.upsert(collection_name=collection, points=pts[i:i+step], wait=True)
            fp.unlink(missing_ok=True)
            ok += 1
        except Exception as e:
            print(f"⚠️ Replay failed for {fp.name}: {e}")
            # Giữ lại file để lần sau thử lại
            pass
    return ok
