# src/db_postgres.py
import time
from typing import List, Dict, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, DBAPIError

from src.config import POSTGRES_URL

# -----------------------------
# Engine với pool + keepalive
# -----------------------------
_engine = None

def _build_engine():
    # Neon yêu cầu SSL qua URL (?sslmode=require).
    # Bật keepalive + pre_ping để tránh "stale connection".
    return create_engine(
        POSTGRES_URL,
        future=True,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        pool_recycle=300,  # tái chế kết nối mỗi 5 phút
        connect_args={
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
        },
    )

def get_engine():
    global _engine
    if _engine is None:
        _engine = _build_engine()
    return _engine

def _reset_engine():
    global _engine
    try:
        if _engine is not None:
            _engine.dispose()
    finally:
        _engine = None


# -----------------------------
# Helpers: retry + backoff
# -----------------------------
def _exec_with_retry(fn, *args, retries: int = 4, backoff: float = 0.8, **kwargs):
    """
    Thực thi hàm DB với retry khi gặp lỗi kết nối (SSL closed, Connection reset, v.v.).
    """
    delay = 0.5
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except (OperationalError, DBAPIError) as e:
            msg = str(e).lower()
            transient = any(
                s in msg
                for s in [
                    "ssl connection has been closed",
                    "server closed the connection",
                    "connection reset",
                    "connection aborted",
                    "could not connect",
                    "timeout",
                ]
            )
            if attempt == retries or not transient:
                # Hết retry hoặc không phải lỗi tạm thời -> ném ra
                raise
            # Reset engine + backoff rồi thử lại
            _reset_engine()
            time.sleep(delay)
            delay *= (1.0 + backoff)


# -----------------------------
# Schema
# -----------------------------
def init_metadata_table():
    def _work():
        eng = get_engine()
        with eng.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS fitai_metadata(
                    id TEXT PRIMARY KEY,
                    text_excerpt TEXT,
                    source TEXT,
                    bmi_category TEXT,
                    gender TEXT,
                    goal TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """))
    _exec_with_retry(_work)


# -----------------------------
# Insert theo batch + retry
# -----------------------------
def insert_metadata(items: List[Dict], batch_size: int = 500):
    """
    Chèn metadata theo batch, ON CONFLICT DO NOTHING để tránh trùng id.
    Có retry & reset engine khi gặp lỗi kết nối.
    items: [{id, excerpt, source, bmi, gender, goal}, ...]
    """
    if not items:
        return

    def _insert_batch(batch: List[Dict]):
        eng = get_engine()
        with eng.begin() as conn:
            conn.execute(text("""
                INSERT INTO fitai_metadata (id, text_excerpt, source, bmi_category, gender, goal)
                VALUES (:id, :excerpt, :source, :bmi, :gender, :goal)
                ON CONFLICT (id) DO NOTHING;
            """), batch)

    # Chia nhỏ và chèn lần lượt (mỗi batch có retry riêng)
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        _exec_with_retry(_insert_batch, batch)


# -----------------------------
# Truy vấn hỗ trợ lọc/hiển thị
# -----------------------------
def get_ids_by_filters(
    bmi: Optional[str] = None,
    gender: Optional[str] = None,
    goal: Optional[str] = None,
    limit: int = 5000
) -> List[str]:
    """
    Trả về danh sách point-id thỏa bộ lọc metadata (giới hạn 'limit' để tránh filter quá dài).
    """
    def _work():
        eng = get_engine()
        conds = []
        params = {}
        if bmi:
            conds.append("bmi_category = :bmi")
            params["bmi"] = bmi
        if gender:
            conds.append("gender = :gender")
            params["gender"] = gender
        if goal:
            conds.append("goal = :goal")
            params["goal"] = goal
        where = ("WHERE " + " AND ".join(conds)) if conds else ""
        sql = f"SELECT id FROM fitai_metadata {where} LIMIT :lim"
        params["lim"] = limit
        with eng.begin() as conn:
            rows = conn.execute(text(sql), params).scalars().all()
            return list(rows)
    return _exec_with_retry(_work)

def get_metadata_by_ids(ids: List[str]) -> List[Dict]:
    """
    Lấy lại excerpt/source theo danh sách id để bổ sung khi payload Qdrant thiếu.
    """
    if not ids:
        return []

    def _work():
        eng = get_engine()
        with eng.begin() as conn:
            rows = conn.execute(text("""
                SELECT id, text_excerpt, source, bmi_category, gender, goal
                FROM fitai_metadata
                WHERE id = ANY(:ids)
            """), {"ids": ids}).mappings().all()
            return [dict(r) for r in rows]

    return _exec_with_retry(_work)
