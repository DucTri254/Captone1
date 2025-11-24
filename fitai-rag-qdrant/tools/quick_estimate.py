# tools/quick_estimate.py
import os
import math
import argparse
import pandas as pd
from pathlib import Path
from typing import Iterable, List, Tuple
from dotenv import load_dotenv

# Dùng lại splitter giống build_index
from src.utils import split_text
from src.config import CSV_PATHS  # đọc các path từ .env

# ---------- HyperLogLog (ước lượng số phần tử duy nhất) ----------
# Triển khai HLL nhỏ gọn, không phụ thuộc lib ngoài
import hashlib
class HLL:
    # p=14 -> m=16384 bộ đếm, sai số ~1.04/sqrt(m) ~ 0.8%
    def __init__(self, p: int = 14):
        assert 4 <= p <= 18
        self.p = p
        self.m = 1 << p
        self.reg = [0] * self.m

    @staticmethod
    def _hash64(x: bytes) -> int:
        # sha1 -> lấy 64-bit
        return int.from_bytes(hashlib.sha1(x).digest()[:8], "big", signed=False)

    def add(self, s: str):
        h = self._hash64(s.encode("utf-8"))
        idx = h >> (64 - self.p)               # index register
        w = (h << self.p) & ((1 << 64) - 1)    # remaining bits
        # ρ(w): số bit 0 từ trái + 1
        rank = 1
        while w & (1 << 63) == 0 and rank <= 64 - self.p:
            rank += 1
            w = (w << 1) & ((1 << 64) - 1)
        if rank > self.reg[idx]:
            self.reg[idx] = rank

    def count(self) -> float:
        m = self.m
        alpha_m = 0.7213 / (1 + 1.079 / m)
        Z = sum(2.0 ** (-r) for r in self.reg)
        E = alpha_m * (m ** 2) / Z

        # small-range correction
        V = self.reg.count(0)
        if E <= 2.5 * m and V > 0:
            E = m * math.log(m / V)

        # large-range correction không quá cần thiết cho data cỡ vừa
        return E

# ----------------- Ước lượng chunk & unique -----------------
def yield_chunks_from_row(values, chunk_size: int, chunk_overlap: int, min_len: int) -> Iterable[str]:
    text = " | ".join([str(v) for v in values if str(v).strip().lower() not in ("nan", "", "none")])
    if not text:
        return
    for ch in split_text(text, chunk_size, chunk_overlap):
        if len(ch) >= min_len:
            yield ch

def estimate_file(csv_path: str, chunk_size: int, chunk_overlap: int, min_len: int, csv_chunk_rows: int) -> Tuple[int,int,float]:
    """
    Trả về:
      - rows: số dòng
      - total_chunks: tổng chunk sinh ra (chưa khử trùng)
      - unique_chunks_est: ước lượng số chunk duy nhất (HLL)
    """
    # Đếm rows nhanh
    try:
        # dùng iterator để không ăn RAM
        rows = 0
        total_chunks = 0
        hll = HLL(p=14)

        try:
            reader = pd.read_csv(csv_path, chunksize=csv_chunk_rows)
        except UnicodeDecodeError:
            reader = pd.read_csv(csv_path, chunksize=csv_chunk_rows, encoding="latin-1")

        for df_chunk in reader:
            rows += len(df_chunk)
            for row in df_chunk.itertuples(index=False, name=None):
                for ch in yield_chunks_from_row(row, chunk_size, chunk_overlap, min_len):
                    total_chunks += 1
                    hll.add(ch)  # ước lượng unique
        unique_est = hll.count()
        return rows, total_chunks, unique_est
    except Exception as e:
        print(f"⚠️ Estimate failed for {os.path.basename(csv_path)}: {e}")
        return 0, 0, 0.0

def fmt_int(x) -> str:
    try:
        return f"{int(x):,}"
    except:
        return str(x)

def main():
    load_dotenv()
    ap = argparse.ArgumentParser(description="Quick estimate for indexing cost/time.")
    ap.add_argument("--chunk_size", type=int, default=int(os.getenv("CHUNK_SIZE", "700")))
    ap.add_argument("--chunk_overlap", type=int, default=int(os.getenv("CHUNK_OVERLAP", "60")))
    ap.add_argument("--min_chunk_len", type=int, default=int(os.getenv("MIN_CHUNK_LEN", "90")))
    ap.add_argument("--csv_chunk_rows", type=int, default=int(os.getenv("CSV_CHUNK_ROWS", "20000")))
    ap.add_argument("--eps", type=float, default=float(os.getenv("EMBED_EPS", "120.0")),
                    help="Embeddings per second (mặc định 120 vec/s ~ CPU quantized). Ví dụ GPU 3050: 350–550")
    ap.add_argument("--qps", type=float, default=float(os.getenv("QDRANT_UPSERT_QPS", "200.0")),
                    help="Qdrant upserts per second (payload batch). Tham số ước lượng.")
    ap.add_argument("--per_point_ms", type=float, default=float(os.getenv("QDRANT_PER_POINT_MS", "0.10")),
                    help="Thời gian mỗi point (ms) khi upsert, ước lượng tổng thời gian.")
    ap.add_argument("--show_each", action="store_true", help="Hiển thị chi tiết từng file")
    args = ap.parse_args()

    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap
    min_len = args.min_chunk_len
    csv_chunk_rows = args.csv_chunk_rows

    # Tập CSV từ config
    paths: List[str] = [p for p in CSV_PATHS if p and os.path.exists(p)]
    if not paths:
        print("❌ Không tìm thấy CSV. Kiểm tra .env / đường dẫn.")
        for idx, p in enumerate(CSV_PATHS, 1):
            print(f"  {idx}. {p}")
        return

    print("=== QUICK ESTIMATE (no embeddings done) ===")
    print(f"CHUNK_SIZE={chunk_size}, OVERLAP={chunk_overlap}, MIN_LEN={min_len}, CSV_CHUNK_ROWS={csv_chunk_rows}")
    print(f"Assumed throughput: EPS={args.eps:.1f} vec/s | Qdrant per-point={args.per_point_ms:.2f} ms")

    grand_rows = 0
    grand_chunks = 0
    grand_unique = 0.0

    per_file = []
    for p in paths:
        rows, total_chunks, unique_est = estimate_file(p, chunk_size, chunk_overlap, min_len, csv_chunk_rows)
        grand_rows += rows
        grand_chunks += total_chunks
        grand_unique += unique_est  # gộp theo *ước lượng trên từng file* (xấp xỉ nhẹ)
        per_file.append((os.path.basename(p), rows, total_chunks, unique_est))

    if args.show_each:
        print("\nPer-file:")
        print(f"{'File':45s} | {'Rows':>10s} | {'Chunks_raw':>12s} | {'Unique_est':>12s}")
        print("-"*90)
        for name, rows, chunks, uniq in per_file:
            print(f"{name:45s} | {fmt_int(rows):>10s} | {fmt_int(chunks):>12s} | {fmt_int(round(uniq)):>12s}")

    # Ước lượng thời gian
    # Thời gian embed ~ unique vectors / EPS
    est_embed_sec = (grand_unique / max(1.0, args.eps))
    # Thời gian upsert ~ unique vectors * per_point_ms
    est_upsert_sec = (grand_unique * (args.per_point_ms / 1000.0))

    total_sec = est_embed_sec + est_upsert_sec

    def hms(sec: float) -> str:
        sec = int(sec)
        h = sec // 3600
        m = (sec % 3600) // 60
        s = sec % 60
        if h > 0:
            return f"{h}h {m}m {s}s"
        if m > 0:
            return f"{m}m {s}s"
        return f"{s}s"

    print("\n=== SUMMARY ===")
    print(f"Files:                     {len(paths)}")
    print(f"Total rows:                {fmt_int(grand_rows)}")
    print(f"Total chunks (raw):        {fmt_int(grand_chunks)}")
    print(f"Unique chunks (HLL est.):  ~{fmt_int(round(grand_unique))}")
    print(f"Embed time (est):          {hms(est_embed_sec)}  @ {args.eps:.1f} vec/s")
    print(f"Qdrant upsert time (est):  {hms(est_upsert_sec)}  @ {args.per_point_ms:.2f} ms/point")
    print(f"TOTAL (est):               {hms(total_sec)}")
    print("\nNote:")
    print("- Unique_est là ước lượng ≈0.8% sai số (HLL p=14).")
    print("- Thời gian thực tế phụ thuộc batch-size, mạng và Qdrant rate-limit.")
    print("- Điều chỉnh --eps nếu bạn chạy GPU (ví dụ: --eps 450).")

if __name__ == "__main__":
    main()
