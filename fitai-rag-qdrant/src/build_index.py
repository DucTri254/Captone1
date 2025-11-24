import os
from pathlib import Path  # phải import trước khi set HF_HOME

# Chặn import vision/audio của Transformers (tránh lôi torchvision)
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ["DISABLE_TRANSFORMERS_AV"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))

import time
import gc
from typing import List, Dict, Iterable, Set
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import httpx
import httpcore
from qdrant_client.http.exceptions import ResponseHandlingException

from src.db_postgres import init_metadata_table, insert_metadata
from src.config import (
    QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION,
    EMBEDDING_MODEL, CSV_PATHS
)
from src.utils import md5_id, split_text
from src.net_utils import dns_warmup, spool_points, replay_spool

# =========================
# TUNABLES (có thể override bằng .env)
# =========================
CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", "60"))
MIN_CHUNK_LEN  = int(os.getenv("MIN_CHUNK_LEN", "90"))

CSV_CHUNK_ROWS = int(os.getenv("CSV_CHUNK_ROWS", "20000"))
INIT_EMB_BATCH = int(os.getenv("INIT_EMB_BATCH", "128"))
MIN_EMB_BATCH  = int(os.getenv("MIN_EMB_BATCH", "16"))
UPSERT_BATCH   = int(os.getenv("UPSERT_BATCH", "768"))
META_BATCH     = int(os.getenv("META_BATCH", "1000"))

RETRY_TIMES    = int(os.getenv("RETRY_TIMES", "4"))
RETRY_BACKOFF  = float(os.getenv("RETRY_BACKOFF", "1.6"))

USE_ONNX_CPU   = os.getenv("USE_ONNX_CPU", "1") == "1"
USE_DQ_CPU     = os.getenv("USE_DQ_CPU", "0") == "1"  # Dynamic Quantization nếu không dùng ONNX
CPU_WORKERS    = int(os.getenv("CPU_WORKERS", "0"))    # 0 = auto
MAX_SEQ_LEN    = int(os.getenv("MAX_SEQ_LEN", "512"))

HF_HUB_OFFLINE = os.getenv("HF_HUB_OFFLINE", "0") == "1"

NET_ERRS = (
    httpx.ConnectError, httpx.ReadTimeout,
    httpcore.ConnectError, httpcore.ReadTimeout,
    ResponseHandlingException, TimeoutError, OSError
)

# =========================
# Encoder backends
# =========================
class GPUEncoder:
    """Encoder cho GPU sử dụng SentenceTransformer (BGE-M3) với adaptive batch."""
    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Loading embedding model: {model_name} on {self.device} (fp16={self.device=='cuda'})")
        self.model = SentenceTransformer(model_name, device=self.device)
        if self.device == "cuda":
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

    def adaptive_encode(self, texts: List[str], init_batch: int, min_batch: int = MIN_EMB_BATCH) -> List[List[float]]:
        batch = max(min_batch, init_batch)
        while True:
            try:
                with torch.inference_mode():
                    vecs = self.model.encode(
                        texts,
                        batch_size=batch,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        normalize_embeddings=True,
                    )
                return vecs.tolist()
            except RuntimeError as e:
                msg = str(e).lower()
                if "out of memory" in msg or "cuda" in msg:
                    if batch <= min_batch:
                        # Fallback CPU cho batch này
                        print("⚠️ CUDA OOM ở batch nhỏ nhất → thử CPU cho batch này.")
                        torch.cuda.empty_cache()
                        cpu_model = SentenceTransformer(self.model._first_module().auto_model.name_or_path, device="cpu")
                        with torch.inference_mode():
                            vecs = cpu_model.encode(
                                texts,
                                batch_size=max(8, min_batch // 2),
                                convert_to_numpy=True,
                                show_progress_bar=False,
                                normalize_embeddings=True,
                            )
                        del cpu_model
                        gc.collect()
                        return vecs.tolist()
                    batch = max(min_batch, batch // 2)
                    print(f"⚠️ OOM → giảm batch xuống {batch} và thử lại…")
                    torch.cuda.empty_cache()
                    continue
                raise

    def encode(self, texts: List[str], init_batch: int = INIT_EMB_BATCH) -> List[List[float]]:
        return self.adaptive_encode(texts, init_batch, MIN_EMB_BATCH)

    def dim(self) -> int:
        return len(self.encode(["probe"], init_batch=16)[0])

class ONNXEncoderCPU:
    """Encoder ONNXRuntime cho CPU (nhanh hơn PyTorch CPU)."""
    def __init__(self, model_name: str):
        print(f"⚡ Using ONNXRuntime backend on CPU for {model_name}")
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        # export=True: nếu chưa có ONNX thì export tự động
        self.model = ORTModelForFeatureExtraction.from_pretrained(model_name, export=True)
        self.max_len = MAX_SEQ_LEN

    @staticmethod
    def _mean_pooling(last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = (last_hidden_state * input_mask_expanded).sum(dim=1)
        sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, texts: List[str], init_batch: int = 512) -> List[List[float]]:
        out_vectors = []
        bs = max(64, min(init_batch * 2, 1024))  # CPU chịu batch lớn
        for i in range(0, len(texts), bs):
            batch_texts = texts[i:i+bs]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt"
            )
            with torch.inference_mode():
                outputs = self.model(**inputs)
                last_hidden = outputs.last_hidden_state
                pooled = self._mean_pooling(last_hidden, inputs["attention_mask"])
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)  # L2 normalize
                out_vectors.extend(pooled.cpu().tolist())
        return out_vectors

    def dim(self) -> int:
        return len(self.encode(["probe"], init_batch=64)[0])

class TorchCPUEncoderDQ:
    """Encoder CPU PyTorch + Dynamic Quantization (nếu không dùng ONNX)."""
    def __init__(self, model_name: str):
        print(f"⚙️ Using PyTorch CPU with dynamic quantization for {model_name}")
        self.st_model = SentenceTransformer(model_name, device="cpu")
        try:
            base = self.st_model._first_module().auto_model
            self.quant_model = torch.quantization.quantize_dynamic(
                base, {torch.nn.Linear}, dtype=torch.qint8
            )
            self.st_model._first_module().auto_model = self.quant_model
            print("✅ Applied dynamic quantization on Linear layers.")
        except Exception as e:
            print(f"⚠️ Dynamic quantization failed (continue without): {e}")

        os.environ["OMP_NUM_THREADS"] = str(max(1, (os.cpu_count() or 4)))
        os.environ["MKL_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]

    def encode(self, texts: List[str], init_batch: int = 512) -> List[List[float]]:
        return self.st_model.encode(
            texts,
            batch_size=max(64, min(init_batch, 512)),
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()

    def dim(self) -> int:
        return len(self.encode(["probe"], init_batch=64)[0])

def make_encoder(model_name: str):
    if torch.cuda.is_available():
        return GPUEncoder(model_name)
    # CPU fallback
    if USE_ONNX_CPU:
        try:
            return ONNXEncoderCPU(model_name)
        except Exception as e:
            print(f"⚠️ ONNX backend init failed: {e} → fallback to Torch CPU")
    if USE_DQ_CPU:
        return TorchCPUEncoderDQ(model_name)
    enc = SentenceTransformer(model_name, device="cpu")
    print(f"ℹ️ Using plain PyTorch CPU for {model_name}")
    return enc  # bare ST model (has .encode)

# =========================
# Qdrant helpers
# =========================
def ensure_collection(client: QdrantClient, dim: int):
    cols = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in cols:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )
        print(f"✅ Created Qdrant collection '{QDRANT_COLLECTION}' ({dim} dims)")
    else:
        print(f"⚙️ Collection '{QDRANT_COLLECTION}' already exists.")

def make_client() -> QdrantClient:
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        prefer_grpc=False,
        timeout=300,
        check_compatibility=False
    )

# =========================
# Pipeline funcs
# =========================
def yield_chunks_from_row(values) -> Iterable[str]:
    text = " | ".join([str(v) for v in values if str(v).strip().lower() not in ("nan", "", "none")])
    if not text:
        return
    for ch in split_text(text, CHUNK_SIZE, CHUNK_OVERLAP):
        if len(ch) >= MIN_CHUNK_LEN:
            yield ch

def upsert_with_retry(client: QdrantClient, points: List[PointStruct], collection: str) -> int:
    """
    Upsert với retry. Nếu hết retry mà vẫn lỗi mạng thì ghi spool để replay sau.
    Trả về số điểm đã 'xử lý' (upsert thành công hoặc đã spool).
    """
    if not points:
        return 0
    delay = 0.5
    for attempt in range(1, RETRY_TIMES + 1):
        try:
            client.upsert(collection_name=collection, points=points, wait=True)
            return len(points)
        except NET_ERRS as e:
            if attempt == RETRY_TIMES:
                # mạng không ổn -> ghi spool
                flat = []
                for p in points:
                    flat.append({"id": p.id, "vector": p.vector, "payload": p.payload})
                fp = spool_points(collection, flat)
                print(f"🟠 Network down → spooled {len(points)} pts to {fp.name}")
                return len(points)
            print(f"⚠️ upsert fail (attempt {attempt}/{RETRY_TIMES}): {e} → retry after {delay:.1f}s")
            time.sleep(delay)
            delay = min(delay * RETRY_BACKOFF, 12.0)
    return 0

def encode_batch(encoder, texts: List[str], init_batch=INIT_EMB_BATCH) -> List[List[float]]:
    if hasattr(encoder, "encode") and type(encoder).__name__ in ("GPUEncoder", "ONNXEncoderCPU", "TorchCPUEncoderDQ"):
        return encoder.encode(texts, init_batch)
    return encoder.encode(
        texts,
        batch_size=min(512, max(64, init_batch)),
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).tolist()

def get_dim(encoder) -> int:
    if hasattr(encoder, "dim"):
        return encoder.dim()
    return len(encode_batch(encoder, ["probe"], init_batch=16)[0])

def index_one_csv(csv_path: str, model_name: str) -> int:
    """Worker-safe: khởi tạo encoder/client/engine trong mỗi tiến trình."""
    source = os.path.basename(csv_path)
    print(f"\n🔹 Indexing {source}")
    total_vec = 0
    seen: Set[str] = set()
    meta_buffer: List[Dict] = []

    if HF_HUB_OFFLINE:
        os.environ["HF_HUB_OFFLINE"] = "1"

    # Encoder + Qdrant client
    encoder = make_encoder(model_name)
    client = make_client()

    # Trước khi làm gì, cố replay spool còn tồn
    try:
        replayed = replay_spool(client, QDRANT_COLLECTION)
        if replayed:
            print(f"🔁 Replayed {replayed} spool file(s) → Qdrant")
    except Exception:
        pass

    # Lấy dim và đảm bảo collection (mỗi worker check an toàn)
    dim = get_dim(encoder)
    ensure_collection(client, dim)

    # Đọc CSV theo lô
    try:
        reader = pd.read_csv(csv_path, chunksize=CSV_CHUNK_ROWS)
    except UnicodeDecodeError:
        reader = pd.read_csv(csv_path, chunksize=CSV_CHUNK_ROWS, encoding="latin-1")

    # Cho phép hạ batch khi thiếu RAM
    upsert_batch = UPSERT_BATCH
    emb_batch = INIT_EMB_BATCH

    for df_chunk in reader:
        to_embed: List[str] = []
        payloads: List[Dict] = []
        ids: List[str] = []

        for row in tqdm(df_chunk.itertuples(index=False, name=None), total=len(df_chunk), desc=source, leave=False):
            for ch in yield_chunks_from_row(row):
                pid = md5_id(ch)
                if pid in seen:
                    continue
                seen.add(pid)
                to_embed.append(ch)
                ids.append(pid)
                payloads.append({"text_excerpt": ch[:220], "source": source})

                # khi buffer lớn, encode & upsert
                if len(to_embed) >= emb_batch * 6:
                    try:
                        vectors = encode_batch(encoder, to_embed, emb_batch)
                    except MemoryError:
                        emb_batch = max(MIN_EMB_BATCH, emb_batch // 2)
                        upsert_batch = max(128, upsert_batch // 2)
                        print(f"⚠️ MemoryError → hạ INIT_EMB_BATCH={emb_batch}, UPSERT_BATCH={upsert_batch}")
                        vectors = encode_batch(encoder, to_embed, emb_batch)

                    points = [PointStruct(id=i, vector=v, payload=p) for i, v, p in zip(ids, vectors, payloads)]
                    for j in range(0, len(points), upsert_batch):
                        total_vec += upsert_with_retry(client, points[j:j + upsert_batch], QDRANT_COLLECTION)

                    # metadata
                    for ch_text, pid in zip(to_embed, ids):
                        meta_buffer.append({"id": pid, "excerpt": ch_text[:200], "source": source,
                                            "bmi": None, "gender": None, "goal": None})
                        if len(meta_buffer) >= META_BATCH:
                            insert_metadata(meta_buffer)
                            meta_buffer.clear()

                    to_embed.clear(); payloads.clear(); ids.clear()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    gc.collect()

        # Flush phần còn lại
        if to_embed:
            try:
                vectors = encode_batch(encoder, to_embed, emb_batch)
            except MemoryError:
                emb_batch = max(MIN_EMB_BATCH, emb_batch // 2)
                upsert_batch = max(128, upsert_batch // 2)
                print(f"⚠️ MemoryError → hạ INIT_EMB_BATCH={emb_batch}, UPSERT_BATCH={upsert_batch}")
                vectors = encode_batch(encoder, to_embed, emb_batch)

            points = [PointStruct(id=i, vector=v, payload=p) for i, v, p in zip(ids, vectors, payloads)]
            for j in range(0, len(points), upsert_batch):
                total_vec += upsert_with_retry(client, points[j:j + upsert_batch], QDRANT_COLLECTION)

            for ch_text, pid in zip(to_embed, ids):
                meta_buffer.append({"id": pid, "excerpt": ch_text[:200], "source": source,
                                    "bmi": None, "gender": None, "goal": None})

            to_embed.clear(); payloads.clear(); ids.clear()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gc.collect()

        if meta_buffer:
            insert_metadata(meta_buffer)
            meta_buffer.clear()

    print(f"✅ Done {source}: +{total_vec:,} vectors")
    return total_vec

def _safe_index_worker(p: str, model_name: str) -> int:
    try:
        return index_one_csv(p, model_name)
    except Exception as e:
        print(f"✖ Worker failed on {os.path.basename(p)}: {e.__class__.__name__}: {e}")
        return 0

def _mp_init():
    os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
    os.environ["DISABLE_TRANSFORMERS_AV"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =========================
# Orchestrator
# =========================
def build_index(csv_paths: List[str]):
    # Schema Postgres
    init_metadata_table()

    # Danh sách file
    base_paths = [p for p in csv_paths if p and os.path.exists(p)]
    if not base_paths:
        print("❌ No CSV files found. Check .env paths.")
        return

    # DNS warm-up (báo sớm nếu DNS lỗi)
    try:
        dns_warmup(QDRANT_URL, tries=6, delay=0.8)
        print("✅ DNS warm-up: OK")
    except Exception as e:
        print(f"❌ DNS warm-up failed: {e}")

    # Chuẩn bị ONNX trước (nếu dùng CPU + ONNX)
    if not torch.cuda.is_available() and USE_ONNX_CPU:
        try:
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            from transformers import AutoTokenizer
            AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
            ORTModelForFeatureExtraction.from_pretrained(EMBEDDING_MODEL, export=True)
            print("✅ ONNX pre-export done")
        except Exception as e:
            print(f"⚠️ ONNX pre-export failed (will export in worker): {e}")

    on_gpu = torch.cuda.is_available()
    if on_gpu:
        # GPU: chạy tuần tự (GPU là nút cổ chai)
        print("🟢 GPU detected → single-process indexing")
        total = 0
        for p in base_paths:
            total += index_one_csv(p, EMBEDDING_MODEL)
        print(f"\n🎯 TOTAL upserted: {total:,}")
        return

    # CPU fallback: chạy đa tiến trình theo file (Windows bắt buộc ở trong hàm)
    n_files = len(base_paths)
    if CPU_WORKERS <= 0:
        cpu_workers_eff = max(1, (os.cpu_count() or 4) // 2)
    else:
        cpu_workers_eff = max(1, CPU_WORKERS)

    workers = min(cpu_workers_eff, n_files)
    print(f"🟡 CPU fallback → multi-process with {workers} worker(s) for {n_files} file(s)")

    total = 0
    if workers == 1:
        for p in base_paths:
            total += _safe_index_worker(p, EMBEDDING_MODEL)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_mp_init) as pool:
            futs = {pool.submit(_safe_index_worker, p, EMBEDDING_MODEL): p for p in base_paths}
            for fut in as_completed(futs):
                try:
                    total += fut.result()
                except Exception as e:
                    print(f"✖ A worker crashed: {e.__class__.__name__}: {e}")

    print(f"\n🎯 TOTAL upserted: {total:,}")

if __name__ == "__main__":
    build_index(CSV_PATHS)
    