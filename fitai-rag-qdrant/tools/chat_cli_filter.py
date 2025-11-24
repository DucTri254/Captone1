# tools/chat_cli_filter.py
import os
import time
import torch
from typing import Optional, List

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, HasIdCondition, PointIdsList

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION, EMBEDDING_MODEL
from src.db_postgres import init_metadata_table, get_ids_by_filters, get_metadata_by_ids

# ---- Qdrant + Embedder ----
def make_qdrant() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=False, timeout=60)

def make_embedder():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    return model, device

# ---- LLM (HuggingFace - gọn nhẹ) ----
def make_llm(model_name: Optional[str] = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = model_name or os.getenv("LLM_MODEL", "Qwen/Qwen2.5-1.8B-Instruct")
    tok = AutoTokenizer.from_pretrained(name)
    dtype = torch.float16 if device == "cuda" else torch.float32
    mdl = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device)
    return tok, mdl, device

def generate_answer(tokenizer, model, device, prompt: str, max_new_tokens: int = 256) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # tách phần sau "Answer:" nếu có
    if "Answer:" in text:
        text = text.split("Answer:", 1)[-1].strip()
    return text.strip()

# ---- Retrieval có lọc bằng danh sách id từ Postgres ----
def retrieve_context(client: QdrantClient, embedder, query: str, top_k: int = 4,
                     bmi: Optional[str] = None, gender: Optional[str] = None, goal: Optional[str] = None):
    qvec = embedder.encode([query], normalize_embeddings=True)[0]

    allowed_ids: List[str] = get_ids_by_filters(bmi=bmi, gender=gender, goal=goal, limit=5000)
    qfilter = None
    if allowed_ids:
        # Qdrant filter theo id
        qfilter = Filter(must=[HasIdCondition(has_id=PointIdsList(points=allowed_ids))])

    hits = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=qvec,
        limit=top_k,
        query_filter=qfilter
    )
    # Lấy context: ưu tiên text_excerpt trong payload; nếu thiếu thì truy Postgres
    contexts = []
    missing_ids = []
    for h in hits:
        ex = (h.payload or {}).get("text_excerpt")
        if ex:
            contexts.append(ex)
        else:
            missing_ids.append(h.id)

    if missing_ids:
        recs = get_metadata_by_ids(missing_ids)
        for r in recs:
            if r.get("text_excerpt"):
                contexts.append(r["text_excerpt"])

    return "\n---\n".join(contexts) if contexts else ""

def chat_once(query: str, bmi: Optional[str] = None, gender: Optional[str] = None, goal: Optional[str] = None, top_k: int = 4):
    init_metadata_table()
    client = make_qdrant()
    embedder, _ = make_embedder()
    tokenizer, model, device = make_llm()

    ctx = retrieve_context(client, embedder, query, top_k=top_k, bmi=bmi, gender=gender, goal=goal)
    if not ctx and (bmi or gender or goal):
        return "(Không tìm thấy tài liệu khớp bộ lọc — thử nới lỏng filter hoặc tăng giới hạn.)"

    prompt = f"""You are FitAI, a smart fitness assistant.
Use the context to answer briefly and practically. If unsure, say so.

Context:
{ctx or "(no context retrieved)"}

Question: {query}
Answer:"""
    return generate_answer(tokenizer, model, device, prompt)

def main():
    print("🧠 FitAI-RAG CLI (with Neon filters). Gõ 'exit' để thoát.")
    print("Bạn có thể đặt filter nhanh: ví dụ 'gender=male bmi=normal goal=muscle_gain'")
    while True:
        raw = input("\nYou: ").strip()
        if not raw or raw.lower() in {"exit", "quit"}:
            break
        # parse filter inline
        parts = raw.split()
        filters = {"bmi": None, "gender": None, "goal": None}
        query_tokens = []
        for p in parts:
            if p.startswith("bmi="):
                filters["bmi"] = p.split("=", 1)[1]
            elif p.startswith("gender="):
                filters["gender"] = p.split("=", 1)[1]
            elif p.startswith("goal="):
                filters["goal"] = p.split("=", 1)[1]
            else:
                query_tokens.append(p)
        query = " ".join(query_tokens).strip() or raw
        t0 = time.time()
        ans = chat_once(query, **filters)
        print(f"\n🤖 {ans}")
        print(f"(took {time.time()-t0:.2f}s)")

if __name__ == "__main__":
    main()
