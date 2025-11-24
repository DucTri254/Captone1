import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import torch, ollama
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.config import *
from src.smart_profile import infer_profile, predict_bmr, predict_tdee, personalize_region
from src.advanced_prompt import build_prompt
from src.rag_reasoner import expand_query

app = FastAPI(title="FitAI ULTRA RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
_embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)
_qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

class ChatReq(BaseModel):
    query: str
    top_k: int = TOP_K

class ChatResp(BaseModel):
    answer: str
    contexts: list


@app.post("/chat", response_model=ChatResp)
def chat(req: ChatReq):

    # 1) AUTO QUERY EXPANSION
    queries = expand_query(req.query)

    # 2) ENSEMBLE RETRIEVAL
    all_ctx = []
    for q in queries:
        qvec = _embedder.encode(q, normalize_embeddings=True).tolist()
        res = _qdrant.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=qvec,
            limit=req.top_k,
            with_payload=True
        )
        all_ctx.extend(res)

    # Dedup
    seen = set()
    final_ctx = []
    for p in all_ctx:
        text = p.payload.get("text_excerpt", "")
        if text not in seen:
            seen.add(text)
            final_ctx.append(p)

    # Build context text
    ctx_text = ""
    for p in final_ctx:
        line = f"[{p.payload.get('source','?')} | score={p.score:.3f}] {p.payload.get('text_excerpt','')}\n----\n"
        if len(ctx_text) + len(line) > CTX_BUDGET:
            break
        ctx_text += line

    # 3) SMART PROFILE ENGINE
    profile = infer_profile(req.query)
    bmr = predict_bmr(profile.get("weight"), 170, profile.get("age", 25), profile.get("gender"))
    tdee = predict_tdee(bmr)

    # 4) BUILD ADVANCED PROMPT
    prompt = build_prompt(ctx_text, req.query, profile, bmr, tdee)

    # 5) LLM RESPONSE
    out = ollama.chat(model=LLM_MODEL, messages=[
        {"role": "system", "content": "Bạn là FitAI Ultra."},
        {"role": "user", "content": prompt}
    ])

    answer = out["message"]["content"]

    return ChatResp(answer=answer, contexts=[
        {"text": p.payload.get("text_excerpt", ""), "score": p.score}
        for p in final_ctx
    ])
