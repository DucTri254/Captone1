# src/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langdetect import detect
import ollama

from src.hybrid_retriever import hybrid_search, get_model, get_client
from src.config import OLLAMA_MODEL

app = FastAPI(title="FitAI Hybrid RAG")

@app.on_event("startup")
def _startup():
    get_model()
    get_client()
    # Kiểm tra model trong Ollama (chịu lỗi format khác nhau giữa phiên bản)
    try:
        info = ollama.list()
        models = info.get("models", info if isinstance(info, list) else [])
        names = []
        for m in models:
            if isinstance(m, dict) and "name" in m:
                names.append(m["name"])
            elif isinstance(m, str):
                names.append(m)
        if OLLAMA_MODEL not in names:
            print(f"⚠️ Ollama model '{OLLAMA_MODEL}' not found. Run:  ollama pull {OLLAMA_MODEL}")
        else:
            print(f"✅ Ollama model available: {OLLAMA_MODEL}")
    except Exception as e:
        print(f"⚠️ Ollama check failed: {e}")
    print("✅ Initialized model & Qdrant client")

class ChatRequest(BaseModel):
    question: str
    k: int = 5

@app.post("/chat")
def chat(req: ChatRequest):
    hits = hybrid_search(req.question, req.k)
    ctx = "\n\n".join([h["text"] for h in hits if h["text"]])
    lang = "Vietnamese" if detect(req.question) == "vi" else "English"
    prompt = f"Answer in {lang}.\nContext:\n{ctx}\n\nQuestion: {req.question}\nAnswer:"
    try:
        res = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama error: {e}")
    return {"answer": res["message"]["content"], "sources": hits}

@app.get("/health")
def health():
    return {"ok": True}
