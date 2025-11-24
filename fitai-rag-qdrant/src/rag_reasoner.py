from sentence_transformers import SentenceTransformer

def expand_query(raw_query: str) -> list:
    """
    Tự mở rộng truy vấn — mô phỏng Claude RAG Auto Expansion
    """
    expansions = [
        raw_query,
        f"chi tiết: {raw_query}",
        f"phân tích chuyên sâu: {raw_query}",
        f"nguyên nhân và giải pháp: {raw_query}",
        f"lời khuyên fitness của chuyên gia: {raw_query}",
    ]
    return expansions
