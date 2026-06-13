# rag_pipeline/embeddings.py

from langchain_community.embeddings import HuggingFaceBgeEmbeddings

def get_embedding_model():
    """โหลด BAAI/bge-m3 — รองรับทั้งไทยและอังกฤษ"""
    print("Loading BAAI/bge-m3 embedding model...")

    model = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},   # เปลี่ยนเป็น "cuda" ถ้ามี GPU
        encode_kwargs={
            "normalize_embeddings": True,  # จำเป็นสำหรับ cosine similarity
            "batch_size": 16,
        },
    )

    print("Model loaded.")
    return model