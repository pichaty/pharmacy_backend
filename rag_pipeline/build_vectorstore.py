# rag_pipeline/build_vectorstore.py

import os
import pickle
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

load_dotenv()

# ------------------------------------------------------------------ #
# Paths
# ------------------------------------------------------------------ #
BASE_DIR    = Path(__file__).parent.parent
MD_DIR      = BASE_DIR / "markdown_output"
CHROMA_DIR  = BASE_DIR / "chroma_db"
BM25_PATH   = BASE_DIR / "bm25_index.pkl"

MD_FILES = {
    "aafp":     MD_DIR / "aafp.md",
    "thai_uri": MD_DIR / "thai_uri.md",
}

# ------------------------------------------------------------------ #
# Step 1: โหลดและ Chunk Markdown ตาม Header
# ------------------------------------------------------------------ #
def load_and_chunk(source_name: str, md_path: Path) -> list[Document]:
    print(f"\nChunking: {md_path.name}")
    import re
    text = md_path.read_text(encoding="utf-8")

    # หา page markers ก่อนลบออก
    # เก็บว่าแต่ละ position ใน text อยู่ใน page ไหน
    page_map = {}  # {char_position: page_num}
    for match in re.finditer(r'<!-- page (\d+) -->', text):
        page_num = int(match.group(1))
        page_map[match.start()] = page_num

    # ลบ comments ออกจาก text
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

    # แบ่ง chunk ตาม Markdown headers
    headers_to_split = [
        ("#",  "h1"),
        ("##", "h2"),
    ]
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split,
        strip_headers=False,
    )
    chunks = splitter.split_text(text)

    # กำหนด pdf filename
    pdf_filename = {
        "aafp":     "AAFP_2022_Original.pdf",
        "thai_uri": "Thai URI Children.pdf",
    }.get(source_name, "")

    # เพิ่ม metadata
    docs = []
    char_pos = 0  # ติดตาม position ใน text
    for chunk in chunks:
        if len(chunk.page_content.strip()) < 50:
            continue

        # หา page number จาก position
        # ใช้ page markers ที่ใกล้ที่สุดก่อน chunk นี้
        page_num = 1
        for pos, pg in sorted(page_map.items()):
            if pos <= char_pos:
                page_num = pg
        char_pos += len(chunk.page_content)

        h2 = chunk.metadata.get("h2", "")
        h3 = chunk.metadata.get("h3", "")

        context_prefix = ""
        if h2:
            context_prefix += f"หัวข้อ: {h2}\n"
        if h3:
            context_prefix += f"หัวข้อย่อย: {h3}\n"

        if context_prefix:
            chunk.page_content = context_prefix + chunk.page_content

        chunk.metadata["source"]   = source_name
        chunk.metadata["language"] = "th" if source_name == "thai_uri" else "en"
        chunk.metadata["page"]     = page_num
        chunk.metadata["pdf_file"] = pdf_filename
        docs.append(chunk)

    # Merge TABLE chunks เข้ากับ section ก่อนหน้า
    merged_docs = []
    i = 0
    while i < len(docs):
        current = docs[i]
        h2 = current.metadata.get("h2", "")
        
        # ถ้าเป็น TABLE chunk ให้ merge เข้ากับ chunk ก่อนหน้า
        if h2.startswith("TABLE") and merged_docs:
            prev = merged_docs[-1]
            prev.page_content += "\n\n" + current.page_content
            i += 1
            continue
        
        merged_docs.append(current)
        i += 1

    print(f"  → {len(docs)} chunks (after merge: {len(merged_docs)})")
    return merged_docs



# ------------------------------------------------------------------ #
# Step 2: Embed และบันทึกลง ChromaDB
# ------------------------------------------------------------------ #
def build_vectorstore(all_docs: list[Document]):
    print("\nLoading embedding model (BAAI/bge-m3)...")
    print("(ครั้งแรกอาจใช้เวลาโหลด model ~1-2 นาทีค่ะ)")


    embed_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 8,
        },
    )

    print(f"\nBuilding ChromaDB ({len(all_docs)} chunks)...")
    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embed_model,
        persist_directory=str(CHROMA_DIR),
        collection_name="uri_knowledge",
    )
    print(f"ChromaDB saved → {CHROMA_DIR}")
    return vectorstore, embed_model

# ------------------------------------------------------------------ #
# Step 3: สร้าง BM25 Index สำหรับ Keyword Search
# ------------------------------------------------------------------ #
def build_bm25(all_docs: list[Document]):
    print("\nBuilding BM25 index...")
    tokenized = [doc.page_content.split() for doc in all_docs]
    bm25 = BM25Okapi(tokenized)

    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "docs": all_docs}, f)
    print(f"BM25 saved → {BM25_PATH}")
    return bm25

# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    # โหลดและ chunk ทุกไฟล์
    all_docs = []
    for source_name, md_path in MD_FILES.items():
        if not md_path.exists():
            print(f"File not found: {md_path}")
            continue
        docs = load_and_chunk(source_name, md_path)
        all_docs.extend(docs)

    print(f"\nTotal chunks: {len(all_docs)}")

    # สร้าง vectorstore และ BM25
    vectorstore, embed_model = build_vectorstore(all_docs)
    bm25 = build_bm25(all_docs)

    # ทดสอบ query
    print("\nทดสอบ query...")
    query = "amoxicillin ขนาดยาสำหรับเด็ก"
    results = vectorstore.similarity_search(query, k=3)
    print(f"\nผลลัพธ์สำหรับ: '{query}'")
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Source:  {doc.metadata.get('source')}")
        print(f"Page:    {doc.metadata.get('page', 'N/A')}")
        print(f"PDF:     {doc.metadata.get('pdf_file', 'N/A')}")
        print(f"Section: {doc.metadata.get('h2', 'N/A')}")
        print(doc.page_content[:200])

    print("\nDone! Vector store ready 🎉")