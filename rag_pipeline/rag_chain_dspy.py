# rag_pipeline/rag_chain_dspy.py

import os
import pickle
from pathlib import Path
from dotenv import load_dotenv

import dspy
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()

BASE_DIR   = Path(__file__).parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"
BM25_PATH  = BASE_DIR / "bm25_index.pkl"
OPTIMIZED_PROMPT_PATH = BASE_DIR / "optimized_prompt.json"

# ------------------------------------------------------------------ #
# ตั้งค่า DSPy
# ------------------------------------------------------------------ #
gemini = dspy.LM(
    model="gemini/gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
)

def _configure_dspy():
    try:
        dspy.configure(lm=gemini)
    except RuntimeError:
        pass  # ถ้า configure แล้วข้ามไปเลย

_configure_dspy()


# ------------------------------------------------------------------ #
# DSPy Signature และ Module (เหมือนใน dspy_optimize.py)
# ------------------------------------------------------------------ #
class PharmacyRAG(dspy.Signature):
    """คุณคือผู้ช่วยเภสัชกรที่เชี่ยวชาญด้านโรคติดเชื้อทางเดินหายใจส่วนบน (URI)
    ช่วยเภสัชกรหน้าร้านตัดสินใจเรื่องยาโดยอิงจาก clinical guidelines เป็นหลัก

    === ลักษณะการตอบ ===
    - ตอบเหมือนเภสัชกรคุยกับเพื่อนร่วมวิชาชีพ กระชับ ตรงประเด็น ไม่เป็นทางการเกินไป
    - ตรวจจับเพศจากคำในคำถาม: "ครับ" → ตอบด้วย "ครับ" ตลอด / "ค่ะ" → ตอบด้วย "ค่ะ" ตลอด
    - ถ้าไม่มีสัญญาณเพศ → ใช้ "ค่ะ" เป็น default
    - ห้ามสลับคำลงท้าย ใช้แค่อย่างเดียวตลอด conversation
    - ใช้ภาษาไทยธรรมชาติ เช่น "จากที่เล่ามา..." "เคสนี้เข้าได้กับ..." "แนะนำให้..."
    - ห้ามพูดกับผู้ป่วยหรือผู้ปกครองโดยตรง ให้พูดกับเภสัชกรเสมอ
    - ตอบในรูปแบบที่อ่านง่าย เช่น
      วินิจฉัย: ...
      ยาที่แนะนำ: ...
      ขนาดยา: ...
      ระยะเวลา: ...
      คำเตือน: ...
      อ้างอิง: ...

    === กฎการตัดสินใจ ===
    ถ้าข้อมูลไม่ครบ:
    → ถามกลับก่อนเสมอ อย่างน้อย 2-3 ข้อ
    → ห้ามตอบหรือแนะนำยาก่อนที่จะรู้ อายุ และ/หรือ น้ำหนักเด็ก
    → ห้ามตอบก่อนที่จะรู้ประวัติแพ้ยาในเคสที่อาจต้องจ่ายยาปฏิชีวนะ
    → อธิบายสั้นๆ ว่าทำไมต้องรู้ข้อมูลนั้น
    → ถามแบบเป็นธรรมชาติตามบริบท ห้ามถาม template ซ้ำๆ
    → ถ้าถามไปแล้วได้ข้อมูลบางส่วน ให้ตอบด้วยข้อมูลที่มีได้เลย
    → ถ้าข้อมูลพอวินิจฉัย viral URI ได้ชัดเจน ให้ตอบเลยไม่ต้องรอครบ 100%

    ถ้าอาการเข้าได้กับ Viral URI:
    → ปฏิเสธยาปฏิชีวนะพร้อมอธิบายเหตุผลสั้นๆ
    → แนะนำ Supportive care เช่น Paracetamol น้ำเกลือล้างจมูก ดื่มน้ำมากๆ พักผ่อน
    → บอกอาการที่ต้องกลับมาพบแพทย์

    ถ้าต้องให้ยาปฏิชีวนะ:
    → ระบุให้ครบ: ชื่อยา + ขนาด mg/kg ถ้าเป็นเด็ก + วิธีใช้ + ระยะเวลา
    → ถ้ามีน้ำหนัก → คำนวณขนาดยาจาก weight จริงๆ
    → ถ้าไม่มีน้ำหนัก → ระบุเป็น mg/kg/day ได้เลย
    → เตือนไม่ให้หยุดยาก่อนครบกำหนด
    → ห้ามใช้ขนาดยาเด็ก (mg/kg) กับผู้ใหญ่

    ถ้าเป็นกรณีฉุกเฉิน epiglottitis, RPA, หายใจลำบาก:
    → แนะนำส่ง ER ทันที ไม่จ่ายยา

    ถ้ามีประวัติแพ้ยา:
    → แยก Type 1 (anaphylaxis/urticaria) vs Non-type 1 (ผื่นธรรมดา)
    → เลือกยาทางเลือกตาม context ที่ได้รับ
    → ห้ามใช้ขนาดยาเด็ก (mg/kg) กับผู้ใหญ่

    === การใช้ข้อมูล ===
    - ถ้ามีใน context → ใช้ context เป็นหลักเสมอ อ้างอิง guideline + เลขหน้าทุกครั้ง
    - ถ้าไม่มีใน context → ตอบจากความรู้ทางคลินิกทั่วไปได้ แต่ต้องระบุว่า "(ข้อมูลทั่วไป ไม่ได้อยู่ใน guideline ที่มี)"
    - ห้ามเดาขนาดยาหรือชื่อยาที่ไม่มีใน context โดยเด็ดขาด

    === การอ้างอิง ===
    - อ้างอิง guideline และหน้าทุกครั้งท้ายคำตอบ
    - ดูเลขหน้าจาก [แหล่งที่ X] ใน context
    - รูปแบบ: (AAFP 2022 หน้า X) หรือ (แนวทางการดูแลรักษาโรคติดเชื้อเฉียบพลันระบบหายใจในเด็ก พ.ศ. 2562 หน้า X)
    - ห้ามอ้างแค่ชื่อ guideline โดยไม่มีเลขหน้า"""

    context: str = dspy.InputField(desc="เนื้อหาจาก clinical guidelines")
    question: str = dspy.InputField(desc="คำถามจากเภสัชกร")
    answer: str = dspy.OutputField(desc="คำตอบที่ถูกต้องตาม guideline")



class PharmacyModule(dspy.Module):
    def __init__(self):
        self.generate = dspy.ChainOfThought(PharmacyRAG)

    def forward(self, context, question):
        return self.generate(context=context, question=question)

# ------------------------------------------------------------------ #
# โหลด Optimized Module
# ------------------------------------------------------------------ #
def load_optimized_module():
    module = PharmacyModule()
    module.load(str(OPTIMIZED_PROMPT_PATH))
    print("Optimized DSPy module loaded.")
    return module

# ------------------------------------------------------------------ #
# โหลด Retrievers
# ------------------------------------------------------------------ #
def load_retrievers():
    print("Loading embedding model...")
    embed_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_API_KEY")

    if qdrant_url and qdrant_key:
        # ใช้ Qdrant Cloud
        print("Using Qdrant Cloud...")
        from qdrant_client import QdrantClient
        from langchain_qdrant import QdrantVectorStore
        qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
        vectorstore = QdrantVectorStore(
            client=qdrant_client,
            collection_name="uri_knowledge",
            embedding=embed_model,
            content_payload_key="page_content",
        )
    else:
        # ใช้ ChromaDB (local)
        print("Using ChromaDB (local)...")
        from langchain_chroma import Chroma
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embed_model,
            collection_name="uri_knowledge",
        )

    with open(BM25_PATH, "rb") as f:
        bm25_data = pickle.load(f)

    print("Retrievers loaded.")
    return vectorstore, bm25_data["bm25"], bm25_data["docs"]

# ------------------------------------------------------------------ #
# Hybrid Search
# ------------------------------------------------------------------ #
def hybrid_search(query: str, vectorstore, bm25, docs, k: int = 5) -> list:
    vector_results = vectorstore.similarity_search(query, k=k)
    vector_ids = {doc.page_content[:100] for doc in vector_results}

    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_idx = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:k]
    bm25_results = [docs[i] for i in top_bm25_idx]

    combined = list(vector_results)
    for doc in bm25_results:
        if doc.page_content[:100] not in vector_ids:
            combined.append(doc)

    return combined[:k + 2]

# ------------------------------------------------------------------ #
# Format Context
# ------------------------------------------------------------------ #
def format_context(docs: list, query: str = "") -> str:
    import re
    child_patterns = [r"เด็ก", r"อายุ\s*\d+\s*ขวบ", r"อายุ\s*\d+\s*เดือน", r"\d+\s*ขวบ"]
    is_child_case = any(re.search(p, query) for p in child_patterns)

    if not is_child_case:
        # เรียง AAFP ก่อน แล้วตามด้วย thai_uri
        docs = sorted(docs, key=lambda d: 0 if d.metadata.get("source") == "aafp" else 1)

    context_parts = []
    for i, doc in enumerate(docs):
        source  = doc.metadata.get("source", "unknown")
        page    = doc.metadata.get("page", "?")
        section = doc.metadata.get("h2", "") or doc.metadata.get("h3", "")

        source_label = {
            "aafp":     "AAFP 2022",
            "thai_uri": "แนวทางการดูแลรักษาโรคติดเชื้อเฉียบพลันระบบหายใจในเด็ก พ.ศ. 2562",
        }.get(source, source)

        context_parts.append(
            f"[แหล่งที่ {i+1}] {source_label} หน้า {page}"
            + (f" | หัวข้อ: {section}" if section else "")
            + f"\n{doc.page_content}\n"
        )

    return "\n---\n".join(context_parts)

# ------------------------------------------------------------------ #
# Ask — ใช้ DSPy Module
# ------------------------------------------------------------------ #
def ask(query: str, vectorstore, bm25, docs,
        module, history: list[dict] = None) -> tuple[str, list[dict]]:

    if history is None:
        history = []

    retrieved = hybrid_search(query, vectorstore, bm25, docs)
    context = format_context(retrieved, query)
    print("=== SOURCE ORDER ===")
    for doc in retrieved:
        print(doc.metadata.get("source"), "หน้า", doc.metadata.get("page"))
    print("===================")

    # เพิ่ม history เข้าไปใน question
    if history:
        history_text = "บทสนทนาก่อนหน้า:\n"
        for turn in history:
            history_text += f"เภสัชกร: {turn['user']}\n"
            history_text += f"ผู้ช่วย: {turn['assistant']}\n"
        full_question = f"{history_text}\nคำถามปัจจุบัน: {query}"
    else:
        full_question = query

    result = module(context=context, question=full_question)
    answer = result.answer

    history.append({
        "user":      query,
        "assistant": answer,
    })

    return answer, history

# ------------------------------------------------------------------ #
# Main — Interactive
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    vectorstore, bm25, docs = load_retrievers()
    module = load_optimized_module()

    print("\n" + "="*60)
    print("Pharmacy Chatbot — DSPy Version")
    print("พิมพ์ 'exit' เพื่อออก | 'new' เพื่อเริ่มเคสใหม่")
    print("="*60)

    history = []

    while True:
        query = input("\nเภสัชกร: ").strip()

        if query.lower() == "exit":
            print("ออกจากระบบแล้วค่ะ")
            break

        if query.lower() == "new":
            history = []
            print("เริ่มเคสใหม่แล้วค่ะ")
            continue

        if not query:
            continue

        answer, history = ask(query, vectorstore, bm25, docs, module, history)
        print(f"\nผู้ช่วย: {answer}")