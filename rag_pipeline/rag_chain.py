# rag_pipeline/rag_chain.py

import os
import pickle
from pathlib import Path
from dotenv import load_dotenv

from google import genai
from google.genai import types
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

BASE_DIR   = Path(__file__).parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"
BM25_PATH  = BASE_DIR / "bm25_index.pkl"

# ------------------------------------------------------------------ #
# System Prompt สำหรับเภสัชกร
# ------------------------------------------------------------------ #
PHARMACY_SYSTEM_PROMPT = """คุณคือผู้ช่วยเภสัชกรที่เชี่ยวชาญด้านโรคติดเชื้อทางเดินหายใจส่วนบน (URI)
ช่วยเภสัชกรหน้าร้านตัดสินใจเรื่องยาโดยอิงจาก 2 แนวทางนี้เท่านั้น:
1. AAFP 2022 — ใช้กับผู้ป่วยทุกวัย ทั้งผู้ใหญ่และเด็ก ครอบคลุมทุกโรค URI
2. แนวทางการดูแลรักษาโรคติดเชื้อเฉียบพลันระบบหายใจในเด็ก พ.ศ. 2562 — ใช้กับเด็ก

## ลักษณะการตอบ
- ตอบเหมือนเภสัชกรคุยกับเพื่อนร่วมวิชาชีพ กระชับ ตรงประเด็น ไม่เป็นทางการเกินไป
- ใช้คำลงท้ายว่า "ค่ะ" หรือ "ครับ" เท่านั้น ห้ามใช้ "จ้ะ" หรือคำลงท้ายอื่น
- ใช้ภาษาไทยที่เป็นธรรมชาติ เช่น "จากที่เล่ามา..." "น้องเข้าได้กับ..." "แนะนำให้..."
- ไม่ต้องขึ้นต้นด้วย "จากอาการที่คุณแม่เล่ามา ลูกชายของคุณมีอาการ..." แบบเป็นทางการ
- ตอบในรูปแบบที่อ่านง่าย โดยใช้หัวข้อชัดเจน เช่น
  วินิจฉัย: ...
  ยาที่แนะนำ: ...
  ขนาดยา: ...
  ระยะเวลา: ...
  คำเตือน: ...
  อ้างอิง: ...

## กฎการตัดสินใจ

**ถ้าข้อมูลไม่ครบ:**
→ ถามกลับก่อนเสมอ อย่างน้อย 2-3 ข้อ ได้แก่ อายุ น้ำหนัก ประวัติแพ้ยา ระยะเวลาอาการ ความรุนแรง
→ อธิบายสั้นๆ ว่าทำไมต้องรู้ข้อมูลนั้น

**ถ้าอาการเข้าได้กับ Viral URI:**
→ ปฏิเสธยาปฏิชีวนะพร้อมอธิบายเหตุผลสั้นๆ
→ แนะนำ Supportive care เสมอ เช่น
   - Paracetamol ถ้ามีไข้หรือปวด (ระบุขนาดยาด้วย)
   - น้ำเกลือล้างจมูก / หยอดจมูก
   - ดื่มน้ำมากๆ พักผ่อน
   - บอกอาการที่ต้องกลับมาพบแพทย์

**ถ้าต้องให้ยาปฏิชีวนะ:**
→ ระบุให้ครบเสมอ: ชื่อยา + ขนาด (mg/kg ถ้าเป็นเด็ก) + วิธีใช้ + ระยะเวลา
→ ถ้ามีน้ำหนัก → คำนวณขนาดยาจาก weight จริงๆ
→ ถ้าไม่มีน้ำหนัก → ระบุเป็น mg/kg/day ได้เลย ไม่ต้องถามน้ำหนักซ้ำ
   เช่น "Amoxicillin 80-90 mg/kg/day แบ่ง BID × 10 วัน"
→ เตือนไม่ให้หยุดยาก่อนครบกำหนด

**ถ้าเป็นกรณีฉุกเฉิน (epiglottitis, RPA, หายใจลำบาก):**
→ แนะนำส่ง ER ทันที ไม่จ่ายยา

**ถ้ามีประวัติแพ้ยา:**
→ แยก Type 1 (anaphylaxis/urticaria) vs Non-type 1 (ผื่นธรรมดา)
→ เลือกยาทางเลือกให้เหมาะสม

## การอ้างอิง
- อ้างอิง guideline และหน้าทุกครั้งท้ายคำตอบ
- ถ้าใช้ทั้งสอง guideline ให้อ้างทั้งคู่
- รูปแบบ: (AAFP 2022 หน้า X) หรือ (แนวทางการดูแลรักษาโรคติดเชื้อเฉียบพลันระบบหายใจในเด็ก พ.ศ. 2562 หน้า X)

## ข้อห้าม
- ห้ามตอบนอกเหนือจาก context ที่ได้รับ
- ห้ามเดาหรือสรุปเองโดยไม่มีใน guideline
- ถ้าไม่มีข้อมูลใน context ให้บอกตรงๆ ว่า "ไม่พบในแนวทางที่มี"
"""
# ------------------------------------------------------------------ #
# โหลด Vectorstore และ BM25
# ------------------------------------------------------------------ #
def load_retrievers():
    print("Loading embedding model...")
    embed_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

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
# Hybrid Search: Vector + BM25
# ------------------------------------------------------------------ #
def hybrid_search(query: str, vectorstore, bm25, docs, k: int = 5) -> list:
    # Vector search
    vector_results = vectorstore.similarity_search(query, k=k)
    vector_ids = {doc.page_content[:100] for doc in vector_results}

    # BM25 search
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_idx = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:k]
    bm25_results = [docs[i] for i in top_bm25_idx]

    # รวมผล โดยไม่ซ้ำ
    combined = list(vector_results)
    for doc in bm25_results:
        if doc.page_content[:100] not in vector_ids:
            combined.append(doc)

    return combined[:k + 2]  # เผื่อไว้นิดนึง

# ------------------------------------------------------------------ #
# สร้าง Context จาก Retrieved Docs
# ------------------------------------------------------------------ #
def format_context(docs: list) -> str:
    context_parts = []
    for i, doc in enumerate(docs):
        source   = doc.metadata.get("source", "unknown")
        page     = doc.metadata.get("page", "?")
        pdf_file = doc.metadata.get("pdf_file", "")
        section  = doc.metadata.get("h2", "") or doc.metadata.get("h3", "")

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
# ถามคำถาม
# ------------------------------------------------------------------ #
def ask(query: str, vectorstore, bm25, docs,
        history: list[dict] = None) -> tuple[str, list[dict]]:

    if history is None:
        history = []

    # ดึง context
    retrieved = hybrid_search(query, vectorstore, bm25, docs)
    context   = format_context(retrieved)

    # สร้าง history text
    history_text = ""
    if history:
        history_text = "บทสนทนาก่อนหน้า:\n"
        for turn in history:
            history_text += f"เภสัชกร: {turn['user']}\n"
            history_text += f"ผู้ช่วย: {turn['assistant']}\n"
        history_text += "\n"

    prompt = f"""บริบทจาก Clinical Guidelines:
{context}

{history_text}คำถามปัจจุบัน:
{query}

กรุณาตอบโดยอิงจากบริบทและบทสนทนาก่อนหน้าด้วย"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=PHARMACY_SYSTEM_PROMPT,
            temperature=0,
        ),
        contents=prompt,
    )

    answer = response.text

    # อัปเดต history
    history.append({
        "user":      query,
        "assistant": answer,
    })

    return answer, history
# ------------------------------------------------------------------ #
# Main — ทดสอบ 3 เคส
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    vectorstore, bm25, docs = load_retrievers()

    print("\n" + "="*60)
    print("Pharmacy Chatbot — URI Assistant")
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

        answer, history = ask(query, vectorstore, bm25, docs, history)
        print(f"\nผู้ช่วย: {answer}")