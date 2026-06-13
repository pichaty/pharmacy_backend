# rag_pipeline/pdf_to_markdown.py

import os
import base64
import time
from pathlib import Path
from google import genai
from dotenv import load_dotenv
import fitz  # pymupdf - แค่ใช้แปลง PDF → รูป ไม่ได้อ่าน text

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

DATA_DIR    = Path(__file__).parent.parent / "data"
OUTPUT_DIR  = Path(__file__).parent.parent / "markdown_output"
OUTPUT_DIR.mkdir(exist_ok=True)

PDF_FILES = {
    "aafp":     DATA_DIR / "AAFP_2022_Original.pdf",
    "thai_uri": DATA_DIR / "Thai URI Children.pdf",
}

# ------------------------------------------------------------------ #
# Prompt สำหรับหน้าเนื้อหาทั่วไป
# ------------------------------------------------------------------ #
# PROMPT_TEXT = """คุณเป็นผู้เชี่ยวชาญด้านการแปลงเอกสารทางการแพทย์
# แปลงเนื้อหาในรูปนี้ให้เป็น Markdown ที่สมบูรณ์ โดยมีกฎดังนี้:

# 1. หัวข้อหลัก (ชื่อโรค / บทหลัก) → ใช้ ## 
# 2. หัวข้อย่อย (สาเหตุ / อาการ / การรักษา / การวินิจฉัย) → ใช้ ###
# 3. หัวข้อย่อยลงไปอีก → ใช้ ####
# 4. ตาราง (เช่น ตารางขนาดยา) → แปลงเป็น Markdown table ให้ครบถ้วนทุก cell
# 5. รายการ bullet → ใช้ -
# 6. ตัวเลขขนาดยา หน่วย และระยะเวลา → ต้องแม่นยำ 100% ห้ามเดา
# 7. ศัพท์ทางการแพทย์ภาษาอังกฤษ → คงไว้ตามเดิม
# 8. ถ้าหน้านี้เป็นหน้าปก / สารบัญ / บรรณานุกรม → ตอบแค่ [SKIP]
# 9. ห้ามเพิ่มเนื้อหาที่ไม่มีในรูป
# 10. ตอบเป็น Markdown เท่านั้น ไม่ต้องมีคำอธิบายนำหน้า
# """

# # ------------------------------------------------------------------ #
# # Prompt พิเศษสำหรับหน้าที่มี Flowchart / Decision tree
# # ------------------------------------------------------------------ #
# PROMPT_FLOWCHART = """คุณเป็นผู้เชี่ยวชาญด้านการแปลงเอกสารทางการแพทย์
# หน้านี้มี Flowchart หรือ Decision tree การรักษา
# แปลงให้เป็น Markdown โดยใช้รูปแบบ If-Then ดังนี้:

# **ชื่อแผนภูมิ** (ถ้ามี)

# - ถ้า [เงื่อนไข] → [การกระทำ/การวินิจฉัย]
#   - ถ้า [เงื่อนไข่ย่อย] → [การกระทำ]
#   - ถ้า [เงื่อนไขย่อย] → [การกระทำ]
# - ถ้า [เงื่อนไข] → [การกระทำ]

# กฎ:
# 1. ต้องครอบคลุมทุก branch ของ flowchart
# 2. ตัวเลขและยาต้องแม่นยำ 100%
# 3. ตอบเป็น Markdown เท่านั้น
# """

PROMPT_UNIFIED = """คุณเป็นผู้เชี่ยวชาญด้านการแปลงเอกสารทางการแพทย์เป็น Markdown
ตรวจสอบรูปนี้และทำตามกฎต่อไปนี้:

**ถ้าหน้านี้เป็นหน้าปก / ISBN / รายชื่อคณะกรรมการ / บรรณานุกรม / รายการอ้างอิง:**
→ ตอบแค่ [SKIP]

**ถ้าหน้านี้มีเนื้อหาทางการแพทย์ (ข้อความ, ตาราง, หรือ flowchart):**
→ แปลงเป็น Markdown โดยใช้กฎเหล่านี้:

1. คัดลอกข้อความให้ตรงกับต้นฉบับ ห้ามแปลหรือสรุปเป็นภาษาอื่น
2. ถ้าต้นฉบับเป็นภาษาอังกฤษ → ตอบเป็นภาษาอังกฤษ
3. ถ้าต้นฉบับเป็นภาษาไทย → ตอบเป็นภาษาไทย
4. หัวข้อหลัก → ## | หัวข้อย่อย → ### | หัวข้อย่อยลงไป → ####
5. ตาราง → Markdown table ครบทุก cell ตัวเลขยาต้องแม่นยำ 100%
6. ถ้ามี Flowchart หรือ Decision tree → แปลงเป็น if-then ในภาษาเดิมของเอกสาร
7. ห้ามเพิ่มคำอธิบายหรือความเห็นของตัวเอง
8. ตอบเป็น Markdown เท่านั้น
"""

def pdf_page_to_image_bytes(pdf_path: Path, page_num: int, dpi: int = 150) -> bytes:
    """แปลง 1 หน้า PDF เป็น PNG bytes"""
    doc = fitz.open(str(pdf_path))
    page = doc[page_num]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    doc.close()
    return img_bytes

# def has_flowchart(pdf_path: Path, page_num: int) -> bool:
#     """เช็คคร่าวๆ ว่าหน้านี้น่าจะมี flowchart ไหม (ดูจากจำนวนรูปในหน้า)"""
#     doc = fitz.open(str(pdf_path))
#     page = doc[page_num]
#     images = page.get_images()
#     doc.close()
#     return len(images) > 0

def gemini_extract_page(img_bytes: bytes) -> str:
    image_part = {
        "mime_type": "image/png",
        "data": base64.b64encode(img_bytes).decode("utf-8"),
    }
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[PROMPT_UNIFIED, image_part],
    )
    text = response.text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])
    return text

def convert_pdf_to_markdown(source_name: str, pdf_path: Path) -> Path:
    """แปลง PDF ทั้งไฟล์ → Markdown file"""
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    doc.close()

    print(f"\n{'='*50}")
    print(f"Processing: {pdf_path.name} ({total_pages} pages)")
    print(f"{'='*50}")

    all_markdown = []
    skipped = 0

    for page_num in range(total_pages):
        print(f"  Page {page_num + 1}/{total_pages}", end=" ... ")

        try:
            img_bytes = pdf_page_to_image_bytes(pdf_path, page_num)
            md_text = gemini_extract_page(img_bytes)

            if md_text == "[SKIP]":
                print("skipped")
                skipped += 1
            else:
                # ใส่ comment บอกหน้าไว้ตรวจสอบ
                all_markdown.append(f"\n<!-- page {page_num + 1} -->\n{md_text}")
                print("done")

        except Exception as e:
            print(f"ERROR: {e}")
            all_markdown.append(f"\n<!-- page {page_num + 1} ERROR: {e} -->\n")

        # Rate limit — Gemini Flash free tier: 15 req/min
        time.sleep(4)

    output_path = OUTPUT_DIR / f"{source_name}.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_markdown))

    print(f"\nSaved → {output_path}")
    print(f"Pages processed: {total_pages - skipped}, Skipped: {skipped}")
    return output_path


if __name__ == "__main__":
    for source_name, pdf_path in PDF_FILES.items():
        if not pdf_path.exists():
            print(f"File not found: {pdf_path}")
            continue
        convert_pdf_to_markdown(source_name, pdf_path)
    print("\nAll PDFs converted!")