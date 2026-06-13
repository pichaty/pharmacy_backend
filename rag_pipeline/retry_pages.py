# rag_pipeline/retry_pages.py

import os
import base64
import time
from pathlib import Path
# import google.generativeai as genai
from google import genai
from dotenv import load_dotenv
import fitz

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

OUTPUT_DIR = Path(__file__).parent.parent / "markdown_output"
DATA_DIR   = Path(__file__).parent.parent / "data"

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

# ระบุหน้าที่ต้องรันซ้ำ (page number จริงๆ เริ่มจาก 1)
RETRY_PAGES = {
    "aafp":     [6],
    # "thai_uri": [46],
}
def pdf_page_to_image_bytes(pdf_path, page_num, dpi=150):
    doc = fitz.open(str(pdf_path))
    page = doc[page_num]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    doc.close()
    return img_bytes

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

def retry_pages():
    pdf_map = {
        "thai_uri": DATA_DIR / "Thai URI Children.pdf",
        "aafp":     DATA_DIR / "AAFP_2022_Original.pdf",
    }

    for source_name, pages in RETRY_PAGES.items():
        pdf_path = pdf_map[source_name]
        md_path  = OUTPUT_DIR / f"{source_name}.md"

        content = md_path.read_text(encoding="utf-8")

        for page_num in pages:
            print(f"Retrying {source_name} page {page_num}...", end=" ")
            try:
                img_bytes = pdf_page_to_image_bytes(pdf_path, page_num - 1)
                md_text   = gemini_extract_page(img_bytes)

                # แทนที่ ERROR comment เดิมด้วยเนื้อหาใหม่
                old = f"<!-- page {page_num} ERROR:"
                # หา block เดิมแล้วแทนที่
                lines = content.split("\n")
                new_lines = []
                skip = False
                for line in lines:
                    if line.startswith(f"<!-- page {page_num} ERROR:"):
                        new_lines.append(f"\n<!-- page {page_num} -->\n{md_text}")
                        skip = True
                    elif skip and line.startswith("<!-- page "):
                        skip = False
                        new_lines.append(line)
                    elif not skip:
                        new_lines.append(line)
                content = "\n".join(new_lines)
                print("done")

            except Exception as e:
                print(f"ERROR again: {e}")

            time.sleep(4)

        md_path.write_text(content, encoding="utf-8")
        print(f"Updated → {md_path}")

if __name__ == "__main__":
    retry_pages()