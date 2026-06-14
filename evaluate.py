# evaluate.py
# วางไฟล์นี้ที่ root folder: pharmacy_chatbot/evaluate.py

import os
import pickle
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from google.genai import types

load_dotenv()

import sys
sys.path.append(str(Path(__file__).parent))
from rag_pipeline.rag_chain_dspy import load_retrievers, ask, load_optimized_module
from google import genai
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# ------------------------------------------------------------------ #
# Test Set — 25 เคส (DSPy ไม่เคยเห็นเลย)
# ------------------------------------------------------------------ #
TEST_CASES = [
    # ===== ง่าย (6 เคส) =====
    {
        "id": "easy_1", "level": "ง่าย",
        "input": "แม่พาลูกชายอายุ 3 ขวบมาร้านยา บอกว่าเป็นหวัดมา 2 วัน มีน้ำมูกใส ไอเล็กน้อย ไข้ 37.8°C ไม่มีหูอื้อ ไม่เจ็บคอมาก ไม่มีน้ำมูกข้นเขียว ขอยาแก้อักเสบ เพราะคิดว่าจะทำให้หายเร็วขึ้น",
    },
    {
        "id": "easy_4", "level": "ง่าย",
        "input": "แม่พาลูกสาว 5 ขวบมาร้านยา น้ำมูกเปลี่ยนเป็นสีเขียวมา 1 วัน เริ่มป่วย 3 วันก่อน ไข้ 38°C ไม่ปวดหน้าผาก ไม่ปวดแก้ม แม่บอกน้ำมูกเขียวแสดงว่ามีเชื้อแบคทีเรียแน่นอน",
    },
    {
        "id": "easy_7", "level": "ง่าย",
        "input": "เด็กอายุ 10 ปี มาด้วยอาการเจ็บคอ กลืนเจ็บ มีไข้สูง 38.5°C ตรวจพบฝ้าขาวที่ต่อมทอนซิล ต่อมน้ำเหลืองที่คอส่วนหน้าโตและกดเจ็บ โดยไม่มีอาการไอ",
    },
    {
        "id": "easy_10", "level": "ง่าย",
        "input": "เด็กชายอายุ 10 ปี น้ำหนัก 34 kg มีไข้ 39°C เจ็บคอมาก ไม่มีไอ ไม่มีน้ำมูก ตรวจพบจุดหนองบนต่อมทอนซิล และต่อมน้ำเหลืองบริเวณหน้าลำคอโต",
    },
    {
        "id": "easy_11", "level": "ง่าย",
        "input": "เด็กชายอายุ 5 ปี น้ำหนัก 18 kg ปวดหูข้างขวาเล็กน้อย ไข้ 38°C มา 1 วัน ไม่มีของเหลวไหลออกจากหู ยังเล่นได้ปกติ",
    },
    {
        "id": "easy_12", "level": "ง่าย",
        "input": "เด็กหญิงอายุ 8 ปี น้ำหนัก 27 kg น้ำมูกเหลือง ไอ คัดจมูก มา 4 วัน ไม่มีกดเจ็บที่บริเวณใบหน้า ไม่มีไข้สูง",
    },

    # ===== ปานกลาง (5 เคส) =====
    {
        "id": "medium_3", "level": "ปานกลาง",
        "input": "ชายอายุ 35 ปี มีน้ำมูกข้น ปวดหน้าผากและแก้มข้างซ้ายมาได้ 12 วัน เริ่มเป็นหวัดแล้วค่อยๆ แย่ลง ไข้ 37.8°C กดบริเวณไซนัสแก้มซ้ายเจ็บ ไม่แพ้ยาใด",
    },
    {
        "id": "medium_6", "level": "ปานกลาง",
        "input": "เด็กอายุ 7 ปี มาด้วยอาการเจ็บคอ ไข้ 38.2°C ไม่มีอาการไอ แต่ตรวจไม่พบฝ้าขาวที่ทอนซิล และต่อมน้ำเหลืองไม่โต",
    },
    {
        "id": "medium_9", "level": "ปานกลาง",
        "input": "เด็กหญิงอายุ 18 เดือน ไข้ 39.5°C ร้องกวนมาก ดึงหูสองข้าง ตรวจพบเยื่อแก้วหูโป่งทั้งสองข้าง",
    },
    {
        "id": "medium_11", "level": "ปานกลาง",
        "input": "หญิงอายุ 52 ปี คัดจมูก น้ำมูกเหลืองข้น ปวดหน้าผาก ปวดบริเวณโหนกแก้ม ไข้ 37.5°C มาได้ 12 วัน ช่วง 5-6 วันแรกเหมือนหวัดธรรมดา จากนั้นเริ่มดีขึ้น แต่ 3 วันที่ผ่านมาอาการกลับแย่ลงอีกครั้ง ไม่มีประวัติแพ้ยา",
    },
    {
        "id": "medium_12", "level": "ปานกลาง",
        "input": "เด็กชายอายุ 7 ปี น้ำหนัก 25 kg เจ็บหูข้างซ้ายมา 2 วัน ไข้ 38.8°C ร้องไห้และดึงหูตัวเอง ไม่มีหนองไหลออกจากหู ไม่มีประวัติ allergy เคยได้รับ amoxicillin ครั้งสุดท้ายเมื่อ 3 เดือนก่อน",
    },

    # ===== ยาก (4 เคส) =====
    {
        "id": "hard_2", "level": "ยาก",
        "input": "แม่พาลูก 7 ขวบมาร้านยา ลูกได้รับการวินิจฉัย AOM จากแพทย์เมื่อ 5 วันก่อน สั่ง amoxicillin 80 mg/kg/วัน รับประทานครบ 5 วันแล้ว แต่เด็กยังปวดหูอยู่ ไข้ไม่ลด แม่ขอซื้อ amoxicillin เพิ่ม น้ำหนักเด็ก 25 kg",
    },
    {
        "id": "hard_5", "level": "ยาก",
        "input": "ผู้ป่วยปวดศีรษะและคัดจมูกจากหวัดมา 5 วันแล้วอาการเริ่มดีขึ้น แต่ในวันที่ 6 กลับมามีไข้สูงขึ้นมาใหม่ 39.1°C น้ำมูกเปลี่ยนเป็นหนองข้นมากขึ้น และปวดใบหน้ารุนแรงติดต่อกันมา 3 วัน",
    },
    {
        "id": "hard_8", "level": "ยาก",
        "input": "เด็กอายุ 6 ปี มีอาการไข้สูง เจ็บคอรุนแรง กลืนลำบาก มีอาการน้ำลายไหลยืด เสียงพูดอู้อี้ หายใจเข้ามีเสียงฮึด และชอบนั่งโน้มตัวไปข้างหน้า",
    },
    {
        "id": "hard_10", "level": "ยาก",
        "input": "เด็กหญิงอายุ 8 ปี น้ำหนัก 28 kg วินิจฉัย AOM ข้างขวา ไข้ 38.2°C อาการปวดหูรุนแรง ไม่มีแพ้ยา แพทย์สั่ง amoxicillin แต่พ่อแจ้งว่าเด็กได้รับ amoxicillin มาได้ 3 วันแต่อาการยังไม่ดีขึ้น ยังมีไข้อยู่",
    },
    {
        "id": "hard_11", "level": "ยาก",
        "input": "หญิงอายุ 38 ปี ปวดหน้า น้ำมูกเหลืองข้น ไข้ 39.2°C มาได้ 4 วัน อาการรุนแรงตั้งแต่ต้น มีประวัติแพ้ penicillin แบบรุนแรง anaphylaxis เมื่อ 5 ปีก่อน แพทย์วินิจฉัยเป็น Acute Bacterial Rhinosinusitis",
    },

    # ===== Negative (5 เคส) =====
    {
        "id": "neg_2", "level": "negative",
        "input": "หญิง 28 ปี มีใบสั่ง Penicillin V 500 mg วันละ 2 ครั้ง × 10 วัน สำหรับ GABHS pharyngitis บอกว่าอยากเปลี่ยนเป็น amoxicillin วันละครั้ง สะดวกกว่า",
    },
    {
        "id": "neg_5", "level": "negative",
        "input": "ผู้ป่วยชายอายุ 26 ปี เสียงแหบ มีน้ำมูกใส คัดจมูกเล็กน้อย เจ็บคอเล็กน้อย เป็นมา 2 วัน ไม่มีไข้ ขอซื้อยาแก้อักเสบ amoxicillin",
    },
    {
        "id": "neg_7", "level": "negative",
        "input": "ผู้ป่วยหญิงอายุ 32 ปี เจ็บคอมากมา 1 วัน มีน้ำมูกใส มีไข้ต่ำๆ 37.4°C ไอคอกแคกตลอดเวลา ตรวจคอไม่พบจุดหนอง ทอนซิลแดงเล็กน้อย ต่อมน้ำเหลืองไม่โต ขอซื้อ Amoxicillin วันละครั้ง",
    },
    {
        "id": "neg_8", "level": "negative",
        "input": "ผู้ป่วยหญิงอายุ 45 ปี คัดจมูก น้ำมูกข้นสีเขียวเหลือง ปวดหน่วงๆ บริเวณแก้มและหัวคิ้วมา 5 วัน ไม่มีไข้ ขอยา Augmentin",
    },
    {
        "id": "neg_11", "level": "negative",
        "input": "หญิงอายุ 25 ปี มาพร้อม prescription จากคลินิก วินิจฉัย Common Cold น้ำมูกใส ไอ จาม ไม่มีไข้ มาได้ 2 วัน แต่ผู้ป่วยมาแจ้งว่าแพทย์สั่งให้มาซื้อยาฆ่าเชื้อ",
    },

    # ===== ไม่สมบูรณ์ (5 เคส) =====
    {
        "id": "incomplete_2", "level": "ไม่สมบูรณ์",
        "input": "ชายคนหนึ่งเดินเข้ามาร้านยา บอกสั้นๆ ว่า เจ็บคอครับ ขอ amoxicillin หน่อย เคยกินแล้วหาย",
    },
    {
        "id": "incomplete_5", "level": "ไม่สมบูรณ์",
        "input": "คนไข้มีอาการเจ็บคอมาก กลืนน้ำลายเหมือนมีหนามแทง ไข้สูง 38.8°C ส่องกระจกดูเห็นทอนซิลโตบวมแดงและมีจุดขาวๆ ขึ้นเต็มเลย แนะนำยาปฏิชีวนะให้หน่อยค่ะ",
    },
    {
        "id": "incomplete_8", "level": "ไม่สมบูรณ์",
        "input": "แม่อายุประมาณ 30 ปี พาเด็กชายมาร้านยา บอกว่า ลูกปวดหู ร้องไห้มา 1 วัน มีไข้ เด็กดูงอแง ไม่สามารถบอกอายุชัดเจน แม่ขอยาแก้ปวดหู กับยาปฏิชีวนะ",
    },
    {
        "id": "incomplete_10", "level": "ไม่สมบูรณ์",
        "input": "หญิงอายุ 42 ปี มาร้านยาพร้อม prescription วินิจฉัย Acute Bacterial Rhinosinusitis แพทย์สั่ง Amoxicillin/Clavulanate 875/125 mg BID × 7 วัน ผู้ป่วยบอกว่าเคยแพ้ยาปฏิชีวนะ ไม่แน่ใจว่ายาตัวไหน",
    },
]

# ------------------------------------------------------------------ #
# Expected outputs จากเฉลยพี่เภสัช
# ------------------------------------------------------------------ #
EXPECTED_OUTPUTS = {
    "easy_1":  "ยาที่ให้ได้: Paracetamol สำหรับลดไข้, น้ำเกลือหยอดจมูกช่วยให้จมูกโล่ง, ดูดน้ำมูกด้วยลูกยาง ห้ามให้เด็ก <4 ปี: ยาแก้ไอ-ยาลดน้ำมูก-ยาแก้คัดจมูก (antihistamine, pseudoephedrine, dextromethorphan)",
    "easy_4":  "Supportive care: Paracetamol ลดไข้, น้ำเกลือหยอดจมูก, ดื่มน้ำมาก, สังเกตอาการ นัดกลับมาหรือไปพบแพทย์ถ้าอาการไม่ดีขึ้นใน 10 วัน หรือแย่ลงกะทันหัน",
    "easy_7":  "คะแนน Modified Centor = 5 คะแนน สั่งจ่ายยาปฏิชีวนะ Penicillin V หรือ Amoxicillin นาน 10 วัน",
    "easy_10": "Modified Centor = 5 คะแนน ต้องให้ยาปฏิชีวนะ ยาที่ควรได้รับ: amoxicillin 10 วัน",
    "easy_11": "AOM อาการปวดหูไม่มาก ให้ยาแก้ปวดในทุกราย แต่ยังไม่แนะนำให้ยาปฏิชีวนะ watchful waiting 48 ชั่วโมง",
    "easy_12": "watchful waiting ให้ยาตามอาการ เฝ้าดูอาการไซนัสอักเสบ หากอาการยังไม่ดีขึ้นใน 7 วัน",
    "medium_3":  "First-line: Amoxicillin/clavulanate (Augmentin) 500 mg ทุก 8 ชั่วโมง หรือ 875 mg ทุก 12 ชั่วโมง 5-7 วัน",
    "medium_6":  "คะแนน Modified Centor = 3 คะแนน ควรทำ RADT ถ้าผลบวกจึงค่อยสั่งจ่ายยาปฏิชีวนะ ถ้าลบในเด็กต้องทำ Throat culture",
    "medium_9":  "AOM อายุ <2 ปี ให้ Amoxicillin 80-90 mg/kg/day แบ่งให้วันละ 2-3 ครั้ง 7-10 วัน",
    "medium_11": "เข้าได้กับ double sickening ของ ABRS ควรได้รับยา ATB amoxicillin/clavulanate",
    "medium_12": "AOM เคยได้ amoxicillin ใน 3 เดือน ให้ Amoxicillin 80-90 mg/kg/day 7-10 วัน",
    "hard_2":  "Treatment failure ให้ Amoxicillin/clavulanate 90 mg/kg/วัน แบ่ง 2 ครั้ง คำนวณ: 90×25=2,250 mg/วัน → 1,125 mg ทุก 12 ชั่วโมง",
    "hard_5":  "Double sickening ให้ Amoxicillin/clavulanate 500 mg ทุก 8 ชั่วโมง หรือ 875 mg ทุก 12 ชั่วโมง 5-7 วัน",
    "hard_8":  "Epiglottitis รีบส่งโรงพยาบาล IV antibiotics เช่น Ceftriaxone หรือ Ampicillin/sulbactam",
    "hard_10": "Treatment failure ดูว่าเป็น high dose หรือไม่ ถ้าไม่ใช่เปลี่ยนเป็น high dose ถ้าเป็น high dose แล้วเปลี่ยนเป็น Amoxicillin/clavulanate",
    "hard_11": "Anaphylaxis ห้ามใช้ amoxicillin/clavulanate ให้ Doxycycline 100 mg วันละ 2 ครั้ง 5-7 วัน",
    "neg_2":  "ไม่ควรเปลี่ยนยาโดยไม่ผ่านแพทย์ การเปลี่ยน Rx เป็นหน้าที่ของแพทย์ผู้สั่ง",
    "neg_5":  "ห้ามจ่ายยาปฏิชีวนะ อาการเข้าได้กับ Acute Laryngitis เกิดจากไวรัส ให้ Supportive care",
    "neg_7":  "ปฏิเสธ Amoxicillin Modified Centor Score 0 คะแนน อาการเจ็บคอเกิดจากไวรัส ให้ Supportive care",
    "neg_8":  "ปฏิเสธการจ่ายยา เพิ่งเป็นมา 5 วัน ไม่ถึง 10 วัน ยังเข้าข่าย Viral Rhinosinusitis ให้ Supportive care",
    "neg_11": "Common cold เป็นการติดเชื้อไวรัส ไม่มีข้อบ่งใช้ยาปฏิชีวนะ ยืนยันไม่จ่ายยาปฏิชีวนะ",
    "incomplete_2":  "ต้องถามกลับ: มีไอหรือไม่ มีไข้หรือไม่ ต่อมน้ำเหลืองกดเจ็บหรือไม่ มีหนองที่ทอนซิลหรือไม่ อายุเท่าไหร่",
    "incomplete_5":  "ต้องถามกลับ: ผู้ป่วยอายุเท่าไหร่ มีไอหรือน้ำมูกร่วมด้วยไหม ต่อมน้ำเหลืองโตหรือกดเจ็บไหม มีประวัติแพ้ยาไหม",
    "incomplete_8":  "ต้องถามกลับ: อายุและน้ำหนักเด็ก ปวดหูข้างเดียวหรือสองข้าง ไข้สูงแค่ไหน มีหนองไหลออกจากหูไหม เคยได้ amoxicillin ใน 30 วันไหม",
    "incomplete_10": "ต้องถามกลับ: จำได้ไหมว่าแพ้ยาอะไร อาการแพ้เป็นอย่างไร เกิดขึ้นนานแค่ไหน ต้องรักษาด้วยยาอะไร เคยกินยากลุ่มเดิมอีกไหม",
}

# ------------------------------------------------------------------ #
# LLM-as-Judge
# ------------------------------------------------------------------ #
def judge_answer(question: str, expected: str, answer: str) -> tuple[bool, str]:
    judge_prompt = f"""คุณเป็นเภสัชกรผู้เชี่ยวชาญที่ตรวจคำตอบของ AI chatbot ด้านยา URI

คำถามจากเภสัชกร:
{question}

เฉลยที่ถูกต้อง:
{expected}

คำตอบของ bot:
{answer}

กรุณาตัดสินว่าคำตอบของ bot ถูกต้องตามหลักการแพทย์และเภสัชกรรมไหม โดยพิจารณาจาก:
1. การตัดสินใจจ่ายยา/ไม่จ่ายยาถูกต้องไหม
2. ชนิดยา ขนาดยา และระยะเวลาถูกต้องไหม
3. มีการแนะนำ supportive care ที่เหมาะสมไหม
4. ถ้าข้อมูลไม่ครบ มีการถามกลับไหม

ตอบในรูปแบบนี้เท่านั้น:
PASS หรือ FAIL
เหตุผล: (อธิบายสั้นๆ 1-2 ประโยค)"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(temperature=0),
        contents=judge_prompt,
    )

    if response is None or response.text is None:
        time.sleep(10)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(temperature=0),
            contents=judge_prompt,
        )

    result_text = response.text.strip() if response and response.text else "FAIL"
    is_pass = result_text.upper().startswith("PASS")
    reason = result_text.split("\n")[1] if "\n" in result_text else result_text
    return is_pass, reason

# ------------------------------------------------------------------ #
# Evaluate
# ------------------------------------------------------------------ #
def evaluate(vectorstore, bm25, docs, module):
    results = []
    passed = 0
    total = len(TEST_CASES)

    print(f"\nเริ่มทดสอบ {total} เคส (LLM-as-Judge)...")
    print("หมายเหตุ: ใช้เฉพาะ test set ที่ DSPy ไม่เคยเห็น\n")

    for i, case in enumerate(TEST_CASES):
        print(f"[{i+1}/{total}] {case['id']}...", end=" ", flush=True)

        answer, _ = ask(case["input"], vectorstore, bm25, docs, module)
        expected = EXPECTED_OUTPUTS.get(case["id"], "")
        is_pass, reason = judge_answer(case["input"], expected, answer)

        if is_pass:
            passed += 1
            print(f"✅ {reason}")
        else:
            print(f"❌ {reason}")

        results.append({
            "id":       case["id"],
            "level":    case["level"],
            "input":    case["input"],
            "expected": expected,
            "answer":   answer,
            "pass":     is_pass,
            "reason":   reason,
        })

        time.sleep(5)

    accuracy = passed / total * 100
    print(f"\n{'='*50}")
    print(f"ผลการทดสอบ (clean test set): {passed}/{total} เคส ({accuracy:.1f}%)")
    print(f"{'='*50}")

    for level in ["ง่าย", "ปานกลาง", "ยาก", "negative", "ไม่สมบูรณ์"]:
        level_cases  = [r for r in results if r["level"] == level]
        level_passed = sum(1 for r in level_cases if r["pass"])
        if level_cases:
            print(f"  {level}: {level_passed}/{len(level_cases)}")

    output_path = Path("evaluation_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total":    total,
                "passed":   passed,
                "accuracy": round(accuracy, 1),
                "note":     "Clean test set — DSPy ไม่เคยเห็น 25 เคสนี้",
            },
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\nบันทึกผลไว้ที่ → {output_path}")
    return results

if __name__ == "__main__":
    vectorstore, bm25, docs = load_retrievers()
    module = load_optimized_module()
    evaluate(vectorstore, bm25, docs, module)