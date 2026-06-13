import sys
import os
from pathlib import Path
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json

# เพิ่ม path ของ pharmacy_chatbot
RAG_PATH = Path('/pharmacy_chatbot')
sys.path.insert(0, str(RAG_PATH))

from rag_pipeline.rag_chain_dspy import load_retrievers, ask, load_optimized_module
from .models import ChatSession, ChatMessage

# โหลด RAG ครั้งเดียวตอนเริ่ม
print("Loading RAG pipeline...")
vectorstore, bm25, docs = load_retrievers()
module = load_optimized_module()
print("RAG pipeline loaded.")


def index(request):
    """หน้าหลัก Chat"""
    sessions = ChatSession.objects.all()[:20]
    return render(request, 'chatbot/index.html', {'sessions': sessions})


@csrf_exempt
@require_http_methods(["POST"])
def new_session(request):
    """สร้าง session ใหม่"""
    session = ChatSession.objects.create()
    return JsonResponse({
        'session_id': str(session.id),
        'title': session.title
    })


def get_session(request, session_id):
    """ดึงข้อความใน session"""
    session = get_object_or_404(ChatSession, id=session_id)
    messages = session.messages.all()
    return JsonResponse({
        'session_id': str(session.id),
        'title': session.title,
        'messages': [
            {'role': m.role, 'content': m.content}
            for m in messages
        ]
    })


@csrf_exempt
@require_http_methods(["POST"])
def chat(request, session_id):
    """รับข้อความและตอบกลับ"""
    session = get_object_or_404(ChatSession, id=session_id)
    data = json.loads(request.body)
    user_message = data.get('message', '')

    # บันทึก user message
    ChatMessage.objects.create(
        session=session,
        role='user',
        content=user_message
    )

    

   # ดึง history จาก DB
    all_messages = list(session.messages.all())
    history = []
    for i in range(0, len(all_messages) - 1, 2):
        if (all_messages[i].role == 'user' and 
            i + 1 < len(all_messages) and 
            all_messages[i+1].role == 'assistant'):
            history.append({
                'user': all_messages[i].content,
                'assistant': all_messages[i+1].content
            })

    # ถามค่ะ
    answer, _ = ask(
        user_message, vectorstore, bm25, docs, module, history
    )
    # บันทึก assistant message
    ChatMessage.objects.create(
        session=session,
        role='assistant',
        content=answer
    )

    # อัพเดท title ถ้ายังเป็น default
    if session.title == "เคสใหม่":
        session.title = user_message[:50]
        session.save()

    return JsonResponse({'answer': answer})


from django.http import FileResponse
import os

def serve_guideline(request, name):
    page = request.GET.get('page', '1')  # รับ ?page=22
    files = {
        'aafp': '/pharmacy_chatbot/data/AAFP_2022_Original.pdf',
        'thai_uri': '/pharmacy_chatbot/data/Thai URI Children.pdf',
    }
    path = files.get(name)
    if path and os.path.exists(path):
        return FileResponse(open(path, 'rb'), content_type='application/pdf')
    from django.http import Http404
    raise Http404


import threading
import json

# เก็บ status การรัน
test_status = {'status': 'idle', 'results': [], 'summary': {}}

def testcase_page(request):
    """หน้า Test Case Dashboard"""
    return render(request, 'chatbot/testcase.html')

def run_testcase(request):
    """รัน evaluate แบบ background"""
    if request.method != 'POST':
        from django.http import HttpResponseNotAllowed
        return HttpResponseNotAllowed(['POST'])

    global test_status
    test_status = {'status': 'running', 'results': [], 'summary': {}}

    def run():
        global test_status
        try:
            sys.path.insert(0, str(RAG_PATH))
            from evaluate import TEST_CASES, EXPECTED_OUTPUTS, judge_answer

            total = len(TEST_CASES)
            passed = 0
            results = []

            for case in TEST_CASES:
                answer, _ = ask(case['input'], vectorstore, bm25, docs, module, [])
                expected = EXPECTED_OUTPUTS.get(case['id'], '')
                is_pass, reason = judge_answer(case['input'], expected, answer)

                if is_pass:
                    passed += 1

                results.append({
                    'id': case['id'],
                    'level': case['level'],
                    'input': case['input'],
                    'expected': expected,
                    'answer': answer,
                    'pass': is_pass,
                    'reason': reason,
                    'score': 1.0 if is_pass else 0.0,
                })

                test_status['results'] = results
                test_status['summary'] = {
                    'total': total,
                    'passed': passed,
                    'accuracy': round(passed / len(results) * 100, 1),
                }

                import time
                time.sleep(5)

            test_status['status'] = 'done'

        except Exception as e:
            test_status['status'] = 'error'
            print(f"Evaluate error: {e}")

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    return JsonResponse({'status': 'started'})

def get_testcase_results(request):
    """ดึงผล test cases"""
    return JsonResponse({
        'status': test_status['status'],
        'results': test_status['results'],
        'summary': test_status['summary'],
    })


def get_all_testcases(request):
    try:
        from all_testcases import ALL_TEST_CASES
        cases = [
            {
                'id': c['id'],
                'level': c['level'],
                'input': c['input'],
                'expected': c.get('expected', ''),
            }
            for c in ALL_TEST_CASES
        ]
        return JsonResponse({'cases': cases})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def run_single_testcase(request):
    """รัน test case ทีละอัน"""
    try:
        data = json.loads(request.body)
        case_id = data.get('id')
        input_text = data.get('input')
        expected = data.get('expected', '')

        from evaluate import judge_answer

        answer, _ = ask(input_text, vectorstore, bm25, docs, module, [])
        is_pass, reason = judge_answer(input_text, expected, answer)

        return JsonResponse({
            'id': case_id,
            'answer': answer,
            'pass': is_pass,
            'reason': reason,
            'score': 1.0 if is_pass else 0.0,
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
