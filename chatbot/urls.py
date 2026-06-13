from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/session/new/', views.new_session, name='new_session'),
    path('api/session/<uuid:session_id>/', views.get_session, name='get_session'),
    path('api/chat/<uuid:session_id>/', views.chat, name='chat'),
    path('api/guideline/<str:name>/', views.serve_guideline, name='serve_guideline'),
    path('testcase/', views.testcase_page, name='testcase'),
    path('api/testcase/run/', views.run_testcase, name='run_testcase'),
    path('api/testcase/results/', views.get_testcase_results, name='get_testcase_results'),
    path('api/testcase/cases/', views.get_all_testcases, name='get_all_testcases'),
    path('api/testcase/run-single/', views.run_single_testcase, name='run_single_testcase'),
]