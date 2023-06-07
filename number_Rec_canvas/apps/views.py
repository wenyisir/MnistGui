from django.shortcuts import render, HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import os

# Create your views here.
@csrf_exempt
def process_data(request):
    if request.method == 'GET':
        return render(request, 'canvas.html')
    else:
        data = json.loads(request.body.decode('utf-8'))  # 解析JSON数据
        print(data)
        from NumberReco import number_recogntion
        labels = number_recogntion(data)
        print(f'number is: {labels}')
        response_data = {'result': labels}
        return JsonResponse(response_data)

