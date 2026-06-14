FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download embedding model ตอน build ไม่ต้องโหลดตอน runtime
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3')"

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "python manage.py migrate && gunicorn pharmacy_backend.wsgi:application --bind 0.0.0.0:8000 --workers 1 --timeout 300 --preload"]
