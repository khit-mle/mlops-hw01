FROM python:3.12

WORKDIR /app

COPY requirements-lab3.txt .
RUN pip install --no-cache-dir -r requirements-lab3.txt

COPY src/ .
COPY models/ ./models/

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
