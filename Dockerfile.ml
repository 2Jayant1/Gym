FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements-ml.txt ./requirements-ml.txt
RUN pip install --no-cache-dir -r requirements-ml.txt

COPY serving/ ./serving/
COPY training/ ./training/
COPY configs/ ./configs/
COPY train.py ./train.py

EXPOSE 8001

CMD ["python", "-m", "uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8001"]
