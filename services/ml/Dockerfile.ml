FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app/services/ml

# Install pinned requirements
COPY services/ml/requirements-ml.txt ./requirements-ml.txt
RUN pip install --no-cache-dir -r requirements-ml.txt

# Copy ML service code
COPY services/ml/ ./

EXPOSE 8001

CMD ["python", "-m", "uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8001"]
