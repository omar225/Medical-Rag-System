# Two stages: build the React UI with Node, then run FastAPI (which also serves the UI) on Python.
FROM node:22-alpine AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    OLLAMA_BASE_URL=http://ollama:11434 \
    RETRIEVAL_INDEX_DIR=/app/storage/vector_store/dense \
    INDEX_BUILD_WAIT_SECONDS=900 \
    FRONTEND_DIST=/app/frontend/dist
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY data ./data
COPY --from=frontend /frontend/dist ./frontend/dist
RUN useradd --create-home --uid 1000 app && mkdir -p storage/vector_store && chown -R app:app storage
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["uvicorn", "medical_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
