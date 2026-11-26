# Persistent F1 Pit-Wall Backend (FastAPI + 4 Hz SignalR WebSockets + faster-whisper)
# Deployable on Render, Railway, Fly.io, Cloud Run, or local Docker.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install full backend dependencies (includes websockets + faster-whisper + pyarrow)
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r backend/requirements.txt

# Copy contract, backend (with pre-trained ML .joblib + pre_race_raw.parquet), and PWA shell
COPY contract ./contract
COPY backend ./backend
COPY frontend ./frontend

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/api/health', timeout=4)" || exit 1

CMD ["sh", "-c", "uvicorn backend.server.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
