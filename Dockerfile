# =============================================================================
# Digital Twin Career Engine — Dockerfile
# =============================================================================
# Сборка:  docker build -t career-engine .
# Запуск:  docker run -p 8501:8501 --env-file .env career-engine
# (или через docker-compose up --build)
# =============================================================================

FROM python:3.11-slim

# --- системные зависимости (нужны для pdfplumber, telethon и компиляции C-ext) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpoppler-cpp-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- зависимости Python ---
# Сначала копируем только файлы зависимостей, чтобы Docker кешировал этот слой
COPY requirements.txt requirements.docker.txt ./

# Установка всех зависимостей (core + optional — всё в одном слое)
RUN pip install --no-cache-dir -r requirements.docker.txt

# --- исходный код проекта ---
COPY . .

# --- порт Streamlit ---
EXPOSE 8501

# --- healthcheck (используется docker-compose depends_on + CI) ---
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -sf http://localhost:8501/_stcore/health || exit 1

# --- запуск ---
CMD ["streamlit", "run", "app/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none"]
