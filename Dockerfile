FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1

WORKDIR /srv
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY frontend ./frontend
COPY scripts ./scripts
COPY eval ./eval

# the app writes the dataset and embedding cache under data/, so that folder must be writable
RUN useradd --create-home --uid 1000 signal && mkdir -p data/cache && chown -R signal /srv
USER signal

ENV PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/me' % os.environ.get('PORT','8000'), timeout=4)"

# one worker on purpose: rate limits and the loaded dataset live in memory
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
