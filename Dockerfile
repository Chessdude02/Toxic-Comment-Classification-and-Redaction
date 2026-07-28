# Toxic Comment Classification & Redaction — Flask web app / API
#
# Ships the redaction web app (src/toxicity_web_app.py). It does NOT bundle a
# trained model, the Jigsaw dataset, or GloVe embeddings (all gitignored) —
# without a trained model at /app/saved_models, the app automatically falls
# back to the rule-based heuristic detector (src/heuristic_fallback.py)
# instead of failing outright. Train via the notebook and mount the result
# at /app/saved_models to get the trained classifier's predictions instead.

FROM python:3.11-slim

WORKDIR /app

# System deps for scientific-Python wheels that occasionally need a compiler.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

ENV PYTHONUNBUFFERED=1
# Cloud platforms (Render, Railway, Fly.io, ...) inject their own $PORT;
# 5000 is just the local-run default.
ENV PORT=5000

WORKDIR /app/src

# gunicorn, not the Flask dev server used by `python toxicity_web_app.py` -
# that server isn't designed for production traffic. Long --timeout because
# the first request after a cold start pays for TensorFlow's import and
# model loading. Shell form so $PORT expands at container start.
#
# --workers 1: each gunicorn worker is a separate process that loads its own
# full copy of TensorFlow + the model into memory. On a 512MB-RAM free-tier
# instance (Render, Railway, etc.), 2 workers reliably triggers an OOM kill -
# confirmed in practice ("Out of memory (used over 512Mi)"). --threads 4
# gives request concurrency without duplicating that memory, since threads
# share one process's memory. Override WEB_CONCURRENCY if you deploy
# somewhere with more RAM and want real multi-process concurrency instead.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers ${WEB_CONCURRENCY:-1} --threads 4 --timeout 120 toxicity_web_app:app"]
