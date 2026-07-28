# Toxic Comment Classification & Redaction — Flask web app / API
#
# Ships the redaction web app (src/toxicity_web_app.py). It does NOT bundle a
# trained model, the Jigsaw dataset, or GloVe embeddings (all gitignored) —
# without a trained model at /app/saved_models, the toxicity-check endpoints
# return HTTP 503 rather than mock predictions. Train via the notebook first
# and mount the result at /app/saved_models to get live predictions.

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

WORKDIR /app/src
CMD ["python", "toxicity_web_app.py"]
