FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY api/requirements-docker.txt /tmp/requirements.txt

# CPU-only PyTorch: no CUDA runtime.
RUN pip install --no-cache-dir torch \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    -r /tmp/requirements.txt

# Compatible CPU build of torchvision.
RUN pip install --no-cache-dir --no-deps --force-reinstall \
    torchvision==0.29.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --no-deps peft==0.21.2

# Separate scikit-learn version for serialized TF-IDF models.
RUN python -m venv --system-site-packages /opt/tfidf-venv && \
    /opt/tfidf-venv/bin/pip install --no-cache-dir "scikit-learn==1.6.1"


COPY api/ ./api/
COPY src/ ./src/
COPY app/ ./app/
COPY configs/ ./configs/

# SciBERT B — verified local checkpoint
COPY deploy-assets/scibert_B/ /opt/evidencite-models/scibert_B/
ENV EVIDENCITE_SCIBERT_B_PATH=/opt/evidencite-models/scibert_B

# SciBERT A — Configuration A
COPY deploy-assets/scibert_A/ /opt/evidencite-models/scibert_A/
ENV EVIDENCITE_SCIBERT_A_PATH=/opt/evidencite-models/scibert_A

RUN mkdir -p data/raw data/processed

# TF-IDF models trained with scikit-learn 1.6.1.
COPY deploy-assets/tfidf_A/ /opt/evidencite-models/tfidf_A/
COPY deploy-assets/tfidf_B/ /opt/evidencite-models/tfidf_B/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
