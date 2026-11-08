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

COPY api/ ./api/
COPY src/ ./src/
COPY app/ ./app/
COPY configs/ ./configs/

RUN mkdir -p data/raw data/processed

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
