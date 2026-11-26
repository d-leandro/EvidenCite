#!/usr/bin/env bash
# Run ON EC2 via SSM, where Docker is available. No Hugging Face token required for public model.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE=/srv/evidencite/models/granite_base
install -d -m 0755 "$BASE"
docker run --rm \
  -v "$BASE:/models" \
  -v "$ROOT/tools/download_granite_base.py:/download.py:ro" \
  -e HF_HUB_DISABLE_TELEMETRY=1 \
  python:3.12-slim sh -ec \
  'pip install --disable-pip-version-check -q "huggingface_hub>=0.34,<2" && python /download.py /models'
