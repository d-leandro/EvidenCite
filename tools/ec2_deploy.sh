#!/usr/bin/env bash
# Intended for AWS SSM Run Command on the EC2 instance, not local WSL.
set -euo pipefail

BUCKET="${EVIDENCITE_MODELS_BUCKET:-}"
if [[ ! "$BUCKET" =~ ^[a-z0-9][a-z0-9.-]{2,62}$ ]]; then
  echo '[STOP] Configure EVIDENCITE_MODELS_BUCKET con el bucket privado.' >&2
  exit 2
fi
if [[ "$(id -u)" != 0 ]]; then
  echo '[STOP] Este script requiere root en EC2 (sudo).' >&2
  exit 2
fi
if [[ ! -s /srv/evidencite/secrets/.env ]]; then
  echo '[STOP] Falta /srv/evidencite/secrets/.env en EC2.' >&2
  exit 2
fi
if ! grep -q '^GEMINI_API_KEY=.' /srv/evidencite/secrets/.env; then
  echo '[STOP] Falta GEMINI_API_KEY en el envfile protegido de EC2.' >&2
  exit 2
fi
chmod 600 /srv/evidencite/secrets/.env

cd /srv/evidencite/repo
for model in scibert_A scibert_B tfidf_A tfidf_B granite_base granite_A granite_B; do
  echo "[S3] Recuperando ${model}"
  mkdir -p "/srv/evidencite/models/$model"
  aws s3 sync "s3://${BUCKET}/${model}/" "/srv/evidencite/models/${model}/" --only-show-errors
done

for model in scibert_A scibert_B; do
  test -s "/srv/evidencite/models/$model/config.json"
  test -s "/srv/evidencite/models/$model/model.safetensors"
done
for model in tfidf_A tfidf_B; do
  test -s "/srv/evidencite/models/$model/tfidf_pipeline.joblib"
  test -s "/srv/evidencite/models/$model/input_representation.json"
done
test -s /srv/evidencite/models/granite_base/config.json
find /srv/evidencite/models/granite_base -name '*.safetensors' -size +0c -print -quit | grep -q .
for model in granite_A granite_B; do
  test -s "/srv/evidencite/models/$model/adapter_config.json"
done

echo '[OK] Artefactos mínimos presentes; iniciando Compose CLOUD'
docker compose --env-file /srv/evidencite/secrets/.env -f compose.cloud.yaml config --quiet
docker compose --env-file /srv/evidencite/secrets/.env -f compose.cloud.yaml up -d --build --wait --wait-timeout 1800

echo '[OK] EvidenCite desplegado. Interfaz solo vía SSM en localhost:8080.'
