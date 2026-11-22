#!/usr/bin/env bash
# Run on the EC2 instance through SSM as root, AFTER configuring S3 and Gemini locally.
set -euo pipefail

BUCKET="${EVIDENCITE_MODELS_BUCKET:-}"
[[ "$BUCKET" =~ ^[a-z0-9][a-z0-9.-]{2,62}$ ]] || { echo '[STOP] EVIDENCITE_MODELS_BUCKET no configurado'; exit 2; }
[[ "$(id -u)" == 0 ]] || { echo '[STOP] Se necesita root/sudo'; exit 2; }
ENV_FILE=/srv/evidencite/secrets/.env
[[ -s "$ENV_FILE" ]] || { echo '[STOP] Crear el envfile de Gemini en EC2 primero'; exit 2; }
chmod 600 "$ENV_FILE"
# Do not print the secret, even for debugging.

cd /srv/evidencite/repo
for model in tfidf_A tfidf_B scibert_A scibert_B; do
  mkdir -p "/srv/evidencite/models/$model"
  echo "[S3] $model"
  aws s3 sync "s3://$BUCKET/$model/" "/srv/evidencite/models/$model/" --only-show-errors
 done
for model in tfidf_A tfidf_B; do
  test -s "/srv/evidencite/models/$model/tfidf_pipeline.joblib"
  test -s "/srv/evidencite/models/$model/input_representation.json"
 done
for model in scibert_A scibert_B; do
  test -s "/srv/evidencite/models/$model/config.json"
  test -s "/srv/evidencite/models/$model/model.safetensors"
  test -s "/srv/evidencite/models/$model/tokenizer.json"
  test -s "/srv/evidencite/models/$model/tokenizer_config.json"
 done

install -d -m 0700 /srv/evidencite/metrics
# Check that the configuration is valid without showing resolved environment variables.
docker compose --env-file "$ENV_FILE" -f compose.academy.yaml config --quiet
# Only changes this isolated Compose project; does not affect local WSL Granite.
docker compose --env-file "$ENV_FILE" -f compose.academy.yaml up -d --build --wait --wait-timeout 1800

curl --fail --silent --show-error --max-time 10 http://127.0.0.1:8080/health
printf '\n[OK] EvidenCite Academy Lite saludable. SSM port-forward: localhost:8080\n'
