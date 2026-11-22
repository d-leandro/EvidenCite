#!/usr/bin/env bash
# Upload only model artifacts, never text datasets, logs or credentials.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo 'Uso: bash tools/upload_models_s3.sh <bucket-modelos-de-terraform>' >&2
  exit 2
fi
BUCKET="$1"
if [[ ! "$BUCKET" =~ ^[a-z0-9][a-z0-9.-]{2,62}$ ]]; then
  echo 'Nombre de bucket inválido.' >&2
  exit 2
fi
ROOT="$(git rev-parse --show-toplevel)"
BASE="${HOME}/evidencite-mlflow-backup/test-revisado"

declare -a names=(scibert_A scibert_B tfidf_A tfidf_B granite_base granite_A granite_B)
declare -a dirs=(
  "$ROOT/deploy-assets/scibert_A"
  "$ROOT/deploy-assets/scibert_B"
  "$ROOT/deploy-assets/tfidf_A"
  "$ROOT/deploy-assets/tfidf_B"
  "$ROOT/deploy-assets/granite_base"
  "$BASE/granite_A/artifacts"
  "$BASE/granite_B/artifacts"
)

for i in "${!names[@]}"; do
  if [[ ! -d "${dirs[$i]}" ]] || [[ -z "$(find "${dirs[$i]}" -type f -print -quit)" ]]; then
    echo "[STOP] Faltan artefactos: ${names[$i]} -> ${dirs[$i]}" >&2
    exit 2
  fi
done

echo 'Subiremos únicamente los siete directorios de modelos.'
for i in "${!names[@]}"; do
  echo "[UPLOAD] ${names[$i]} -> s3://$BUCKET/${names[$i]}/"
  aws s3 sync "${dirs[$i]}/" "s3://${BUCKET}/${names[$i]}/" \
    --exclude '*.lock' --only-show-errors
 done
 echo '[OK] Artefactos cargados en S3 (revisar costos de almacenamiento).'
