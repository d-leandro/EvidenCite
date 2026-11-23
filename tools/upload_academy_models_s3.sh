#!/usr/bin/env bash
# Upload only inference assets that the AWS Academy Lite deployment needs.
# No Granite weights, private datasets, or credentials.
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo 'Uso: bash tools/upload_academy_models_s3.sh <bucket-privado>' >&2
  exit 2
fi
BUCKET="$1"
if [[ ! "$BUCKET" =~ ^[a-z0-9][a-z0-9.-]{2,62}$ ]]; then
  echo '[STOP] Nombre de bucket inválido' >&2
  exit 2
fi
ROOT="$(git rev-parse --show-toplevel)"
NAMES=(tfidf_A tfidf_B scibert_A scibert_B)
for model in "${NAMES[@]}"; do
  directory="$ROOT/deploy-assets/$model"
  if [[ ! -d "$directory" ]]; then
    echo "[STOP] Falta $directory" >&2
    exit 2
  fi
  if [[ "$model" == tfidf_* ]]; then
    for filename in tfidf_pipeline.joblib input_representation.json; do
      [[ -s "$directory/$filename" ]] || { echo "[STOP] Falta $directory/$filename" >&2; exit 2; }
    done
  else
    for filename in config.json model.safetensors tokenizer.json tokenizer_config.json; do
      [[ -s "$directory/$filename" ]] || { echo "[STOP] Falta $directory/$filename" >&2; exit 2; }
    done
  fi
done

for model in "${NAMES[@]}"; do
  echo "[UPLOAD] $model -> s3://$BUCKET/$model/"
  aws s3 sync "$ROOT/deploy-assets/$model/" "s3://$BUCKET/$model/" \
    --exclude '*.lock' --exclude '*.log' --only-show-errors
 done
 echo '[OK] Solo los 4 artefactos TF-IDF y SciBERT se copiaron a S3.'
