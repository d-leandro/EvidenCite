#!/usr/bin/env bash
# Upload ONLY the two verified public-safe model adapters; never upload datasets or credentials.
set -euo pipefail
export AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1 AWS_PAGER=''
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
BUCKET="evidencite-demo-${ACCOUNT}-models"
aws s3api head-bucket --bucket "$BUCKET" >/dev/null
for MODE in A B; do
  SOURCE="$HOME/evidencite-mlflow-backup/test-revisado/granite_${MODE}/artifacts"
  for required in adapter_model.safetensors adapter_config.json; do
    test -s "$SOURCE/$required" || { echo "[STOP] Missing $SOURCE/$required" >&2; exit 2; }
  done
  echo "[UPLOAD] Granite adapter $MODE -> s3://$BUCKET/granite_$MODE/"
  aws s3 sync "$SOURCE/" "s3://$BUCKET/granite_$MODE/" \
    --no-follow-symlinks --only-show-errors \
    --exclude '*' \
    --include 'adapter_config.json' \
    --include 'adapter_model.safetensors' \
    --include 'tokenizer*' \
    --include 'special_tokens_map.json' \
    --include 'vocab.*' \
    --include 'merges.txt' \
    --include 'added_tokens.json' \
    --include 'chat_template.jinja'
done
printf '[OK] Granite A/B adapters uploaded to the existing persistent model bucket\n'
