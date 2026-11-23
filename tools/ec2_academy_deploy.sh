#!/usr/bin/env bash
# Run on the EC2 instance through SSM as root. No manual .env setup required.
set -euo pipefail

BUCKET="${EVIDENCITE_MODELS_BUCKET:-}"
[[ "$BUCKET" =~ ^[a-z0-9][a-z0-9.-]{2,62}$ ]] || { echo '[STOP] EVIDENCITE_MODELS_BUCKET no configurado'; exit 2; }
[[ "$(id -u)" == 0 ]] || { echo '[STOP] Se necesita root/sudo'; exit 2; }
# Resolve the *current* EC2 public IP using IMDSv2. No hard-coded hostname.
TOKEN="$(curl -fsS --retry 3 --connect-timeout 3 --max-time 10 \
  -X PUT -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' \
  http://169.254.169.254/latest/api/token)"
PUBLIC_IP="$(curl -fsS --retry 3 --connect-timeout 3 --max-time 10 \
  -H "X-aws-ec2-metadata-token: $TOKEN" \
  http://169.254.169.254/latest/meta-data/public-ipv4)"
[[ "$PUBLIC_IP" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] || { echo '[STOP] IP publica no valida'; exit 2; }
DOMAIN="evidencite.${PUBLIC_IP//./-}.sslip.io"
unset TOKEN

# Retrieve SecureString with the EC2 instance role. Never send API keys over SSM commands
# or Terraform variables. Store the Compose env only on EC2 with mode 0600.
umask 077
install -d -m 0700 /srv/evidencite/secrets
ENV_FILE=/srv/evidencite/secrets/.env
GEMINI_API_KEY="$(aws ssm get-parameter \
  --region us-east-1 \
  --name /evidencite/prod/gemini-api-key \
  --with-decryption \
  --query Parameter.Value --output text)"
[[ -n "$GEMINI_API_KEY" && "$GEMINI_API_KEY" != 'None' ]] || { echo '[STOP] Gemini key unavailable'; exit 2; }
export GEMINI_API_KEY
python3 - "$ENV_FILE" "$DOMAIN" <<'PYENV'
import os
import sys
import tempfile
from pathlib import Path

path = Path(sys.argv[1])
domain = sys.argv[2]
key = os.environ['GEMINI_API_KEY']
# Compose-compatible single-quoted dotenv value; no interpolation of $ characters.
def quote(value):
    if any(c in value for c in "\r\n"):
        raise ValueError('Unexpected multiline secret')
    return "'" + value.replace('\\', '\\\\').replace("'", "\\'") + "'"
fd, tmp = tempfile.mkstemp(prefix='.env.', dir=path.parent, text=True)
try:
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as file:
        file.write(f'EVIDENCITE_DOMAIN={domain}\n')
        file.write('GOOGLE_GENAI_USE_VERTEXAI=false\n')
        file.write('GEMINI_API_KEY=' + quote(key) + '\n')
    os.replace(tmp, path)
finally:
    if os.path.exists(tmp):
        os.unlink(tmp)
PYENV
unset GEMINI_API_KEY

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

# Nginx resolves the API container IP at startup.
# Refresh its upstream after Docker recreates the API.
docker compose --env-file "$ENV_FILE" -f compose.academy.yaml restart gateway

# Allow the restarted gateway to accept connections before health checks.
for attempt in $(seq 1 15); do
  if curl -fsS --max-time 5 http://127.0.0.1:8080/health >/dev/null 2>&1; then
    break
  fi
  if [ "$attempt" -eq 15 ]; then
    echo '[STOP] Gateway unavailable after restart' >&2
    exit 1
  fi
  sleep 2
done

curl --fail --silent --show-error --max-time 10 http://127.0.0.1:8080/health
# Verify that the API reports all six real model configurations as available.
python3 - <<'PYCHECK'
import json
from urllib.request import urlopen
with urlopen('http://127.0.0.1:8080/api/v1/models', timeout=10) as response:
    models = json.load(response)['models']
expected = {'tfidf-a', 'tfidf-b', 'scibert-a', 'scibert-b', 'gemini-a', 'gemini-b'}
ready = {m['id'] for m in models if m['available']}
missing = sorted(expected - ready)
if missing:
    raise SystemExit('[STOP] Unavailable models: ' + ', '.join(missing))
print('[OK] Six model configurations registered')
PYCHECK
printf '\n[OK] EvidenCite deployed: https://%s\n' "$DOMAIN"
