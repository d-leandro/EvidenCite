#!/usr/bin/env bash
# Store tunnel endpoint and shared token in SSM. Never print or commit the token.
set -euo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^https://[A-Za-z0-9.-]+$ ]]; then
  echo 'Uso: bash tools/granite_publish_aws.sh https://subdominio.trycloudflare.com' >&2
  exit 2
fi
ENV_FILE="$HOME/.config/evidencite/granite-local.env"
[[ -f "$ENV_FILE" ]] || { echo "[STOP] Falta $ENV_FILE" >&2; exit 2; }
TOKEN="$(python3 - "$ENV_FILE" <<'PYTOKEN'
import json,sys
for line in open(sys.argv[1], encoding='utf-8'):
    if line.startswith('GRANITE_SHARED_TOKEN='):
        print(json.loads(line.partition('=')[2]))
        break
PYTOKEN
)"
[[ "${#TOKEN}" -ge 32 ]] || { echo '[STOP] Token inválido' >&2; exit 2; }
# Uses your AWS_PROFILE (evidencite-lab). These params are account-scoped in us-east-1.
aws ssm put-parameter --region us-east-1 \
  --name /evidencite/prod/granite-url --type String --value "$1" --overwrite >/dev/null
aws ssm put-parameter --region us-east-1 \
  --name /evidencite/prod/granite-shared-token --type SecureString --value "$TOKEN" --overwrite >/dev/null
unset TOKEN
printf '[OK] Cloudflare URL stored in SSM: %s\n' "$1"
echo '[OK] Token stored as SSM SecureString (hidden)'
echo '[NEXT] Commit and push patch; run Deploy EvidenCite AWS EC2 on feature/app-infrastructure.'
