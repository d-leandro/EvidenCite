#!/usr/bin/env bash
# Collect ONLY approved, lightweight trained TF-IDF artifacts for the GitHub rubric.
# Never git-add automatically; never fetch or commit data, secrets or heavy weights.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo '[STOP] Ejecuta desde un repositorio Git.' >&2; exit 2;
}
[[ -f "$ROOT/compose.aws.yaml" && -f "$ROOT/README.md" ]] || {
  echo '[STOP] La carpeta actual no parece EvidenCite.' >&2; exit 2;
}

MAX_BYTES=$((45*1024*1024))  # Limit below GitHub 100 MiB hard cap.
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
AWS_REGION="${AWS_REGION:-us-east-1}"
ACCOUNT=''
BUCKET=''
INCLUDED=0
SKIPPED=0

function s3_bucket() {
  if [[ -z "$BUCKET" ]]; then
    if ! command -v aws >/dev/null 2>&1; then
      echo '[STOP] No existe AWS CLI y faltan modelos locales.' >&2; return 1
    fi
    ACCOUNT="$(aws sts get-caller-identity --query Account --output text)" || return 1
    [[ "$ACCOUNT" =~ ^[0-9]{12}$ ]] || { echo '[STOP] Cuenta AWS inválida.' >&2; return 1; }
    BUCKET="evidencite-demo-${ACCOUNT}-models"
    echo "[S3] Bucket privado detectado: $BUCKET"
  fi
}

for MODE in A B; do
  NAME="tfidf_${MODE}"
  LOCAL_DIR="$ROOT/deploy-assets/$NAME"
  SRC="$LOCAL_DIR"
  echo
  echo "===== $NAME ====="
  if [[ ! -s "$SRC/tfidf_pipeline.joblib" || ! -s "$SRC/input_representation.json" ]]; then
    s3_bucket || { echo "[WARN] No se pudo recuperar $NAME" >&2; SKIPPED=$((SKIPPED+1)); continue; }
    SRC="$WORK/$NAME"
    mkdir -p "$SRC"
    echo "[S3] Copiando exclusivamente los dos artefactos requeridos de $NAME"
    if ! aws s3 cp "s3://$BUCKET/$NAME/tfidf_pipeline.joblib" "$SRC/tfidf_pipeline.joblib" --region "$AWS_REGION" --only-show-errors || \
       ! aws s3 cp "s3://$BUCKET/$NAME/input_representation.json" "$SRC/input_representation.json" --region "$AWS_REGION" --only-show-errors; then
      echo "[WARN] Falló la lectura de S3: $NAME. Revisa credenciales, región y ruta." >&2
      SKIPPED=$((SKIPPED+1))
      continue
    fi
  else
    echo '[LOCAL] Encontrados los artefactos en deploy-assets/'
  fi

  python3 - "$SRC/input_representation.json" <<'PY'
import json,sys
with open(sys.argv[1],encoding='utf-8') as f: json.load(f)
PY
  BYTES="$(stat -c '%s' "$SRC/tfidf_pipeline.joblib")"
  if (( BYTES > MAX_BYTES )); then
    echo "[WARN] $NAME mide ${BYTES} bytes, supera 45 MiB. NO lo subiremos a Git."
    echo '       Utiliza almacenamiento externo autorizado y documenta acceso/hash.'
    SKIPPED=$((SKIPPED+1))
    continue
  fi

  DEST="$ROOT/models/$NAME"
  mkdir -p "$DEST"
  cp -- "$SRC/tfidf_pipeline.joblib" "$DEST/tfidf_pipeline.joblib"
  cp -- "$SRC/input_representation.json" "$DEST/input_representation.json"
  chmod 0644 "$DEST/tfidf_pipeline.joblib" "$DEST/input_representation.json"
  echo "[OK] Modelo preparado: $NAME (${BYTES} bytes)"
  sha256sum "$DEST/tfidf_pipeline.joblib" "$DEST/input_representation.json"
  INCLUDED=$((INCLUDED+1))
done

printf '\n[RESUMEN] %s configuraciones preparadas; %s omitidas.\n' "$INCLUDED" "$SKIPPED"
if (( INCLUDED > 0 )); then
  echo 'Revisa tamaños y procedencia ANTES de preparar el commit:'
  echo "  git status -sb"
  echo "  git add -f models/README.md models/tfidf_*/tfidf_pipeline.joblib models/tfidf_*/input_representation.json"
  echo 'NOTA: Si solo preparaste una configuración, ajusta la lista de git add a los archivos realmente presentes.'
else
  echo '[INFO] No se copiaron pesos; conserva models/README.md como catálogo verificable.'
fi
