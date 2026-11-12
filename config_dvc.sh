#!/usr/bin/env bash
# Configura el remoto DVC (Google Drive) en .dvc/config.local del repo desde el que se ejecuta.
# Requiere ID_CARPETA (el ID de la carpeta de Drive) y una forma de entrar:
# - GDRIVE_CLIENT_ID y GDRIVE_CLIENT_SECRET: el cliente OAuth; el primer pull o push abre el navegador.
# - GDRIVE_SA_JSON: la llave JSON de una cuenta de servicio con acceso a la carpeta; no usa navegador (Colab).
# Usa el dvc del PATH si hay uno y, si no, el del proyecto con `uv run dvc`.
set -euo pipefail

error() { echo "Error: $*" >&2; exit 1; }

[[ -n "${ID_CARPETA:-}" ]] || error "define ID_CARPETA antes de ejecutar este script."
if [[ -n "${GDRIVE_SA_JSON:-}" ]]; then
  [[ -f "$GDRIVE_SA_JSON" ]] || error "no existe GDRIVE_SA_JSON=$GDRIVE_SA_JSON."
elif [[ -z "${GDRIVE_CLIENT_ID:-}" || -z "${GDRIVE_CLIENT_SECRET:-}" ]]; then
  error "define GDRIVE_CLIENT_ID y GDRIVE_CLIENT_SECRET (o GDRIVE_SA_JSON) antes de ejecutar este script."
fi
if command -v dvc >/dev/null 2>&1; then dvc=(dvc); else dvc=(uv run dvc); fi
modificar() { "${dvc[@]}" remote modify --local gdrive-evidencite "$@"; }

"${dvc[@]}" remote add --local -d -f gdrive-evidencite "gdrive://$ID_CARPETA"
if [[ -n "${GDRIVE_SA_JSON:-}" ]]; then
  # Ruta absoluta: DVC busca la llave desde la carpeta en que se ejecuta.
  llave="$(cd "$(dirname "$GDRIVE_SA_JSON")" && pwd)/$(basename "$GDRIVE_SA_JSON")"
  modificar gdrive_use_service_account true
  modificar gdrive_service_account_json_file_path "$llave"
  echo "Remoto DVC configurado con la cuenta de servicio."
else
  modificar gdrive_client_id "$GDRIVE_CLIENT_ID"
  modificar gdrive_client_secret "$GDRIVE_CLIENT_SECRET"
  echo "Remoto DVC configurado. Ahora ejecuta: uv run dvc pull"
fi
