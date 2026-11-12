#!/usr/bin/env bash
# Funciones comunes de correr_experimento.sh y probar_variantes.sh. No se ejecuta solo: lo cargan los dos scripts con
# `source`, después de definir REPO (la raíz del repo) y el arreglo VARIABLES (los CLAVE=valor de la corrida).

PY="${PYTHON:-python3}"
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1 MLFLOW_DISABLE_AGENT_HINT=1 MLFLOW_DISABLE_TELEMETRY=true

# El último valor de una CLAVE en VARIABLES (los argumentos ganan); si no está, la variable de entorno; si no, $2.
valor() {
  local clave="$1" defecto="${2:-}" v="" par
  for par in "${VARIABLES[@]}"; do [[ "$par" == "$clave="* ]] && v="${par#*=}"; done
  [[ -z "$v" ]] && v="${!clave:-}"
  echo "${v:-$defecto}"
}

# Comprime src/, configs/ y config_dvc.sh del repo en $1, bajo la carpeta EvidenCite/, para que en Colab quede
# /content/EvidenCite/src/... (`colab upload` sube un solo archivo). Salta cachés y salidas de corridas.
empaquetar_codigo() {
  "$PY" - "$REPO" "$1" <<'PY'
import sys, zipfile
from pathlib import Path
repo, destino = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
saltar = {"__pycache__", ".ipynb_checkpoints"}
with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as z:
    for carpeta in ("src", "configs"):
        for archivo in sorted((repo / carpeta).rglob("*")):
            relativo = archivo.relative_to(repo)
            if archivo.is_file() and not saltar & set(relativo.parts) and archivo.suffix != ".pyc":
                z.write(archivo, Path("EvidenCite") / relativo)
    z.write(repo / "config_dvc.sh", "EvidenCite/config_dvc.sh")  # configura DVC en la sesión
print(f"  código: src/, configs/ y config_dvc.sh ({destino.stat().st_size / 1024:.0f} KB)")
PY
}

# Carpeta del repo con los datos versionados con DVC: DATOS_DVC (data/dataset por defecto, o data/labeled), sin "./",
# "/" final ni ".dvc".
carpeta_datos() {
  local c
  c="$(valor DATOS_DVC data/dataset)"
  c="${c#./}"; c="${c%/}"
  echo "${c%.dvc}"
}

# URL https del repo (el remoto origin), sin usuario ni token: la que clona la sesión de Colab, que no tiene llaves.
url_repo() {
  git -C "$REPO" remote get-url origin 2>/dev/null |
    sed -E 's#^git@([^:]+):#https://\1/#; s#^ssh://git@([^/:]+)(:[0-9]+)?/#https://\1/#; s#^(https?://)[^/@]+@#\1#'
}

# Valida, antes de crear la sesión, lo que la sesión necesita para bajar los datos con DVC. Deja en DATOS_COLAB las
# variables que se le pasan y en LLAVE la llave que se sube. En Colab los datos solo salen de DVC: si falta algo, falla.
# - DATOS_TAG (obligatorio): el tag de git con la versión de los datos; tiene que estar en GitHub.
# - GDRIVE_SA_JSON (obligatorio, variable de entorno): la llave JSON de la cuenta de servicio, con acceso de lectura
#   a la carpeta de Drive.
# - DATOS_DVC: la carpeta que se baja (carpeta_datos).
# - ID_CARPETA: el ID de la carpeta de Drive; si falta, se lee de .dvc/config.local (la que deja config_dvc.sh).
preparar_datos_colab() {
  local tag datos carpeta url llave="${GDRIVE_SA_JSON:-}" estado=0
  tag="$(valor DATOS_TAG)"; datos="$(carpeta_datos)"; carpeta="$(valor ID_CARPETA)"; url="$(url_repo)"
  if [[ -n "$(valor DATASET)$(valor DATOS_DIR)" ]]; then
    echo "DATASET y DATOS_DIR solo valen en local: en Colab los datos salen de DVC (DATOS_TAG)"; return 1
  fi
  [[ -n "$tag" ]] || { echo "Falta DATOS_TAG: el tag con la versión de los datos (p. ej. labeled-v1)"; return 1; }
  [[ -n "$llave" ]] || { echo "Falta GDRIVE_SA_JSON: la ruta de la llave JSON de la cuenta de servicio"; return 1; }
  llave="${llave/#\~/$HOME}"
  [[ -f "$llave" ]] || { echo "No existe GDRIVE_SA_JSON=$llave"; return 1; }
  [[ -n "$carpeta" ]] || carpeta="$(sed -n 's#^[[:space:]]*url[[:space:]]*=[[:space:]]*gdrive://##p' \
                                      "$REPO/.dvc/config.local" 2>/dev/null | head -1 | tr -d '[:space:]')"
  [[ -n "$carpeta" ]] || { echo "Falta ID_CARPETA: el ID de la carpeta de Google Drive del remoto DVC"; return 1; }
  [[ -n "$url" ]] || { echo "No encontré la URL del remoto origin del repo"; return 1; }
  # Se consulta sin credenciales, como lo va a clonar la sesión.
  GIT_TERMINAL_PROMPT=0 GIT_ASKPASS='' SSH_ASKPASS='' git -c credential.helper= -c core.askPass= \
    ls-remote --exit-code --tags "$url" "refs/tags/$tag" >/dev/null 2>&1 || estado=$?
  case "$estado" in
    0) ;;
    2) echo "El tag $tag no está en $url: súbelo con \`git push origin $tag\`"; return 1 ;;
    *) echo "No pude consultar $url sin credenciales, y la sesión tampoco podría clonarlo (¿repo privado o sin red?)"
       return 1 ;;
  esac
  LLAVE="$llave"
  # shellcheck disable=SC2034  # la usan los scripts que cargan este archivo
  DATOS_COLAB=("DATOS_TAG=$tag" "DATOS_DVC=$datos" "ID_CARPETA=$carpeta" "REPO_GIT=$url")
  echo "  datos: $datos del tag $tag (la sesión los baja con DVC de gdrive://$carpeta)"
}

# Instala las dependencias en la sesión $1 (DVC incluido) y sube el código que dejó empaquetar_codigo en $2 y la llave
# de la cuenta de servicio. La llave va como archivo y no con --env, porque la CLI guarda los --env en su historial.
subir_a_sesion() {
  colab install -s "$1" -r "$REPO/scripts/requirements-colab.txt" &&
  colab upload -s "$1" "$LLAVE" /content/cuenta_servicio.json &&
  colab upload -s "$1" "$2/evidencite.zip" /content/evidencite.zip
}

# Commit del código y si src/ o configs/ tienen cambios sin commit, para las etiquetas de MLflow: deja en TRAZAS
# CODIGO_COMMIT y CODIGO_SIN_COMMIT (vacío si el repo no está en git).
# shellcheck disable=SC2034  # TRAZAS la usan los scripts que cargan este archivo
trazas_codigo() {
  local commit
  TRAZAS=()
  commit="$(git -C "$REPO" rev-parse HEAD 2>/dev/null)" || return 0
  TRAZAS=("CODIGO_COMMIT=$commit" "CODIGO_SIN_COMMIT=false")
  [[ -z "$(git -C "$REPO" status --porcelain -- src configs)" ]] || TRAZAS[1]="CODIGO_SIN_COMMIT=true"
}

# Carpeta local del dataset para las corridas en esta máquina: DATASET (carpeta, o un .zip que se descomprime en una
# carpeta temporal, DATOS_TMP, que el script borra al terminar) o la carpeta de datos del repo (carpeta_datos), que
# se baja con `uv run dvc pull`.
datos_locales() {
  local origen="${DATASET:-$REPO/$(carpeta_datos)}"
  if [[ -f "$origen" && "$origen" == *.zip ]]; then
    DATOS_TMP="$(mktemp -d)"
    "$PY" -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$origen" "$DATOS_TMP"
    origen="$(dirname "$(find "$DATOS_TMP" -name train.jsonl -not -path '*__MACOSX*' | head -1)")"
  fi
  if [[ ! -f "$origen/train.jsonl" ]]; then
    echo "No encontré train/dev/test.jsonl en $origen: bájalos con \`uv run dvc pull\` o define DATASET" >&2
    return 1
  fi
  (cd "$origen" && pwd)
}

# Servidor de MLflow: la IP escrita en mlflow_ip.txt (junto a donde se lanza el script; gana, para poder cambiarla a
# mitad de una tanda) o MLFLOW_IP (argumento o variable). Vacío si no hay ninguna: la corrida avisa y guarda en local.
ip_mlflow() {
  local ip=""
  [[ -s mlflow_ip.txt ]] && ip="$(tr -d '[:space:]' < mlflow_ip.txt)"
  echo "${ip:-$(valor MLFLOW_IP)}"
}

# Dónde quedaron los resultados, según la línea final de python -m src.modeling en el registro $1:
# mlflow (todo en MLflow, nada que bajar), ya (estaba en MLflow y no se repitió), local (zip en SALIDAS_DIR) o fallo.
destino_resultados() {
  if grep -qa "RESULTADOS EN MLFLOW:" "$1"; then echo mlflow
  elif grep -qa "YA EN MLFLOW:" "$1"; then echo ya
  elif grep -qa "RESULTADOS LOCALES:" "$1"; then echo local
  else echo fallo; fi
}

# Opciones de `colab new` según la GPU: el backend rechaza --high-mem con G4
opciones_sesion() {
  [[ "$1" != G4 ]] && echo "--high-mem" || true
}