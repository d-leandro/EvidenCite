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

# Comprime src/ y configs/ del repo en $1, bajo la carpeta EvidenCite/, para que en Colab quede
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
print(f"  código: src/ y configs/ ({destino.stat().st_size / 1024:.0f} KB)")
PY
}

# Deja en $1 el zip del dataset para subirlo: DATASET (un .zip o una carpeta con train/dev/test.jsonl), o
# ./dataset.zip, o data/dataset del repo. Una carpeta se comprime en el momento.
preparar_dataset() {
  local destino="$1" origen="${DATASET:-}"
  if [[ -z "$origen" ]]; then
    for c in dataset.zip "$REPO/data/dataset"; do
      if [[ -f "$c" || -f "$c/train.jsonl" ]]; then origen="$c"; break; fi
    done
  fi
  [[ -n "$origen" ]] || { echo "No encontré el dataset: define DATASET=ruta/dataset.zip (o una carpeta con train/dev/test.jsonl)"; return 1; }
  if [[ -f "$origen" ]]; then
    cp "$origen" "$destino"
  else
    [[ -f "$origen/train.jsonl" ]] || { echo "DATASET=$origen no tiene train/dev/test.jsonl"; return 1; }
    "$PY" - "$origen" "$destino" <<'PY'
import sys, zipfile
from pathlib import Path
origen, destino = Path(sys.argv[1]), Path(sys.argv[2])
with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as z:
    for s in ("train", "dev", "test"):
        z.write(origen / f"{s}.jsonl", f"dataset/{s}.jsonl")
PY
  fi
  echo "  datos: $origen"
}

# Carpeta local del dataset para las corridas en esta máquina: DATASET (carpeta, o un .zip que se descomprime en una
# carpeta temporal, DATOS_TMP, que el script borra al terminar) o data/dataset del repo.
datos_locales() {
  local origen="${DATASET:-$REPO/data/dataset}"
  if [[ -f "$origen" && "$origen" == *.zip ]]; then
    DATOS_TMP="$(mktemp -d)"
    "$PY" -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$origen" "$DATOS_TMP"
    origen="$(dirname "$(find "$DATOS_TMP" -name train.jsonl -not -path '*__MACOSX*' | head -1)")"
  fi
  [[ -f "$origen/train.jsonl" ]] || { echo "No encontré train/dev/test.jsonl en $origen: define DATASET" >&2; return 1; }
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