#!/usr/bin/env bash
# Una corrida simple de src/modeling, en Colab con la CLI o en esta máquina.
#
# Se asume un servidor de MLflow: MLFLOW_IP=ip:puerto, o la IP escrita en mlflow_ip.txt junto a donde se lanza el
# script. Todo queda en la corrida de MLflow (métricas, curvas, archivos, salida de la consola y el mejor modelo, en la
# columna Models) y nada en local: no se descarga nada de Colab. Si el servidor no responde, la corrida lo advierte,
# guarda en local y el script baja el zip a resultados/.
#
# En Colab: crea la sesión, instala requirements-colab.txt, sube src/ y configs/ (comprimidos: `colab upload` solo sube
# archivos) y el dataset, corre lanzar_colab.py con cada CLAVE=valor como variable de entorno y apaga la sesión. Por
# defecto pasa PRECISION=auto (bf16 en A100, L4 y G4; fp16 en T4) y AGRUPAR_LONGITUD=1, las optimizaciones que lora.zip
# aplicaba siempre; se cambian con PRECISION=fp32 AGRUPAR_LONGITUD=0.
# En local (local en vez de la GPU): corre `python -m src.modeling` aquí mismo. Sin MODELO usa el modelo de humo de la
# familia (Qwen3 o BERT minúsculo) en modo mini: verifica el pipeline y lo que sube a MLflow en segundos, sin GPU.
#
# Uso:   scripts/correr_experimento.sh <G4|A100|L4|T4|local> [CLAVE=valor ...]
#        scripts/correr_experimento.sh G4 MODELO=Qwen/Qwen3-8B-Base LORA_TIPO=dora CONFIGURACION=B LR=1e-4 SCHEDULER=cosine
#        scripts/correr_experimento.sh G4 FAMILIA=scibert CONFIGURACION=B           # rama encoder (SciBERT por defecto)
#        scripts/correr_experimento.sh local                                        # humo LoRA, modo mini
#        scripts/correr_experimento.sh local FAMILIA=scibert                        # humo SciBERT, modo mini
# Los parámetros (CLAVE=valor) están en src/modeling/__main__.py y en src/modeling/MIGRACION_LORA.md.
# Variables del script: MLFLOW_IP, DATASET (zip o carpeta; por defecto ./dataset.zip o data/dataset del repo), SESION,
# TIMEOUT y HF_TOKEN. La salida se ve en la consola; solo si algo falla o queda en local se guarda en registros/.
set -uo pipefail

GPU="${1:?Uso: $0 <G4|A100|L4|T4|local> [CLAVE=valor ...]}"; shift
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIABLES=("$@")
source "$REPO/scripts/_comun.sh"

FAMILIA="$(valor FAMILIA lora)"
case "$FAMILIA" in
  lora)    NOMBRE="lora_$(valor LORA_TIPO lora)_$(valor CONFIGURACION A)_$(date +%m%d-%H%M%S)" ;;
  scibert) NOMBRE="scibert_$(valor CONFIGURACION A)_$(date +%m%d-%H%M%S)" ;;
  tfidf)   NOMBRE="tfidf_$(valor CONFIGURACION A)_$(date +%m%d-%H%M%S)" ;;
  *)       echo "FAMILIA debe ser lora, scibert o tfidf"; exit 1 ;;
esac
NOMBRE="$(valor NOMBRE_CORRIDA "$NOMBRE")"
IP_MLFLOW="$(ip_mlflow)"
[[ -n "$IP_MLFLOW" ]] && echo "MLflow: $IP_MLFLOW" || echo "Sin MLFLOW_IP ni mlflow_ip.txt: la corrida guardará en local"
REGISTRO="$(mktemp)"   # la salida de la corrida; se conserva en registros/ solo si hace falta

# Al terminar: si todo quedó en MLflow no queda nada en local; si no, el registro va a registros/<nombre>.log.
terminar() {
  case "$(destino_resultados "$REGISTRO")" in
    mlflow) rm -f "$REGISTRO"; echo "Listo: todo quedó en MLflow (corrida $NOMBRE); nada en local." ;;
    local)  mkdir -p registros && mv "$REGISTRO" "registros/$NOMBRE.log"
            echo "Resultados en local: resultados/$NOMBRE.zip · salida en registros/$NOMBRE.log" ;;
    *)      mkdir -p registros && mv "$REGISTRO" "registros/$NOMBRE.log"; echo "Falló: revisa registros/$NOMBRE.log"
            return 1 ;;
  esac
}

# ---------------------------------------------------------------------------------------------------------------- local
if [[ "$GPU" == local ]]; then
  SALIDAS="$(mktemp -d)"   # carpeta de la corrida mientras entrena; con MLflow queda vacía
  trap 'rm -rf "$SALIDAS" "${DATOS_TMP:-}"' EXIT
  PREVIAS=(FAMILIA="$FAMILIA" SALIDAS_DIR="$SALIDAS")
  [[ "$FAMILIA" != tfidf ]] && PREVIAS+=(MODO="$(valor MODO mini)" MODELO="$(valor MODELO humo)")
  [[ -z "$(valor DATOS_DIR)" ]] && { DATOS="$(datos_locales)" || exit 1; PREVIAS+=(DATOS_DIR="$DATOS"); }
  [[ -n "$IP_MLFLOW" ]] && PREVIAS+=(MLFLOW_IP="$IP_MLFLOW")
  echo "Corrida $NOMBRE · $FAMILIA · local"
  ( cd "$REPO" && env "${PREVIAS[@]}" "${VARIABLES[@]}" NOMBRE_CORRIDA="$NOMBRE" "$PY" -m src.modeling ) \
    2>&1 | tee "$REGISTRO"
  if [[ "$(destino_resultados "$REGISTRO")" == local ]]; then
    mkdir -p resultados && cp "$SALIDAS/$NOMBRE.zip" "resultados/$NOMBRE.zip"
  fi
  terminar
  exit $?
fi

# ---------------------------------------------------------------------------------------------------------------- Colab
SESION="${SESION:-evidencite}"
TIMEOUT="${TIMEOUT:-14400}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
empaquetar_codigo "$TMP/evidencite.zip" || exit 1
preparar_dataset "$TMP/dataset.zip" || exit 1

# Valores de la corrida; lo que se pase como CLAVE=valor va después y los reemplaza.
COMUNES=(FAMILIA="$FAMILIA")
[[ "$FAMILIA" != tfidf ]] && COMUNES+=(PRECISION=auto AGRUPAR_LONGITUD=1)
ENTORNO=("${COMUNES[@]}" "${VARIABLES[@]}" "NOMBRE_CORRIDA=$NOMBRE")
[[ -n "$IP_MLFLOW" ]] && ENTORNO+=("MLFLOW_IP=$IP_MLFLOW")
[[ -n "${HF_TOKEN:-}" ]] && ENTORNO+=("HF_TOKEN=$HF_TOKEN")      # la CLI lo guarda en su historial
ENV=(); for par in "${ENTORNO[@]}"; do ENV+=(--env "$par"); done
echo "Corrida $NOMBRE · $FAMILIA · sesión $SESION ($GPU)"

colab new -s "$SESION" --gpu "$GPU" || exit 1
trap 'colab stop -s "$SESION" >/dev/null 2>&1; rm -rf "$TMP"' EXIT    # se apaga al terminar, aunque algo falle
colab install -s "$SESION" -r "$REPO/scripts/requirements-colab.txt" || exit 1
colab upload -s "$SESION" "$TMP/dataset.zip" /content/dataset.zip || exit 1
colab upload -s "$SESION" "$TMP/evidencite.zip" /content/evidencite.zip || exit 1
colab exec -s "$SESION" -f "$REPO/scripts/lanzar_colab.py" --timeout "$TIMEOUT" "${ENV[@]}" 2>&1 | tee "$REGISTRO"
if [[ "$(destino_resultados "$REGISTRO")" == local ]]; then
  # Sin MLflow, la corrida quedó en la sesión: se baja antes de apagarla.
  mkdir -p resultados
  if ! colab download -s "$SESION" "/content/salidas/$NOMBRE.zip" "$PWD/resultados/$NOMBRE.zip"; then
    trap 'rm -rf "$TMP"' EXIT
    mkdir -p registros && mv "$REGISTRO" "registros/$NOMBRE.log"
    echo "La descarga falló, pero el zip existe en la sesión, que dejo ENCENDIDA. Bájalo y después apágala:"
    echo "  colab download -s $SESION /content/salidas/$NOMBRE.zip \"$PWD/resultados/$NOMBRE.zip\""
    echo "  colab stop -s $SESION"
    exit 1
  fi
fi
terminar
