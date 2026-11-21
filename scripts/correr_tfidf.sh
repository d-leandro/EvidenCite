#!/usr/bin/env bash
# TF-IDF + regresión logística (FAMILIA=tfidf) con las configuraciones A y B, en una sola sesión.
#
# Es un envoltorio de probar_variantes.sh: una sesión (instala, sube el código y baja los datos con DVC una sola vez),
# una corrida de MLflow por configuración (tfidf_logreg_A_lrna y tfidf_logreg_B_lrna) y salta la que ya esté
# terminada en MLflow. TF-IDF no tiene variantes ni tasa de aprendizaje: VARIANTES=logreg y LRS=na solo completan el
# nombre de la corrida. No usa GPU, pero la CLI de Colab no acepta una sesión sin GPU: se pide la más barata (T4).
#
# Uso:   scripts/correr_tfidf.sh [T4|L4|G4|A100|local] [CLAVE=valor ...]
#        SOLO_MLFLOW=1 scripts/correr_tfidf.sh                       # A y B con los datos del tag test-revisado
#        CONFIGS=B scripts/correr_tfidf.sh T4 C_PENALTY=10           # solo B y otra regularización
#        DATOS_TAG=labeled-v2 scripts/correr_tfidf.sh                # otra versión de los datos
#
# Por defecto: DATOS_TAG=test-revisado, DATOS_DVC=data/labeled, INCLUIR_SECCION=0, MLFLOW_ESPERA_MIN=120 y
# MLFLOW_EXPERIMENTO=EvidenCite_A6_A7_Evaluacion_Comparativa. Cada uno se cambia con una variable de entorno o con
# CLAVE=valor (gana el último). Hiperparámetros de TF-IDF, también como CLAVE=valor: NGRAM_MIN, NGRAM_MAX, MIN_DF,
# MAX_DF, MAX_FEATURES, C_PENALTY, MAX_ITER.
# Las variables de probar_variantes.sh siguen valiendo, en el entorno: GDRIVE_SA_JSON (obligatoria en Colab), MLFLOW_IP,
# SOLO_MLFLOW, TIMEOUT, SESION (tfidf por defecto) y CONFIGS ("A B" por defecto). Si MLFLOW_TRACKING_USERNAME y
# MLFLOW_TRACKING_PASSWORD están en el entorno, se pasan a la sesión.
# En local (primer argumento `local`) los datos salen de DATASET o de la carpeta DATOS_DVC del repo, no del tag.
set -uo pipefail

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$AQUI/probar_variantes.sh" ]] || { echo "No encuentro probar_variantes.sh junto a $0"; exit 1; }

GPU=T4
if [[ $# -gt 0 && "$1" != *=* ]]; then GPU="$1"; shift; fi
[[ "$GPU" == local ]] && echo "local: los datos salen de DATASET o de ${DATOS_DVC:-data/labeled} en el repo (DATOS_TAG no se usa)" >&2

# Listas de probar_variantes.sh (las lee del entorno).
export CONFIGS="${CONFIGS:-A B}" SESION="${SESION:-tfidf}"
export MODELOS=tfidf VARIANTES=logreg LRS=na

# Lo que se le pasa a cada corrida. Va antes de "$@": lo que el usuario escriba como CLAVE=valor lo reemplaza.
ARGS=(FAMILIA=tfidf MODO=completo
      "INCLUIR_SECCION=${INCLUIR_SECCION:-0}"
      "DATOS_TAG=${DATOS_TAG:-test-revisado}" "DATOS_DVC=${DATOS_DVC:-data/labeled}"
      "MLFLOW_EXPERIMENTO=${MLFLOW_EXPERIMENTO:-EvidenCite_A6_A7_Evaluacion_Comparativa}"
      "MLFLOW_ESPERA_MIN=${MLFLOW_ESPERA_MIN:-120}")
[[ -n "${MLFLOW_TRACKING_USERNAME:-}" ]] && ARGS+=("MLFLOW_TRACKING_USERNAME=$MLFLOW_TRACKING_USERNAME")
[[ -n "${MLFLOW_TRACKING_PASSWORD:-}" ]] && ARGS+=("MLFLOW_TRACKING_PASSWORD=$MLFLOW_TRACKING_PASSWORD")

exec "$AQUI/probar_variantes.sh" "$GPU" "${ARGS[@]}" "$@"
