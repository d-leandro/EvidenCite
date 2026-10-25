#!/usr/bin/env bash
# Una sola corrida de lora.py en Colab con la CLI: crea la sesión, instala, sube los datos y los módulos, entrena,
# baja el zip y apaga la sesión. Para barridos está correr_lora.sh; esta usa sus mismos valores por defecto.
# Se lanza desde la carpeta de lora.py, con los demás módulos, requirements.txt y dataset.zip al lado.
#
# Uso: ./correr_una.sh <G4|A100|L4|T4> [CLAVE=valor ...]     (cada CLAVE=valor le llega a lora.py como variable)
#      ./correr_una.sh G4 MODELO=Qwen/Qwen3-8B-Base CONFIGURACION=B LR=1e-4
#      ./correr_una.sh G4 MODELO=Qwen/Qwen3-4B-Base NOMBRE_CORRIDA=prueba_4b MLFLOW_IP=54.225.26.128:8080
set -euo pipefail

GPU="${1:?Uso: $0 <G4|A100|L4|T4> [CLAVE=valor ...]}"; shift
SESION="${SESION:-otra}"                                   # nombre de la sesión de Colab
NOMBRE="corrida_$(date +%Y%m%d_%H%M%S)"                    # nombre del zip, salvo que se pase NOMBRE_CORRIDA=...
for par in "$@"; do case "$par" in NOMBRE_CORRIDA=*) NOMBRE="${par#*=}" ;; esac; done
for archivo in lora.py requirements.txt dataset.zip; do
  [[ -f "$archivo" ]] || { echo "Falta $archivo: lanza el script desde la carpeta de lora.py"; exit 1; }
done

# Los valores del barrido; lo que se pase como CLAVE=valor va después y los reemplaza
VARIABLES=(MODO=completo EPOCAS=1 EVALS_POR_EPOCA=1 PACIENCIA=5 SCHEDULER=wsd EVALUAR_TEST=1 BATCH_EVAL=16)
case "$GPU" in                                             # memoria para modelos de 7–8B
  G4)        VARIABLES+=(GRAD_CHECKPOINTING=0 BATCH=4 ACUMULACION=4) ;;
  A100|H100) VARIABLES+=(GRAD_CHECKPOINTING=1) ;;
  *)         VARIABLES+=(GRAD_CHECKPOINTING=1 BATCH=8 ACUMULACION=2) ;;
esac
VARIABLES+=("HF_TOKEN=${HF_TOKEN:-}" "NOMBRE_CORRIDA=$NOMBRE" "$@")   # la CLI guarda HF_TOKEN en su historial
ENV=(); for par in "${VARIABLES[@]}"; do ENV+=(--env "$par"); done

mkdir -p resultados registros
colab new -s "$SESION" --gpu "$GPU"                                    # VM con GPU
trap 'colab stop -s "$SESION"' EXIT                                     # se apaga al terminar, aunque algo falle
colab install -s "$SESION" -r requirements.txt                          # accelerate, bitsandbytes, torchao, mlflow-skinny
colab upload -s "$SESION" dataset.zip /content/dataset.zip              # los datos
for f in *.py; do colab upload -s "$SESION" "$f" "/content/$f"; done    # lora.py y los módulos que importa
colab exec -s "$SESION" -f lora.py --timeout 14400 "${ENV[@]}" 2>&1 | tee "registros/$NOMBRE.log"   # entrena
colab download -s "$SESION" "/content/salidas/$NOMBRE.zip" "resultados/$NOMBRE.zip" ||
  { echo "No hay zip: revisa registros/$NOMBRE.log"; exit 1; }
echo "Listo: resultados/$NOMBRE.zip"
