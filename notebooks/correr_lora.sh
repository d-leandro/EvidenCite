#!/usr/bin/env bash
# Barrido de baseline_lora.ipynb en Colab con la CLI, para comparar con correr_algoritmos.sh (RL): cada
# modelo × variante de adaptador, en modo completo, con un zip por corrida en
# resultados/<modelo>_<variante>_sft.zip.
#
# Por defecto entrena el mismo adaptador que el notebook de RL (QLoRA, r=16, alpha 32, dropout 0, todas las
# capas lineales), así la única diferencia entre los dos barridos es el método de entrenamiento: clasificación
# supervisada con cabeza (baseline_lora) frente a RL generativo (baseline_rl).
#
# Uso:   ./correr_lora.sh A100          (o G4)
# Ajustes por variable de entorno (opcionales):
#   MODELOS="HuggingFaceTB/SmolLM2-360M Qwen/Qwen3-1.7B-Base"   modelos, separados por espacios (los mismos que en RL)
#   VARIANTES="qlora lora dora"                                   variantes de LORA_TIPO
#   EXTRA="--env EPOCAS=3 --env LR=1e-4"                          cualquier otro parámetro del notebook
#   HF_TOKEN=hf_...                                               solo para modelos con acceso restringido
# Requiere dataset.zip y requirements.txt junto al notebook.
set -eo pipefail

GPU="${1:-A100}"
SESION="lora"
MODELOS="${MODELOS:-HuggingFaceTB/SmolLM2-360M}"
VARIANTES="${VARIANTES:-qlora}"
TOKEN=()
if [ -n "${HF_TOKEN:-}" ]; then TOKEN=(--env "HF_TOKEN=$HF_TOKEN"); fi

mkdir -p resultados
colab new -s "$SESION" --gpu "$GPU"
trap 'colab stop -s "$SESION" || true' EXIT
colab install -s "$SESION" -r requirements.txt
colab upload -s "$SESION" dataset.zip /content/dataset.zip

for MODELO in $MODELOS; do
  for VARIANTE in $VARIANTES; do
    NOMBRE="${MODELO##*/}_${VARIANTE}_sft"
    echo "== $NOMBRE"
    # shellcheck disable=SC2086  # EXTRA se separa en palabras a propósito
    colab exec -s "$SESION" -f baseline_lora.ipynb --timeout 14400 \
      --env MODO=completo --env MODELO="$MODELO" --env LORA_TIPO="$VARIANTE" \
      --env LORA_R=16 --env LORA_ALPHA=32 --env LORA_DROPOUT=0 --env LORA_MODULOS=auto \
      --env NOMBRE_CORRIDA="$NOMBRE" \
      ${TOKEN[@]+"${TOKEN[@]}"} $EXTRA || echo "AVISO: colab exec terminó con error en $NOMBRE"
    # Se descarga el zip con el nombre de la corrida, no ultima_corrida.zip: si la corrida falló, no se baja la anterior.
    colab download -s "$SESION" "/content/salidas/$NOMBRE.zip" "resultados/$NOMBRE.zip" \
      || echo "AVISO: no hay resultados de $NOMBRE; revisa baseline_lora_output.ipynb"
  done
done
