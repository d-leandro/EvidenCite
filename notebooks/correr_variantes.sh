#!/usr/bin/env bash
# Entrena baseline_lora.ipynb en modo completo para cada modelo × configuración × variante de LoRA × tasa.
# Usa una sola sesión de Colab: instala, sube los datos y descarga cada modelo una vez, y reinicia el kernel
# entre corridas para liberar la GPU.
#
# Uso:   ./correr_variantes.sh <G4|A100> [sesión] [modelo ...]
#        ./correr_variantes.sh G4                                   # una sesión, los modelos de MODELOS
#        ./correr_variantes.sh G4 s8b Qwen/Qwen3-8B-Base             # dos sesiones en paralelo, cada una
#        ./correr_variantes.sh G4 s4b Qwen/Qwen3-4B-Base             # en su terminal y con su modelo
# Salida: resultados/<modelo>_<variante>_<A|B>_lr<tasa>.zip y, por corrida, el registro y el notebook con salidas en registros/.
# Si lo vuelves a lanzar, salta las combinaciones que ya tienen zip en resultados/.
#
# MLflow (opcional): cada corrida se registra en el experimento EXPERIMENTO con el nombre del zip.
#        MLFLOW_IP=52.73.34.221:8080 EXPERIMENTO=lora_barrido ./correr_variantes.sh G4
# Si el servidor se cae, la corrida sigue entrenando y reintenta; al terminar lo espera hasta MLFLOW_ESPERA_MIN minutos
# (20) desde la caída. Si al reiniciarlo cambió la IP, escríbela en mlflow_ip.txt junto al script (para las corridas
# siguientes) y súbela a la sesión para la corrida en curso: colab upload -s <sesión> mlflow_ip.txt /content/mlflow_ip.txt
set -uo pipefail

GPU="${1:?Uso: $0 <G4|A100> [sesión] [modelo ...]}"
SESION="${2:-lora}"
TIMEOUT=14400                      # segundos máximos por corrida (la celda 13); si se pasa, la corrida se pierde
MODELOS=(Qwen/Qwen3-8B-Base Qwen/Qwen3-4B-Base)
if (( $# > 2 )); then MODELOS=("${@:3}"); fi   # los modelos pasados como argumento reemplazan la lista
CONFIGS=(A B)
VARIANTES=(lora)  # LoRA como control + LoRA+, EVA y DoRA (README, sección 5)
LRS=(5e-5 1e-4 2e-4)                # cada variante tiene su tasa óptima: comparar solo con la tasa fija sesga el resultado
COMUNES=(--env MODO=completo --env EPOCAS=8 --env EVALS_POR_EPOCA=2 --env PACIENCIA=5  --env "HF_TOKEN=${HF_TOKEN:-}"
         --env SCHEDULER=wsd --env EVALUAR_TEST=1 --env BATCH_EVAL=16)
MLFLOW_IP="${MLFLOW_IP:-}"                        # vacío: sin MLflow
EXPERIMENTO="${EXPERIMENTO:-baseline_lora}"
MLFLOW_ESPERA_MIN="${MLFLOW_ESPERA_MIN:-20}"
ip_mlflow() {                                     # la IP de mlflow_ip.txt, si existe, gana sobre MLFLOW_IP
  local ip=""
  [[ -s mlflow_ip.txt ]] && ip=$(tr -d '[:space:]' < mlflow_ip.txt)
  echo "${ip:-$MLFLOW_IP}"
}

# Memoria para modelos de 7–8B (sin esto, con BATCH=16 y sin gradient checkpointing no cabe en ninguna de las dos)
case "$GPU" in
  G4)        MEMORIA=(--env GRAD_CHECKPOINTING=0 --env BATCH=4 --env ACUMULACION=4) ;;
  A100|H100) MEMORIA=(--env GRAD_CHECKPOINTING=1) ;;
  *)         MEMORIA=(--env GRAD_CHECKPOINTING=1 --env BATCH=8 --env ACUMULACION=2) ;;
esac

mkdir -p resultados registros
FALLIDAS="registros/fallidas_$SESION.txt"
rm -f "$FALLIDAS"                                   # las fallidas se reintentan en cada lanzamiento
trap 'colab stop -s "$SESION" >/dev/null 2>&1' EXIT   # libera la GPU al terminar, también con Ctrl+C

preparar_sesion() {
  colab new -s "$SESION" --gpu "$GPU" &&
  colab install -s "$SESION" -r requirements.txt &&
  colab upload -s "$SESION" dataset.zip /content/dataset.zip
}

preparar_sesion || { echo "No se pudo preparar la sesión de Colab '$SESION'"; exit 1; }
if [[ -n "$(ip_mlflow)" ]]; then
  echo "MLflow: experimento $EXPERIMENTO · servidor $(ip_mlflow)"
else
  echo "MLflow: desactivado (define MLFLOW_IP para usarlo)"
fi

for modelo in "${MODELOS[@]}"; do
  for config in "${CONFIGS[@]}"; do
    for variante in "${VARIANTES[@]}"; do
    for lr in "${LRS[@]}"; do
      nombre="${modelo##*/}_${variante}_${config}_lr${lr}"
      if [[ -f "resultados/$nombre.zip" ]]; then
        echo "== $nombre: ya está en resultados/, la salto"
        continue
      fi
      echo "== $nombre · sesión $SESION · $(date '+%Y-%m-%d %H:%M')"
      colab restart-kernel -s "$SESION" || true      # libera la memoria de la corrida anterior
      # Una copia del notebook por corrida: colab exec deja las salidas junto a ella (<nombre>_output.ipynb),
      # así dos sesiones en paralelo no se pisan el archivo.
      cuaderno="registros/$nombre.ipynb"
      cp baseline_lora.ipynb "$cuaderno"
      ip=$(ip_mlflow)
      args=(-s "$SESION" -f "$cuaderno" --timeout "$TIMEOUT" "${COMUNES[@]}" "${MEMORIA[@]}"
            --env "MODELO=$modelo" --env "LORA_TIPO=$variante" --env "CONFIGURACION=$config"
            --env "LR=$lr" --env "NOMBRE_CORRIDA=$nombre"
            --env "MLFLOW_IP=$ip" --env "MLFLOW_EXPERIMENTO=$EXPERIMENTO" --env "MLFLOW_ESPERA_MIN=$MLFLOW_ESPERA_MIN")
      if ! colab exec "${args[@]}" 2>&1 | tee "registros/$nombre.log"; then
        echo "   la sesión se perdió: creo otra y repito esta corrida"
        colab stop -s "$SESION" >/dev/null 2>&1
        preparar_sesion && colab exec "${args[@]}" 2>&1 | tee "registros/$nombre.log"
      fi
      # Se baja el zip con el nombre de esta corrida, no ultima_corrida.zip: si la corrida falló, no existe y
      # no se confunde con la anterior. Se baja a .part y se renombra al final: un zip a medias no cuenta como hecho.
      if colab download -s "$SESION" "/content/salidas/$nombre.zip" "resultados/$nombre.zip.part" &&
         mv -f "resultados/$nombre.zip.part" "resultados/$nombre.zip"; then
        echo "   listo: resultados/$nombre.zip"
      else
        echo "$nombre" >> "$FALLIDAS"
        echo "   FALLÓ: revisa registros/$nombre.log"
      fi
    done
    done
  done
done

echo "Terminado ($SESION). Corridas fallidas: $(cat "$FALLIDAS" 2>/dev/null | wc -l | tr -d ' ')"
