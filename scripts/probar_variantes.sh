#!/usr/bin/env bash
# Prueba variantes de LoRA: cada combinación de variante × tasa × configuración × modelo es una corrida aparte de
# MLflow (en RUNNING mientras entrena, con su mejor modelo en la columna Models). Es correr_lora.sh de lora.zip sobre
# src/modeling.
#
# Se asume un servidor de MLflow: MLFLOW_IP=ip:puerto, o la IP escrita en mlflow_ip.txt junto a donde se lanza el
# script (si cambia a mitad de la tanda, basta con reescribir el archivo). Con servidor no se descarga nada de Colab ni
# queda nada en local, y una combinación que ya está terminada en MLflow se salta. Sin servidor, cada corrida lo
# advierte, guarda en local y el script baja su zip a resultados/ (y salta las que ya tienen zip ahí).
#
# En Colab: una sola sesión. Instala, sube el dataset y el código una vez, reinicia el kernel entre corridas para
# liberar la GPU y, si se pierde la sesión, crea otra y repite la corrida. Pasa PRECISION=auto y AGRUPAR_LONGITUD=1
# (se cambian con CLAVE=valor).
# En local (local en vez de la GPU): las 10 variantes con el Qwen3 de humo en modo mini, una corrida por variante.
# Verifica que cada una entrene y suba a MLflow, en segundos y sin GPU.
#
# Uso:   scripts/probar_variantes.sh <G4|A100|L4|T4|local> [CLAVE=valor ...]
#        scripts/probar_variantes.sh G4                                       # lora, lora_plus, eva y dora con Qwen3-8B, B y 1e-4
#        VARIANTES="lora dora" LRS="5e-5 1e-4" CONFIGS="A B" scripts/probar_variantes.sh G4 EPOCAS=4 SCHEDULER=cosine
#        MODELOS="Qwen/Qwen3-4B-Base" SESION=s4b scripts/probar_variantes.sh G4  # otra sesión en paralelo
#        scripts/probar_variantes.sh local                                    # las 10 variantes con el humo
# Listas (separadas por espacios): VARIANTES, LRS, CONFIGS y MODELOS. Cada CLAVE=valor llega a todas las corridas.
set -uo pipefail

GPU="${1:?Uso: $0 <G4|A100|L4|T4|local> [CLAVE=valor ...]}"; shift
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIABLES=("$@")
source "$REPO/scripts/_comun.sh"

if [[ "$GPU" == local ]]; then
  read -ra VARIANTES <<< "${VARIANTES:-lora rslora dora pissa olora eva lora_plus lora_fa vera qlora}"
  read -ra LRS <<< "${LRS:-1e-4}"; read -ra CONFIGS <<< "${CONFIGS:-A}"; read -ra MODELOS <<< "${MODELOS:-humo}"
else
  read -ra VARIANTES <<< "${VARIANTES:-lora lora_plus eva dora}"     # LoRA como control + LoRA+, EVA y DoRA
  read -ra LRS <<< "${LRS:-1e-4}"; read -ra CONFIGS <<< "${CONFIGS:-B}"; read -ra MODELOS <<< "${MODELOS:-Qwen/Qwen3-8B-Base}"
fi
EXPERIMENTO="$(valor MLFLOW_EXPERIMENTO variantes_lora)"
FALLIDAS=()
if [[ -n "$(ip_mlflow)" ]]; then echo "MLflow: experimento $EXPERIMENTO · servidor $(ip_mlflow)"
else echo "Sin MLFLOW_IP ni mlflow_ip.txt: las corridas guardarán en local"; fi

# Después de cada corrida: con MLflow no queda nada; si quedó en local o falló, el registro va a registros/.
cerrar_corrida() {   # $1 nombre · $2 registro temporal
  case "$(destino_resultados "$2")" in
    mlflow) rm -f "$2"; echo "   listo en MLflow" ;;
    ya)     rm -f "$2"; echo "   ya estaba terminada en MLflow: la salto" ;;
    local)  mkdir -p registros && mv "$2" "registros/$1.log"; echo "   en local: resultados/$1.zip" ;;
    *)      mkdir -p registros && mv "$2" "registros/$1.log"; FALLIDAS+=("$1"); echo "   FALLÓ: revisa registros/$1.log" ;;
  esac
}

resumen_final() {
  echo "Terminado ($1). Corridas fallidas: ${#FALLIDAS[@]}${FALLIDAS[*]:+ (${FALLIDAS[*]})}"
}

# ---------------------------------------------------------------------------------------------------------------- local
if [[ "$GPU" == local ]]; then
  SALIDAS="$(mktemp -d)"
  trap 'rm -rf "$SALIDAS" "${DATOS_TMP:-}"' EXIT
  [[ -z "$(valor DATOS_DIR)" ]] && { DATOS="$(datos_locales)" || exit 1; }
  for modelo in "${MODELOS[@]}"; do for config in "${CONFIGS[@]}"; do for variante in "${VARIANTES[@]}"; do for lr in "${LRS[@]}"; do
    nombre="${modelo##*/}_${variante}_${config}_lr${lr}"
    [[ -f "resultados/$nombre.zip" ]] && { echo "== $nombre: ya está en resultados/, la salto"; continue; }
    echo "== $nombre"
    previas=(FAMILIA=lora SALIDAS_DIR="$SALIDAS" MODO="$(valor MODO mini)" SALTAR_EXISTENTES=1
             MLFLOW_EXPERIMENTO="$EXPERIMENTO")
    [[ -n "${DATOS:-}" ]] && previas+=(DATOS_DIR="$DATOS")
    [[ -n "$(ip_mlflow)" ]] && previas+=(MLFLOW_IP="$(ip_mlflow)")
    registro="$(mktemp)"
    ( cd "$REPO" &&
      env "${previas[@]}" "${VARIABLES[@]}" MODELO="$modelo" LORA_TIPO="$variante" CONFIGURACION="$config" LR="$lr" \
        NOMBRE_CORRIDA="$nombre" "$PY" -m src.modeling ) > "$registro" 2>&1
    tr '\r' '\n' < "$registro" | grep -a "macro-F1 (9)\|^AVISO\|MLflowNoEncontrado" | sed 's/^ */   /' | cut -c1-140
    if [[ "$(destino_resultados "$registro")" == local ]]; then
      mkdir -p resultados && cp "$SALIDAS/$nombre.zip" "resultados/$nombre.zip"
    fi
    cerrar_corrida "$nombre" "$registro"
  done; done; done; done
  resumen_final local
  exit 0
fi

# ---------------------------------------------------------------------------------------------------------------- Colab
SESION="${SESION:-variantes}"
TIMEOUT="${TIMEOUT:-14400}"   # segundos máximos por corrida; si se pasa, la corrida se pierde
TMP="$(mktemp -d)"
trap 'colab stop -s "$SESION" >/dev/null 2>&1; rm -rf "$TMP"' EXIT   # libera la GPU al terminar, también con Ctrl+C
empaquetar_codigo "$TMP/evidencite.zip" || exit 1
preparar_dataset "$TMP/dataset.zip" || exit 1
COMUNES=(FAMILIA=lora PRECISION=auto AGRUPAR_LONGITUD=1 SALTAR_EXISTENTES=1 "MLFLOW_EXPERIMENTO=$EXPERIMENTO")
[[ -n "${HF_TOKEN:-}" ]] && COMUNES+=("HF_TOKEN=$HF_TOKEN")      # la CLI lo guarda en su historial

preparar_sesion() {
  colab new $(opciones_sesion "$GPU") -s "$SESION" --gpu "$GPU" &&
  colab install -s "$SESION" -r "$REPO/scripts/requirements-colab.txt" &&
  colab upload -s "$SESION" "$TMP/dataset.zip" /content/dataset.zip &&
  colab upload -s "$SESION" "$TMP/evidencite.zip" /content/evidencite.zip
}
preparar_sesion || { echo "No se pudo preparar la sesión de Colab '$SESION'"; exit 1; }

for modelo in "${MODELOS[@]}"; do for config in "${CONFIGS[@]}"; do for variante in "${VARIANTES[@]}"; do for lr in "${LRS[@]}"; do
  nombre="${modelo##*/}_${variante}_${config}_lr${lr}"
  [[ -f "resultados/$nombre.zip" ]] && { echo "== $nombre: ya está en resultados/, la salto"; continue; }
  echo "== $nombre · sesión $SESION · $(date '+%Y-%m-%d %H:%M')"
  colab restart-kernel -s "$SESION" || true      # libera la memoria de la corrida anterior
  entorno=("${COMUNES[@]}" "${VARIABLES[@]}" "MODELO=$modelo" "LORA_TIPO=$variante" "CONFIGURACION=$config" "LR=$lr"
           "NOMBRE_CORRIDA=$nombre")
  [[ -n "$(ip_mlflow)" ]] && entorno+=("MLFLOW_IP=$(ip_mlflow)")
  args=(-s "$SESION" -f "$REPO/scripts/lanzar_colab.py" --timeout "$TIMEOUT")
  for par in "${entorno[@]}"; do args+=(--env "$par"); done
  registro="$(mktemp)"
  # colab exec solo falla si se pierde la sesión; un error de Python sale en el registro y la corrida cuenta como
  # fallida porque no dice dónde quedaron sus resultados.
  if ! colab exec "${args[@]}" 2>&1 | tee "$registro"; then
    echo "   la sesión se perdió: creo otra y repito esta corrida"
    colab stop -s "$SESION" >/dev/null 2>&1
    preparar_sesion && colab exec "${args[@]}" 2>&1 | tee "$registro"
  fi
  if [[ "$(destino_resultados "$registro")" == local ]]; then
    # Sin MLflow, la corrida quedó en la sesión: se baja a .part y se renombra al final (un zip a medias no cuenta).
    mkdir -p resultados
    if ! { colab download -s "$SESION" "/content/salidas/$nombre.zip" "resultados/$nombre.zip.part" &&
           mv -f "resultados/$nombre.zip.part" "resultados/$nombre.zip"; }; then
      echo "LA DESCARGA FALLÓ" >> "$registro"; sed -i 's/RESULTADOS LOCALES:/(sin descargar)/' "$registro"
    fi
  fi
  cerrar_corrida "$nombre" "$registro"
done; done; done; done
resumen_final "$SESION"
