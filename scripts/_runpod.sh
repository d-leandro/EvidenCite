#!/usr/bin/env bash
# PROVEEDOR=runpod: redefine `colab` como una función que traduce cada subcomando que usan correr_experimento.sh y
# probar_variantes.sh a runpodctl + ssh/scp, de modo que esos scripts corren sin cambios en un pod de RunPod. Un pod por
# SESION; sus datos de conexión (id, IP, puerto) quedan en $RP_ESTADO/<sesión>. No se ejecuta solo: lo carga _comun.sh.
# Requiere: runpodctl con su API key (`runpodctl doctor`), RUNPOD_API_KEY exportada y tu llave SSH registrada en RunPod.
RP_IMAGEN="${RP_IMAGEN:-runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404}"   # aporta sshd, Python 3.12 y git; sus paquetes no se usan
RP_CUDA="${RP_CUDA:-13.0}"    # CUDA mínima del driver del host: el torch de requirements-runpod.txt es para CUDA 13
RP_DISCO="${RP_DISCO:-150}"   # GB de disco del contenedor: entorno virtual, pesos del modelo, caché de Hugging Face y datos
RP_ESTADO="${RP_ESTADO:-$HOME/.cache/evidencite-runpod}"; mkdir -p "$RP_ESTADO"
# ConnectTimeout cubre también el saludo de sshd: sin él, si el puerto acepta la conexión pero sshd no responde
# ("kex_exchange_identification: read: Connection timed out"), ssh/scp esperan el timeout de TCP del kernel (minutos).
RP_SSH=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ServerAliveInterval=30 -o BatchMode=yes -o ConnectTimeout=20)
[[ -n "${RP_LLAVE:-}" ]] && RP_SSH+=(-i "$RP_LLAVE" -o IdentitiesOnly=yes)   # llave privada si no es la de ssh por defecto
declare -A RP_GPU=([B200]="NVIDIA B200"  [G4]="NVIDIA RTX PRO 6000 Blackwell Server Edition" [A100]="NVIDIA A100-SXM4-80GB" [L4]="NVIDIA L4" [H100]="NVIDIA H100 NVL" [H200]="NVIDIA H200")
# IP pública y puerto SSH del pod $RP_ID, de la REST API v2 de RunPod (api.runpod.io/v2; la v1, rest.runpod.io, se
# retiró el 15-11-2026). En v2 vienen en ssh.direct; mientras el pod arranca, ssh.direct es null y esto imprime "- -".
# También acepta el formato de v1 (publicIp + portMappings), por si RunPod lo devuelve en otra ruta.
RP_IP_PUERTO='import json, os, re, sys
def pod(x):  # el pod con nuestro id, sea la respuesta un pod o una lista {"pods": [...]}
    if isinstance(x, dict):
        if x.get("id") == os.environ["RP_ID"]:
            return x
        x = list(x.values())
    for v in x if isinstance(x, list) else []:
        r = pod(v)
        if r:
            return r
try:
    p = pod(json.load(sys.stdin)) or {}
except ValueError:
    p = {}
d = (p.get("ssh") or {}).get("direct") or {}
host, puerto = d.get("host") or d.get("ip"), d.get("port")
if not (host and puerto) and d.get("command"):  # p. ej. "ssh root@1.2.3.4 -p 40123"
    m = re.search(r"@([\w.-]+)\s+-p\s*(\d+)", d["command"])
    host, puerto = m.groups() if m else (None, None)
if not (host and puerto) and p.get("publicIp"):
    host, puerto = p["publicIp"], (p.get("portMappings") or {}).get("22")
print(host or "-", puerto or "-")'
# Reintenta un ssh/scp que no llegó a conectar (sshd del pod aún arrancando o reiniciándose). Con -255 solo reintenta
# si el código es 255 (fallo de conexión de ssh), para no repetir un comando remoto que sí corrió y falló.
_rp_reintentar() {
  local solo255=0 n r; [[ "$1" == -255 ]] && { solo255=1; shift; }
  for n in 1 2 3 4 5; do
    "$@"; r=$?
    ((r == 0)) && return 0
    ((solo255 && r != 255)) && return "$r"
    ((n < 5)) && { echo "  SSH al pod falló (intento $n/5, código $r): reintento en 15 s" >&2; sleep 15; }
  done
  return "$r"
}
rp_api() { curl -fsS -m 20 -H "Authorization: Bearer ${RUNPOD_API_KEY:?exporta RUNPOD_API_KEY}" "https://api.runpod.io/v2/$1"; }
# `colab install`: entorno virtual limpio en /opt/ev con requirements-runpod.txt, sin tocar los paquetes de la imagen;
# gcc y los headers de Python los usa torch.compile. Falla si torch no ve la GPU.
RP_PREPARAR='set -e
(sleep 30 ; timeout 120 apt-get update -qq -o Acquire::http::Timeout=20 && DEBIAN_FRONTEND=noninteractive timeout 300 apt-get install -y -qq build-essential python3-dev) >/dev/null 2>&1 || true
command -v uv >/dev/null || PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install -q uv
[ -x /opt/ev/bin/python ] || uv venv -q --python python3 /opt/ev
VIRTUAL_ENV=/opt/ev uv pip install -q -r /content/requirements-runpod.txt
/opt/ev/bin/python -c "import torch; assert torch.cuda.is_available(), torch.__version__; print(torch.__version__, torch.cuda.get_device_name(0))"'

# Crea el pod de la sesión $1 con la GPU de Colab $2 y espera a tener SSH. Si falla después de crearlo, lo borra.
_rp_crear() {
  local nombre="evidencite-$1-$(date +%s)" id ip="-" puerto="-"
  [[ -n "${RP_GPU[$2]:-}" ]] || { echo "La GPU $2 no tiene equivalente en RunPod (usa G4, A100 o L4)" >&2; return 1; }
  [[ -n "${RUNPOD_API_KEY:-}" ]] || { echo "Exporta RUNPOD_API_KEY: la IP del pod se pide a la API de RunPod" >&2; return 1; }
  runpodctl pod create --name "$nombre" --image "$RP_IMAGEN" --gpu-id "${RP_GPU[$2]}" --container-disk-in-gb "$RP_DISCO" \
    --ports 22/tcp --cloud-type SECURE --min-cuda-version "$RP_CUDA" >/dev/null || return 1
  id="$(runpodctl pod list --name "$nombre" --output json | "$PY" -c 'import json,sys; print(json.load(sys.stdin)[0]["id"])')" || return 1
  echo "$id - -" > "$RP_ESTADO/$1"   # desde ya, para que `stop` y la limpieza al salir lo borren aunque falle lo que sigue
  for _ in $(seq 60); do   # hasta 10 min por la IP pública y el puerto SSH
    read -r ip puerto < <({ rp_api "pods/$id" || rp_api pods; } 2>/dev/null | RP_ID="$id" "$PY" -c "$RP_IP_PUERTO") || true
    [[ "${ip:--}" != - && "${puerto:--}" != - ]] && break; ip=-; puerto=-; sleep 10
  done
  echo "$id $ip $puerto" > "$RP_ESTADO/$1"
  for _ in $(seq 30); do   # sshd tarda unos segundos más que la IP
    [[ "$ip" != - ]] && ssh "${RP_SSH[@]}" -p "$puerto" "root@$ip" 'mkdir -p /content' 2>/dev/null && return 0; sleep 5
  done
  echo "El pod $id no quedó accesible por SSH (¿llave sin registrar en RunPod o distinta a la que usa ssh? RP_LLAVE=ruta): lo borro" >&2; runpodctl pod delete "$id" >/dev/null; rm -f "$RP_ESTADO/$1"; return 1
}

colab() {
  local cmd="$1" s="" gpu="" f="" req="" t=14400 env=() pos=(); shift
  while (($#)); do case "$1" in
    -s) s="$2"; shift 2 ;; --gpu) gpu="$2"; shift 2 ;; -f) f="$2"; shift 2 ;; -r) req="$2"; shift 2 ;;
    --timeout) t="$2"; shift 2 ;; --env) env+=("$2"); shift 2 ;; --high-mem) shift ;; *) pos+=("$1"); shift ;;
  esac; done
  [[ "$cmd" == new ]] && { _rp_crear "$s" "$gpu"; return; }
  local ID HOST PUERTO; read -r ID HOST PUERTO < "$RP_ESTADO/$s" 2>/dev/null || { echo "No hay sesión '$s'" >&2; return 1; }
  local remoto=(ssh -n "${RP_SSH[@]}" -p "$PUERTO" "root@$HOST") copiar=(scp -q "${RP_SSH[@]}" -P "$PUERTO")
  case "$cmd" in
    install)  _rp_reintentar "${copiar[@]}" "$req" "$(dirname "$req")/requirements-runpod.txt" "root@$HOST:/content/" &&
              _rp_reintentar -255 "${remoto[@]}" "$RP_PREPARAR" ;;
    upload)   _rp_reintentar "${copiar[@]}" "${pos[0]}" "root@$HOST:${pos[1]}" ;;
    download) _rp_reintentar "${copiar[@]}" "root@$HOST:${pos[0]}" "${pos[1]}" ;;
    exec)     "${copiar[@]}" "$f" "root@$HOST:/content/_lanzar.py" || return 1
              "${remoto[@]}" "cd /content && timeout $t env RUNPOD_POD_ID=$ID $(printf '%q ' "${env[@]}") /opt/ev/bin/python _lanzar.py"
              [[ $? -ne 255 ]] ;;   # como la CLI de Colab: solo falla si se pierde la sesión; un error de Python queda en el registro
    restart-kernel) ;;              # cada exec es un proceso nuevo: la GPU queda libre al terminar
    stop)     if runpodctl pod delete "$ID" >/dev/null; then rm -f "$RP_ESTADO/$s"; echo "Pod $ID borrado" >&2
              else echo "NO se pudo borrar el pod $ID: bórralo a mano (runpodctl pod delete $ID) o seguirá cobrando" >&2; return 1; fi ;;
    *)        echo "colab $cmd: no está implementado para RunPod" >&2; return 1 ;;
  esac
}
