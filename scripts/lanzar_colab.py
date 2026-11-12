"""Lanzador para la CLI de Colab: colab exec -s <sesión> -f scripts/lanzar_colab.py --env FAMILIA=lora --env ...

`colab exec -f` manda el código de este archivo al kernel, que lo corre en /content. `colab upload` sube un solo
archivo, no carpetas, así que los scripts suben src/, configs/ y config_dvc.sh comprimidos en /content/evidencite.zip
y la llave de la cuenta de servicio en /content/cuenta_servicio.json. Este lanzador descomprime el código, clona el
repo en el tag de los datos (DATOS_TAG) en /content/datos, configura DVC con config_dvc.sh y la llave, baja con
`dvc pull` la carpeta DATOS_DVC, entra a /content/EvidenCite y corre `python -m src.modeling` con las variables de
entorno de --env y DATOS_DIR en esos datos. Si algo de los datos falla, la corrida falla. El zip de la corrida queda
en /content/salidas/<nombre>.zip.
"""

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# Variables que deben fijarse antes de importar torch, transformers y huggingface_hub.
if sys.platform == "linux" and not any(v in os.environ for v in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # menos fragmentación de la memoria de CUDA
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")  # las barras con \r ensucian la salida de la CLI
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

RAIZ = Path("/content") if Path("/content").is_dir() else Path.cwd()
REPO = RAIZ / "EvidenCite"
CLON = RAIZ / "datos"  # el repo clonado en el tag de los datos
LLAVE = RAIZ / "cuenta_servicio.json"


def _extraer(zip_: Path, destino: Path) -> None:
    """Extrae el zip en `destino`, sin carpetas de macOS y rechazando rutas que salgan del destino."""
    with zipfile.ZipFile(zip_) as z:
        miembros = [m for m in z.namelist() if not m.startswith("__MACOSX")]
        for m in miembros:
            if not (destino / m).resolve().is_relative_to(destino.resolve()):
                raise ValueError(f"Ruta insegura dentro de {zip_.name}: {m}")
        z.extractall(destino, miembros)


def _correr(comando: list[str], cwd: Path, **entorno: str) -> None:
    """Corre el comando, muestra su salida y falla si el comando falla.

    Lo que imprime un subproceso no llega a la CLI desde el kernel, así que se captura y se imprime aquí.
    """
    r = subprocess.run(comando, cwd=cwd, capture_output=True, text=True, check=False, env={**os.environ, **entorno})
    salida = (r.stdout + r.stderr).strip()
    if salida:
        print(salida)
    if r.returncode != 0:
        raise RuntimeError(f"Falló `{' '.join(comando)}` (código {r.returncode}): revisa la salida de arriba")


def _bajar_datos() -> Path:
    """Baja con DVC la carpeta DATOS_DVC del tag DATOS_TAG y devuelve su ruta.

    Clona el repo (REPO_GIT) en ese tag, configura el remoto de Google Drive (ID_CARPETA) con config_dvc.sh y la
    llave de la cuenta de servicio y hace `dvc pull`. Si la sesión ya tiene el clon en ese tag no lo repite, y DVC
    solo descarga lo que falta.
    """
    faltan = [v for v in ("DATOS_TAG", "DATOS_DVC", "ID_CARPETA", "REPO_GIT") if not os.environ.get(v)]
    if faltan:
        raise RuntimeError(
            f"Faltan {', '.join(faltan)}: lanza la corrida con correr_experimento.sh o probar_variantes.sh"
        )
    if not LLAVE.is_file():
        raise FileNotFoundError(f"No está la llave de la cuenta de servicio en {LLAVE}")
    dvc = shutil.which("dvc", path=os.pathsep.join([str(Path(sys.executable).parent), os.environ.get("PATH", "")]))
    if dvc is None:
        raise RuntimeError("DVC no está instalado en la sesión: revisa scripts/requirements-colab.txt")
    tag = os.environ["DATOS_TAG"]
    datos = os.environ["DATOS_DVC"].strip().removeprefix("./").rstrip("/").removesuffix(".dvc")

    origen = f"{os.environ['REPO_GIT']} {tag}"
    marca = CLON / ".git" / "origen_datos"  # el repo y el tag del clon
    if not marca.is_file() or marca.read_text(encoding="utf-8").strip() != origen:
        shutil.rmtree(CLON, ignore_errors=True)
        clonar = ["git", "-c", "advice.detachedHead=false", "clone", "--quiet", "--depth", "1", "--branch", tag]
        _correr([*clonar, os.environ["REPO_GIT"], str(CLON)], RAIZ)
        marca.write_text(origen, encoding="utf-8")
    if not (CLON / f"{datos}.dvc").is_file():
        raise FileNotFoundError(f"El tag {tag} no tiene {datos}.dvc: revisa DATOS_TAG y DATOS_DVC")
    # El config_dvc.sh del código subido (el del tag puede ser anterior), con el dvc de la sesión y no `uv run dvc`.
    camino = os.pathsep.join([str(Path(dvc).parent), os.environ.get("PATH", "")])
    _correr(["bash", str(REPO / "config_dvc.sh")], CLON, GDRIVE_SA_JSON=str(LLAVE), PATH=camino)
    _correr([dvc, "pull", f"{datos}.dvc"], CLON, DVC_NO_ANALYTICS="1")

    carpeta = CLON / datos
    faltan = [f"{s}.jsonl" for s in ("train", "dev", "test") if not (carpeta / f"{s}.jsonl").is_file()]
    if faltan:
        raise FileNotFoundError(f"{carpeta} no tiene {', '.join(faltan)}")
    print(f"Datos: {datos} del tag {tag}")
    return carpeta


if (RAIZ / "evidencite.zip").is_file():
    for carpeta in ("src", "configs"):  # el código de esta corrida reemplaza al de corridas anteriores
        shutil.rmtree(REPO / carpeta, ignore_errors=True)
    _extraer(RAIZ / "evidencite.zip", RAIZ)
os.environ["DATOS_DIR"] = str(_bajar_datos())
os.environ.setdefault("SALIDAS_DIR", str(RAIZ / "salidas"))

os.chdir(REPO)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
# Si el kernel ya importó src en una corrida anterior, se olvida para importar el código recién descomprimido.
for nombre in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
    del sys.modules[nombre]

from src.modeling.__main__ import main  # noqa: E402

resultado = main()
