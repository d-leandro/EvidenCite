"""Lanzador para la CLI de Colab: colab exec -s <sesión> -f scripts/lanzar_colab.py --env FAMILIA=lora --env ...

`colab exec -f` manda el código de este archivo al kernel, que lo corre en /content. `colab upload` sube un solo
archivo, no carpetas, así que los scripts suben src/ y configs/ comprimidos en /content/evidencite.zip y el dataset en
/content/dataset.zip; este lanzador los descomprime, entra a /content/EvidenCite y corre `python -m src.modeling` con
las variables de entorno de --env. El zip de la corrida queda en /content/salidas/<nombre>.zip.
"""

import os
import shutil
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


def _extraer(zip_: Path, destino: Path) -> None:
    """Extrae el zip en `destino`, sin carpetas de macOS y rechazando rutas que salgan del destino."""
    with zipfile.ZipFile(zip_) as z:
        miembros = [m for m in z.namelist() if not m.startswith("__MACOSX")]
        for m in miembros:
            if not (destino / m).resolve().is_relative_to(destino.resolve()):
                raise ValueError(f"Ruta insegura dentro de {zip_.name}: {m}")
        z.extractall(destino, miembros)


if (RAIZ / "evidencite.zip").is_file():
    for carpeta in ("src", "configs"):  # el código de esta corrida reemplaza al de corridas anteriores
        shutil.rmtree(REPO / carpeta, ignore_errors=True)
    _extraer(RAIZ / "evidencite.zip", RAIZ)
if (RAIZ / "dataset.zip").is_file() and not os.environ.get("DATOS_DIR"):
    destino = RAIZ / "dataset_extraido"
    if not destino.is_dir() or (RAIZ / "dataset.zip").stat().st_mtime > destino.stat().st_mtime:
        shutil.rmtree(destino, ignore_errors=True)  # se extrae una vez por dataset.zip subido
        _extraer(RAIZ / "dataset.zip", destino)
    train = next(p for p in sorted(destino.rglob("train.jsonl")))
    os.environ["DATOS_DIR"] = str(train.parent)
os.environ.setdefault("SALIDAS_DIR", str(RAIZ / "salidas"))

os.chdir(REPO)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
# Si el kernel ya importó src en una corrida anterior, se olvida para importar el código recién descomprimido.
for nombre in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
    del sys.modules[nombre]

from src.modeling.__main__ import main  # noqa: E402

resultado = main()
