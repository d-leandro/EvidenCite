"""Entorno: versiones de las librerías, token de Hugging Face y detección de Colab (celda 1 del notebook).

en_colab() se llama una sola vez, en el bloque principal de lora.py, y su valor llega a las funciones
como parámetro.
"""

import os
import platform
from importlib import metadata

import peft
import torch
import transformers
from huggingface_hub import constants as hf_constants
from packaging.version import Version

try:
    from google.colab import userdata as colab_userdata  # solo existe en Colab
except ImportError:
    colab_userdata = None


# Versiones por debajo de las cuales el notebook no funciona: torch 2.6 lo exige transformers 5, transformers 4.56
# introdujo el argumento dtype de from_pretrained y peft 0.16 trae LoRA-FA. No se fija ninguna versión exacta.
VERSIONES_MINIMAS = {"torch": "2.6", "transformers": "4.56", "peft": "0.16"}


def titulo(texto: str) -> None:
    """Imprime un encabezado de sección para separar las etapas en la salida."""
    print(f"\n== {texto} " + "=" * max(3, 70 - len(texto)))


def en_colab() -> bool:
    """Devuelve True si el notebook corre en un runtime de Google Colab, desde la interfaz web o desde la CLI."""
    return "COLAB_RELEASE_TAG" in os.environ or colab_userdata is not None


def versiones() -> dict:
    """Devuelve la versión de Python y de cada librería del notebook (None si no está instalada)."""
    v = {"python": platform.python_version(), "torch": torch.__version__, "cuda_de_torch": torch.version.cuda,
         "transformers": transformers.__version__, "peft": peft.__version__}
    for paquete in ("accelerate", "bitsandbytes", "triton", "scikit-learn", "huggingface-hub", "mlflow", "mlflow-skinny"):
        try:
            v[paquete] = metadata.version(paquete)
        except metadata.PackageNotFoundError:
            v[paquete] = None
    return v


def verificar_versiones() -> dict:
    """Imprime las versiones instaladas y avisa de las que están por debajo de VERSIONES_MINIMAS.

    No instala ni fija nada: en Colab se usan las versiones que trae el runtime.
    """
    v = versiones()
    print("  " + " · ".join(f"{k} {val}" for k, val in v.items() if val))
    for paquete, minima in VERSIONES_MINIMAS.items():
        if Version(v[paquete]) < Version(minima):
            print(f"AVISO: {paquete} {v[paquete]} es anterior a {minima}, la mínima que necesita el notebook.")
    return v


def leer_hf_token(colab: bool) -> tuple[str | None, str]:
    """Busca el token de Hugging Face y devuelve (token, origen). El origen es un texto para el registro.

    - En Colab lo lee del secret HF_TOKEN (panel de la llave en la interfaz web, con acceso activado para el
      notebook). Los secrets solo los entrega la interfaz web: desde la CLI la petición se agota a los 10 s.
      Si el secret no está disponible, usa la variable de entorno HF_TOKEN (con la CLI: --env HF_TOKEN=...).
    - En local lo lee de la variable de entorno HF_TOKEN.
    """
    token_entorno = os.environ.get("HF_TOKEN", "").strip()
    if not (colab and colab_userdata is not None):
        return (token_entorno, "variable de entorno HF_TOKEN") if token_entorno else (None, "no hay variable de entorno HF_TOKEN")
    try:
        token = (colab_userdata.get("HF_TOKEN") or "").strip()
        if token:
            return token, "secret HF_TOKEN de Colab"
        motivo = "el secret HF_TOKEN está vacío"
    except Exception as e:  # noqa: BLE001  (las clases de error de google.colab cambian entre versiones)
        motivo = {
            "SecretNotFoundError": "no existe el secret HF_TOKEN",
            "NotebookAccessError": "el notebook no tiene acceso al secret HF_TOKEN (actívalo en el panel de secrets)",
            "TimeoutException": "el secret no respondió: Colab solo entrega secrets a la interfaz web, no a la CLI",
        }.get(type(e).__name__, f"no se pudo leer el secret HF_TOKEN ({type(e).__name__}: {e})")
    if token_entorno:
        return token_entorno, f"variable de entorno HF_TOKEN ({motivo})"
    return None, motivo


def configurar_hf_token(colab: bool) -> str:
    """Deja el token listo para las descargas de Hugging Face y devuelve de dónde salió. Nunca lo imprime.

    Con token, lo pone en la variable de entorno HF_TOKEN, que huggingface_hub consulta antes que cualquier otra
    fuente. Sin token en Colab, desactiva la búsqueda implícita de huggingface_hub para que no vuelva a pedir el
    secret por su cuenta (otros 10 s de espera y un aviso); los modelos públicos se descargan igual.
    """
    token, origen = leer_hf_token(colab)
    if token:
        os.environ["HF_TOKEN"] = token
        print(f"  token de Hugging Face: {origen}")
    else:
        if colab:
            hf_constants.HF_HUB_DISABLE_IMPLICIT_TOKEN = True
        print(f"  sin token de Hugging Face ({origen}): solo se pueden descargar modelos públicos")
    return origen
