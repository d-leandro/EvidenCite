"""Módulo de ingestión del dataset MultiCite hacia data/raw/multicite/.

Descarga de forma determinista los splits de clasificación
desde el repositorio oficial de GitHub (allenai/multicite).
"""

import argparse
import json
import logging
from pathlib import Path
import time
from typing import Dict, List, Optional

from pydantic import BaseModel, Field
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MultiCiteIngestionConfig(BaseModel):
    """Configuración para la ingestión del dataset MultiCite."""

    base_url: str = Field(
        default="https://raw.githubusercontent.com/allenai/multicite/master/data",
        description="URL base del repositorio oficial en GitHub",
    )
    subset: str = Field(
        default="classification_3_context",
        description="Subconjunto de contexto (ej. classification_3_context, classification_1_context)",
    )
    output_dir: Path = Field(
        default=Path("data/raw/multicite"),
        description="Ruta destino inmutable para almacenar los archivos crudos",
    )
    include_full: bool = Field(
        default=False,
        description="Indica si se descarga además el archivo monolítico full-v20210918.json",
    )
    force: bool = Field(
        default=False,
        description="Fuerza la re-descarga incluso si los archivos locales ya existen",
    )
    timeout_seconds: int = Field(
        default=30,
        description="Timeout de conexión y lectura para peticiones HTTP en segundos",
    )
    max_retries: int = Field(
        default=3,
        description="Número máximo de reintentos ante errores de conexión o HTTP 5xx",
    )
    backoff_factor: float = Field(
        default=2.0,
        description="Factor de incremento para el tiempo de espera exponencial entre reintentos",
    )


class MultiCiteDownloader:
    """Orquestador de descarga y validación estructural de MultiCite."""

    SPLIT_FILES: List[str] = ["train.json", "dev.json", "test.json"]
    FULL_FILE: str = "full-v20210918.json"

    def __init__(self, config: Optional[MultiCiteIngestionConfig] = None) -> None:
        self.config = config or MultiCiteIngestionConfig()
        self.output_dir = Path(self.config.output_dir)

    def download_file(self, url: str, dest_path: Path) -> Path:
        """Descarga un archivo vía HTTP con reintentos y exponential backoff.

        Args:
            url: URL remota del archivo.
            dest_path: Ruta local de destino.

        Returns:
            Path al archivo descargado.

        Raises:
            requests.HTTPError: Si la petición falla de forma persistente.
            ValueError: Si la URL no existe (HTTP 404).
        """
        for attempt in range(1, self.config.max_retries + 1):
            try:
                logger.info("Descargando: %s -> %s (intento %d/%d)", url, dest_path.name, attempt, self.config.max_retries)
                response = requests.get(url, timeout=self.config.timeout_seconds)

                if response.status_code == 404:
                    raise ValueError(f"Recurso no encontrado en GitHub (HTTP 404): {url}. Verifica el nombre del subconjunto.")

                response.raise_for_status()

                dest_path.write_bytes(response.content)
                logger.info("Guardado exitosamente: %s (%d bytes)", dest_path.name, len(response.content))
                return dest_path

            except (requests.RequestException, requests.Timeout) as exc:
                if attempt == self.config.max_retries:
                    logger.error("Error definitivo al descargar %s tras %d intentos: %s", url, attempt, exc)
                    raise
                wait_time = self.config.backoff_factor ** attempt
                logger.warning("Fallo transitorio al descargar %s: %s. Reintentando en %.1fs...", url, exc, wait_time)
                time.sleep(wait_time)

        return dest_path

    @staticmethod
    def validate_split_file(file_path: Path) -> int:
        """Valida que el archivo JSON contenga una lista de instancias de clasificación válidas.

        Returns:
            Cantidad total de registros en el split.

        Raises:
            ValueError: Si el archivo no es un JSON válido o no cumple con el esquema esperado.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"El archivo {file_path.name} no contiene una lista JSON.")

        for i, item in enumerate(data):
            if not isinstance(item, dict) or "id" not in item or "x" not in item or "y" not in item:
                raise ValueError(f"Registro inválido en {file_path.name} índice {i}: faltan llaves 'id', 'x' o 'y'.")

        return len(data)

    @staticmethod
    def validate_full_file(file_path: Path) -> int:
        """Valida que el archivo full sea un diccionario JSON indexado por paper_id.

        Returns:
            Cantidad de papers en el dump completo.

        Raises:
            ValueError: Si el archivo no es un diccionario JSON.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"El archivo {file_path.name} no contiene un diccionario JSON de artículos.")

        return len(data)

    def run(self) -> Dict[str, int]:
        """Ejecuta el pipeline de descarga y validación.

        Returns:
            Diccionario que mapea cada archivo procesado con su conteo de registros.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        files_to_download: Dict[str, str] = {
            split: f"{self.config.base_url}/{self.config.subset}/{split}"
            for split in self.SPLIT_FILES
        }

        if self.config.include_full:
            url_full = f"{self.config.base_url}/{self.FULL_FILE}"
            files_to_download[self.FULL_FILE] = url_full

        results: Dict[str, int] = {}

        for filename, url in files_to_download.items():
            target_path = self.output_dir / filename

            # Idempotencia: omitir descarga si el archivo ya existe y no se forzó
            if target_path.exists() and not self.config.force:
                logger.info("El archivo %s ya existe en disco. Omitiendo descarga.", filename)
            else:
                self.download_file(url, target_path)

            # Validación de formato
            if filename == self.FULL_FILE:
                record_count = self.validate_full_file(target_path)
            else:
                record_count = self.validate_split_file(target_path)

            results[filename] = record_count
            logger.info("Validado %s: %d registros.", filename, record_count)

        return results


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Descarga el dataset MultiCite hacia data/raw/multicite/ de forma reproducible."
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="classification_3_context",
        help="Subconjunto de contexto a descargar (default: classification_3_context)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/multicite"),
        help="Directorio destino inmutable (default: data/raw/multicite)",
    )
    parser.add_argument(
        "--include-full",
        action="store_true",
        default=False,
        help="Descargar además el archivo completo full-v20210918.json (~46 MB)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Forzar la re-descarga de archivos existentes",
    )
    return parser.parse_args()


def main() -> None:
    """Punto de entrada de ejecución CLI."""
    args = parse_args()
    config = MultiCiteIngestionConfig(
        subset=args.subset,
        output_dir=args.output_dir,
        include_full=args.include_full,
        force=args.force,
    )
    downloader = MultiCiteDownloader(config)
    results = downloader.run()
    logger.info("Ingestión completada con éxito. Archivos procesados: %s", results)


if __name__ == "__main__":
    main()
