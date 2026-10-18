"""Módulo de ingestión de unarXive hacia data/raw/unarxive/.

Descarga y descomprime shards JSONL desde Zenodo o la muestra oficial,
permitiendo extracción selectiva por años recientes (ej. 2022, 2021).
"""

import argparse
import json
import logging
from pathlib import Path
import shutil
import tarfile
import time
from typing import Dict, List, Optional

from pydantic import BaseModel, Field
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


class UnarxiveIngestionConfig(BaseModel):
    """Configuración para la ingestión de shards de unarXive."""

    mode: str = Field(
        default="sample",
        description="Modo de ingestión: 'sample' (muestra oficial ~10MB), 'zenodo' (open subset 4.84GB) o 'custom'",
    )
    sample_url: str = Field(
        default="https://github.com/IllDepence/unarXive/raw/master/doc/unarXive_data_sample.tar.gz",
        description="URL del shard de muestra oficial en GitHub",
    )
    zenodo_open_url: str = Field(
        default="https://zenodo.org/api/records/7752615/files/unarXive_230324_open_subset.tar.xz/content",
        description="URL del subconjunto abierto de Zenodo (~4.84 GB)",
    )
    custom_url: Optional[str] = Field(
        default=None,
        description="URL directa cuando mode='custom'",
    )
    years: Optional[List[str]] = Field(
        default=["2022", "2021"],
        description="Años a extraer selectivamente (ej. ['2022', '2021']). Si es None, extrae todos.",
    )
    output_dir: Path = Field(
        default=Path("data/raw/unarxive"),
        description="Directorio destino para los shards descomprimidos",
    )
    keep_archive: bool = Field(
        default=False,
        description="Conservar el archivo comprimido (.tar.xz/.tar.gz) tras la extracción",
    )
    force: bool = Field(
        default=False,
        description="Forzar descarga y extracción aunque los shards ya existan",
    )
    timeout_seconds: int = Field(
        default=120,
        description="Timeout de lectura por petición HTTP en segundos",
    )
    max_retries: int = Field(
        default=3,
        description="Número máximo de reintentos ante fallas de red",
    )
    backoff_factor: float = Field(
        default=2.0,
        description="Factor de espera exponencial entre reintentos",
    )
    chunk_size_bytes: int = Field(
        default=4 * 1024 * 1024,
        description="Tamaño de bloque en bytes para streaming de descarga (default: 4MB)",
    )
    progress_interval_bytes: int = Field(
        default=100 * 1024 * 1024,
        description="Intervalo en bytes para reportar progreso de descarga en logs (default: 100MB)",
    )


class UnarxiveDownloader:
    """Orquestador de descarga, descompresión selectiva y validación de unarXive."""

    def __init__(self, config: Optional[UnarxiveIngestionConfig] = None) -> None:
        self.config = config or UnarxiveIngestionConfig()
        self.output_dir = Path(self.config.output_dir)

    def get_download_url(self) -> str:
        """Determina la URL a descargar según el modo configurado."""
        if self.config.mode == "sample":
            return self.config.sample_url
        elif self.config.mode == "zenodo":
            return self.config.zenodo_open_url
        elif self.config.mode == "custom":
            if not self.config.custom_url:
                raise ValueError("Se especificó mode='custom' pero custom_url es None.")
            return self.config.custom_url
        else:
            raise ValueError(f"Modo no reconocido: {self.config.mode}. Opciones válidas: 'sample', 'zenodo', 'custom'.")

    def download_file(self, url: str, dest_path: Path) -> Path:
        """Descarga un archivo remoto vía HTTP con streaming en bloques y reintentos.

        Args:
            url: URL del archivo remoto.
            dest_path: Ruta destino local.

        Returns:
            Path al archivo descargado.

        Raises:
            requests.HTTPError: Si la petición falla tras agotar reintentos.
            ValueError: Si la URL no existe (HTTP 404).
        """
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        for attempt in range(1, self.config.max_retries + 1):
            try:
                logger.info("Descargando: %s -> %s (intento %d/%d)", url, dest_path.name, attempt, self.config.max_retries)
                response = requests.get(url, stream=True, timeout=self.config.timeout_seconds)

                if response.status_code == 404:
                    raise ValueError(f"Recurso no encontrado (HTTP 404): {url}")

                response.raise_for_status()

                content_length = response.headers.get("content-length")
                total_size = int(content_length) if content_length and content_length.isdigit() else None

                total_downloaded = 0
                last_logged = 0
                start_time = time.time()
                last_log_time = start_time

                with open(dest_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=self.config.chunk_size_bytes):
                        if chunk:
                            f.write(chunk)
                            total_downloaded += len(chunk)

                            if total_downloaded - last_logged >= self.config.progress_interval_bytes:
                                now = time.time()
                                elapsed_interval = max(now - last_log_time, 0.001)
                                speed_mb_s = (total_downloaded - last_logged) / (1024 * 1024 * elapsed_interval)
                                mb_downloaded = total_downloaded / (1024 * 1024)

                                if total_size:
                                    mb_total = total_size / (1024 * 1024)
                                    pct = (total_downloaded / total_size) * 100
                                    logger.info(
                                        "Progreso descarga: %.1f MB / %.1f MB (%.1f%%) - Velocidad: %.2f MB/s",
                                        mb_downloaded,
                                        mb_total,
                                        pct,
                                        speed_mb_s,
                                    )
                                else:
                                    logger.info(
                                        "Progreso descarga: %.1f MB descargados - Velocidad: %.2f MB/s",
                                        mb_downloaded,
                                        speed_mb_s,
                                    )

                                last_logged = total_downloaded
                                last_log_time = now

                elapsed_total = max(time.time() - start_time, 0.001)
                avg_speed = (total_downloaded / (1024 * 1024)) / elapsed_total
                logger.info(
                    "Descarga completada: %s (%.2f MB en %.1fs, promedio %.2f MB/s)",
                    dest_path.name,
                    total_downloaded / (1024 * 1024),
                    elapsed_total,
                    avg_speed,
                )
                return dest_path

            except (requests.RequestException, requests.Timeout) as exc:
                if attempt == self.config.max_retries:
                    logger.error("Error definitivo al descargar %s tras %d intentos: %s", url, attempt, exc)
                    raise
                wait_time = self.config.backoff_factor ** attempt
                logger.warning("Fallo transitorio al descargar %s: %s. Reintentando en %.1fs...", url, exc, wait_time)
                time.sleep(wait_time)

        return dest_path

    def extract_archive(
        self,
        archive_path: Path,
        target_dir: Path,
        years: Optional[List[str]] = None,
    ) -> List[Path]:
        """Descomprime un archivo tar (.tar.gz, .tar.xz) filtrando por años si aplica.

        Deposita los archivos .jsonl directamente en target_dir (aplanando rutas internas).

        Args:
            archive_path: Ruta al archivo comprimido.
            target_dir: Directorio destino donde depositar los .jsonl.
            years: Lista opcional de años a extraer (ej. ['2022', '2021']).

        Returns:
            Lista de rutas Path a los shards .jsonl extraídos.
        """
        target_dir.mkdir(parents=True, exist_ok=True)
        extracted_shards: List[Path] = []

        # Normalizar años a sufijos de 2 dígitos (ej. '2022' -> '22')
        year_suffixes = [y[-2:] for y in years] if years else []

        logger.info("Abriendo archivo comprimido: %s", archive_path.name)
        with tarfile.open(archive_path, "r:*") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".jsonl"):
                    continue

                # Filtrar por año si se especificó
                if year_suffixes:
                    matches_year = any(
                        f"/{y}/" in member.name
                        or member.name.startswith(f"{y}/")
                        or f"_{y}" in member.name
                        for y in year_suffixes
                    )
                    if not matches_year:
                        continue

                filename = Path(member.name).name
                dest_file = target_dir / filename

                f_in = tar.extractfile(member)
                if f_in:
                    with open(dest_file, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                    extracted_shards.append(dest_file)
                    if len(extracted_shards) % 50 == 0 or len(extracted_shards) == 1:
                        logger.info("Extraído shard #%d: %s (%.2f MB)", len(extracted_shards), filename, dest_file.stat().st_size / (1024 * 1024))

        logger.info("Extracción finalizada. Total de shards extraídos: %d", len(extracted_shards))
        return extracted_shards

    @staticmethod
    def validate_shard(shard_path: Path, max_records_to_check: int = 5) -> int:
        """Valida que el archivo JSONL sea legible y cumpla con el esquema básico de unarXive.

        Args:
            shard_path: Ruta al shard .jsonl a verificar.
            max_records_to_check: Cantidad de registros iniciales a validar estructuralmente.

        Returns:
            Cantidad total de artículos leídos en el shard.

        Raises:
            ValueError: Si el archivo está vacío o los registros no contienen 'paper_id' o 'body_text'.
        """
        total_records = 0
        with open(shard_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line_str = line.strip()
                if not line_str:
                    continue

                if total_records < max_records_to_check:
                    try:
                        record = json.loads(line_str)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"JSON inválido en {shard_path.name} línea {line_idx + 1}: {exc}") from exc

                    if not isinstance(record, dict) or "paper_id" not in record or "body_text" not in record:
                        raise ValueError(
                            f"Registro inválido en {shard_path.name} línea {line_idx + 1}: faltan llaves 'paper_id' o 'body_text'."
                        )

                total_records += 1

        if total_records == 0:
            raise ValueError(f"El shard {shard_path.name} está vacío.")

        return total_records

    def find_existing_shards(self) -> List[Path]:
        """Encuentra shards .jsonl existentes en output_dir que coincidan con los años solicitados."""
        if not self.output_dir.exists():
            return []

        all_shards = list(self.output_dir.glob("*.jsonl"))
        if not self.config.years or self.config.mode == "sample":
            return [s for s in all_shards if s.stat().st_size > 0]

        year_suffixes = [y[-2:] for y in self.config.years]
        matching_shards = [
            s
            for s in all_shards
            if s.stat().st_size > 0
            and any(f"_{y}" in s.name or f"arXiv_src_{y}" in s.name for y in year_suffixes)
        ]
        return matching_shards

    def run(self) -> Dict[str, int]:
        """Ejecuta el pipeline completo de descarga, descompresión selectiva y validación.

        Returns:
            Diccionario que mapea cada shard extraído con su conteo de artículos validados.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Idempotencia: comprobar si ya existen shards locales válidos
        existing_shards = self.find_existing_shards()
        if existing_shards and not self.config.force:
            logger.info("Se encontraron %d shards ya existentes en disco. Omitiendo descarga.", len(existing_shards))
            results: Dict[str, int] = {}
            for s in existing_shards:
                count = self.validate_shard(s)
                results[s.name] = count
                logger.info("Validado shard existente %s: %d artículos.", s.name, count)
            return results

        # Determinar URL y nombre temporal de archivo
        url = self.get_download_url()
        archive_name = "unarxive_download.tar.gz" if url.endswith(".tar.gz") else "unarxive_download.tar.xz"
        temp_archive_path = self.output_dir / archive_name

        # Descargar archivo si no existe localmente
        if not temp_archive_path.exists() or temp_archive_path.stat().st_size == 0:
            self.download_file(url, temp_archive_path)
        else:
            logger.info(
                "Archivo comprimido ya existente en disco: %s (%.2f MB). Omitiendo descarga.",
                temp_archive_path.name,
                temp_archive_path.stat().st_size / (1024 * 1024),
            )

        # Extraer selectivamente por año
        try:
            years_filter = None if self.config.mode == "sample" else self.config.years
            extracted_shards = self.extract_archive(
                archive_path=temp_archive_path,
                target_dir=self.output_dir,
                years=years_filter,
            )
        finally:
            # Limpiar archivo comprimido a menos que se indique keep_archive
            if not self.config.keep_archive and temp_archive_path.exists():
                logger.info("Eliminando archivo comprimido temporal: %s", temp_archive_path.name)
                temp_archive_path.unlink()

        # Validar shards extraídos
        results: Dict[str, int] = {}
        for shard_path in extracted_shards:
            count = self.validate_shard(shard_path)
            results[shard_path.name] = count
            logger.info("Validado shard %s: %d artículos.", shard_path.name, count)

        return results


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Descarga y extrae shards de unarXive hacia data/raw/unarxive/."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sample", "zenodo", "custom"],
        default="sample",
        help="Modo de ingestión: 'sample' (muestra oficial ~10MB), 'zenodo' (open subset 4.84GB) o 'custom'",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        default=["2022", "2021"],
        help="Años a extraer del subconjunto (ej. 2022 2021).",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="URL directa del archivo a descargar cuando --mode custom.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/unarxive"),
        help="Directorio destino inmutable (default: data/raw/unarxive)",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        default=False,
        help="Conservar el archivo comprimido (.tar.gz/.tar.xz) tras la extracción",
    )
    parser.add_argument(
        "--chunk-size-mb",
        type=float,
        default=4.0,
        help="Tamaño de bloque en MB para el streaming de descarga (default: 4.0)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Forzar la re-descarga y re-extracción de shards existentes",
    )
    return parser.parse_args()


def main() -> None:
    """Punto de entrada de ejecución CLI."""
    args = parse_args()
    config = UnarxiveIngestionConfig(
        mode=args.mode,
        years=args.years,
        custom_url=args.url,
        output_dir=args.output_dir,
        keep_archive=args.keep_archive,
        chunk_size_bytes=int(args.chunk_size_mb * 1024 * 1024),
        force=args.force,
    )
    downloader = UnarxiveDownloader(config)
    results = downloader.run()
    logger.info("Ingestión de unarXive finalizada con éxito. Shards disponibles: %s", list(results.keys()))


if __name__ == "__main__":
    main()
