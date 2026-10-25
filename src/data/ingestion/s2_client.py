"""Cliente para la API de Semantic Scholar con rate limiting y caché SQLite.

Permite buscar artículos por título para resolver abstracts (Configuración A)
y extraer identificadores canónicos de arXiv (Configuración B).
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


class S2ClientConfig(BaseModel):
    """Configuración para el cliente de Semantic Scholar Graph API."""

    base_url: str = Field(
        default="https://api.semanticscholar.org/graph/v1",
        description="URL base de la API Graph de Semantic Scholar",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Clave x-api-key (si es None, se lee de SEMANTIC_SCHOLAR_API_KEY)",
    )
    cache_path: Optional[Path] = Field(
        default=Path("data/raw/s2_cache.db"),
        description="Ruta a la base de datos SQLite para caché persistente de consultas",
    )
    rate_limit_delay: float = Field(
        default=10.0,
        description="Segundos mínimos de espera entre peticiones sucesivas",
    )
    timeout_seconds: int = Field(
        default=30,
        description="Timeout de petición HTTP en segundos",
    )
    max_retries: int = Field(
        default=5,
        description="Número máximo de reintentos ante 429 o errores transitorios 5xx",
    )
    backoff_factor: float = Field(
        default=2.0,
        description="Factor de espera exponencial entre reintentos",
    )


class S2PaperMetadata(BaseModel):
    """Metadatos bibliográficos normalizados retornados por Semantic Scholar."""

    paper_id: str = Field(..., description="Identificador único SHA-1 en Semantic Scholar")
    title: str = Field(..., description="Título normalizado del artículo")
    abstract: Optional[str] = Field(default=None, description="Texto del abstract (indispensable para Config A)")
    year: Optional[int] = Field(default=None, description="Año de publicación")
    arxiv_id: Optional[str] = Field(default=None, description="Identificador canónico de arXiv (ej. '1910.05453')")
    doi: Optional[str] = Field(default=None, description="Identificador DOI si existe")
    authors: List[str] = Field(default_factory=list, description="Lista de nombres de autores")


def _load_dotenv_if_present() -> None:
    """Carga variables desde el archivo .env si existe en el entorno local."""
    for candidate in [Path(".env"), Path(__file__).resolve().parents[3] / ".env"]:
        if candidate.exists():
            try:
                for line in candidate.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k not in os.environ:
                            os.environ[k] = v
            except Exception:
                pass


class SemanticScholarClient:
    """Cliente oficial para la API de Semantic Scholar con caché SQLite y rate limiting."""

    FIELDS: str = "paperId,title,abstract,externalIds,year,authors"

    def __init__(self, config: Optional[S2ClientConfig] = None) -> None:
        _load_dotenv_if_present()
        self.config = config or S2ClientConfig()
        self.api_key = self.config.api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")

        # Ajustar delay por defecto a 10.0s para el endpoint de búsqueda libre (100% de éxito empírico)
        if self.api_key and self.config.rate_limit_delay in (1.0, 1.2):
            self.rate_limit_delay = 10.0
        else:
            self.rate_limit_delay = self.config.rate_limit_delay

        self.last_request_time: float = 0.0
        self._init_cache()

    def _init_cache(self) -> None:
        """Inicializa la tabla SQLite para caché local de consultas si está configurada."""
        if not self.config.cache_path:
            return

        cache_path = Path(self.config.cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(cache_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_cache (
                    cache_key TEXT PRIMARY KEY,
                    paper_data TEXT,
                    created_at REAL
                )
                """
            )
            conn.commit()

    def _get_from_cache(self, key: str) -> Tuple[bool, Optional[S2PaperMetadata]]:
        """Recupera un resultado previo desde la base de datos SQLite.

        Returns:
            Tupla (is_cached, paper_metadata).
            - (False, None): No está en caché (cache miss).
            - (True, None): Está en caché como 'no encontrado' (tombstone).
            - (True, S2PaperMetadata): Encontrado y recuperado de caché.
        """
        if not self.config.cache_path or not Path(self.config.cache_path).exists():
            return False, None

        with sqlite3.connect(self.config.cache_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT paper_data FROM paper_cache WHERE cache_key = ?", (key,))
            row = cursor.fetchone()

        if not row or not row[0]:
            return False, None

        try:
            data = json.loads(row[0])
            if data.get("__not_found__"):
                return True, None
            return True, S2PaperMetadata(**data)
        except (json.JSONDecodeError, TypeError, ValueError):
            return False, None

    def _save_to_cache(self, key: str, paper: Optional[S2PaperMetadata]) -> None:
        """Almacena una respuesta en la caché local SQLite."""
        if not self.config.cache_path:
            return

        payload = paper.model_dump() if paper else {"__not_found__": True}
        data_str = json.dumps(payload)

        with sqlite3.connect(self.config.cache_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO paper_cache (cache_key, paper_data, created_at)
                VALUES (?, ?, ?)
                """,
                (key, data_str, time.time()),
            )
            conn.commit()

    def _request_with_retry(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Realiza peticiones HTTP GET respetando rate limit y reintentos ante 429 / 5xx."""
        # Control proactivo de tasa
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)

        headers: Dict[str, str] = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        for attempt in range(1, self.config.max_retries + 1):
            try:
                self.last_request_time = time.time()
                response = requests.get(
                    endpoint,
                    headers=headers,
                    params=params,
                    timeout=self.config.timeout_seconds,
                )

                if response.status_code == 200:
                    return response.json()

                if response.status_code == 404:
                    return None

                if response.status_code in (401, 403):
                    raise ValueError(
                        f"Autenticación fallida en Semantic Scholar (HTTP {response.status_code}). Verifica tu API key."
                    )

                if response.status_code == 429:
                    if attempt == self.config.max_retries:
                        logger.warning(
                            "Rate limit alcanzado (HTTP 429) tras %d intentos en %s.",
                            self.config.max_retries,
                            endpoint,
                        )
                        response.raise_for_status()

                    retry_after = response.headers.get("Retry-After")
                    try:
                        if retry_after is not None:
                            val = float(retry_after)
                            wait_time = val if val < 1.0 else val + 1.0
                        else:
                            wait_time = max(15.0, 15.0 * attempt)
                    except (ValueError, TypeError):
                        wait_time = max(15.0, 15.0 * attempt)

                    logger.warning(
                        "Rate limit alcanzado (HTTP 429) en intento %d/%d. Esperando %.1fs para enfriamiento...",
                        attempt,
                        self.config.max_retries,
                        wait_time,
                    )
                    time.sleep(wait_time)
                    self.last_request_time = time.time()
                    continue

                response.raise_for_status()

            except (requests.RequestException, requests.Timeout) as exc:
                self.last_request_time = time.time()
                if attempt == self.config.max_retries:
                    logger.error("Error definitivo en petición a %s tras %d intentos: %s", endpoint, attempt, exc)
                    raise
                wait_time = max(15.0, 15.0 * attempt)
                logger.warning("Falla transitoria en %s: %s. Reintentando en %.1fs...", endpoint, exc, wait_time)
                time.sleep(wait_time)
                self.last_request_time = time.time()

        return None

    @staticmethod
    def _parse_paper_dict(item: Dict[str, Any]) -> S2PaperMetadata:
        """Normaliza un diccionario de respuesta de S2 en un objeto S2PaperMetadata."""
        paper_id = item.get("paperId", "")
        title = item.get("title", "")
        abstract = item.get("abstract")
        year = item.get("year")

        external_ids = item.get("externalIds") or {}
        arxiv_id = external_ids.get("ArXiv") or external_ids.get("arxiv")
        doi = external_ids.get("DOI") or external_ids.get("doi")

        authors_list: List[str] = []
        for author in item.get("authors", []):
            if isinstance(author, dict) and author.get("name"):
                authors_list.append(author["name"])

        return S2PaperMetadata(
            paper_id=paper_id,
            title=title,
            abstract=abstract,
            year=year,
            arxiv_id=arxiv_id,
            doi=doi,
            authors=authors_list,
        )

    def search_paper_by_title(self, title: str) -> Optional[S2PaperMetadata]:
        """Busca un artículo por título o cadena textual de cita.

        Args:
            title: Título o referencia textual a buscar.

        Returns:
            Objeto S2PaperMetadata del mejor candidato o None si no hubo resultados.
        """
        clean_title = title.strip()
        if not clean_title:
            return None

        cache_key = f"search:{clean_title.lower()}"
        is_cached, cached = self._get_from_cache(cache_key)
        if is_cached:
            return cached

        endpoint = f"{self.config.base_url}/paper/search"
        params = {"query": clean_title, "limit": 1, "fields": self.FIELDS}

        try:
            data = self._request_with_retry(endpoint, params=params)
        except requests.RequestException as exc:
            logger.warning("Error de red o rate limit en S2 para '%s': %s", clean_title, exc)
            return None

        if not data or not data.get("data"):
            self._save_to_cache(cache_key, None)
            return None

        best_match = data["data"][0]
        metadata = self._parse_paper_dict(best_match)
        self._save_to_cache(cache_key, metadata)
        return metadata

    def get_paper_by_id(self, paper_id: str) -> Optional[S2PaperMetadata]:
        """Consulta un artículo por ID de Semantic Scholar, DOI o arXiv (ej. 'ARXIV:1910.05453').

        Args:
            paper_id: Identificador canónico con o sin prefijo.

        Returns:
            Objeto S2PaperMetadata o None si no existe.
        """
        clean_id = paper_id.strip()
        if not clean_id:
            return None

        cache_key = f"id:{clean_id.lower()}"
        is_cached, cached = self._get_from_cache(cache_key)
        if is_cached:
            return cached

        endpoint = f"{self.config.base_url}/paper/{clean_id}"
        params = {"fields": self.FIELDS}

        try:
            data = self._request_with_retry(endpoint, params=params)
        except requests.RequestException as exc:
            logger.warning("Error de red o rate limit en S2 para '%s': %s", clean_id, exc)
            return None

        if not data or "paperId" not in data:
            self._save_to_cache(cache_key, None)
            return None

        metadata = self._parse_paper_dict(data)
        self._save_to_cache(cache_key, metadata)
        return metadata

    def resolve_arxiv_id(self, title_or_query: str) -> Optional[str]:
        """Utilidad directa: busca un paper y retorna únicamente su arxiv_id si existe."""
        paper = self.search_paper_by_title(title_or_query)
        return paper.arxiv_id if paper else None


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos para consultas de prueba."""
    parser = argparse.ArgumentParser(
        description="Consulta la API de Semantic Scholar para resolver abstracts y arXiv IDs."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--title", type=str, help="Título del artículo a buscar")
    group.add_argument("--id", type=str, help="Identificador canónico (e.g. ARXIV:1910.05453 o S2 SHA)")
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("data/raw/s2_cache.db"),
        help="Ruta a la base de datos de caché SQLite (default: data/raw/s2_cache.db)",
    )
    return parser.parse_args()


def main() -> None:
    """Punto de entrada de ejecución CLI para consultas directas."""
    args = parse_args()
    config = S2ClientConfig(cache_path=args.cache_path)
    client = SemanticScholarClient(config)

    if args.title:
        logger.info("Buscando artículo por título: '%s'", args.title)
        paper = client.search_paper_by_title(args.title)
    else:
        logger.info("Consultando artículo por ID: '%s'", args.id)
        paper = client.get_paper_by_id(args.id)

    if not paper:
        logger.warning("No se encontró ningún artículo coincidente.")
    else:
        logger.info("Resultado resuelto:")
        print(json.dumps(paper.model_dump(), indent=2))


if __name__ == "__main__":
    main()
