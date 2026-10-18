"""Pipeline Orquestador de Datos End-to-End (Capa 1 a Capa 4 / Actividad A1).

Orquesta la transformación secuencial de datos científicos desde su forma cruda
hasta los datasets de modelado finales, garantizando la idempotencia, el checkpointing
y la paridad estricta 1:1 entre Configuración A y Configuración B.

Etapas del Grafo Acíclico Dirigido (DAG):
1. structuring: data/raw/unarxive -> data/structured/papers.jsonl (CS filtrado)
2. seminal: Pre-calentamiento S2 y catálogo seminal (146 papers) -> s2_cache.db + papers.jsonl
3. mining: Extracción de contextos citantes (m=3) -> data/processed/citing_contexts.jsonl
4. chunking: Particionado granular (<=110 palabras) -> data/processed/cited_chunks.jsonl
5. builder: Ensamble 1:1, recuperación Top-3 SciBERT y split anti-fuga -> data/dataset/

"""

import argparse
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

# Asegurar que la raíz del proyecto esté en sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.data.builder import BuilderConfig, DatasetBuilder
from src.data.chunking import ChunkingConfig, ChunkingPipeline
from src.data.ingestion.s2_client import S2ClientConfig, SemanticScholarClient
from src.data.mining import MiningConfig, MiningPipeline
from src.data.seminal_resolver import (
    ingest_missing_seminal_papers,
    load_seminal_catalog,
    resolve_and_cache_seminal_papers,
)
from src.data.structuring import StructuringConfig, UnarxiveStructurer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

STAGE_NAMES = ["structuring", "seminal", "mining", "chunking", "builder"]


@dataclass
class StageResult:
    """Resultado de la ejecución de una etapa del pipeline."""

    name: str
    status: str  # "COMPLETED", "SKIPPED", "FAILED", "DRY-RUN"
    duration_seconds: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class PipelineConfig:
    """Configuración maestra del pipeline orquestador de datos."""

    raw_unarxive_dir: Path = Path("data/raw/unarxive")
    raw_multicite_dir: Path = Path("data/raw/multicite")
    s2_cache_path: Path = Path("data/raw/s2_cache.db")
    structured_papers_path: Path = Path("data/structured/papers.jsonl")
    citing_contexts_path: Path = Path("data/processed/citing_contexts.jsonl")
    cited_chunks_path: Path = Path("data/processed/cited_chunks.jsonl")
    dataset_dir: Path = Path("data/dataset")
    builder_config_path: Path = Path("configs/builder.yaml")
    retrieval_config_path: Path = Path("configs/retrieval.yaml")
    seed: int = 42
    force: bool = False
    skip_seminal: bool = False

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "PipelineConfig":
        """Carga la configuración desde un archivo YAML externo."""
        path = Path(yaml_path)
        if not path.exists():
            logger.warning("Archivo de configuración %s no encontrado. Usando defaults.", path)
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls(
            raw_unarxive_dir=Path(data.get("raw_unarxive_dir", cls.raw_unarxive_dir)),
            raw_multicite_dir=Path(data.get("raw_multicite_dir", cls.raw_multicite_dir)),
            s2_cache_path=Path(data.get("s2_cache_path", cls.s2_cache_path)),
            structured_papers_path=Path(data.get("structured_papers_path", cls.structured_papers_path)),
            citing_contexts_path=Path(data.get("citing_contexts_path", cls.citing_contexts_path)),
            cited_chunks_path=Path(data.get("cited_chunks_path", cls.cited_chunks_path)),
            dataset_dir=Path(data.get("dataset_dir", cls.dataset_dir)),
            builder_config_path=Path(data.get("builder_config_path", cls.builder_config_path)),
            retrieval_config_path=Path(data.get("retrieval_config_path", cls.retrieval_config_path)),
            seed=int(data.get("seed", cls.seed)),
            force=bool(data.get("force", cls.force)),
            skip_seminal=bool(data.get("skip_seminal", cls.skip_seminal)),
        )


class DataPipeline:
    """Orquestador end-to-end del pipeline de datos científicos."""

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()

    def validate_preconditions(self, stages_to_run: List[str]) -> List[str]:
        """Verifica que los insumos requeridos para las etapas existan en disco (Fail-Fast, RF-02)."""
        errors = []
        if "structuring" in stages_to_run:
            if not self.config.raw_unarxive_dir.exists():
                errors.append(f"Directorio crudo de unarXive no encontrado: {self.config.raw_unarxive_dir}")

        if "mining" in stages_to_run:
            if not self.config.raw_multicite_dir.exists():
                errors.append(f"Directorio crudo de MultiCite no encontrado: {self.config.raw_multicite_dir}")

        return errors

    def should_skip_stage(self, stage_name: str) -> Tuple[bool, str]:
        """Evalúa si una etapa puede omitirse debido a la existencia de sus artefactos (RF-03)."""
        if self.config.force:
            return False, "Ejecución forzada (--force activo)"

        if stage_name == "structuring":
            if self.config.structured_papers_path.exists() and self.config.structured_papers_path.stat().st_size > 0:
                return True, f"Artefacto existente: {self.config.structured_papers_path}"

        elif stage_name == "seminal":
            if self.config.skip_seminal:
                return True, "Omitido explícitamente vía --skip-seminal"
            # Si papers.jsonl ya existe y s2_cache.db existe con tamaño > 0, se puede omitir si no es forzado
            if self.config.s2_cache_path.exists() and self.config.s2_cache_path.stat().st_size > 0:
                return True, f"Caché S2 existente: {self.config.s2_cache_path}"

        elif stage_name == "mining":
            if self.config.citing_contexts_path.exists() and self.config.citing_contexts_path.stat().st_size > 0:
                return True, f"Artefacto existente: {self.config.citing_contexts_path}"

        elif stage_name == "chunking":
            if self.config.cited_chunks_path.exists() and self.config.cited_chunks_path.stat().st_size > 0:
                return True, f"Artefacto existente: {self.config.cited_chunks_path}"

        elif stage_name == "builder":
            instances_path = self.config.dataset_dir / "instances.jsonl"
            metrics_path = self.config.dataset_dir / "dataset_metrics.json"
            if instances_path.exists() and instances_path.stat().st_size > 0 and metrics_path.exists():
                return True, f"Dataset consolidado existente: {instances_path}"

        return False, "Artefacto ausente o incompleto"

    def run_stage_structuring(self) -> Dict[str, Any]:
        """Ejecuta la estructuración de shards de unarXive hacia papers.jsonl."""
        cfg = StructuringConfig(
            input_dir=self.config.raw_unarxive_dir,
            output_file=self.config.structured_papers_path,
            force=self.config.force,
        )
        structurer = UnarxiveStructurer(cfg)
        return structurer.run()

    def run_stage_seminal(self) -> Dict[str, Any]:
        """Ejecuta la resolución de artículos seminales hacia s2_cache.db y papers.jsonl."""
        s2_config = S2ClientConfig(cache_path=self.config.s2_cache_path)
        client = SemanticScholarClient(s2_config)
        catalog = load_seminal_catalog()

        resolved = resolve_and_cache_seminal_papers(client, catalog)
        ingested = ingest_missing_seminal_papers(self.config.structured_papers_path, catalog)
        return {"catalog_size": len(catalog), "resolved_s2": resolved, "ingested_papers": ingested}

    def run_stage_mining(self) -> Dict[str, Any]:
        """Ejecuta el minado de contextos citantes hacia citing_contexts.jsonl."""
        cfg = MiningConfig(
            multicite_dir=self.config.raw_multicite_dir,
            papers_file=self.config.structured_papers_path,
            output_file=self.config.citing_contexts_path,
            s2_cache_path=self.config.s2_cache_path,
            force=self.config.force,
        )
        miner = MiningPipeline(cfg)
        return miner.run()

    def run_stage_chunking(self) -> Dict[str, Any]:
        """Ejecuta el particionado discursivo hacia cited_chunks.jsonl."""
        cfg = ChunkingConfig(
            citing_contexts_path=self.config.citing_contexts_path,
            papers_path=self.config.structured_papers_path,
            output_path=self.config.cited_chunks_path,
            force=self.config.force,
        )
        chunker = ChunkingPipeline(cfg)
        return chunker.run()

    def run_stage_builder(self) -> Dict[str, Any]:
        """Ejecuta el ensamble final y particionado anti-fuga hacia data/dataset/."""
        builder_cfg = BuilderConfig.from_yaml(self.config.builder_config_path)
        builder_cfg.input_contexts_path = self.config.citing_contexts_path
        builder_cfg.papers_path = self.config.structured_papers_path
        builder_cfg.chunks_path = self.config.cited_chunks_path
        builder_cfg.retrieval_config_path = self.config.retrieval_config_path
        builder_cfg.output_dir = self.config.dataset_dir
        builder_cfg.seed = self.config.seed
        builder_cfg.force = self.config.force

        builder = DatasetBuilder(builder_cfg)
        return builder.build_dataset()

    def execute(
        self,
        stages: Optional[List[str]] = None,
        from_stage: Optional[str] = None,
        single_stage: Optional[str] = None,
        dry_run: bool = False,
    ) -> List[StageResult]:
        """Ejecuta el pipeline según el plan y las etapas seleccionadas."""
        # 1. Determinar lista ordenada de etapas a evaluar
        if single_stage:
            if single_stage not in STAGE_NAMES:
                raise ValueError(f"Etapa inválida: '{single_stage}'. Opciones: {STAGE_NAMES}")
            selected_stages = [single_stage]
        elif from_stage:
            if from_stage not in STAGE_NAMES:
                raise ValueError(f"Etapa inválida: '{from_stage}'. Opciones: {STAGE_NAMES}")
            start_idx = STAGE_NAMES.index(from_stage)
            selected_stages = STAGE_NAMES[start_idx:]
        elif stages:
            for s in stages:
                if s not in STAGE_NAMES:
                    raise ValueError(f"Etapa inválida: '{s}'. Opciones: {STAGE_NAMES}")
            selected_stages = [s for s in STAGE_NAMES if s in stages]
        else:
            selected_stages = list(STAGE_NAMES)

        # 2. Validar pre-condiciones iniciales (Fail-Fast)
        if not dry_run:
            precondition_errors = self.validate_preconditions(selected_stages)
            if precondition_errors:
                for err in precondition_errors:
                    logger.error("Pre-condición fallida: %s", err)
                raise RuntimeError(f"Fallaron {len(precondition_errors)} pre-condición(es) iniciales del pipeline.")

        logger.info("=" * 68)
        logger.info("           INICIANDO PIPELINE ORQUESTADOR DE DATOS (A1)           ")
        logger.info("=" * 68)
        logger.info("Etapas programadas: %s", " -> ".join(selected_stages))
        logger.info("Modo de ejecución: %s", "DRY-RUN (Simulación)" if dry_run else ("FORZADO (--force)" if self.config.force else "INCREMENTAL / CHECKPOINT"))
        logger.info("Semilla pseudoaleatoria: %d", self.config.seed)
        logger.info("-" * 68)

        results: List[StageResult] = []

        # Mapa de métodos ejecutores
        stage_runners = {
            "structuring": self.run_stage_structuring,
            "seminal": self.run_stage_seminal,
            "mining": self.run_stage_mining,
            "chunking": self.run_stage_chunking,
            "builder": self.run_stage_builder,
        }

        total_stages = len(selected_stages)

        for idx, stage_name in enumerate(selected_stages, 1):
            logger.info("\n[%d/%d] Evaluando etapa: %s", idx, total_stages, stage_name.upper())

            skip, reason = self.should_skip_stage(stage_name)

            if dry_run:
                status = "SKIPPED (Existe)" if skip else "TO EXECUTE"
                logger.info("  [DRY-RUN] Estado: %s (%s)", status, reason)
                results.append(StageResult(name=stage_name, status=f"DRY-RUN: {status}", details={"reason": reason}))
                continue

            if skip:
                logger.info("  Etapa omitida (%s). Pasando a la siguiente.", reason)
                results.append(StageResult(name=stage_name, status="SKIPPED", details={"reason": reason}))
                continue

            # Ejecución real de la etapa
            runner = stage_runners[stage_name]
            t_start = time.time()
            try:
                logger.info("  Ejecutando %s...", stage_name)
                details = runner()
                duration = time.time() - t_start
                logger.info("  Etapa %s COMPLETADA con éxito en %.2f s.", stage_name, duration)
                results.append(
                    StageResult(
                        name=stage_name,
                        status="COMPLETED",
                        duration_seconds=duration,
                        details=details if isinstance(details, dict) else {},
                    )
                )
            except Exception as exc:
                duration = time.time() - t_start
                logger.exception("  ERROR en etapa %s tras %.2f s: %s", stage_name, duration, exc)
                results.append(
                    StageResult(
                        name=stage_name,
                        status="FAILED",
                        duration_seconds=duration,
                        error=str(exc),
                    )
                )
                logger.error("\nPipeline interrumpido debido a falla en la etapa '%s'.", stage_name)
                break

        self.print_summary(results)
        return results

    def print_summary(self, results: List[StageResult]) -> None:
        """Imprime la tabla consolidada de telemetría y resultados en consola."""
        print("\n" + "=" * 68)
        print("          RESUMEN DE EJECUCIÓN DEL PIPELINE DE DATOS          ")
        print("=" * 68)
        print(f"{'ETAPA':<15} | {'ESTADO':<15} | {'DURACIÓN':<12} | {'DETALLES'}")
        print("-" * 68)

        total_time = 0.0
        for r in results:
            dur_str = f"{r.duration_seconds:.2f} s" if r.duration_seconds > 0 else "-"
            detail_str = ""
            if r.status == "COMPLETED":
                if r.name == "builder" and "total_instances" in r.details:
                    detail_str = f"{r.details['total_instances']} pares (1:1)"
                elif r.name == "mining" and "total_records" in r.details:
                    detail_str = f"{r.details['total_records']} contextos"
                elif r.name == "chunking" and "total_chunks" in r.details:
                    detail_str = f"{r.details['total_chunks']} chunks"
                elif r.name == "seminal" and "catalog_size" in r.details:
                    detail_str = f"{r.details['catalog_size']} papers seminales"
            elif r.status == "SKIPPED":
                detail_str = r.details.get("reason", "Artefacto presente")
            elif r.status == "FAILED":
                detail_str = f"Error: {r.error}"
            else:
                detail_str = r.details.get("reason", "")

            print(f"{r.name:<15} | {r.status:<15} | {dur_str:<12} | {detail_str}")
            total_time += r.duration_seconds

        print("-" * 68)
        print(f"Tiempo Total de Cómputo: {total_time:.2f} s")
        print("=" * 68 + "\n")


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Pipeline orquestador end-to-end de datos científicos (Actividad A1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=Path("configs/data_pipeline.yaml"), help="Ruta al archivo YAML de configuración")
    parser.add_argument("--all", action="store_true", help="Ejecutar todas las etapas del pipeline")
    parser.add_argument("--force", action="store_true", help="Fuerza la sobreescritura de artefactos existentes")
    parser.add_argument("--dry-run", action="store_true", help="Inspecciona el plan de ejecución y artefactos sin procesar datos")
    parser.add_argument("--from-stage", type=str, choices=STAGE_NAMES, default=None, help="Inicia la ejecución a partir de la etapa indicada")
    parser.add_argument("--stage", type=str, choices=STAGE_NAMES, default=None, help="Ejecuta únicamente la etapa especificada")
    parser.add_argument("--skip-seminal", action="store_true", help="Omite la resolución de Semantic Scholar")
    parser.add_argument("--seed", type=int, default=None, help="Sobrescribe la semilla pseudoaleatoria")
    return parser.parse_args()


def main() -> None:
    """Punto de entrada de línea de comandos."""
    args = parse_args()

    # Cargar configuración desde YAML o defaults
    config = PipelineConfig.from_yaml(args.config)
    if args.force:
        config.force = True
    if args.skip_seminal:
        config.skip_seminal = True
    if args.seed is not None:
        config.seed = args.seed

    pipeline = DataPipeline(config)
    results = pipeline.execute(
        from_stage=args.from_stage,
        single_stage=args.stage,
        dry_run=args.dry_run,
    )

    # Si alguna etapa falló, salir con código de error
    if any(r.status == "FAILED" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
