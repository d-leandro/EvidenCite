"""Orquestador principal del pre-etiquetado asistido por panel de jueces (Actividad A2).

Uso:
    python -m src.labeling.judge_panel --config configs/labeling.yaml --sample 60
"""

import argparse
from collections import Counter
import json
import logging
from pathlib import Path
import random
import sys
from typing import Dict, List, Optional

import yaml

from src.labeling.io import load_instances, write_instances_by_split
from src.labeling.judges import build_judges
from src.labeling.metrics import compute_labeling_metrics
from src.labeling.panel import aggregate_votes, apply_decision
from src.labeling.schemas import LabelingConfig, PanelVote

logger = logging.getLogger("src.labeling.judge_panel")


def setup_logging(log_path: Path) -> None:
    """Configura logging a consola y al archivo persistente data/logs/labeling.log (RNF-05)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )
    for noisy in ("httpx", "httpx2", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Define los argumentos de ejecución (--config, --limit, --sample, --judges, --force, --cache-only)."""
    parser = argparse.ArgumentParser(description="A2: Pre-etiquetado asistido por panel de jueces LLM.")
    parser.add_argument("--config", type=Path, default=Path("configs/labeling.yaml"), help="Ruta al archivo YAML de configuración.")
    parser.add_argument("--limit", type=int, default=None, help="Limita a procesar las primeras N instancias (ideal smoke tests).")
    parser.add_argument("--sample", type=int, default=None, help="Toma una muestra aleatoria de N instancias con la semilla de la configuración.")
    parser.add_argument("--judges", type=str, default=None, help="Lista de IDs de jueces separados por coma (ej. 'llama31_8b,qwen25_7b').")
    parser.add_argument("--force", action="store_true", help="Ignora cachés previas y fuerza re-inferencia completa.")
    parser.add_argument("--cache-only", action="store_true", help="Agrega solo con los votos en caché, sin consultar a los jueces.")
    args = parser.parse_args(argv)
    if args.force and args.cache_only:
        parser.error("--force y --cache-only no se pueden usar juntos")
    return args


def main(argv: Optional[List[str]] = None) -> None:
    """Carga configuración, ejecuta jueces secuencialmente, agrega decisiones y persiste resultados."""
    args = parse_args(argv)
    with open(args.config, "r", encoding="utf-8") as f:
        config = LabelingConfig.model_validate(yaml.safe_load(f))
    setup_logging(config.log_path)

    # 1. Cargar instancias exclusivamente desde data/dataset/instances.jsonl
    logger.info("Cargando instancias desde %s ...", config.input_path)
    instances = load_instances(config.input_path)
    if args.limit:
        logger.info("Modo smoke test activado: procesando únicamente las primeras %d instancias", args.limit)
        instances = instances[: args.limit]
    if args.sample and args.sample < len(instances):
        logger.info("Muestra aleatoria de %d instancias (seed=%d)", args.sample, config.seed)
        keep = set(random.Random(config.seed).sample(range(len(instances)), args.sample))
        instances = [inst for i, inst in enumerate(instances) if i in keep]

    # 2. Inicializar jueces y filtrar si el usuario especificó --judges
    judge_configs = config.judges
    if args.judges:
        selected_ids = {j.strip() for j in args.judges.split(",") if j.strip()}
        unknown = selected_ids - {j.id for j in judge_configs}
        if unknown:
            raise SystemExit(f"Jueces no definidos en la configuración: {sorted(unknown)}")
        judge_configs = [j for j in judge_configs if j.id in selected_ids]
    judges = build_judges(judge_configs, config.output_dir / "votes", force=args.force, cache_only=args.cache_only)

    # 3. Inferencia secuencial juez por juez (D14: resiliencia y protección de memoria local)
    votes_by_instance: Dict[str, List[PanelVote]] = {inst.instance_id: [] for inst in instances}
    for judge in judges:
        logger.info("Iniciando votación con juez [%s] (%s) ...", judge.config.id, judge.config.model_name)
        judge_votes = judge.vote_all(instances, progress_desc=judge.config.id)
        for inst in instances:
            vote = judge_votes.get(inst.instance_id)
            if vote is not None:
                votes_by_instance[inst.instance_id].append(vote)

    # 4. Agregación pura y mutación inmutable
    logger.info("Agregando decisiones con umbral min_agreement = %.2f ...", config.aggregation.min_agreement)
    updated_instances = []
    case_counts: Counter = Counter()
    for inst in instances:
        decision = aggregate_votes(
            votes_by_instance[inst.instance_id],
            inst.label,
            inst.annotation_metadata.is_gold,
            config.aggregation,
        )
        case_counts[decision.case] += 1
        updated_instances.append(apply_decision(inst, decision))
    logger.info("Casos de decisión: %s", dict(sorted(case_counts.items())))

    # 5. Cálculo y persistencia de métricas
    logger.info("Calculando métricas de acuerdo, transición y balance ...")
    metrics = compute_labeling_metrics(
        instances, updated_instances, votes_by_instance, config.aggregation.quota_per_class, dict(case_counts)
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = config.output_dir / "labeling_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info("Métricas guardadas en %s", metrics_path)

    # 6. Multiplexación en disco de los splits preservando el orden barajado de data/dataset/
    logger.info("Escribiendo datasets particionados en %s ...", config.output_dir)
    write_instances_by_split(
        updated_instances,
        config.output_dir,
        reference_dir=config.input_path.parent,
    )
    logger.info("Pipeline de etiquetado A2 finalizado.")


if __name__ == "__main__":
    main()
