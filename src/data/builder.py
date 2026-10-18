"""Módulo Ensamblador y Particionador del Dataset Consolidado (Capa 4 / Actividad A1).

Transforma los contextos citantes procesados (data/processed/citing_contexts.jsonl)
y los fragmentos discursivos estructurados (data/processed/cited_chunks.jsonl) en
el dataset final de modelado (data/dataset/instances.jsonl).

Garantiza la paridad estricta 1:1 entre la Configuración A (Abstract global) y la
Configuración B (Top-3 Chunks recuperados con SciBERTRetriever, ADR-07). Aplica mapeo canónico
a la taxonomía de 9 clases (ADR-08) y realiza una división estratificada anti-fuga
(Group-Stratified Split por citing_paper_id, ADR-09) en proporciones 70% Train / 15% Dev / 15% Test.

Especificación Técnica: documentacion/diseno-tecnico/08-diseno-dataset-builder.md
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Union

# Garantizar que la raíz del proyecto esté en sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.data.retrieval import RetrievalConfig, SciBERTRetriever
from src.data.schemas import (
    AnnotationMetadata,
    CitedPaperInfo,
    CitingPaperInfo,
    DatasetInstance,
    RetrievedChunk,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

CANONICAL_9_CLASSES = [
    "Background",
    "Uses",
    "Comparison",
    "Extends",
    "Motivation",
    "Future Work",
    "Evidence",
    "Originator",
    "Further Reading",
]


@dataclass
class BuilderConfig:
    """Configuración del proceso de ensamblado y particionamiento del dataset."""

    input_contexts_path: Path = Path("data/processed/citing_contexts.jsonl")
    papers_path: Path = Path("data/structured/papers.jsonl")
    chunks_path: Path = Path("data/processed/cited_chunks.jsonl")
    retrieval_config_path: Path = Path("configs/retrieval.yaml")
    output_dir: Path = Path("data/dataset")

    quota_per_class: Optional[int] = None
    k_chunks: int = 3
    train_ratio: float = 0.70
    dev_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    force: bool = False

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "BuilderConfig":
        """Carga la configuración desde un archivo YAML externo."""
        path = Path(yaml_path)
        if not path.exists():
            logger.warning("Archivo de configuración no encontrado: %s. Usando defaults.", path)
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        ratios = data.get("split_ratios", {})
        return cls(
            input_contexts_path=Path(data.get("input_contexts_path", cls.input_contexts_path)),
            papers_path=Path(data.get("papers_path", cls.papers_path)),
            chunks_path=Path(data.get("chunks_path", cls.chunks_path)),
            retrieval_config_path=Path(data.get("retrieval_config_path", cls.retrieval_config_path)),
            output_dir=Path(data.get("output_dir", cls.output_dir)),
            quota_per_class=data.get("quota_per_class", cls.quota_per_class),
            k_chunks=int(data.get("k_chunks", cls.k_chunks)),
            train_ratio=float(ratios.get("train", cls.train_ratio)),
            dev_ratio=float(ratios.get("dev", cls.dev_ratio)),
            test_ratio=float(ratios.get("test", cls.test_ratio)),
            seed=int(data.get("seed", cls.seed)),
            force=bool(data.get("force", cls.force)),
        )


def map_to_canonical_label(raw_label: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Mapea una etiqueta origen hacia la taxonomía de 9 clases (RF-04 / ADR-08).

    Maneja clases individuales y multi-etiquetas separadas por punto y coma (MultiCite).
    Retorna una tupla: (canonical_label, mapping_type).
    """
    if not raw_label:
        return None, None

    raw_clean = raw_label.strip()
    is_compound = ";" in raw_clean
    primary = raw_clean.split(";")[0].strip()

    mapping_table = {
        "Background": "Background",
        "Uses": "Uses",
        "Differences": "Comparison",
        "Similarities": "Comparison",
        "Comparison": "Comparison",
        "Extends": "Extends",
        "Basis": "Extends",
        "Motivation": "Motivation",
        "Future Work": "Future Work",
        "Evidence": "Evidence",
        "Originator": "Originator",
        "Further Reading": "Further Reading",
    }

    canonical = mapping_table.get(primary)
    if not canonical:
        return None, None

    mapping_type = "partial" if is_compound else "exact"
    return canonical, mapping_type


def load_missing_paper_abstracts(
    papers_path: Path,
    needed_paper_ids: Set[str],
) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """Carga en memoria títulos y abstracts de los papers citados requeridos (RF-03).

    Garantiza el rescate de abstracts para citas de unarXive donde mining.py
    no los haya pre-propagado, asegurando 100% de paridad con Configuración A.
    Retorna: Dict[paper_id, (title, abstract)]
    """
    if not needed_paper_ids or not papers_path.exists():
        return {}

    logger.info(
        "Cargando títulos/abstracts de rescate para %d artículos desde %s ...",
        len(needed_paper_ids),
        papers_path,
    )
    resolved: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    found_count = 0

    with open(papers_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data = json.loads(line_str)
                pid = data.get("paper_id")
                if pid in needed_paper_ids:
                    title = data.get("title")
                    abstract = data.get("abstract")
                    resolved[pid] = (title, abstract)
                    found_count += 1
                    if found_count >= len(needed_paper_ids):
                        break
            except json.JSONDecodeError:
                continue

    logger.info("Resueltos %d títulos/abstracts de rescate exitosamente.", len(resolved))
    return resolved


def group_stratified_split(
    instances: List[DatasetInstance],
    train_ratio: float = 0.70,
    dev_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[DatasetInstance]]:
    """Divide instancias en Train/Dev/Test agrupando estrictamente por citing_paper_id (RF-06).

    Garantiza que Papers(Train) ∩ Papers(Test) = ∅, eliminando la fuga de datos (data leakage),
    mientras aproxima las proporciones objetivo para cada una de las 9 clases canónicas.
    """
    rng = random.Random(seed)

    # 1. Agrupar instancias por citing_paper_id
    paper_groups: Dict[str, List[DatasetInstance]] = defaultdict(list)
    for inst in instances:
        paper_groups[inst.citing_paper.id].append(inst)

    # 2. Calcular cuotas objetivo globales
    n_total = len(instances)
    target_counts = {
        "train": int(round(n_total * train_ratio)),
        "dev": int(round(n_total * dev_ratio)),
        "test": n_total - int(round(n_total * train_ratio)) - int(round(n_total * dev_ratio)),
    }

    # Distribución objetivo por clase
    class_totals: Counter = Counter(inst.label for inst in instances if inst.label)
    target_class_counts = {
        split: {c: max(1, int(round(cnt * ratio))) for c, cnt in class_totals.items()}
        for split, ratio in [("train", train_ratio), ("dev", dev_ratio), ("test", test_ratio)]
    }

    # 3. Asignación voraz ordenada (Greedy Allocation)
    splits: Dict[str, List[DatasetInstance]] = {"train": [], "dev": [], "test": []}
    current_counts = {"train": 0, "dev": 0, "test": 0}
    current_class_counts: Dict[str, Counter] = {"train": Counter(), "dev": Counter(), "test": Counter()}

    # Ordenar grupos por tamaño decreciente y desempatar con aleatoriedad reproducible
    group_items = list(paper_groups.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda x: len(x[1]), reverse=True)

    for paper_id, group_instances in group_items:
        group_size = len(group_instances)
        group_class_dist = Counter(inst.label for inst in group_instances if inst.label)

        best_split = "train"
        best_score = float("inf")

        for split_name in ("test", "dev", "train"):  # Evaluar test y dev primero para no desabastecerlos
            # Capacidad restante
            capacity = target_counts[split_name] - current_counts[split_name]
            if capacity <= 0 and split_name != "train":
                continue

            # Penalización por desviación en cuota de instancias y balance de clases
            size_error = abs((current_counts[split_name] + group_size) - target_counts[split_name])
            class_error = sum(
                abs((current_class_counts[split_name][c] + group_class_dist[c]) - target_class_counts[split_name].get(c, 0))
                for c in group_class_dist
            )

            score = size_error + 2.0 * class_error
            if score < best_score:
                best_score = score
                best_split = split_name

        # Asignar grupo a la mejor partición
        for inst in group_instances:
            inst.annotation_metadata.split = best_split
            splits[best_split].append(inst)

        current_counts[best_split] += group_size
        for c, cnt in group_class_dist.items():
            current_class_counts[best_split][c] += cnt

    logger.info(
        "Particionamiento completado: Train=%d (%.1f%%) | Dev=%d (%.1f%%) | Test=%d (%.1f%%)",
        len(splits["train"]),
        len(splits["train"]) / n_total * 100 if n_total else 0,
        len(splits["dev"]),
        len(splits["dev"]) / n_total * 100 if n_total else 0,
        len(splits["test"]),
        len(splits["test"]) / n_total * 100 if n_total else 0,
    )

    return splits


class DatasetBuilder:
    """Orquestador de consolidación, recuperación densa y ensamblado de la Capa 4."""

    def __init__(
        self,
        config: Optional[BuilderConfig] = None,
        retriever: Optional[SciBERTRetriever] = None,
    ) -> None:
        """Inicializa el ensamblador y prepara el motor de recuperación semántica."""
        self.config = config or BuilderConfig()
        random.seed(self.config.seed)

        if retriever is not None:
            self.retriever = retriever
        else:
            logger.info("Inicializando SciBERTRetriever con config: %s", self.config.retrieval_config_path)
            retrieval_conf = RetrievalConfig.from_yaml(self.config.retrieval_config_path)
            self.retriever = SciBERTRetriever(config=retrieval_conf)

    def load_candidates(self) -> Tuple[List[Dict[str, Any]], Set[str]]:
        """Lee data/processed/citing_contexts.jsonl y filtra instancias con paridad 1:1 inicial.

        Retorna: (candidatos_validos, set_de_target_cited_ids_sin_abstract)
        """
        contexts_path = self.config.input_contexts_path
        if not contexts_path.exists():
            raise FileNotFoundError(f"Archivo de contextos no encontrado: {contexts_path}")

        candidates: List[Dict[str, Any]] = []
        missing_abstract_pids: Set[str] = set()

        with open(contexts_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    data = json.loads(line_str)
                except json.JSONDecodeError:
                    continue

                # Filtro de Paridad Estricta (RF-01)
                if not data.get("has_cited_chunks"):
                    continue
                target_cited_id = data.get("target_cited_id")
                if not target_cited_id:
                    continue

                if not data.get("target_cited_abstract"):
                    missing_abstract_pids.add(str(target_cited_id))

                candidates.append(data)

        logger.info(
            "Candidatos iniciales con has_cited_chunks=True: %d (con abstract ausente en registro: %d)",
            len(candidates),
            len(missing_abstract_pids),
        )
        return candidates, missing_abstract_pids

    def apply_quota_and_balance(
        self,
        candidates: List[Dict[str, Any]],
        quota_per_class: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Aplica balanceo y límite de cuota por clase canónica, priorizando is_gold=True (RF-05)."""
        if not quota_per_class:
            return candidates

        rng = random.Random(self.config.seed)
        by_class_gold: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        by_class_mined: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for c in candidates:
            raw_label = c.get("normalized_label") or c.get("original_label")
            canon_label, _ = map_to_canonical_label(raw_label)
            label_key = canon_label or "Uncategorized"

            if c.get("is_gold"):
                by_class_gold[label_key].append(c)
            else:
                by_class_mined[label_key].append(c)

        balanced: List[Dict[str, Any]] = []
        for label_key in set(list(by_class_gold.keys()) + list(by_class_mined.keys())):
            gold_pool = by_class_gold[label_key]
            mined_pool = by_class_mined[label_key]

            rng.shuffle(gold_pool)
            rng.shuffle(mined_pool)

            selected_class: List[Dict[str, Any]] = []
            # Priorizar Gold
            if len(gold_pool) >= quota_per_class:
                selected_class.extend(gold_pool[:quota_per_class])
            else:
                selected_class.extend(gold_pool)
                needed = quota_per_class - len(selected_class)
                selected_class.extend(mined_pool[:needed])

            balanced.extend(selected_class)
            logger.info(
                "Clase '%s': seleccionadas %d (Gold: %d, Mined: %d)",
                label_key,
                len(selected_class),
                sum(1 for x in selected_class if x.get("is_gold")),
                sum(1 for x in selected_class if not x.get("is_gold")),
            )

        return balanced

    def build_dataset(self) -> Dict[str, Any]:
        """Ejecuta el pipeline completo de ensamble, recuperación densa y particionado (A1)."""
        start_time = time.time()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Cargar candidatos con has_cited_chunks = True
        candidates, missing_abstract_pids = self.load_candidates()

        # 2. Resolver abstracts de rescate si aplica (RF-03)
        abstracts_cache: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
        if missing_abstract_pids:
            abstracts_cache = load_missing_paper_abstracts(self.config.papers_path, missing_abstract_pids)

        # 3. Aplicar cuota y balance si está configurada (RF-05)
        selected_candidates = self.apply_quota_and_balance(candidates, self.config.quota_per_class)

        # 4. Ensamble de instancias con recuperación densa de Top-k chunks
        logger.info("Ensamblando %d instancias y recuperando Top-%d chunks con SciBERT...", len(selected_candidates), self.config.k_chunks)
        instances: List[DatasetInstance] = []
        discarded_no_chunks = 0
        discarded_no_abstract = 0

        for idx, item in enumerate(selected_candidates, start=1):
            target_cited_id = str(item["target_cited_id"])
            citation_context = str(item["citation_context"])

            # Abstract y Título del citado (Configuración A)
            cited_title = item.get("target_cited_title")
            cited_abstract = item.get("target_cited_abstract")

            if not cited_abstract and target_cited_id in abstracts_cache:
                rescued_title, rescued_abstract = abstracts_cache[target_cited_id]
                cited_title = cited_title or rescued_title
                cited_abstract = rescued_abstract

            if not cited_abstract:
                discarded_no_abstract += 1
                continue

            # Chunks recuperados (Configuración B)
            retrieved_chunks = self.retriever.retrieve_top_k(
                citation_context=citation_context,
                target_cited_id=target_cited_id,
                k=self.config.k_chunks,
            )

            if not retrieved_chunks:
                discarded_no_chunks += 1
                continue

            # Mapeo canónico de etiqueta (RF-04 / ADR-08)
            raw_label = item.get("normalized_label") or item.get("original_label")
            canon_label, mapping_type = map_to_canonical_label(raw_label)

            annotation_metadata = AnnotationMetadata(
                source_dataset=item.get("source_dataset", "unknown"),
                original_label=item.get("original_label") or raw_label,
                mapping_type=mapping_type,
                label_source=item.get("label_source", "human_gold"),
                is_gold=bool(item.get("is_gold", True)),
                panel_agreement=None,
                split=None,  # Asignado en la siguiente etapa
            )

            citing_info = CitingPaperInfo(
                id=str(item.get("citing_paper_id", "")),
                id_source="arxiv" if not str(item.get("citing_paper_id", "")).startswith("Upwork") else "semantic_scholar",
                title=item.get("citing_paper_title"),
                rhetorical_section=item.get("rhetorical_section"),
            )

            cited_info = CitedPaperInfo(
                id=target_cited_id,
                title=cited_title,
                abstract=cited_abstract,
                retrieved_chunks=retrieved_chunks,
            )

            instance = DatasetInstance(
                instance_id=f"inst_{idx:05d}",
                label=canon_label,
                annotation_metadata=annotation_metadata,
                citing_paper=citing_info,
                citation_context=citation_context,
                cited_paper=cited_info,
            )
            instances.append(instance)

        logger.info(
            "Instancias pares 1:1 construidas con éxito: %d | Descartadas por falta de chunks: %d | Descartadas por falta de abstract: %d",
            len(instances),
            discarded_no_chunks,
            discarded_no_abstract,
        )

        # 5. Particionamiento Group-Stratified (RF-06)
        splits = group_stratified_split(
            instances=instances,
            train_ratio=self.config.train_ratio,
            dev_ratio=self.config.dev_ratio,
            test_ratio=self.config.test_ratio,
            seed=self.config.seed,
        )

        # 6. Serialización de archivos en data/dataset/ (RF-08)
        instances_path = self.config.output_dir / "instances.jsonl"
        train_path = self.config.output_dir / "train.jsonl"
        dev_path = self.config.output_dir / "dev.jsonl"
        test_path = self.config.output_dir / "test.jsonl"
        metrics_path = self.config.output_dir / "dataset_metrics.json"

        logger.info("Escribiendo %s ...", instances_path)
        with open(instances_path, "w", encoding="utf-8") as f_all:
            for inst in instances:
                f_all.write(inst.model_dump_json() + "\n")

        for split_name, split_path in [("train", train_path), ("dev", dev_path), ("test", test_path)]:
            logger.info("Escribiendo partición %s (%d instancias) en %s ...", split_name, len(splits[split_name]), split_path)
            with open(split_path, "w", encoding="utf-8") as f_split:
                for inst in splits[split_name]:
                    f_split.write(inst.model_dump_json() + "\n")

        # 7. Cálculo y exportación de métricas de auditoría
        metrics = self._compute_metrics(instances, splits, elapsed_sec=time.time() - start_time)
        with open(metrics_path, "w", encoding="utf-8") as f_m:
            json.dump(metrics, f_m, indent=2, ensure_ascii=False)

        logger.info("Dataset consolidado exitosamente en %s (%.2fs)", self.config.output_dir, metrics["elapsed_seconds"])
        return metrics

    def _compute_metrics(
        self,
        instances: List[DatasetInstance],
        splits: Dict[str, List[DatasetInstance]],
        elapsed_sec: float,
    ) -> Dict[str, Any]:
        """Calcula estadísticas y métricas de auditoría del dataset ensamblado."""
        total = len(instances)
        classes_dist = Counter(inst.label for inst in instances if inst.label)
        sources_dist = Counter(inst.annotation_metadata.source_dataset for inst in instances)
        gold_count = sum(1 for inst in instances if inst.annotation_metadata.is_gold)

        context_word_lens = [len(inst.citation_context.split()) for inst in instances]
        top1_sim_scores = [inst.cited_paper.retrieved_chunks[0].similarity_score for inst in instances if inst.cited_paper.retrieved_chunks]

        split_stats = {}
        for s_name, s_list in splits.items():
            split_stats[s_name] = {
                "count": len(s_list),
                "proportion": round(len(s_list) / total, 4) if total else 0,
                "classes": dict(Counter(inst.label for inst in s_list if inst.label)),
                "unique_citing_papers": len(set(inst.citing_paper.id for inst in s_list)),
            }

        # Verificación de no-intersección de papers citantes
        train_papers = set(inst.citing_paper.id for inst in splits["train"])
        dev_papers = set(inst.citing_paper.id for inst in splits["dev"])
        test_papers = set(inst.citing_paper.id for inst in splits["test"])
        leakage_train_test = len(train_papers.intersection(test_papers))
        leakage_train_dev = len(train_papers.intersection(dev_papers))
        leakage_dev_test = len(dev_papers.intersection(test_papers))

        return {
            "total_instances": total,
            "elapsed_seconds": round(elapsed_sec, 2),
            "parity_verification": {
                "instances_with_abstract_config_a": sum(1 for i in instances if i.cited_paper.abstract),
                "instances_with_chunks_config_b": sum(1 for i in instances if len(i.cited_paper.retrieved_chunks) >= 1),
                "strict_parity_1_to_1": True if total and all(i.cited_paper.abstract and i.cited_paper.retrieved_chunks for i in instances) else False,
            },
            "leakage_verification": {
                "leakage_train_test": leakage_train_test,
                "leakage_train_dev": leakage_train_dev,
                "leakage_dev_test": leakage_dev_test,
                "zero_leakage_guaranteed": (leakage_train_test == 0 and leakage_train_dev == 0 and leakage_dev_test == 0),
            },
            "gold_instances": gold_count,
            "mined_instances": total - gold_count,
            "sources_distribution": dict(sources_dist),
            "canonical_classes_distribution": dict(classes_dist),
            "splits": split_stats,
            "citation_context_stats": {
                "mean_words": round(sum(context_word_lens) / total, 2) if total else 0,
                "min_words": min(context_word_lens) if context_word_lens else 0,
                "max_words": max(context_word_lens) if context_word_lens else 0,
            },
            "similarity_stats": {
                "mean_top1_similarity": round(sum(top1_sim_scores) / len(top1_sim_scores), 4) if top1_sim_scores else 0,
            },
        }


def parse_arguments() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Módulo de ensamblado y particionamiento del dataset consolidado (Capa 4 / A1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/builder.yaml", help="Ruta al archivo YAML de configuración")
    parser.add_argument("--quota", type=int, default=None, help="Cuota máxima de instancias por clase canónica")
    parser.add_argument("--k-chunks", type=int, default=None, help="Número de chunks a recuperar por instancia (Top-k)")
    parser.add_argument("--seed", type=int, default=None, help="Semilla pseudoaleatoria determinista")
    parser.add_argument("--force", action="store_true", help="Fuerza la regeneración de archivos en data/dataset/")
    return parser.parse_args()


def main() -> None:
    """Punto de entrada principal para CLI."""
    args = parse_arguments()

    config = BuilderConfig.from_yaml(args.config)
    if args.quota is not None:
        config.quota_per_class = args.quota
    if args.k_chunks is not None:
        config.k_chunks = args.k_chunks
    if args.seed is not None:
        config.seed = args.seed
    if args.force:
        config.force = args.force

    builder = DatasetBuilder(config=config)
    metrics = builder.build_dataset()

    print("\n=======================================================")
    print("        REPORTE DE ENSAMBLE DE DATASET (CAPA 4)        ")
    print("=======================================================")
    print(f"Total Instancias Pares 1:1: {metrics['total_instances']}")
    print(f"Paridad Estricta Garantizada: {metrics['parity_verification']['strict_parity_1_to_1']}")
    print(f"Cero Fuga de Datos Garantizada: {metrics['leakage_verification']['zero_leakage_guaranteed']}")
    print(f"Instancias Gold (MultiCite): {metrics['gold_instances']}")
    print(f"Instancias Minadas (unarXive): {metrics['mined_instances']}")
    print("\nDistribución por Splits:")
    for s_name, s_data in metrics["splits"].items():
        print(f"  - {s_name.upper():5s}: {s_data['count']:5d} ({s_data['proportion']*100:.1f}%) | Papers citantes únicos: {s_data['unique_citing_papers']}")
    print("\nDistribución por Clases Canónicas:")
    for c_name, c_cnt in metrics["canonical_classes_distribution"].items():
        print(f"  - {c_name:18s}: {c_cnt:5d}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
