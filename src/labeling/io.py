"""Operaciones de lectura y escritura particionada de instancias."""

import json
from pathlib import Path
from typing import Dict, List, Optional

from src.data.schemas import DatasetInstance

SPLITS = ("train", "dev", "test")


def load_instances(input_path: Path) -> List[DatasetInstance]:
    """Carga y valida todas las instancias de data/dataset/instances.jsonl contra DatasetInstance."""
    instances: List[DatasetInstance] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                instances.append(DatasetInstance.model_validate_json(line))
    return instances


def _read_split_ids(split_path: Path) -> List[str]:
    """Extrae la secuencia exacta de instance_id de un archivo JSONL de referencia."""
    ids: List[str] = []
    with open(split_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(json.loads(line)["instance_id"])
    return ids


def write_instances_by_split(
    instances: List[DatasetInstance],
    output_dir: Path,
    reference_dir: Optional[Path] = None,
) -> None:
    """Escribe instances.jsonl y genera train/dev/test preservando el orden exacto de reference_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "instances.jsonl", "w", encoding="utf-8") as f_all:
        for inst in instances:
            f_all.write(inst.model_dump_json() + "\n")

    by_split: Dict[str, Dict[str, DatasetInstance]] = {s: {} for s in SPLITS}
    for inst in instances:
        split = (inst.annotation_metadata.split or "").lower()
        if split in by_split:
            by_split[split][inst.instance_id] = inst

    for split in SPLITS:
        split_map = by_split[split]
        ref_file = (reference_dir / f"{split}.jsonl") if reference_dir is not None else None
        ordered_instances: List[DatasetInstance] = []

        if ref_file is not None and ref_file.exists():
            seen: set[str] = set()
            for iid in _read_split_ids(ref_file):
                if iid in split_map:
                    ordered_instances.append(split_map[iid])
                    seen.add(iid)
            for iid, inst in split_map.items():
                if iid not in seen:
                    ordered_instances.append(inst)
        else:
            ordered_instances = list(split_map.values())

        with open(output_dir / f"{split}.jsonl", "w", encoding="utf-8") as f_split:
            for inst in ordered_instances:
                f_split.write(inst.model_dump_json() + "\n")
