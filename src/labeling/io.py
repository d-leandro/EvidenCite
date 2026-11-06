"""Operaciones de lectura y escritura particionada de instancias."""

from pathlib import Path
from typing import List

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


def write_instances_by_split(
    instances: List[DatasetInstance],
    output_dir: Path,
) -> None:
    """Multiplexa la escritura: escribe instances.jsonl y bifurca a train/dev/test en un único pase."""
    output_dir.mkdir(parents=True, exist_ok=True)
    f_all = open(output_dir / "instances.jsonl", "w", encoding="utf-8")
    split_files = {split: open(output_dir / f"{split}.jsonl", "w", encoding="utf-8") for split in SPLITS}
    try:
        for inst in instances:
            line = inst.model_dump_json() + "\n"
            f_all.write(line)
            split = (inst.annotation_metadata.split or "").lower()
            if split in split_files:
                split_files[split].write(line)
    finally:
        f_all.close()
        for f in split_files.values():
            f.close()
