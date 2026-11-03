"""Módulo de ingestión PyTorch y formateo unificado de entradas (Actividades A4–A7)."""

import hashlib
from pathlib import Path
import re
from typing import Dict, List, Literal, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from src.config.schema import (
     InputRepresentationConfig,
    LABEL_TO_ID,
    resolve_canonical_label,
)
from src.data.schemas import DatasetInstance


def load_jsonl_instances(file_path: Path) -> List[DatasetInstance]:
    """Carga y valida estrictamente cada línea de un archivo JSONL contra DatasetInstance."""
    instances: List[DatasetInstance] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                instances.append(DatasetInstance.model_validate_json(stripped))
    return instances


def compute_split_sha256(file_path: Path) -> str:
    """Calcula el hash SHA-256 de un archivo de partición para auditoría en MLflow."""
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def _join_with_period(parts: List[str]) -> str:
    """Une fragmentos de texto con '. ' evitando duplicar puntos finales."""
    cleaned = [p.strip() for p in parts if p and p.strip()]
    if not cleaned:
        return ""
    result = cleaned[0]
    for part in cleaned[1:]:
        if result.endswith((".", "!", "?")):
            result = f"{result} {part}"
        else:
            result = f"{result}. {part}"
    return result


def format_instance_parts(
    instance: DatasetInstance,
    rep_config: InputRepresentationConfig,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> Tuple[str, Optional[str]]:
    """Construye el par (text_citing, text_cited) según el Eje 1 de representación.

    Reglas de formato acordadas (estilo CiteFusion):
    - Sección retórica seguida de ': ' (ej. 'Introduction: Most of the presented works...').
    - Se preserva 'Unknown Section'; únicamente se omite el prefijo si la sección viene vacía o None.
    - Los fragmentos del citado (Título + Abstract en Config A, o los Top-k Chunks en Config B)
      se separan con '. ' sin duplicar puntos finales.
    """
    raw_context = (instance.citation_context or "").strip()
    if rep_config.strip_cite_tags:
        raw_context = re.sub(r"</?cite>", "", raw_context)

    citing_section = (instance.citing_paper.rhetorical_section or "").strip()
    if rep_config.include_citing_section and citing_section:
        text_citing = f"{citing_section}: {raw_context}"
    else:
        text_citing = raw_context

    if rep_config.configuration == "context_only":
        return text_citing, None

    if rep_config.configuration == "A":
        title = (instance.cited_paper.title or "").strip()
        abstract = (instance.cited_paper.abstract or "").strip()
        text_cited = _join_with_period([title, abstract])
        return text_citing, text_cited

    # Configuración B: Top-k chunks ordenados por rank
    sorted_chunks = sorted(
        instance.cited_paper.retrieved_chunks,
        key=lambda c: c.rank,
    )[: rep_config.top_k_chunks]

    formatted_chunks: List[str] = []
    for chunk in sorted_chunks:
        chunk_sec = (chunk.section_heading or "").strip()
        chunk_body = (chunk.text or "").strip()
        if not chunk_body:
            continue
        if rep_config.include_cited_section and chunk_sec:
            formatted_chunks.append(f"{chunk_sec}: {chunk_body}")
        else:
            formatted_chunks.append(chunk_body)

    if (
        rep_config.enforce_per_chunk_budget
        and tokenizer is not None
        and len(formatted_chunks) > 0
    ):
        citing_token_ids = tokenizer.encode(text_citing, add_special_tokens=False)
        # Reservamos 3 tokens especiales ([CLS], [SEP], [SEP]) + longitud del contexto citante
        reserved_tokens = len(citing_token_ids) + 3
        remaining_budget = max(
            len(formatted_chunks),
            rep_config.max_seq_length - reserved_tokens,
        )
        # Descontamos margen para los separadores '. ' entre chunks (~1 token cada uno)
        per_chunk_quota = max(
            1,
            (remaining_budget - max(0, len(formatted_chunks) - 1)) // len(formatted_chunks),
        )
        budgeted_chunks: List[str] = []
        for fc in formatted_chunks:
            fc_ids = tokenizer.encode(fc, add_special_tokens=False)
            if len(fc_ids) > per_chunk_quota:
                fc = tokenizer.decode(
                    fc_ids[:per_chunk_quota],
                    skip_special_tokens=True,
                ).strip()
            budgeted_chunks.append(fc)
        formatted_chunks = budgeted_chunks

    text_cited = _join_with_period(formatted_chunks)
    return text_citing, text_cited


def format_instance_as_single_text(
    instance: DatasetInstance,
    rep_config: InputRepresentationConfig,
    separator: str = "\n",
) -> str:
    """Formatea una instancia como una única cadena de texto (para TF-IDF o modelos de secuencia única)."""
    text_citing, text_cited = format_instance_parts(instance, rep_config, tokenizer=None)
    if not text_cited:
        return text_citing
    return f"{text_citing}{separator}{text_cited}"


class CitationIntentDataset(Dataset):
    """PyTorch Dataset tipado que procesa instancias de Capa 4 para Configuración A, B o context_only."""

    def __init__(
        self,
        file_path: Optional[Path] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        configuration: Literal["context_only", "A", "B"] = "A",
        max_length: int = 512,
        label_to_id: Optional[Dict[str, int]] = None,
        instances: Optional[List[DatasetInstance]] = None,
        rep_config: Optional[InputRepresentationConfig] = None,
        dynamic_padding: bool = False,
    ) -> None:
        if tokenizer is None:
            raise ValueError("Se requiere una instancia de tokenizer para CitationIntentDataset.")
        if file_path is None and instances is None:
            raise ValueError("Debe proporcionar 'file_path' o 'instances' a CitationIntentDataset.")

        self.tokenizer = tokenizer
        self.rep_config = rep_config or InputRepresentationConfig(
            configuration=configuration,
            max_seq_length=max_length,
        )
        self.configuration = self.rep_config.configuration
        self.max_length = self.rep_config.max_seq_length
        self.label_to_id = label_to_id if label_to_id is not None else dict(LABEL_TO_ID)
        self.dynamic_padding = dynamic_padding
        # Cada instancia se tokeniza una sola vez (lora.zip): las épocas siguientes y las longitudes que usa
        # group_by_length leen de aquí
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}

        if instances is not None:
            self.instances: List[DatasetInstance] = list(instances)
        else:
            assert file_path is not None
            self.instances = load_jsonl_instances(file_path)

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx in self._cache:
            return self._cache[idx]
        inst = self.instances[idx]
        text_citing, text_cited = format_instance_parts(
            inst,
            self.rep_config,
            tokenizer=self.tokenizer,
        )

        padding_strategy: bool | str = False if self.dynamic_padding else "max_length"

        if text_cited is not None:
            # Protegemos text_citing truncando solo el segundo segmento salvo que text_citing
            # por sí solo supere max_length - 2
            citing_len = len(self.tokenizer.encode(text_citing, add_special_tokens=False))
            trunc_strategy = (
                "only_second" if (citing_len + 3) < self.max_length else "longest_first"
            )
            encoding = self.tokenizer(
                text=text_citing,
                text_pair=text_cited,
                padding=padding_strategy,
                truncation=trunc_strategy,
                max_length=self.max_length,
                return_tensors="pt",
            )
        else:
            encoding = self.tokenizer(
                text=text_citing,
                padding=padding_strategy,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

        item: Dict[str, torch.Tensor] = {
            key: val.squeeze(0) for key, val in encoding.items()
        }

        canonical_lbl = resolve_canonical_label(inst.label)
        if canonical_lbl and canonical_lbl in self.label_to_id:
            item["labels"] = torch.tensor(self.label_to_id[canonical_lbl], dtype=torch.long)

        self._cache[idx] = item
        return item

