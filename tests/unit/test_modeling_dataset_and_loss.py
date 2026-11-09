"""Pruebas unitarias para formateo de instancias (Config A/B), CitationIntentDataset y funciones de pérdida."""

from pathlib import Path
from typing import List, Optional

import torch

from src.config.schema import CANONICAL_9_CLASSES, InputRepresentationConfig, LABEL_TO_ID
from src.data.schemas import (
    AnnotationMetadata,
    CitedPaperInfo,
    CitingPaperInfo,
    DatasetInstance,
    RetrievedChunk,
)
from src.modeling.dataset import (
    CitationIntentDataset,
    compute_split_sha256,
    format_instance_as_single_text,
    format_instance_parts,
    load_jsonl_instances,
)
from src.modeling.loss import FocalLoss, build_loss_fn, compute_class_weights


class DummyWordTokenizer:
    """Tokenizador determinista basado en palabras para pruebas unitarias offline."""

    pad_token = "[PAD]"
    eos_token = "[SEP]"
    unk_token = "[UNK]"
    pad_token_id = 0
    eos_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        tokens = [idx + 10 for idx, _ in enumerate(text.split())]
        if add_special_tokens:
            return [1] + tokens + [2]
        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        return " ".join(f"tok{tid}" for tid in token_ids)

    def save_pretrained(self, save_directory: str, **kwargs: object) -> tuple:
        return ()

    def pad(
        self,
        encoded_inputs: List[dict[str, torch.Tensor]],
        padding: bool | str = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        max_len = max(int(item["input_ids"].shape[0]) for item in encoded_inputs)
        if max_length is not None:
            max_len = max(max_len, max_length)
        batch_ids, batch_attn, batch_labels = [], [], []
        for item in encoded_inputs:
            ids = item["input_ids"]
            attn = item["attention_mask"]
            pad_len = max_len - int(ids.shape[0])
            if pad_len > 0:
                ids = torch.cat([ids, torch.zeros(pad_len, dtype=torch.long)])
                attn = torch.cat([attn, torch.zeros(pad_len, dtype=torch.long)])
            batch_ids.append(ids)
            batch_attn.append(attn)
            if "labels" in item:
                batch_labels.append(item["labels"])
        out: dict[str, torch.Tensor] = {
            "input_ids": torch.stack(batch_ids),
            "attention_mask": torch.stack(batch_attn),
        }
        if batch_labels:
            out["labels"] = torch.stack(batch_labels)
        return out

    def __call__(
        self,
        text: str | List[str],
        text_pair: Optional[str] = None,
        padding: bool | str = "max_length",
        truncation: bool | str = True,
        max_length: int = 512,
        return_tensors: str = "pt",
        add_special_tokens: bool = True,
    ) -> dict[str, torch.Tensor]:
        if isinstance(text, list):
            single_items = [
                self(
                    t,
                    padding=False,
                    truncation=truncation,
                    max_length=max_length,
                    return_tensors="pt",
                    add_special_tokens=add_special_tokens,
                )
                for t in text
            ]
            squeezed = [
                {
                    "input_ids": it["input_ids"].squeeze(0),
                    "attention_mask": it["attention_mask"].squeeze(0),
                }
                for it in single_items
            ]
            return self.pad(squeezed, padding=padding, max_length=max_length if padding == "max_length" else None)

        ids_a = self.encode(text, add_special_tokens=False)
        ids_b = self.encode(text_pair, add_special_tokens=False) if text_pair is not None else []

        if text_pair is not None:
            if truncation == "only_second" and (len(ids_a) + len(ids_b) + 3) > max_length:
                allowed_b = max(0, max_length - len(ids_a) - 3)
                ids_b = ids_b[:allowed_b]
            full_ids = [1] + ids_a + [2] + ids_b + [2]
        else:
            full_ids = [1] + ids_a + [2]

        if truncation and len(full_ids) > max_length:
            full_ids = full_ids[: max_length - 1] + [2]

        attn = [1] * len(full_ids)
        if padding == "max_length" and len(full_ids) < max_length:
            pad_len = max_length - len(full_ids)
            full_ids = full_ids + [0] * pad_len
            attn = attn + [0] * pad_len

        return {
            "input_ids": torch.tensor([full_ids], dtype=torch.long),
            "attention_mask": torch.tensor([attn], dtype=torch.long),
        }


def make_synthetic_instance(
    idx: int = 1,
    label: str = "Background",
    citing_section: Optional[str] = "Introduction",
    chunk_text_multiplier: int = 1,
) -> DatasetInstance:
    """Construye una instancia sintética válida con paridad 1:1 (Abstract + 3 Chunks)."""
    return DatasetInstance(
        instance_id=f"inst_{idx:05d}",
        label=label,
        annotation_metadata=AnnotationMetadata(
            source_dataset="multicite",
            original_label=label,
            mapping_type="exact",
            label_source="human_gold",
            is_gold=True,
            split="train",
        ),
        citing_paper=CitingPaperInfo(
            id=f"citing_{idx}",
            id_source="arxiv",
            title="Synthetic Citing Paper",
            rhetorical_section=citing_section,
        ),
        citation_context="Prior studies follow <cite>(Hill et al., 2016)</cite> to learn embeddings.",
        cited_paper=CitedPaperInfo(
            id=f"cited_{idx}",
            title="Learning Phrase Representations",
            abstract="We propose dictionary embeddings for phrase semantics.",
            retrieved_chunks=[
                RetrievedChunk(
                    chunk_id=f"cited_{idx}_s1_c1",
                    section_heading="1 Introduction",
                    text=(" ".join(["first_chunk_word"] * (5 * chunk_text_multiplier))) + ".",
                    similarity_score=0.92,
                    rank=1,
                ),
                RetrievedChunk(
                    chunk_id=f"cited_{idx}_s2_c1",
                    section_heading="2 Methods",
                    text=" ".join(["second_chunk_word"] * (5 * chunk_text_multiplier)),
                    similarity_score=0.89,
                    rank=2,
                ),
                RetrievedChunk(
                    chunk_id=f"cited_{idx}_s3_c1",
                    section_heading="3 Experiments",
                    text=" ".join(["third_chunk_marker"] * 4),
                    similarity_score=0.85,
                    rank=3,
                ),
            ],
        ),
    )


def test_format_instance_parts_citefusion_style_no_brackets() -> None:
    """Valida el estilo CiteFusion (': ' tras sección, '. ' entre chunks, preservación de Unknown Section)."""
    inst = make_synthetic_instance(citing_section="Unknown Section")

    # 1. Con include_citing_section=True preserva Unknown Section sin corchetes
    cfg_ctx_with_sec = InputRepresentationConfig(configuration="context_only", include_citing_section=True)
    citing, cited = format_instance_parts(inst, cfg_ctx_with_sec)
    assert citing == "Unknown Section: Prior studies follow <cite>(Hill et al., 2016)</cite> to learn embeddings."
    assert "[" not in citing and "]" not in citing
    assert cited is None

    # Por defecto include_citing_section=False entrega el contexto aislado sin prefijo de sección
    cfg_ctx_default = InputRepresentationConfig(configuration="context_only")
    citing_default, _ = format_instance_parts(inst, cfg_ctx_default)
    assert citing_default == "Prior studies follow <cite>(Hill et al., 2016)</cite> to learn embeddings."

    # 2. Si citing_section es vacía o None, se omite el prefijo ': ' incluso con include_citing_section=True
    inst_empty_sec = make_synthetic_instance(citing_section="")
    citing_no_sec, _ = format_instance_parts(inst_empty_sec, cfg_ctx_with_sec)
    assert citing_no_sec == "Prior studies follow <cite>(Hill et al., 2016)</cite> to learn embeddings."

    # 3. Configuración A: Título y Abstract separados por '. '
    cfg_a = InputRepresentationConfig(configuration="A", strip_cite_tags=True)
    citing_a, cited_a = format_instance_parts(inst, cfg_a)
    assert "<cite>" not in citing_a and "</cite>" not in citing_a
    assert "(Hill et al., 2016)" in citing_a
    assert cited_a == "Learning Phrase Representations. We propose dictionary embeddings for phrase semantics."

    # 4. Configuración B: Chunks con '{section_heading}: {text}' separados por '. ' sin doble punto
    cfg_b = InputRepresentationConfig(configuration="B", enforce_per_chunk_budget=False)
    _, cited_b = format_instance_parts(inst, cfg_b)
    assert cited_b is not None
    assert cited_b.startswith("1 Introduction: first_chunk_word")
    assert ".. " not in cited_b
    assert ". 2 Methods: second_chunk_word" in cited_b
    assert ". 3 Experiments: third_chunk_marker" in cited_b


def test_enforce_per_chunk_budget_preserves_third_chunk() -> None:
    """Verifica que enforce_per_chunk_budget=True recorte equitativamente cada chunk y preserve el Chunk 3."""
    long_inst = make_synthetic_instance(chunk_text_multiplier=20)
    tokenizer = DummyWordTokenizer()

    cfg_enforced = InputRepresentationConfig(
        configuration="B",
        max_seq_length=45,
        enforce_per_chunk_budget=True,
    )
    citing_e, cited_e = format_instance_parts(long_inst, cfg_enforced, tokenizer=tokenizer)
    assert cited_e is not None
    total_tokens = len(tokenizer.encode(citing_e)) + len(tokenizer.encode(cited_e)) + 3
    assert total_tokens <= 45
    assert cited_e.count(".") >= 2  # Los 3 chunks están presentes


def test_citation_intent_dataset_from_jsonl_and_memory(tmp_path: Path) -> None:
    """Valida CitationIntentDataset cargando desde JSONL en tmp_path y desde lista en memoria."""
    instances = [
        make_synthetic_instance(idx=i + 1, label=CANONICAL_9_CLASSES[i % 9])
        for i in range(9)
    ]
    jsonl_path = tmp_path / "synthetic_train.jsonl"
    jsonl_path.write_text(
        "\n".join(inst.model_dump_json() for inst in instances) + "\n",
        encoding="utf-8",
    )

    loaded = load_jsonl_instances(jsonl_path)
    assert len(loaded) == 9
    assert len(compute_split_sha256(jsonl_path)) == 64

    tokenizer = DummyWordTokenizer()
    ds = CitationIntentDataset(
        file_path=jsonl_path,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        configuration="B",
        max_length=64,
        dynamic_padding=False,
    )
    assert len(ds) == 9
    sample = ds[0]
    assert sample["input_ids"].shape == (64,)
    assert sample["attention_mask"].shape == (64,)
    assert int(sample["labels"].item()) == LABEL_TO_ID["Background"]

    single_text = format_instance_as_single_text(instances[0], InputRepresentationConfig(configuration="A"))
    assert "Learning Phrase Representations" in single_text


def test_class_weights_and_focal_loss() -> None:
    """Verifica el cálculo de pesos ('balanced', 'sqrt_balanced', 'none') y FocalLoss diferenciable."""
    labels = ["Background"] * 20 + ["Comparison"] * 5 + ["Originator"] * 1
    w_none = compute_class_weights(labels, scheme="none")
    assert w_none is None

    w_bal = compute_class_weights(labels, scheme="balanced")
    assert w_bal is not None and w_bal.shape == (9,)
    assert w_bal[LABEL_TO_ID["Originator"]] > w_bal[LABEL_TO_ID["Comparison"]] > w_bal[LABEL_TO_ID["Background"]]

    w_sqrt = compute_class_weights(labels, scheme="sqrt_balanced")
    assert w_sqrt is not None and w_sqrt.shape == (9,)
    # El suavizado sqrt produce razones menos extremas que balanced puro
    ratio_bal = float(w_bal[LABEL_TO_ID["Originator"]] / w_bal[LABEL_TO_ID["Background"]])
    ratio_sqrt = float(w_sqrt[LABEL_TO_ID["Originator"]] / w_sqrt[LABEL_TO_ID["Background"]])
    assert ratio_sqrt < ratio_bal

    logits = torch.randn(6, 9, requires_grad=True)
    targets = torch.tensor([0, 3, 7, 0, 1, 2], dtype=torch.long)

    focal_fn = build_loss_fn("focal", class_weights=w_sqrt, focal_gamma=2.0)
    assert isinstance(focal_fn, FocalLoss)
    loss_val = focal_fn(logits, targets)
    assert loss_val.ndim == 0 and loss_val.item() > 0.0
    loss_val.backward()
    assert logits.grad is not None

