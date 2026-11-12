#!/usr/bin/env python3
"""Orquestador de inferencia para Actividad A5: LLMs Comerciales y de Frontera vía API (OpenAI / Gemini).

Ejecución como módulo:
  # Smoke test (audita payloads, guarda en caché y proyecta costos sin sobreescribir metrics_summary.json):
  python -m src.modeling.llm_api_inference --config configs/llm_api_gpt41_config_a.yaml --limit 5

  # Corrida completa (N=2.798, ejecuta todas las variantes en shot_variants, ej. Zero-Shot y Few-Shot):
  python -m src.modeling.llm_api_inference --config configs/llm_api_gpt41_config_a.yaml
"""

import argparse
import os
from pathlib import Path
import re
import sys
from typing import List, Sequence

# Asegurar que la raíz del repositorio esté en sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.schema import (
    LABEL_TO_ID,
    ExperimentConfig,
    load_experiment_config,
    resolve_canonical_label,
)
from src.data.schemas import DatasetInstance
from src.modeling.dataset import load_jsonl_instances
from src.modeling.evaluate import compute_a6_metrics
from src.modeling.predict import (
    LLMApiPredictor,
    PredictionRecord,
    load_dotenv_if_present,
    run_llm_api_experiment,
    sample_stratified_few_shot_examples,
)


def derive_variant_experiment_name(base_name: str, num_few_shot: int) -> str:
    """Deriva el experiment_name canónico para cada variante de shot_variants (zeroshot vs fewshot)."""
    shot_tag = "zeroshot" if num_few_shot == 0 else ("fewshot" if num_few_shot == 9 else f"{num_few_shot}shot")
    if f"_{shot_tag}" in base_name:
        return base_name
    match = re.search(r"(_config_[ab]|_context_only)$", base_name, flags=re.IGNORECASE)
    if match:
        prefix = base_name[: match.start()]
        suffix = match.group(1)
        return f"{prefix}_{shot_tag}{suffix}"
    return f"{base_name}_{shot_tag}"


def _print_smoke_audit(
    config: ExperimentConfig,
    records: Sequence[PredictionRecord],
    eval_instances: Sequence[DatasetInstance],
    total_dataset_size: int,
    few_shot_examples: Sequence[DatasetInstance],
) -> None:
    """Imprime el reporte de auditoría del smoke test y la proyección financiera/operativa para N=2.798."""
    assert config.llm_api is not None
    llm_cfg = config.llm_api
    n_eval = max(1, len(records))

    true_ids = [LABEL_TO_ID[resolve_canonical_label(inst.label)] for inst in eval_instances]  # type: ignore[index]
    pred_ids = [LABEL_TO_ID[r.predicted_label] for r in records]
    metrics_a6 = compute_a6_metrics(true_ids, pred_ids)

    total_in = sum(r.input_tokens for r in records)
    total_out = sum(r.output_tokens + r.thoughts_tokens for r in records)
    total_thoughts = sum(r.thoughts_tokens for r in records)
    mean_in = total_in / n_eval
    mean_out = total_out / n_eval
    mean_lat_ms = sum(r.latency_ms for r in records) / n_eval
    invalid_count = sum(1 for r in records if not r.is_valid_format)

    smoke_cost_usd = (
        (total_in / 1_000_000.0) * llm_cfg.cost_per_1m_input_tokens_usd
        + (total_out / 1_000_000.0) * llm_cfg.cost_per_1m_output_tokens_usd
    )
    proj_in_tokens = mean_in * total_dataset_size
    proj_out_tokens = mean_out * total_dataset_size
    proj_cost_usd = (
        (proj_in_tokens / 1_000_000.0) * llm_cfg.cost_per_1m_input_tokens_usd
        + (proj_out_tokens / 1_000_000.0) * llm_cfg.cost_per_1m_output_tokens_usd
    )
    proj_minutes = (mean_lat_ms * total_dataset_size) / 1000.0 / 60.0

    print("\n" + "=" * 88)
    print(
        f"SMOKE TEST AUDIT · {config.experiment_name} "
        f"(provider={llm_cfg.provider}, model={llm_cfg.model_name}, "
        f"config={config.input_representation.configuration}, num_few_shot={llm_cfg.num_few_shot})"
    )
    print("=" * 88)
    if few_shot_examples:
        sampled_labels = [resolve_canonical_label(ex.label) for ex in few_shot_examples]
        print(f"• Ejemplares Few-Shot ({len(few_shot_examples)} desde train.jsonl): {sampled_labels}")

    print("\nDetalle por instancia evaluada:")
    for idx, (inst, rec) in enumerate(zip(eval_instances, records), start=1):
        gold_lbl = resolve_canonical_label(inst.label)
        match_mark = "OK" if rec.predicted_label == gold_lbl else "DIFF"
        print(
            f"  [{idx:02d}] id={rec.instance_id} | gold={gold_lbl:<15} | "
            f"pred={rec.predicted_label:<15} ({match_mark}) | conf={rec.confidence:.2f} | "
            f"valid_json={rec.is_valid_format} | tok(in={rec.input_tokens}, out={rec.output_tokens}, "
            f"think={rec.thoughts_tokens}) | lat={rec.latency_ms:.1f} ms"
        )

    print("\nResumen Técnico y Operativo de la Muestra:")
    print(f"  • Instancias evaluadas       : {len(records)} / {total_dataset_size}")
    print(f"  • Macro-F1 (muestra)         : {metrics_a6.macro_f1_present:.4f} (Accuracy: {metrics_a6.micro_f1:.4f})")
    print(f"  • Tasa de formato inválido   : {invalid_count}/{len(records)} ({invalid_count / n_eval:.2%})")
    print(f"  • Thinking tokens emitidos   : {total_thoughts} (thinking_budget={llm_cfg.thinking_budget})")
    print(f"  • Promedio tokens / instancia: {mean_in:.1f} in | {mean_out:.1f} out")
    print(f"  • Latencia media / instancia : {mean_lat_ms:.1f} ms")
    print(f"  • Costo real del smoke test  : ${smoke_cost_usd:.5f} USD")
    print(f"  • Caché persistente          : {llm_cfg.cache_path}")

    print(f"\nProyección para Corrida Completa (N = {total_dataset_size} instancias):")
    print(f"  • Tokens proyectados         : {proj_in_tokens:,.0f} in | {proj_out_tokens:,.0f} out")
    print(f"  • Costo bruto proyectado     : ${proj_cost_usd:.2f} USD (sin descuento de Prompt Caching)")
    if llm_cfg.num_few_shot > 0:
        # En Few-Shot el prefijo estático representa ~80% de los input tokens y recibe 75%-90% de descuento
        discount_factor = 0.75 if llm_cfg.provider == "openai" else 0.90
        cached_in_cost = (
            (proj_in_tokens * 0.80 / 1_000_000.0) * llm_cfg.cost_per_1m_input_tokens_usd * (1.0 - discount_factor)
            + (proj_in_tokens * 0.20 / 1_000_000.0) * llm_cfg.cost_per_1m_input_tokens_usd
        )
        out_cost = (proj_out_tokens / 1_000_000.0) * llm_cfg.cost_per_1m_output_tokens_usd
        print(
            f"  • Costo est. con Prompt Cache: ${cached_in_cost + out_cost:.2f} USD "
            f"(~{int(discount_factor * 100)}% desc. en prefijo Few-Shot)"
        )
    print(f"  • Tiempo secuencial estimado : ~{proj_minutes:.1f} min")
    print("=" * 88 + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orquestador de inferencia Zero-Shot y Few-Shot con LLMs comerciales vía API (Actividad A5)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Ruta al archivo YAML de configuración (ej. configs/llm_api_gpt41_config_a.yaml).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Si se especifica (modo Smoke Test), evalúa únicamente las primeras N instancias "
            "de test.jsonl, guarda en caché y proyecta costos sin sobreescribir metrics_summary.json."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv_if_present()

    base_config = load_experiment_config(args.config)
    if base_config.model_family != "llm_api" or base_config.llm_api is None:
        raise ValueError(
            f"El archivo {args.config} debe tener model_family='llm_api' y bloque llm_api definido."
        )

    llm_cfg = base_config.llm_api
    if llm_cfg.provider in ("openai", "gemini"):
        api_key = os.getenv(llm_cfg.api_key_env, "").strip()
        if not api_key:
            print(
                f"ERROR: La variable de entorno '{llm_cfg.api_key_env}' no está definida o está vacía.\n"
                f"Por favor agrégala en tu archivo .env o expórtala en la terminal antes de ejecutar.",
                file=sys.stderr,
            )
            return 1

    all_test_instances = load_jsonl_instances(base_config.paths.test_file)
    total_dataset_size = len(all_test_instances)
    is_smoke = args.limit is not None and args.limit > 0
    eval_instances = all_test_instances[: args.limit] if is_smoke else all_test_instances

    shot_variants: List[int] = (
        list(llm_cfg.shot_variants)
        if llm_cfg.shot_variants
        else [llm_cfg.num_few_shot]
    )

    for num_shots in shot_variants:
        variant_exp_name = (
            derive_variant_experiment_name(base_config.experiment_name, num_shots)
            if llm_cfg.shot_variants
            else base_config.experiment_name
        )
        variant_llm_cfg = llm_cfg.model_copy(update={"num_few_shot": num_shots})
        variant_config = base_config.model_copy(
            update={
                "experiment_name": variant_exp_name,
                "llm_api": variant_llm_cfg,
            }
        )

        few_shot_examples: List[DatasetInstance] = []
        if num_shots > 0:
            few_shot_examples = sample_stratified_few_shot_examples(
                train_file=variant_config.paths.train_file,
                num_few_shot=num_shots,
                seed=variant_config.seed,
            )

        if is_smoke:
            predictor = LLMApiPredictor(
                llm_config=variant_llm_cfg,
                rep_config=variant_config.input_representation,
            )
            records = predictor.predict_batch(
                eval_instances,
                few_shot_examples=few_shot_examples or None,
            )
            _print_smoke_audit(
                config=variant_config,
                records=records,
                eval_instances=eval_instances,
                total_dataset_size=total_dataset_size,
                few_shot_examples=few_shot_examples,
            )
        else:
            print(
                f"\n>>> Iniciando corrida completa A5: {variant_config.experiment_name} "
                f"(N={len(eval_instances)}, num_few_shot={num_shots}) ..."
            )
            records, metrics_a6, metrics_a7 = run_llm_api_experiment(
                config=variant_config,
                eval_instances=eval_instances,
                few_shot_examples=few_shot_examples or None,
                enable_mlflow=True,
            )
            exp_dir = Path(variant_config.paths.output_dir) / variant_config.experiment_name
            print(
                f"✓ Finalizado {variant_config.experiment_name}: "
                f"Macro-F1(9)={metrics_a6.macro_f1_all_9:.4f} | "
                f"Acc={metrics_a6.micro_f1:.4f} | "
                f"Latencia={metrics_a7.inference_latency_ms_per_instance:.1f} ms | "
                f"Costo=${metrics_a7.estimated_api_cost_usd:.2f} USD | "
                f"Resumen en {exp_dir / 'metrics_summary.json'}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

