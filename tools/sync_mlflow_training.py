#!/usr/bin/env python3
"""Exporta dos experimentos de MLflow a un snapshot portatil para EvidenCite.

Experimento 12: entrenamiento/evaluacion TF-IDF, SciBERT y Granite.
Experimento 13: evaluacion experimental via API (Gemini 3.8 y GPT-4.1-mini).

La respuesta de MLflow incluye parametros efectivos, a diferencia de los valores
predeterminados de los YAML, que una corrida puede haber sobreescrito.

Solo exporta campos permitidos; nunca textos, credenciales, datasets ni tags arbitrarias.
Las tarifas de GPU se aplican en el navegador como escenarios comparativos, no
como gasto real de Colab. No se requiere instalar mlflow ni requests.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

ALLOWED_PARAMS = {
    'model_family', 'configuration', 'model_name', 'provider', 'prompt_mode',
    'num_few_shot', 'temperature', 'thinking_budget', 'thinking_level',
    'pretrained_model_name_or_path', 'task_type', 'learning_rate',
    'per_device_train_batch_size', 'per_device_eval_batch_size',
    'gradient_accumulation_steps', 'num_train_epochs', 'weight_decay',
    'warmup_ratio', 'max_grad_norm', 'classifier_dropout', 'loss_type',
    'class_weight_scheme', 'early_stopping_patience', 'dynamic_padding',
    'fp16', 'bf16', 'seed', 'lr_scheduler', 'evals_per_epoch', 'max_steps',
    'logging_steps', 'gradient_checkpointing', 'group_by_length',
    'attn_implementation', 'lora_variant', 'lora_dropout', 'lora_alpha',
    'r', 'target_modules', 'modules_to_save', 'bias', 'max_seq_length',
    'top_k_chunks', 'include_cited_section', 'max_output_tokens',
    'enforce_structured_output', 'ngram_min', 'ngram_max', 'min_df', 'max_df',
    'max_features', 'sublinear_tf', 'c_penalty', 'class_weight', 'max_iter',
    'cost_per_1m_input_tokens_usd', 'cost_per_1m_cached_input_tokens_usd',
    'cost_per_1m_output_tokens_usd', 'max_retries', 'timeout_seconds',
}
ALLOWED_METRICS = {
    'test_macro_f1_all_9', 'test_accuracy', 'test_micro_f1',
    'test_weighted_f1', 'estimated_api_cost_usd', 'gross_api_cost_usd',
    'net_api_cost_usd', 'total_input_tokens', 'total_output_tokens',
    'total_cached_input_tokens', 'inference_latency_ms_per_instance',
    'cache_hit_rate', 'a7_train_wall_time_sec', 'a7_latency_ms',
    'a7_trainable_ratio_pct', 'best_step', 'best_eval_f1_macro',
    'test_micro_f1', 'test_macro_f1_present', 'dev_macro_f1_all_9',
}

# Se limita la carga de salida de parametros a nombres reconocidos.
VALID_PARAM_NAME = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_.-]{0,90}$')


def classify(run_name: str, experiment: str) -> tuple[str, str | None, str | None]:
    name = run_name.lower()
    mode = re.search(r'(?:^|[_-])([ab])(?:[_-]|$)', name)
    config = mode.group(1).upper() if mode else None
    if experiment == '13':
        if 'gemini' in name:
            return 'Gemini 3.8 Flash (experimental)', 'gemini38', config
        if 'gpt41' in name or 'gpt-4.1' in name:
            return 'GPT-4.1 mini (experimental)', 'gpt41mini', config
        return 'Otros API', None, config
    if 'tfidf' in name:
        return 'TF-IDF + LogReg', 'tfidf', config
    if 'scibert' in name:
        return 'SciBERT', 'scibert', config
    if 'granite' in name:
        return 'Granite + LoRA', 'granite', config
    return 'Otros', None, config


def normalized(run: dict, experiment: str) -> dict:
    info, data = run.get('info') or {}, run.get('data') or {}
    tags = {t['key']: t.get('value') for t in data.get('tags', []) if 'key' in t}
    name = str(info.get('run_name') or tags.get('mlflow.runName') or 'sin_nombre')
    family, prefix, config = classify(name, experiment)
    metrics = {m['key']: m.get('value') for m in data.get('metrics', []) if m.get('key') in ALLOWED_METRICS}
    params = {}
    for param in data.get('params', []):
        key = str(param.get('key', ''))
        # Algunas versiones de logging prefijan los params con 'llm_api.' o 'encoder.'.
        suffix = key.rsplit('.', 1)[-1]
        if VALID_PARAM_NAME.fullmatch(key) and suffix in ALLOWED_PARAMS:
            value = str(param.get('value', ''))
            # Nunca se aceptan valores de longitud extrema o multilinea.
            if len(value) <= 220 and '\n' not in value and '\r' not in value:
                params[key] = value
    start, end = info.get('start_time'), info.get('end_time')
    duration_ms = max(0, int(end) - int(start)) if start is not None and end is not None else None
    # La GPU se registra expresamente en src/modeling/registro_mlflow.py.
    gpu = str(tags.get('gpu') or '')[:110].strip() or None
    shot = re.search(r'(zero|few)shot', name, re.I)
    return {
        'run_id': info.get('run_id'),
        'run_name': name,
        'experiment_id': experiment,
        'model_id': f'{prefix}-{config.lower()}' if prefix and config and experiment == '12' else None,
        'family': family,
        'configuration': config,
        'shot_mode': ('zero-shot' if shot and shot.group(1).lower() == 'zero' else 'few-shot' if shot else None),
        'duration_ms': duration_ms,
        'duration_is_approximate': False,
        'test_macro_f1_all_9': float(metrics['test_macro_f1_all_9']) if metrics.get('test_macro_f1_all_9') is not None else None,
        'metrics': metrics,
        'params': dict(sorted(params.items())),
        'gpu': gpu,
        'status': info.get('status'),
        'is_alternate': 'sin_modelo' in name.lower(),
    }


def fetch_runs(origin: str, experiment_id: str, timeout: int) -> list[dict]:
    url = origin + '/api/2.0/mlflow/runs/search'
    out, page_token = [], None
    for _ in range(30):
        payload = {'experiment_ids': [experiment_id], 'max_results': 200}
        if page_token:
            payload['page_token'] = page_token
        request = Request(url, data=json.dumps(payload).encode(), headers={
            'Content-Type': 'application/json', 'Accept': 'application/json'}, method='POST')
        username = os.getenv("MLFLOW_TRACKING_USERNAME")
        password = os.getenv("MLFLOW_TRACKING_PASSWORD")
        if username and password:
            token = base64.b64encode(
                f"{username}:{password}".encode()
            ).decode()
            request.add_header("Authorization", f"Basic {token}")

        try:
            with urlopen(request, timeout=timeout) as response:
                page = json.load(response)
        except HTTPError as exc:
            raise RuntimeError(f'MLflow {experiment_id}: HTTP {exc.code}; no se modifico el snapshot.') from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(f'MLflow {experiment_id}: no se pudo conectar ({exc}); no se modifico el snapshot.') from exc
        if not isinstance(page.get('runs'), list):
            raise RuntimeError(f'MLflow {experiment_id}: respuesta invalida.')
        out.extend(page['runs'])
        page_token = page.get('next_page_token')
        if not page_token:
            return out
    raise RuntimeError(f'MLflow {experiment_id}: exceso de paginas; no se guardo nada.')


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default=os.getenv('MLFLOW_TRACKING_URI'), help='Servidor actual de MLflow')
    p.add_argument('--timeout', default=25, type=int)
    p.add_argument('--output', default='app/static/mlflow-training.json', type=Path)
    args = p.parse_args()
    if not args.url:
        p.error('Falta --url o MLFLOW_TRACKING_URI')
    parsed = urlsplit(args.url)
    if parsed.scheme not in ('http', 'https') or not parsed.netloc:
        p.error('La URL debe tener protocolo http(s) y host.')
    origin = urlunsplit((parsed.scheme, parsed.netloc, '', '', ''))
    try:
        training = fetch_runs(origin, '12', args.timeout)
        api = fetch_runs(origin, '13', args.timeout)
        # Se esperan al menos las corridas vistas en los dos experimentos.
        if len(training) < 7 or len(api) < 8:
            raise RuntimeError(f'Resultado incompleto: experimento 12={len(training)}, 13={len(api)}. No se sobrescribio el JSON.')
        payload = {
            'schema_version': 2,
            'source': 'mlflow_rest_api',
            'source_note': 'Parametros, metricas y duraciones obtenidos por la API de MLflow; costo de GPU solo escenarios.',
            'exported_at_utc': datetime.now(timezone.utc).isoformat(),
            'experiment_id': '12',
            'experiment_name': 'EvidenCite_A6_A7_Evaluacion_Comparativa',
            'api_experiment_id': '13',
            'api_experiment_name': 'EvidenCite_A5_LLM_API',
            'runs': [normalized(run, '12') for run in training],
            'api_runs': [normalized(run, '13') for run in api],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile('w', dir=args.output.parent, prefix='.mlflow.', suffix='.json', encoding='utf-8', delete=False) as temp:
                json.dump(payload, temp, ensure_ascii=False, indent=2)
                temp_path = Path(temp.name)
            temp_path.replace(args.output)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        print(f'✅ MLflow: {len(training)} runs de entrenamiento + {len(api)} runs de API')
        print(f'✅ Datos escritos a {args.output} (snapshot portatil, sin secretos)')
        for group, records in [('Entrenamiento', payload['runs']), ('API experimental', payload['api_runs'])]:
            print(f'  {group}:')
            for run in records:
                f1 = run['test_macro_f1_all_9']
                print(f"    {run['run_name']}: F1={f1:.4f}" if f1 is not None else f"    {run['run_name']}: F1=N/D", f"params={len(run['params'])}, gpu={run['gpu'] or 'N/D'}")
        return 0
    except (RuntimeError, OSError, ValueError) as exc:
        print(f'❌ {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
