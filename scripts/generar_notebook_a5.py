"""Genera el notebook notebooks/analisis_quirurgico_a5.ipynb para el análisis estadístico y quirúrgico de la Actividad A5."""

from pathlib import Path
import nbformat as nbf


def _md(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def _code(code: str):
    return nbf.v4.new_code_cell(code.strip() + "\n")


def build_notebook() -> nbf.NotebookNode:
    cells = []

    # ------------------------------------------------------------------
    # Portada
    # ------------------------------------------------------------------
    cells.append(_md("""
# Análisis Quirúrgico y Estadístico — Actividad A5 (LLMs Comerciales vía API)

Evaluación exhaustiva sobre el conjunto **Test Gold (`data/labeled/test.jsonl`, $N = 2{.}798$)** bajo el diseño factorial completo $2 \\times 2 \\times 2$:
* **Modelos:** `gpt-4.1-mini` (OpenAI) vs. `gemini-3.8-flash` (Google Vertex AI, `thinking_level="LOW"`).
* **Representación de entrada:** Configuración **A** (Título + Abstract) vs. Configuración **B** (Sección retórica citante + Top-3 *chunks* recuperados).
* **Régimen de *prompting*:** **Zero-Shot** (`fs=0`) vs. **Few-Shot Estratificado con Vecindad Retórica (`fs=9`, `ARQ-09`)**.

Total auditado: **$8 \\times 2{.}798 = 22{.}384$ predicciones reales** registradas en `models/llm_api_cache/*.jsonl`, `models/llm_api_*/metrics_summary.json` y `logs/llm_api_inference_*.log`.
"""))

    # ------------------------------------------------------------------
    # Sección 1: Carga y Reconciliación 1:1
    # ------------------------------------------------------------------
    cells.append(_md("""
## 1. Carga, Reconciliación e Integridad de Artefactos ($N = 2{.}798 \\times 8 = 22{.}384$)

Se cargan exclusivamente las 8 corridas reales verificadas (con `config.json` y `metrics_summary.json`), se enlazan 1:1 con los registros de caché JSONL (`PredictionRecord`) y con la metadata completa del corpus de prueba (`data/labeled/test.jsonl`).
"""))

    cells.append(_code("""
import json
import math
import re
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support

BASE_DIR = Path('..') if Path('../data').exists() else Path('.')
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config.schema import CANONICAL_9_CLASSES, LABEL_TO_ID
from src.modeling.dataset import load_jsonl_instances
from src.modeling.evaluate import compute_a6_metrics, paired_significance_test

FIG_DIR = BASE_DIR / 'reports' / 'figures' / 'a5'
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 140,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 10,
    'axes.titlesize': 11.5,
    'axes.titleweight': 'bold',
})

VARIANTS = [
    ('gpt41_zs_a',   'GPT-4.1-mini · 0-Shot · A',   'llm_api_gpt41mini_zeroshot_config_a',   'gpt41mini_config_a.jsonl',     'gpt-4.1-mini',     0, 'A', ''),
    ('gpt41_zs_b',   'GPT-4.1-mini · 0-Shot · B',   'llm_api_gpt41mini_zeroshot_config_b',   'gpt41mini_config_b.jsonl',     'gpt-4.1-mini',     0, 'B', ''),
    ('gpt41_fs_a',   'GPT-4.1-mini · 9-Shot · A',   'llm_api_gpt41mini_fewshot_config_a',    'gpt41mini_config_a.jsonl',     'gpt-4.1-mini',     9, 'A', '::fs9'),
    ('gpt41_fs_b',   'GPT-4.1-mini · 9-Shot · B',   'llm_api_gpt41mini_fewshot_config_b',    'gpt41mini_config_b.jsonl',     'gpt-4.1-mini',     9, 'B', '::fs9'),
    ('gemini_zs_a',  'Gemini-3.8-Flash · 0-Shot · A', 'llm_api_gemini38flash_zeroshot_config_a', 'gemini38flash_config_a.jsonl', 'gemini-3.8-flash', 0, 'A', ''),
    ('gemini_zs_b',  'Gemini-3.8-Flash · 0-Shot · B', 'llm_api_gemini38flash_zeroshot_config_b', 'gemini38flash_config_b.jsonl', 'gemini-3.8-flash', 0, 'B', ''),
    ('gemini_fs_a',  'Gemini-3.8-Flash · 9-Shot · A', 'llm_api_gemini38flash_fewshot_config_a',  'gemini38flash_config_a.jsonl', 'gemini-3.8-flash', 9, 'A', '::fs9'),
    ('gemini_fs_b',  'Gemini-3.8-Flash · 9-Shot · B', 'llm_api_gemini38flash_fewshot_config_b',  'gemini38flash_config_b.jsonl', 'gemini-3.8-flash', 9, 'B', '::fs9'),
]

COLORS = {
    'gpt41_zs_a':  '#1f77b4',
    'gpt41_zs_b':  '#17becf',
    'gpt41_fs_a':  '#2ca02c',
    'gpt41_fs_b':  '#8c564b',
    'gemini_zs_a': '#ff7f0e',
    'gemini_zs_b': '#d62728',
    'gemini_fs_a': '#9467bd',
    'gemini_fs_b': '#e377c2',
}

test_path = BASE_DIR / 'data' / 'labeled' / 'test.jsonl'
test_raw = [json.loads(ln) for ln in test_path.read_text(encoding='utf-8').splitlines() if ln.strip()]
test_instances = load_jsonl_instances(test_path)
N_TEST = len(test_instances)
y_true = [inst.label for inst in test_instances]
y_true_idx = np.array([LABEL_TO_ID[lbl] for lbl in y_true], dtype=np.int64)

gold_df = pd.DataFrame([
    {
        'instance_id': r['instance_id'],
        'y_true': r['label'],
        'y_true_idx': LABEL_TO_ID[r['label']],
        'citing_section': (r.get('citing_paper') or {}).get('rhetorical_section'),
        'label_source': (r.get('annotation_metadata') or {}).get('label_source'),
        'prior_label_source': (r.get('annotation_metadata') or {}).get('prior_label_source'),
        'panel_agreement': (r.get('annotation_metadata') or {}).get('panel_agreement'),
        'panel_label': (r.get('annotation_metadata') or {}).get('panel_label'),
        'context_chars': len(r.get('citation_context') or ''),
        'context_words': len((r.get('citation_context') or '').split()),
    }
    for r in test_raw
])

print(f'Gold Test Split cargado: {test_path} | N = {N_TEST:,} instancias')
print(f"Cobertura citing_section no nula: {gold_df['citing_section'].notna().sum():,}/{N_TEST:,} ({gold_df['citing_section'].notna().mean()*100:.2f}%)")
print('Distribución label_source:', dict(gold_df['label_source'].value_counts(dropna=False)))
"""))

    cells.append(_code("""
# Carga de cachés JSONL, métricas A6/A7 y configuración exacta + verificación 1:1
caches_by_file = {}
for fname in sorted({v[3] for v in VARIANTS}):
    fpath = BASE_DIR / 'models' / 'llm_api_cache' / fname
    recs = {}
    with fpath.open('r', encoding='utf-8') as f:
        for line_no, ln in enumerate(f, 1):
            if not ln.strip():
                continue
            obj = json.loads(ln)
            rec = obj['record']
            rec['_cache_line'] = line_no
            rec['_cache_key'] = obj['cache_key']
            recs[obj['cache_key']] = rec
    caches_by_file[fname] = recs

runs = {}
rows_all = []
summary_rows = []

for vid, label, exp_dir_name, cache_fname, model_name, shots, cfg, suffix in VARIANTS:
    exp_dir = BASE_DIR / 'models' / exp_dir_name
    cfg_json = json.loads((exp_dir / 'config.json').read_text(encoding='utf-8'))
    ms = json.loads((exp_dir / 'metrics_summary.json').read_text(encoding='utf-8'))
    a6 = ms['test_metrics_a6']
    a7 = ms['operational_metrics_a7']
    preds_ms = ms['test_predictions']
    cache_map = caches_by_file[cache_fname]

    ordered_recs = []
    for idx, iid in enumerate(gold_df['instance_id']):
        ck = f'{iid}::{cfg}::generic::{model_name}{suffix}'
        rec = cache_map[ck]
        assert rec['predicted_label'] == preds_ms[idx], f'Mismatch en {vid} idx={idx}'
        ordered_recs.append(rec)

    a6_re = compute_a6_metrics(y_true, [r['predicted_label'] for r in ordered_recs])
    assert abs(a6_re.macro_f1_all_9 - a6['macro_f1_all_9']) < 1e-12

    probs_mat = np.array(
        [[float(r['probabilities'][c]) for c in CANONICAL_9_CLASSES] for r in ordered_recs],
        dtype=np.float64,
    )
    preds_lbl = [r['predicted_label'] for r in ordered_recs]
    preds_idx = np.array([LABEL_TO_ID[p] for p in preds_lbl], dtype=np.int64)

    runs[vid] = {
        'label': label,
        'exp_name': exp_dir_name,
        'model': model_name,
        'shots': shots,
        'cfg': cfg,
        'config_json': cfg_json,
        'a6': a6,
        'a7': a7,
        'records': ordered_recs,
        'y_pred': preds_lbl,
        'y_pred_idx': preds_idx,
        'probs': probs_mat,
    }

    n_valid_cache_hits = sum(1 for r in ordered_recs if r['cached_input_tokens'] > 0)
    n_thoughts = sum(1 for r in ordered_recs if r['thoughts_tokens'] > 0)
    n_invalid = sum(1 for r in ordered_recs if not r['is_valid_format'])
    lat_arr = np.array([r['latency_ms'] for r in ordered_recs], dtype=np.float64)

    summary_rows.append({
        'vid': vid,
        'Variante': label,
        'Modelo': model_name,
        'Shots': shots,
        'Config': cfg,
        'Macro-F1': a6['macro_f1_all_9'],
        'Weighted-F1': a6['weighted_f1'],
        'Accuracy': a6['micro_f1'],
        'Macro-Prec': a6['macro_precision'],
        'Macro-Rec': a6['macro_recall'],
        'Lat Media (ms)': a7['inference_latency_ms_per_instance'],
        'Lat p50 (ms)': float(np.median(lat_arr)),
        'Lat p95 (ms)': float(np.percentile(lat_arr, 95)),
        'Lat p99 (ms)': float(np.percentile(lat_arr, 99)),
        'Throughput (ej/s)': a7['throughput_instances_per_sec'],
        'Invalid (#)': n_invalid,
        'Invalid Rate': a7['invalid_format_rate'],
        'In Tokens': a7['total_input_tokens'],
        'Cached Tokens': a7['total_cached_input_tokens'],
        'Cache Token %': a7['cache_hit_rate'] * 100.0,
        'Cache Inst %': (n_valid_cache_hits / N_TEST) * 100.0,
        'Out Tokens': a7['total_output_tokens'],
        'Thoughts Tokens': a7['total_thoughts_tokens'],
        'Thoughts Inst (#)': n_thoughts,
        'Thoughts Inst %': (n_thoughts / N_TEST) * 100.0,
        'Costo Neto ($)': a7['estimated_api_cost_usd'],
        'Costo Bruto ($)': a7['gross_api_cost_usd'],
        'Ahorro Caché ($)': a7['gross_api_cost_usd'] - a7['estimated_api_cost_usd'],
    })

    for idx, r in enumerate(ordered_recs):
        rows_all.append({
            'vid': vid,
            'Variante': label,
            'Modelo': model_name,
            'Shots': shots,
            'Config': cfg,
            'instance_id': r['instance_id'],
            'test_idx': idx,
            'y_true': y_true[idx],
            'y_true_idx': int(y_true_idx[idx]),
            'y_pred': r['predicted_label'],
            'y_pred_idx': int(preds_idx[idx]),
            'is_correct': int(r['predicted_label'] == y_true[idx]),
            'confidence': float(r['confidence']),
            'max_prob': float(probs_mat[idx].max()),
            'prob_true': float(probs_mat[idx, y_true_idx[idx]]),
            'is_valid_format': bool(r['is_valid_format']),
            'input_tokens': int(r['input_tokens']),
            'cached_input_tokens': int(r['cached_input_tokens']),
            'uncached_input_tokens': int(r['input_tokens'] - r['cached_input_tokens']),
            'output_tokens': int(r['output_tokens']),
            'thoughts_tokens': int(r['thoughts_tokens']),
            'latency_ms': float(r['latency_ms']),
            'citing_section': gold_df.iloc[idx]['citing_section'],
            'label_source': gold_df.iloc[idx]['label_source'],
            'panel_agreement': gold_df.iloc[idx]['panel_agreement'],
            'context_chars': int(gold_df.iloc[idx]['context_chars']),
            '_cache_line': int(r['_cache_line']),
            '_cache_key': r['_cache_key'],
        })

df_summary = pd.DataFrame(summary_rows)
df_all = pd.DataFrame(rows_all)
print(f'Verificación 1:1 superada en las {len(runs)} variantes ({len(df_all):,} registros totales).')
print(f"Costo total Neto USD = ${df_summary['Costo Neto ($)'].sum():.4f} | Bruto USD = ${df_summary['Costo Bruto ($)'].sum():.4f} | Ahorro Caché = ${df_summary['Ahorro Caché ($)'].sum():.4f}")

# Auditoría de logs HTTP
log_stats = []
for lf in sorted((BASE_DIR / 'logs').glob('llm_api_inference_*.log')):
    if 'runner' in lf.name:
        continue
    txt = lf.read_text(encoding='utf-8', errors='replace')
    n_200 = len(re.findall(r'HTTP/1\\.1 200 OK', txt))
    n_retry_gem = len(re.findall(r'Reintento Gemini', txt))
    n_retry_oai = len(re.findall(r'Retrying request', txt))
    log_stats.append({'Log': lf.name, 'HTTP_200': n_200, 'Reintentos_Gemini': n_retry_gem, 'Reintentos_OpenAI': n_retry_oai})
display(pd.DataFrame(log_stats))

cols_view = [
    'Variante', 'Macro-F1', 'Weighted-F1', 'Accuracy',
    'Lat Media (ms)', 'Lat p50 (ms)', 'Lat p95 (ms)', 'Lat p99 (ms)',
    'Invalid (#)', 'Cache Inst %', 'Cache Token %', 'Thoughts Inst (#)',
    'Costo Neto ($)', 'Costo Bruto ($)'
]
display(df_summary[cols_view].style.format({
    'Macro-F1': '{:.4f}', 'Weighted-F1': '{:.4f}', 'Accuracy': '{:.4f}',
    'Lat Media (ms)': '{:.1f}', 'Lat p50 (ms)': '{:.1f}', 'Lat p95 (ms)': '{:.1f}', 'Lat p99 (ms)': '{:.1f}',
    'Cache Inst %': '{:.2f}%', 'Cache Token %': '{:.2f}%',
    'Costo Neto ($)': '${:.4f}', 'Costo Bruto ($)': '${:.4f}',
}))
"""))

    # ------------------------------------------------------------------
    # Sección 2: Forense de las 2 instancias fallidas
    # ------------------------------------------------------------------
    cells.append(_md("""
## 2. Diagnóstico Forense de las 2 Instancias Fallidas (`is_valid_format == False`)

Conforme a la decisión operativa aprobada, **se preservan intactos los archivos `metrics_summary.json` originales** (cero reinferencia, costo adicional `$0.00`). Se documenta aquí la trazabilidad completa de ambas instancias, su firma estructural de tokens de salida, la evidencia en logs HTTP, sus predicciones en las otras 7 corridas y su impacto real en métricas.
"""))

    cells.append(_code("""
failed = df_all[~df_all['is_valid_format']].copy()
print(f'Total de registros con is_valid_format == False en los 22,384 llamados: {len(failed)}')

forensic_rows = []
for _, row in failed.iterrows():
    iid = row['instance_id']
    g_match = [r for r in test_raw if r['instance_id'] == iid][0]
    other_runs = df_all[(df_all['instance_id'] == iid) & (df_all['vid'] != row['vid'])][
        ['Variante', 'y_pred', 'confidence', 'output_tokens', 'latency_ms']
    ]
    print(f"\\n--- FALLO: {iid} | {row['Variante']} | cache_key={row['_cache_key']} | línea_cache={row['_cache_line']} | test_idx={row['test_idx']} ---")
    print(f"Gold='{row['y_true']}' | Sección='{row['citing_section']}' | label_source={row['label_source']} | panel_agreement={row['panel_agreement']}")
    print(f"Telemetría: input={row['input_tokens']}, cached={row['cached_input_tokens']}, output={row['output_tokens']}, thoughts={row['thoughts_tokens']}, latency_ms={row['latency_ms']:.2f}")
    print(f"Fallback aplicado en código: predicted_label='{row['y_pred']}', confidence={row['confidence']:.6f} (1/9)")
    print('Contexto citante:', g_match['citation_context'])
    display(other_runs.reset_index(drop=True))

    # Impacto contrafactual exacto si el registro fallido hubiera predicho y_true vs su predicción modal Gemini
    vid = row['vid']
    y_pred_cur = runs[vid]['y_pred'][:]
    y_pred_corr = y_pred_cur[:]
    y_pred_corr[row['test_idx']] = row['y_true']
    f1_cur = runs[vid]['a6']['macro_f1_all_9']
    acc_cur = runs[vid]['a6']['micro_f1']
    f1_cf = f1_score(y_true, y_pred_corr, labels=list(CANONICAL_9_CLASSES), average='macro', zero_division=0)
    acc_cf = accuracy_score(y_true, y_pred_corr)
    forensic_rows.append({
        'instance_id': iid,
        'Variante': row['Variante'],
        'cache_key': row['_cache_key'],
        'Línea Caché': row['_cache_line'],
        'Gold': row['y_true'],
        'Fallback': row['y_pred'],
        'Out Tokens': row['output_tokens'],
        'Latencia (ms)': row['latency_ms'],
        'Macro-F1 Observado': f1_cur,
        'Macro-F1 Si Acertada': f1_cf,
        'Δ Macro-F1 Potencial': f1_cf - f1_cur,
        'Δ Accuracy Potencial': acc_cf - acc_cur,
    })

display(pd.DataFrame(forensic_rows).style.format({
    'Latencia (ms)': '{:.2f}',
    'Macro-F1 Observado': '{:.6f}',
    'Macro-F1 Si Acertada': '{:.6f}',
    'Δ Macro-F1 Potencial': '{:+.6f}',
    'Δ Accuracy Potencial': '{:+.6f}',
}))

# Tabla resumen de distribución de output_tokens y detección de singletons anómalos
out_stats = []
for vid, rinfo in runs.items():
    sub = df_all[df_all['vid'] == vid]
    vc = dict(sorted(Counter(sub['output_tokens']).items()))
    inv_tokens = sorted(sub[~sub['is_valid_format']]['output_tokens'].tolist())
    val_tokens = sub[sub['is_valid_format']]['output_tokens']
    out_stats.append({
        'Variante': rinfo['label'],
        'Min Válido': int(val_tokens.min()),
        'Mediana Válida': float(val_tokens.median()),
        'Max Válido': int(val_tokens.max()),
        'Out Tokens en Inválidos': inv_tokens if inv_tokens else '—',
        'Top Modas': dict(Counter(sub['output_tokens']).most_common(4)),
    })
display(pd.DataFrame(out_stats))
"""))

    cells.append(_code("""
# Figura 1: Distribución de output_tokens y evidencia estructural de los 2 fallos de formato
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

# Panel A: Gemini Config A Zero-Shot vs Few-Shot (donde ocurren ambos fallos)
for ax, (vid, title_txt) in zip(
    axes,
    [
        ('gemini_zs_a', 'Gemini-3.8-Flash · 0-Shot · Config A (inst_17318)'),
        ('gemini_fs_a', 'Gemini-3.8-Flash · 9-Shot · Config A (inst_17446)'),
    ],
):
    sub = df_all[df_all['vid'] == vid]
    vc = sub.groupby(['output_tokens', 'is_valid_format']).size().unstack(fill_value=0)
    x_vals = vc.index.to_numpy()
    valid_cnt = vc[True].to_numpy() if True in vc.columns else np.zeros_like(x_vals)
    inv_cnt = vc[False].to_numpy() if False in vc.columns else np.zeros_like(x_vals)

    ax.bar(x_vals, valid_cnt, color='#4c72b0', alpha=0.88, label='Válido (is_valid_format=True)')
    if inv_cnt.sum() > 0:
        # Altura visible auxiliar en eje secundario o anotación sobre la barra real (1)
        ax.bar(x_vals, inv_cnt, bottom=valid_cnt, color='#d62728', edgecolor='black', linewidth=1.2,
               label='Inválido (n=1 singleton)')
        bad_x = x_vals[inv_cnt > 0][0]
        xt = 92 if bad_x < 90 else 104
        yt = 150 if bad_x < 90 else 110
        ax.annotate(
            f'Fallo único\\nout={bad_x} tok\\n(max válido={bad_x-1})',
            xy=(bad_x, 1),
            xytext=(xt, yt),
            ha='center' if bad_x >= 90 else 'left',
            arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.8),
            fontsize=9.2, fontweight='bold', color='#b2182b',
            bbox=dict(boxstyle='round,pad=0.3', fc='#fde0dd', ec='#d62728'),
        )
    ax.set_title(title_txt)
    ax.set_xlabel('output_tokens registrados por la API')
    ax.set_ylabel('Frecuencia (N = 2,798)')
    ax.set_yscale('log')
    ax.set_ylim(0.7, max(valid_cnt) * 2.2)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.legend(loc='upper left', frameon=True, fontsize=8.5)

fig.suptitle('Evidencia Estructural de las 2 Fallas de Formato: Singletons +1 Token por Encima de la Moda JSON Válida',
             fontsize=12.5, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / 'fig01_output_tokens_y_fallos_formato.png', bbox_inches='tight')
plt.show()
"""))

    # ------------------------------------------------------------------
    # Sección 3: Eje 1A - Desempeño Predictivo, Por Clase y Sección
    # ------------------------------------------------------------------
    cells.append(_md("""
## 3. Eje 1A: Desempeño Predictivo Global, por Clase Canónica y por Sección Retórica

Se compara el rendimiento técnico sobre las 9 clases (`Macro-F1`, `Weighted-F1`, `Accuracy`), la vulnerabilidad en las clases de la tríada metodológica (`Improvement`, `Evidence`, `Basis`, `Comparison`) frente a categorías altamente lexicalizadas (`Further Reading`), el impacto de la sección citante y los pares críticos de confusión.
"""))

    cells.append(_code("""
# Tabla F1 por clase across 8 variantes + Macro-F1
f1_class_rows = []
support_dict = runs['gpt41_zs_a']['a6']['per_class_support']
for cls_name in CANONICAL_9_CLASSES:
    row = {'Clase': cls_name, 'Soporte': support_dict[cls_name]}
    for vid, label, *_ in VARIANTS:
        row[label] = runs[vid]['a6']['per_class_f1'][cls_name]
    f1_class_rows.append(row)

df_f1_cls = pd.DataFrame(f1_class_rows)
short_names = {label: f"{m.split('-')[0].upper()}-{s}s-{c}" for _, label, _, _, m, s, c, _ in VARIANTS}
display(df_f1_cls.style.format({c: '{:.4f}' for c in df_f1_cls.columns if c not in ('Clase', 'Soporte')})
        .background_gradient(subset=[c for c in df_f1_cls.columns if c not in ('Clase', 'Soporte')], cmap='YlGnBu', vmin=0.2, vmax=0.85))

# Figura 2: Heatmap F1 por clase + barras comparativas globales
fig = plt.figure(figsize=(16.0, 6.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.15)

ax0 = fig.add_subplot(gs[0, 0])
mat_f1 = df_f1_cls[[v[1] for v in VARIANTS]].to_numpy().T  # 8 x 9
im = ax0.imshow(mat_f1, cmap='YlGnBu', aspect='auto', vmin=0.20, vmax=0.85)
ax0.set_xticks(np.arange(len(CANONICAL_9_CLASSES)))
ax0.set_xticklabels([f"{c}\\n(n={support_dict[c]})" for c in CANONICAL_9_CLASSES], rotation=35, ha='right', fontsize=8.5)
ax0.set_yticks(np.arange(len(VARIANTS)))
ax0.set_yticklabels([v[1] for v in VARIANTS], fontsize=9)
ax0.set_title('F1-Score por Clase Canónica (8 Variantes A5)')
for i in range(mat_f1.shape[0]):
    for j in range(mat_f1.shape[1]):
        val = mat_f1[i, j]
        ax0.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8,
                 color='white' if val > 0.57 else 'black', fontweight='bold')
fig.colorbar(im, ax=ax0, fraction=0.03, pad=0.02, label='F1-Score')

ax1 = fig.add_subplot(gs[0, 1])
y_pos = np.arange(len(VARIANTS))
h = 0.26
ax1.barh(y_pos - h, df_summary['Macro-F1'], height=h, color='#2b5c8f', label='Macro-F1 (9 clases)')
ax1.barh(y_pos,     df_summary['Weighted-F1'], height=h, color='#4ca674', label='Weighted-F1')
ax1.barh(y_pos + h, df_summary['Accuracy'], height=h, color='#e68a2e', label='Accuracy')
for idx, row in df_summary.iterrows():
    ax1.text(row['Macro-F1'] + 0.004, idx - h, f"{row['Macro-F1']:.4f}", va='center', fontsize=8)
    ax1.text(row['Accuracy'] + 0.004, idx + h, f"{row['Accuracy']:.4f}", va='center', fontsize=8)
ax1.set_yticks(y_pos)
ax1.set_yticklabels([])
ax1.invert_yaxis()
ax1.set_xlim(0.32, 0.56)
ax1.set_xlabel('Puntaje')
ax1.set_title('Desempeño Global (Test N = 2,798)')
ax1.grid(axis='x', linestyle='--', alpha=0.35)
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=8.3, frameon=True)

fig.savefig(FIG_DIR / 'fig02_desempeno_global_y_por_clase.png', bbox_inches='tight')
plt.show()
"""))

    cells.append(_code("""
# Análisis por sección retórica citante (citing_section) y principales pares de confusión
def map_macro_section(sec):
    if sec is None or (isinstance(sec, float) and np.isnan(sec)):
        return 'Sin sección (null)'
    s = str(sec).lower()
    if any(k in s for k in ('intro', 'motivation', 'overview', 'prelimin', 'backgr')):
        return 'Intro / Prelim'
    if any(k in s for k in ('related', 'literat', 'prior', 'previous')):
        return 'Related Work'
    if any(k in s for k in ('method', 'approach', 'model', 'architect', 'algorith', 'framework', 'system', 'implement', 'train', 'setup')):
        return 'Method / Setup'
    if any(k in s for k in ('experim', 'evaluat', 'result', 'ablation', 'discuss', 'analysis', 'benchmark', 'compar')):
        return 'Experiments / Eval'
    if any(k in s for k in ('conclu', 'future', 'limit', 'summary')):
        return 'Conclusion / Future'
    return 'Otras secciones'

df_all['macro_section'] = df_all['citing_section'].map(map_macro_section)
gold_df['macro_section'] = gold_df['citing_section'].map(map_macro_section)
sec_counts = gold_df['macro_section'].value_counts()
print('Distribución de macro-secciones en Test Gold:')
print(sec_counts)

sec_perf = []
for ms_name, cnt in sec_counts.items():
    for vid, label, *_ in VARIANTS:
        sub = df_all[(df_all['vid'] == vid) & (df_all['macro_section'] == ms_name)]
        sec_perf.append({
            'Macro-Sección': f"{ms_name} (n={cnt})",
            'ms_raw': ms_name,
            'n': int(cnt),
            'Variante': label,
            'vid': vid,
            'Accuracy': sub['is_correct'].mean(),
            'Macro-F1': f1_score(sub['y_true_idx'], sub['y_pred_idx'], labels=list(range(9)), average='macro', zero_division=0),
        })
df_sec = pd.DataFrame(sec_perf)
piv_sec = df_sec.pivot(index='Macro-Sección', columns='Variante', values='Accuracy')
display(piv_sec[[v[1] for v in VARIANTS]].style.format('{:.4f}').background_gradient(cmap='Blues', axis=None))

# Top pares de confusión en los dos mejores representantes por familia (GPT-4.1-mini 0s-B vs Gemini-3.8-Flash 9s-A)
def top_confusions(vid, top_k=10):
    cm = np.array(runs[vid]['a6']['confusion_matrix'], dtype=int)
    pairs = []
    for i, c_true in enumerate(CANONICAL_9_CLASSES):
        for j, c_pred in enumerate(CANONICAL_9_CLASSES):
            if i != j and cm[i, j] > 0:
                pairs.append((c_true, c_pred, int(cm[i, j]), cm[i, j] / max(1, cm[i].sum()) * 100.0))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pd.DataFrame(pairs[:top_k], columns=['Real', 'Predicho', 'Errores', '% Fila Real'])

print('\\nTop 10 confusiones en Gemini-3.8-Flash · 9-Shot · Config A (mejor global, Macro-F1=0.4811):')
display(top_confusions('gemini_fs_a'))
print('\\nTop 10 confusiones en GPT-4.1-mini · 0-Shot · Config B (mejor OpenAI, Macro-F1=0.4407):')
display(top_confusions('gpt41_zs_b'))

# Figura 3: Matrices de confusión de los mejores exponentes + Accuracy por macro-sección citante
fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.6))

for ax, vid, title_cm in [
    (axes[0], 'gpt41_zs_b',  'GPT-4.1-mini · 0-Shot · B (F1=0.441)'),
    (axes[1], 'gemini_fs_a', 'Gemini-3.8-Flash · 9-Shot · A (F1=0.481)'),
]:
    cm = np.array(runs[vid]['a6']['confusion_matrix'], dtype=float)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0.0, vmax=0.85)
    ax.set_title(title_cm)
    ax.set_xticks(range(9)); ax.set_yticks(range(9))
    ax.set_xticklabels(CANONICAL_9_CLASSES, rotation=40, ha='right', fontsize=8)
    ax.set_yticklabels(CANONICAL_9_CLASSES, fontsize=8)
    ax.set_xlabel('Predicción'); ax.set_ylabel('Etiqueta Real')
    for i in range(9):
        for j in range(9):
            val = cm_norm[i, j]
            if val >= 0.03 or i == j:
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=6.8,
                        color='white' if val > 0.48 else 'black')
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)

ax2 = axes[2]
sec_order = ['Related Work', 'Intro / Prelim', 'Experiments / Eval', 'Method / Setup', 'Otras secciones', 'Conclusion / Future', 'Sin sección (null)']
sec_order = [s for s in sec_order if s in set(df_sec['ms_raw'])]
x_s = np.arange(len(sec_order))
for idx, (vid, short_lbl) in enumerate([
    ('gpt41_zs_a',  'GPT 0s A'),
    ('gpt41_zs_b',  'GPT 0s B'),
    ('gemini_zs_b', 'Gem 0s B'),
    ('gemini_fs_a', 'Gem 9s A'),
    ('gemini_fs_b', 'Gem 9s B'),
]):
    vals = [df_sec[(df_sec['vid'] == vid) & (df_sec['ms_raw'] == s)]['Accuracy'].iloc[0] for s in sec_order]
    ax2.plot(x_s, vals, marker='o', linewidth=1.8, label=short_lbl, color=COLORS[vid])
ax2.set_xticks(x_s)
ax2.set_xticklabels([f"{s}\\n(n={sec_counts[s]})" for s in sec_order], rotation=30, ha='right', fontsize=8)
ax2.set_ylabel('Accuracy por Macro-Sección')
ax2.set_title('Desempeño según Sección del Artículo Citante')
ax2.set_ylim(0.27, 0.67)
ax2.grid(linestyle='--', alpha=0.35)
ax2.legend(fontsize=8, frameon=True, loc='upper right')

fig.tight_layout()
fig.savefig(FIG_DIR / 'fig03_confusiones_criticas_y_seccion_citante.png', bbox_inches='tight')
plt.show()
"""))

    # ------------------------------------------------------------------
    # Sección 4: Significancia Estadística Pareada (Bootstrap + McNemar)
    # ------------------------------------------------------------------
    cells.append(_md(r"""
## 4. Pruebas Formales de Significancia Estadística Pareada ($B = 10{,}000$)

Se ejecuta `src.modeling.evaluate.paired_significance_test` (*Paired Bootstrap* bilateral + prueba exacta/chi2 de *McNemar* sobre discordancias $n_{01}, n_{10}$) para los **12 contrastes ortogonales** del experimento ($4$ pares Config B vs. A, $4$ pares 9-Shot vs. 0-Shot y $4$ pares Gemini vs. GPT), aplicando además ajuste **Holm-Bonferroni** ($\alpha = 0.05$) por comparaciones múltiples.
"""))

    cells.append(_code("""
CONTRASTS = [
    # Bloque 1: Config B - Config A (Hipótesis RAG Retórico)
    ('Config B vs. Config A', 'GPT-4.1-mini · 0-Shot (B vs A)',     'gpt41_zs_a',  'gpt41_zs_b'),
    ('Config B vs. Config A', 'GPT-4.1-mini · 9-Shot (B vs A)',     'gpt41_fs_a',  'gpt41_fs_b'),
    ('Config B vs. Config A', 'Gemini-3.8-Flash · 0-Shot (B vs A)', 'gemini_zs_a', 'gemini_zs_b'),
    ('Config B vs. Config A', 'Gemini-3.8-Flash · 9-Shot (B vs A)', 'gemini_fs_a', 'gemini_fs_b'),
    # Bloque 2: Few-Shot (9) vs Zero-Shot (0)
    ('9-Shot vs. 0-Shot', 'GPT-4.1-mini · Config A (9s vs 0s)',     'gpt41_zs_a',  'gpt41_fs_a'),
    ('9-Shot vs. 0-Shot', 'GPT-4.1-mini · Config B (9s vs 0s)',     'gpt41_zs_b',  'gpt41_fs_b'),
    ('9-Shot vs. 0-Shot', 'Gemini-3.8-Flash · Config A (9s vs 0s)', 'gemini_zs_a', 'gemini_fs_a'),
    ('9-Shot vs. 0-Shot', 'Gemini-3.8-Flash · Config B (9s vs 0s)', 'gemini_zs_b', 'gemini_fs_b'),
    # Bloque 3: Gemini-3.8-Flash vs GPT-4.1-mini
    ('Gemini vs. GPT-4.1-mini', '0-Shot · Config A (Gem vs GPT)',   'gpt41_zs_a',  'gemini_zs_a'),
    ('Gemini vs. GPT-4.1-mini', '0-Shot · Config B (Gem vs GPT)',   'gpt41_zs_b',  'gemini_zs_b'),
    ('Gemini vs. GPT-4.1-mini', '9-Shot · Config A (Gem vs GPT)',   'gpt41_fs_a',  'gemini_fs_a'),
    ('Gemini vs. GPT-4.1-mini', '9-Shot · Config B (Gem vs GPT)',   'gpt41_fs_b',  'gemini_fs_b'),
]

sig_rows = []
for blk, cname, vid_a, vid_b in CONTRASTS:
    res = paired_significance_test(
        y_true=y_true,
        y_pred_a=runs[vid_a]['y_pred'],
        y_pred_b=runs[vid_b]['y_pred'],
        n_bootstrap=10000,
        seed=42,
    )
    sig_rows.append({
        'Bloque': blk,
        'Contraste': cname,
        'vid_a': vid_a,
        'vid_b': vid_b,
        'F1 (Base A)': res.score_a,
        'F1 (Comp B)': res.score_b,
        'Δ F1 (B - A)': res.delta_b_minus_a,
        'IC95_low': res.ci_95_low,
        'IC95_high': res.ci_95_high,
        'IC 95% Bootstrap': f'[{res.ci_95_low:+.4f}, {res.ci_95_high:+.4f}]',
        'p (Bootstrap)': res.p_value_bootstrap,
        'n10 (Solo A ✓)': res.mcnemar_a_only_correct,
        'n01 (Solo B ✓)': res.mcnemar_b_only_correct,
        'p (McNemar)': res.p_value_mcnemar,
        'Sig α=0.05': 'SÍ' if res.is_significant_95 else 'NO',
    })

df_sig_all = pd.DataFrame(sig_rows)

# Corrección Holm-Bonferroni (m=12 pruebas)
for col_p, col_adj in [('p (Bootstrap)', 'Sig Holm-Bonf (Boot)'), ('p (McNemar)', 'Sig Holm-Bonf (McN)')]:
    pvals = df_sig_all[col_p].to_numpy()
    order = np.argsort(pvals)
    m = len(pvals)
    rejected = np.zeros(m, dtype=bool)
    for rank, idx in enumerate(order):
        if pvals[idx] <= 0.05 / (m - rank):
            rejected[idx] = True
        else:
            break
    df_sig_all[col_adj] = np.where(rejected, 'SÍ', 'NO')

display(df_sig_all[[
    'Bloque', 'Contraste', 'F1 (Base A)', 'F1 (Comp B)', 'Δ F1 (B - A)', 'IC 95% Bootstrap',
    'p (Bootstrap)', 'Sig Holm-Bonf (Boot)', 'n10 (Solo A ✓)', 'n01 (Solo B ✓)', 'p (McNemar)', 'Sig Holm-Bonf (McN)'
]].style.format({
    'F1 (Base A)': '{:.4f}', 'F1 (Comp B)': '{:.4f}', 'Δ F1 (B - A)': '{:+.4f}',
    'p (Bootstrap)': '{:.4e}', 'p (McNemar)': '{:.4e}',
}))

# Figura 4: Forest Plot de los 12 contrastes pareados
fig, ax = plt.subplots(figsize=(11.5, 6.4))
y_idx = np.arange(len(df_sig_all))
ax.axvline(0.0, color='black', linestyle='--', linewidth=1.1, alpha=0.7)

blk_colors = {
    'Config B vs. Config A': '#1f77b4',
    '9-Shot vs. 0-Shot': '#ff7f0e',
    'Gemini vs. GPT-4.1-mini': '#2ca02c',
}

for i, r in df_sig_all.iterrows():
    c = blk_colors[r['Bloque']]
    marker = 'D' if r['Sig Holm-Bonf (Boot)'] == 'SÍ' else ('o' if r['Sig α=0.05'] == 'SÍ' else 'x')
    ax.errorbar(
        r['Δ F1 (B - A)'], i,
        xerr=[[r['Δ F1 (B - A)'] - r['IC95_low']], [r['IC95_high'] - r['Δ F1 (B - A)']]],
        fmt=marker, color=c, ecolor=c, elinewidth=2.0, capsize=4, markersize=7,
    )
    p_str = '<1e-4' if r['p (Bootstrap)'] == 0.0 else f"{r['p (Bootstrap)']:.2e}"
    txt_sig = f" {r['Δ F1 (B - A)']:+.4f} (p={p_str}, Holm={r['Sig Holm-Bonf (Boot)']})"
    ax.text(max(r['IC95_high'], r['Δ F1 (B - A)']) + 0.002, i, txt_sig, va='center', fontsize=8.3, color='#222')

ax.axhline(3.5, color='#ccc', linestyle=':', linewidth=1.2)
ax.axhline(7.5, color='#ccc', linestyle=':', linewidth=1.2)
ax.set_yticks(y_idx)
ax.set_yticklabels(df_sig_all['Contraste'], fontsize=9)
ax.invert_yaxis()
ax.set_xlim(-0.040, 0.118)
ax.set_xlabel('Δ Macro-F1 (Comparado − Base) con IC 95% Paired Bootstrap (B = 10,000)')
ax.set_title('Significancia Estadística Pareada en Diseño 2×2×2 (Diamante = Sig. Holm-Bonferroni α=0.05; X = No Sig.)')
ax.grid(axis='x', linestyle='--', alpha=0.35)

fig.tight_layout()
fig.savefig(FIG_DIR / 'fig04_forest_plot_significancia_pareada.png', bbox_inches='tight')
plt.show()
"""))

    # ------------------------------------------------------------------
    # Sección 5: Eje 1C - Calibración Probabilística (Decisión A3 Opción 2)
    # ------------------------------------------------------------------
    cells.append(_md("""
## 5. Eje 1C (Opción 2): Calibración Probabilística (`ECE`, `MCE`, `Brier Score`) y Verificación de `ARQ-09 §5.1`

Conforme a `ARQ-09 §5.1`, las probabilidades de A5 son **confianza verbalizada estructurada** normalizada a $\\Delta^8$ por `NineClassScores.to_dict_canonical()`. Aquí evaluamos su calibración formal con:
* **ECE** (*Expected Calibration Error*, $M=10$ bins equiespaciados sobre confianza top-1 $p_{\\max}$).
* **MCE** (*Maximum Calibration Error*) en bins totales y filtrados ($n_b \\ge 10$ para estabilidad muestral).
* **Brier Score Multiclase** ($\\text{BS} = \\frac{1}{N} \\sum_{i=1}^N \\sum_{c=1}^9 (p_{i,c} - y_{i,c})^2 \\in [0, 2]$) y **NLL** (*Negative Log-Likelihood*).
* **Brecha de Sobreconfianza** ($\\bar{p}_{\\text{err}} - \\bar{p}_{\\text{acierto}}$) y contraste directo **0-Shot vs. 9-Shot con `_RHETORICAL_NEIGHBORS`** (*anclaje discreto* vs. *distribución graduada*).
"""))

    cells.append(_code("""
def compute_calibration_metrics(probs: np.ndarray, y_true_i: np.ndarray, y_pred_i: np.ndarray, n_bins: int = 10):
    conf = probs.max(axis=1)
    correct = (y_pred_i == y_true_i).astype(np.float64)
    n = len(y_true_i)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce_all = 0.0
    mce_min10 = 0.0
    bin_stats = []
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        if b == 0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf > lo) & (conf <= hi)
        cnt = int(mask.sum())
        if cnt > 0:
            acc_b = float(correct[mask].mean())
            conf_b = float(conf[mask].mean())
            gap = abs(acc_b - conf_b)
            ece += (cnt / n) * gap
            mce_all = max(mce_all, gap)
            if cnt >= 10:
                mce_min10 = max(mce_min10, gap)
        else:
            acc_b, conf_b, gap = np.nan, np.nan, np.nan
        bin_stats.append({'bin': b, 'lo': lo, 'hi': hi, 'n': cnt, 'acc': acc_b, 'conf': conf_b, 'gap': gap})

    # Brier Score multiclase (rango [0, 2]) y Top-1 Brier
    one_hot = np.eye(probs.shape[1], dtype=np.float64)[y_true_i]
    brier_multi = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    brier_top1 = float(np.mean((conf - correct) ** 2))

    # NLL y entropía de Shannon media (nats)
    p_true = np.clip(probs[np.arange(n), y_true_i], 1e-12, 1.0)
    nll = float(-np.mean(np.log(p_true)))
    p_safe = np.clip(probs, 1e-12, 1.0)
    entropy_nats = float(np.mean(-np.sum(probs * np.log(p_safe), axis=1)))

    mean_conf = float(conf.mean())
    acc_total = float(correct.mean())
    conf_ok = float(conf[correct == 1].mean()) if (correct == 1).any() else np.nan
    conf_err = float(conf[correct == 0].mean()) if (correct == 0).any() else np.nan

    # Telemetría de anclaje numérico ARQ-09 §5.1
    top1_unique = len(np.unique(np.round(conf, 4)))
    zero_mass_pct = float((probs == 0.0).mean() * 100.0)
    max_probs_top3_share = sum(v for _, v in Counter(np.round(conf, 3)).most_common(3)) / n * 100.0

    return {
        'ECE': float(ece),
        'MCE (todos bins)': float(mce_all),
        'MCE (bins n>=10)': float(mce_min10),
        'Brier Multiclase': brier_multi,
        'Brier Top-1': brier_top1,
        'NLL': nll,
        'Entropía (nats)': entropy_nats,
        'Accuracy': acc_total,
        'Conf Media': mean_conf,
        'Sobreconfianza (Conf-Acc)': mean_conf - acc_total,
        'Conf | Acierto': conf_ok,
        'Conf | Error': conf_err,
        'Δ Conf (Ok-Err)': conf_ok - conf_err,
        'Top3 Modas Conf (%)': max_probs_top3_share,
        'Ceros en Vector (%)': zero_mass_pct,
        'Valores Únicos Conf': top1_unique,
        'bin_stats': pd.DataFrame(bin_stats),
    }

calib_rows = []
calib_bins = {}
for vid, rinfo in runs.items():
    cm = compute_calibration_metrics(rinfo['probs'], y_true_idx, rinfo['y_pred_idx'], n_bins=10)
    calib_bins[vid] = cm.pop('bin_stats')
    calib_rows.append({'vid': vid, 'Variante': rinfo['label'], 'Modelo': rinfo['model'], 'Shots': rinfo['shots'], 'Config': rinfo['cfg'], **cm})

df_calib = pd.DataFrame(calib_rows)
display(df_calib[[
    'Variante', 'ECE', 'MCE (todos bins)', 'MCE (bins n>=10)', 'Brier Multiclase', 'NLL',
    'Accuracy', 'Conf Media', 'Sobreconfianza (Conf-Acc)', 'Conf | Acierto', 'Conf | Error', 'Δ Conf (Ok-Err)',
    'Top3 Modas Conf (%)', 'Ceros en Vector (%)'
]].style.format({
    'ECE': '{:.4f}', 'MCE (todos bins)': '{:.4f}', 'MCE (bins n>=10)': '{:.4f}',
    'Brier Multiclase': '{:.4f}', 'NLL': '{:.4f}', 'Accuracy': '{:.4f}',
    'Conf Media': '{:.4f}', 'Sobreconfianza (Conf-Acc)': '{:+.4f}',
    'Conf | Acierto': '{:.4f}', 'Conf | Error': '{:.4f}', 'Δ Conf (Ok-Err)': '{:+.4f}',
    'Top3 Modas Conf (%)': '{:.1f}%', 'Ceros en Vector (%)': '{:.1f}%',
}))

# Pruebas pareadas Bootstrap/Wilcoxon en Brier Score y Confianza: 9-Shot vs 0-Shot
print('\\nContraste 9-Shot vs. 0-Shot en Brier Score individual (Wilcoxon paired signed-rank):')
for name_pair, v0, v9 in [
    ('GPT-4.1-mini · Config A', 'gpt41_zs_a', 'gpt41_fs_a'),
    ('GPT-4.1-mini · Config B', 'gpt41_zs_b', 'gpt41_fs_b'),
    ('Gemini-3.8-Flash · Config A', 'gemini_zs_a', 'gemini_fs_a'),
    ('Gemini-3.8-Flash · Config B', 'gemini_zs_b', 'gemini_fs_b'),
]:
    one_hot = np.eye(9, dtype=np.float64)[y_true_idx]
    b0 = np.sum((runs[v0]['probs'] - one_hot) ** 2, axis=1)
    b9 = np.sum((runs[v9]['probs'] - one_hot) ** 2, axis=1)
    stat_w, p_w = stats.wilcoxon(b9, b0)
    print(f" • {name_pair:<28}: Brier 0s={b0.mean():.4f} → 9s={b9.mean():.4f} (Δ={b9.mean()-b0.mean():+.4f}, Wilcoxon p={p_w:.3e})")
"""))

    cells.append(_code("""
# Figura 5: Diagramas de Fiabilidad (Reliability Curves) + Comparativa ECE / MCE / Brier
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.8))

# Panel 0,0 y 0,1: Reliability curves para GPT-4.1-mini y Gemini-3.8-Flash
for ax, (model_filter, title_m) in zip(
    [axes[0, 0], axes[0, 1]],
    [('gpt-4.1-mini', 'GPT-4.1-mini · Curva de Fiabilidad (10 Bins)'),
     ('gemini-3.8-flash', 'Gemini-3.8-Flash · Curva de Fiabilidad (10 Bins)')]
):
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.2, alpha=0.6, label='Calibración perfecta (y = x)')
    for vid, label, _, _, mod, sh, cfg, _ in VARIANTS:
        if mod != model_filter:
            continue
        bs = calib_bins[vid]
        mask = bs['n'] >= 10
        row_c = df_calib[df_calib['vid'] == vid].iloc[0]
        ax.plot(bs.loc[mask, 'conf'], bs.loc[mask, 'acc'], marker='o', linewidth=2.0,
                color=COLORS[vid], label=f"{sh}-Shot {cfg} (ECE={row_c['ECE']:.3f}, BS={row_c['Brier Multiclase']:.3f})")
    ax.set_xlim(0.25, 1.02); ax.set_ylim(0.05, 1.00)
    ax.set_xlabel('Confianza Media en el Bin (solo bins con n ≥ 10)')
    ax.set_ylabel('Accuracy Empírica en el Bin')
    ax.set_title(title_m)
    ax.grid(linestyle='--', alpha=0.35)
    ax.legend(loc='upper left', fontsize=8.3, frameon=True)

# Panel 1,0: Barras ECE & MCE (bins n >= 10)
ax_ece = axes[1, 0]
x_v = np.arange(len(VARIANTS))
w = 0.36
ax_ece.bar(x_v - w/2, df_calib['ECE'], width=w, color='#1f77b4', label='ECE (10 bins)')
ax_ece.bar(x_v + w/2, df_calib['MCE (bins n>=10)'], width=w, color='#d62728', alpha=0.85, label='MCE (bins n ≥ 10)')
for idx, r in df_calib.iterrows():
    ax_ece.text(idx - w/2, r['ECE'] + 0.008, f"{r['ECE']:.3f}", ha='center', va='bottom', fontsize=7.5, rotation=90)
    ax_ece.text(idx + w/2, r['MCE (bins n>=10)'] + 0.008, f"{r['MCE (bins n>=10)']:.3f}", ha='center', va='bottom', fontsize=7.5, rotation=90)
ax_ece.set_xticks(x_v)
ax_ece.set_xticklabels([f"{r['Modelo'].split('-')[0].upper()}\\n{r['Shots']}s-{r['Config']}" for _, r in df_calib.iterrows()], fontsize=8.5)
ax_ece.set_ylim(0, max(df_calib['MCE (bins n>=10)']) * 1.25)
ax_ece.set_ylabel('Error de Calibración')
ax_ece.set_title('ECE y MCE (Impacto de Few-Shot ARQ-09)')
ax_ece.grid(axis='y', linestyle='--', alpha=0.35)
ax_ece.legend(loc='upper right', fontsize=8.5, frameon=True)

# Panel 1,1: Brier Score Multiclase + Brecha de Sobreconfianza
ax_bs = axes[1, 1]
ax_bs.bar(x_v - w/2, df_calib['Brier Multiclase'], width=w, color='#2ca02c', label='Brier Score Multiclase')
ax_bs.bar(x_v + w/2, df_calib['Sobreconfianza (Conf-Acc)'], width=w, color='#ff7f0e', label='Sobreconfianza (Conf − Acc)')
for idx, r in df_calib.iterrows():
    ax_bs.text(idx - w/2, r['Brier Multiclase'] + 0.015, f"{r['Brier Multiclase']:.3f}", ha='center', va='bottom', fontsize=7.5, rotation=90)
    ax_bs.text(idx + w/2, r['Sobreconfianza (Conf-Acc)'] + 0.015, f"{r['Sobreconfianza (Conf-Acc)']:+.3f}", ha='center', va='bottom', fontsize=7.5, rotation=90)
ax_bs.set_xticks(x_v)
ax_bs.set_xticklabels([f"{r['Modelo'].split('-')[0].upper()}\\n{r['Shots']}s-{r['Config']}" for _, r in df_calib.iterrows()], fontsize=8.5)
ax_bs.set_ylim(0, max(df_calib['Brier Multiclase']) * 1.25)
ax_bs.set_ylabel('Puntaje')
ax_bs.set_title('Brier Score Multiclase y Brecha de Sobreconfianza')
ax_bs.grid(axis='y', linestyle='--', alpha=0.35)
ax_bs.legend(loc='upper right', fontsize=8.5, frameon=True)

fig.tight_layout()
fig.savefig(FIG_DIR / 'fig05_calibracion_curvas_fiabilidad.png', bbox_inches='tight')
plt.show()
"""))

    cells.append(_code("""
# Figura 6: Verificación empírica de ARQ-09 §5.1 (Anclaje discreto 0-Shot vs. Few-Shot + Confianza en Aciertos vs. Errores)
fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))

# Panel A: Top valores discretos de confidence en GPT-4.1-mini Config B (0-Shot vs 9-Shot) y Gemini Config A
ax = axes[0]
bins_h = np.linspace(0.10, 1.01, 37)
for vid, lbl in [
    ('gpt41_zs_b',  'GPT-4.1-mini 0-Shot B (anclaje discreto)'),
    ('gpt41_fs_b',  'GPT-4.1-mini 9-Shot B (ARQ-09)'),
    ('gemini_zs_a', 'Gemini 0-Shot A'),
    ('gemini_fs_a', 'Gemini 9-Shot A (ARQ-09)'),
]:
    vals = df_all[df_all['vid'] == vid]['confidence'].to_numpy()
    ax.hist(vals, bins=bins_h, histtype='step', linewidth=2.0, density=False, color=COLORS[vid], label=lbl)
ax.set_xlabel('Confianza Top-1 verbalizada (`confidence`)')
ax.set_ylabel('Frecuencia de instancias (N = 2,798)')
ax.set_title('Anclaje Numérico Discreto (0-Shot) vs. Desplazamiento 9-Shot (ARQ-09 §5.1)')
ax.grid(linestyle='--', alpha=0.35)
ax.legend(loc='upper left', fontsize=8.2, frameon=True)

# Panel B: Confianza promedio en Aciertos vs Errores (las 8 variantes)
ax = axes[1]
x_v = np.arange(len(VARIANTS))
ok_means = [df_all[(df_all['vid'] == v[0]) & (df_all['is_correct'] == 1)]['confidence'].mean() for v in VARIANTS]
err_means = [df_all[(df_all['vid'] == v[0]) & (df_all['is_correct'] == 0)]['confidence'].mean() for v in VARIANTS]
w = 0.36
ax.bar(x_v - w/2, ok_means, width=w, color='#2ca02c', label='Aciertos (is_correct=1)')
ax.bar(x_v + w/2, err_means, width=w, color='#d62728', alpha=0.82, label='Errores (is_correct=0)')
for idx in x_v:
    diff = ok_means[idx] - err_means[idx]
    ax.text(idx, max(ok_means[idx], err_means[idx]) + 0.015, f'Δ={diff:+.3f}', ha='center', fontsize=8, fontweight='bold')
ax.set_xticks(x_v)
ax.set_xticklabels([f"{v[4].split('-')[0].upper()}\\n{v[5]}s-{v[6]}" for v in VARIANTS], fontsize=8.3)
ax.set_ylim(0.55, 0.96)
ax.set_ylabel('Confianza Media')
ax.set_title('Discriminación de Confianza: Aciertos vs. Errores')
ax.grid(axis='y', linestyle='--', alpha=0.35)
ax.legend(loc='lower right', fontsize=8.5, frameon=True)

fig.tight_layout()
fig.savefig(FIG_DIR / 'fig06_anclaje_numerico_y_confianza_acierto_error.png', bbox_inches='tight')
plt.show()
"""))

    # ------------------------------------------------------------------
    # Sección 6: Eje 2 - Latencia y Colas
    # ------------------------------------------------------------------
    cells.append(_md(r"""
## 6. Eje 2: Latencia y Análisis de Colas (`latency_ms`)

Evaluación no paramétrica completa de `latency_ms` por modelo, configuración y modo (`p50`, `p95`, `p99`, media, desviación estándar, rango intercuartílico y análisis de *outliers* extremos $>5\text{ s}$, $>10\text{ s}$ y $>30\text{ s}$).
"""))

    cells.append(_code("""
lat_rows = []
for vid, rinfo in runs.items():
    arr = df_all[df_all['vid'] == vid]['latency_ms'].to_numpy()
    lat_rows.append({
        'Variante': rinfo['label'],
        'Media (ms)': float(arr.mean()),
        'Std (ms)': float(arr.std(ddof=1)),
        'Mín (ms)': float(arr.min()),
        'p25 (ms)': float(np.percentile(arr, 25)),
        'p50 (ms)': float(np.percentile(arr, 50)),
        'p75 (ms)': float(np.percentile(arr, 75)),
        'p95 (ms)': float(np.percentile(arr, 95)),
        'p99 (ms)': float(np.percentile(arr, 99)),
        'Máx (ms)': float(arr.max()),
        '>5s (#)': int((arr > 5000).sum()),
        '>10s (#)': int((arr > 10000).sum()),
        '>30s (#)': int((arr > 30000).sum()),
    })

df_lat = pd.DataFrame(lat_rows)
display(df_lat.style.format({
    'Media (ms)': '{:.1f}', 'Std (ms)': '{:.1f}', 'Mín (ms)': '{:.1f}',
    'p25 (ms)': '{:.1f}', 'p50 (ms)': '{:.1f}', 'p75 (ms)': '{:.1f}',
    'p95 (ms)': '{:.1f}', 'p99 (ms)': '{:.1f}', 'Máx (ms)': '{:.1f}',
}))

print('\\nTop 10 outliers absolutos de latencia en los 22,384 llamados:')
display(df_all.nlargest(10, 'latency_ms')[
    ['Variante', 'instance_id', 'latency_ms', 'input_tokens', 'cached_input_tokens', 'thoughts_tokens', 'output_tokens', 'is_valid_format', 'y_true', 'y_pred']
].reset_index(drop=True))

# Figura 7: ECDF logarítmica + Boxplot/Percentiles de latencia
fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.4))

ax0 = axes[0]
for vid, label, *_ in VARIANTS:
    arr = np.sort(df_all[df_all['vid'] == vid]['latency_ms'].to_numpy())
    ecdf = np.arange(1, len(arr) + 1) / len(arr)
    ax0.plot(arr, ecdf, linewidth=1.8, color=COLORS[vid], label=label)
for ref_p, rlbl in [(0.50, 'p50'), (0.95, 'p95'), (0.99, 'p99')]:
    ax0.axhline(ref_p, color='gray', linestyle=':', linewidth=1.0)
    ax0.text(460, ref_p + 0.01, rlbl, fontsize=8.5, color='#444')
ax0.axvline(2000, color='#555', linestyle='--', linewidth=0.9, alpha=0.6)
ax0.axvline(10000, color='#b2182b', linestyle='--', linewidth=0.9, alpha=0.6)
ax0.set_xscale('log')
ax0.set_xlim(400, 65000)
ax0.set_xlabel('Latencia por instancia (ms, escala log)')
ax0.set_ylabel('Probabilidad Acumulada (ECDF)')
ax0.set_title('ECDF de Latencia y Colas (p50 / p95 / p99)')
ax0.grid(True, which='both', linestyle='--', alpha=0.3)
ax0.legend(loc='lower right', fontsize=8.0, frameon=True)

ax1 = axes[1]
data_bp = [df_all[df_all['vid'] == v[0]]['latency_ms'].to_numpy() for v in VARIANTS]
bp = ax1.boxplot(data_bp, vert=True, patch_artist=True, showfliers=False, widths=0.55)
for patch, (vid, *_) in zip(bp['boxes'], VARIANTS):
    patch.set_facecolor(COLORS[vid]); patch.set_alpha(0.65)
for idx, r in df_lat.iterrows():
    ax1.scatter([idx + 1], [r['p95 (ms)']], color='#d62728', marker='^', s=45, zorder=4, label='p95' if idx == 0 else '')
    ax1.scatter([idx + 1], [r['p99 (ms)']], color='black', marker='x', s=45, zorder=4, label='p99' if idx == 0 else '')
ax1.set_yscale('log')
ax1.set_xticks(range(1, len(VARIANTS) + 1))
ax1.set_xticklabels([f"{v[4].split('-')[0].upper()}\\n{v[5]}s-{v[6]}" for v in VARIANTS], fontsize=8.3)
ax1.set_ylabel('Latencia (ms, escala log)')
ax1.set_title('Distribución IQR (cajas sin fliers) + Marcadores p95 y p99')
ax1.grid(axis='y', which='both', linestyle='--', alpha=0.3)
ax1.legend(loc='upper left', fontsize=8.5, frameon=True)

fig.tight_layout()
fig.savefig(FIG_DIR / 'fig07_latencia_ecdf_y_colas.png', bbox_inches='tight')
plt.show()
"""))

    # ------------------------------------------------------------------
    # Sección 7: Eje 3 - Tokens, Prompt Caching y Costos
    # ------------------------------------------------------------------
    cells.append(_md("""
## 7. Eje 3: Tokens, Eficiencia de *Prompt Caching* y Relación con Costo / Latencia

Se comparan las distribuciones de `input_tokens`, `cached_input_tokens` y `output_tokens`, el contraste drástico entre el cacheo automático de prefijos de OpenAI (~83–84% de tokens en 9-Shot) y el cacheo implícito oportunista de Gemini (~4.3–13.4% de tokens en 9-Shot), el ahorro real en USD y el efecto empírico del caché sobre `latency_ms` dentro de las variantes 9-Shot.
"""))

    cells.append(_code("""
tok_rows = []
for vid, rinfo in runs.items():
    sub = df_all[df_all['vid'] == vid]
    hit_mask = sub['cached_input_tokens'] > 0
    hits = sub[hit_mask]
    misses = sub[~hit_mask]
    p_u = np.nan
    if len(hits) > 0 and len(misses) > 0:
        _, p_u = stats.mannwhitneyu(hits['latency_ms'], misses['latency_ms'], alternative='two-sided')
    tok_rows.append({
        'Variante': rinfo['label'],
        'In Media': sub['input_tokens'].mean(),
        'In p50': sub['input_tokens'].median(),
        'In p95': np.percentile(sub['input_tokens'], 95),
        'Inst Cache > 0 (%)': hit_mask.mean() * 100.0,
        '% Tokens Cacheados': sub['cached_input_tokens'].sum() / sub['input_tokens'].sum() * 100.0,
        'Out Media': sub['output_tokens'].mean(),
        'Lat Hit p50 (ms)': hits['latency_ms'].median() if len(hits) else np.nan,
        'Lat Miss p50 (ms)': misses['latency_ms'].median() if len(misses) else np.nan,
        'Mann-Whitney p (Hit vs Miss)': p_u,
        'Costo Bruto ($)': df_summary.loc[df_summary['vid'] == vid, 'Costo Bruto ($)'].iloc[0],
        'Costo Neto ($)': df_summary.loc[df_summary['vid'] == vid, 'Costo Neto ($)'].iloc[0],
        'Ahorro (%)': (1.0 - df_summary.loc[df_summary['vid'] == vid, 'Costo Neto ($)'].iloc[0] /
                       df_summary.loc[df_summary['vid'] == vid, 'Costo Bruto ($)'].iloc[0]) * 100.0,
    })

df_tok = pd.DataFrame(tok_rows)
display(df_tok.style.format({
    'In Media': '{:.1f}', 'In p50': '{:.0f}', 'In p95': '{:.0f}',
    'Inst Cache > 0 (%)': '{:.2f}%', '% Tokens Cacheados': '{:.2f}%', 'Out Media': '{:.1f}',
    'Lat Hit p50 (ms)': '{:.1f}', 'Lat Miss p50 (ms)': '{:.1f}',
    'Mann-Whitney p (Hit vs Miss)': '{:.3e}',
    'Costo Bruto ($)': '${:.4f}', 'Costo Neto ($)': '${:.4f}', 'Ahorro (%)': '{:.2f}%',
}))

# Figura 8: Tokens cacheados vs no cacheados + Costo Bruto/Neto + Latencia Cache Hit vs Miss
fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2))

# Panel A: Millones de tokens (Uncached vs Cached vs Output+Thoughts)
ax0 = axes[0]
x_v = np.arange(len(VARIANTS))
uncached_m = [(df_all[df_all['vid'] == v[0]]['uncached_input_tokens'].sum()) / 1e6 for v in VARIANTS]
cached_m   = [(df_all[df_all['vid'] == v[0]]['cached_input_tokens'].sum()) / 1e6 for v in VARIANTS]
out_th_m   = [((df_all[df_all['vid'] == v[0]]['output_tokens'].sum() + df_all[df_all['vid'] == v[0]]['thoughts_tokens'].sum()) / 1e6) for v in VARIANTS]
ax0.bar(x_v, uncached_m, color='#d95f02', label='Input no cacheado (M)')
ax0.bar(x_v, cached_m, bottom=uncached_m, color='#1b9e77', label='Input cacheado (M)')
ax0.bar(x_v, out_th_m, bottom=np.array(uncached_m)+np.array(cached_m), color='#7570b3', label='Output + Thoughts (M)')
for i, pct in enumerate(df_tok['% Tokens Cacheados']):
    tot = uncached_m[i] + cached_m[i] + out_th_m[i]
    if pct > 0.5:
        ax0.text(i, tot + 0.35, f'{pct:.1f}%\\ncache', ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#1b9e77')
ax0.set_xticks(x_v)
ax0.set_xticklabels([f"{v[4].split('-')[0].upper()}\\n{v[5]}s-{v[6]}" for v in VARIANTS], fontsize=8)
ax0.set_ylabel('Millones de Tokens (N = 2,798)')
ax0.set_title('Descomposición de Tokens y Tasa de Caché')
ax0.set_ylim(0, max(np.array(uncached_m)+np.array(cached_m)+np.array(out_th_m))*1.18)
ax0.grid(axis='y', linestyle='--', alpha=0.35)
ax0.legend(loc='upper left', fontsize=8.0, frameon=True)

# Panel B: Costo Bruto vs Neto USD
ax1 = axes[1]
w = 0.36
ax1.bar(x_v - w/2, df_tok['Costo Bruto ($)'], width=w, color='#a6bddb', edgecolor='#2b5c8f', label='Costo Bruto (sin caché)')
ax1.bar(x_v + w/2, df_tok['Costo Neto ($)'],  width=w, color='#1f77b4', label='Costo Neto (con caché)')
for idx, r in df_tok.iterrows():
    ax1.text(idx + w/2, r['Costo Neto ($)'] + 0.25, f"${r['Costo Neto ($)']:.2f}", ha='center', va='bottom', fontsize=7.5, fontweight='bold')
ax1.set_xticks(x_v)
ax1.set_xticklabels([f"{v[4].split('-')[0].upper()}\\n{v[5]}s-{v[6]}" for v in VARIANTS], fontsize=8)
ax1.set_ylabel('Costo por corrida completa (USD)')
ax1.set_title('Impacto Económico del Prompt Caching')
ax1.set_ylim(0, df_tok['Costo Bruto ($)'].max() * 1.18)
ax1.grid(axis='y', linestyle='--', alpha=0.35)
ax1.legend(loc='upper left', fontsize=8.0, frameon=True)

# Panel C: Latencia Hits vs Misses en las 4 variantes 9-Shot
ax2 = axes[2]
fs_vids = ['gpt41_fs_a', 'gpt41_fs_b', 'gemini_fs_a', 'gemini_fs_b']
pos = np.arange(len(fs_vids))
hit_meds, miss_meds = [], []
for v in fs_vids:
    sub = df_all[df_all['vid'] == v]
    hit_meds.append(sub[sub['cached_input_tokens'] > 0]['latency_ms'].median())
    miss_meds.append(sub[sub['cached_input_tokens'] == 0]['latency_ms'].median())
ax2.bar(pos - w/2, miss_meds, width=w, color='#e34a33', alpha=0.85, label='Cache Miss (cached=0) p50')
ax2.bar(pos + w/2, hit_meds,  width=w, color='#31a354', alpha=0.88, label='Cache Hit (cached>0) p50')
for i in pos:
    ax2.text(i - w/2, miss_meds[i] + 40, f'{miss_meds[i]:.0f}', ha='center', fontsize=8)
    ax2.text(i + w/2, hit_meds[i]  + 40, f'{hit_meds[i]:.0f}',  ha='center', fontsize=8)
ax2.set_xticks(pos)
ax2.set_xticklabels(['GPT 9s-A', 'GPT 9s-B', 'Gem 9s-A', 'Gem 9s-B'], fontsize=8.5)
ax2.set_ylabel('Latencia Mediana p50 (ms)')
ax2.set_title('Efecto del Caché en Latencia Mediana (9-Shot)')
ax2.set_ylim(0, max(max(hit_meds), max(miss_meds)) * 1.18)
ax2.grid(axis='y', linestyle='--', alpha=0.35)
ax2.legend(loc='upper left', fontsize=8.0, frameon=True)

fig.tight_layout()
fig.savefig(FIG_DIR / 'fig08_tokens_cache_eficiencia_y_costos.png', bbox_inches='tight')
plt.show()
"""))

    # ------------------------------------------------------------------
    # Sección 8: Eje 4 - Análisis de Razonamiento (thoughts_tokens)
    # ------------------------------------------------------------------
    cells.append(_md("""
## 8. Eje 4: Análisis de Razonamiento (`thoughts_tokens > 0` vs. `thoughts_tokens == 0`)

Se cuantifica en qué modelos, configuraciones y modos se activa el canal de razonamiento interno (`thoughts_tokens > 0`), su supresión casi total al pasar de 0-Shot a 9-Shot en Gemini-3.8-Flash (`thinking_level="LOW"`), y se evalúa si la activación ocurre en casos más ambiguos o difíciles y si mejora o penaliza la precisión, confianza y latencia. Se reportan explícitamente los tamaños de subgrupo $n$ para evitar sobreinterpretar muestras diminutas.
"""))

    cells.append(_code("""
# 1. Tabla cuantitativa de activación thoughts_tokens por variante
th_summary = []
for vid, rinfo in runs.items():
    sub = df_all[df_all['vid'] == vid]
    pos = sub[sub['thoughts_tokens'] > 0]
    n_pos = len(pos)
    th_summary.append({
        'Variante': rinfo['label'],
        'n (thoughts > 0)': n_pos,
        '% Activación': n_pos / N_TEST * 100.0,
        'n (thoughts == 0)': N_TEST - n_pos,
        'Total Thoughts': int(sub['thoughts_tokens'].sum()),
        'Media | >0': float(pos['thoughts_tokens'].mean()) if n_pos else 0.0,
        'p50 | >0': float(pos['thoughts_tokens'].median()) if n_pos else 0.0,
        'p95 | >0': float(np.percentile(pos['thoughts_tokens'], 95)) if n_pos else 0.0,
        'Máx': int(sub['thoughts_tokens'].max()),
    })
display(pd.DataFrame(th_summary).style.format({
    '% Activación': '{:.2f}%', 'Media | >0': '{:.1f}', 'p50 | >0': '{:.1f}', 'p95 | >0': '{:.1f}'
}))

# 2. Comparativa Macro-F1, Accuracy, Confianza y Latencia p50/p95: thoughts > 0 vs == 0 en las 4 variantes Gemini
th_comp_rows = []
for vid in ['gemini_zs_a', 'gemini_zs_b', 'gemini_fs_a', 'gemini_fs_b']:
    sub = df_all[df_all['vid'] == vid]
    lbl = runs[vid]['label']
    g_pos = sub[sub['thoughts_tokens'] > 0]
    g_zer = sub[sub['thoughts_tokens'] == 0]
    # Test de diferencia de proporciones de acierto (Fisher/Chi2) y latencia (Mann-Whitney)
    ct = [[int(g_pos['is_correct'].sum()), int(len(g_pos) - g_pos['is_correct'].sum())],
          [int(g_zer['is_correct'].sum()), int(len(g_zer) - g_zer['is_correct'].sum())]]
    _, p_acc = stats.fisher_exact(ct)
    _, p_lat = stats.mannwhitneyu(g_pos['latency_ms'], g_zer['latency_ms'], alternative='two-sided')
    for grp_name, g in [('thoughts > 0', g_pos), ('thoughts == 0', g_zer)]:
        # Nota: macro_f1 sobre las 9 clases; advertencia si el subgrupo tiene clases con soporte 0
        present_cls = sorted(g['y_true_idx'].unique().tolist())
        f1_all9 = f1_score(g['y_true_idx'], g['y_pred_idx'], labels=list(range(9)), average='macro', zero_division=0)
        f1_pres = f1_score(g['y_true_idx'], g['y_pred_idx'], labels=present_cls, average='macro', zero_division=0) if present_cls else 0.0
        th_comp_rows.append({
            'Variante': lbl,
            'Subgrupo': grp_name,
            'n': len(g),
            'Clases Pres.': len(present_cls),
            'Accuracy': g['is_correct'].mean(),
            'Macro-F1 (9)': f1_all9,
            'Macro-F1 (pres.)': f1_pres,
            'Conf Media': g['confidence'].mean(),
            'Input Med (tok)': g['input_tokens'].mean(),
            'Lat p50 (ms)': g['latency_ms'].median(),
            'Lat p95 (ms)': np.percentile(g['latency_ms'], 95),
            'p Accuracy (>0 vs =0)': p_acc,
            'p Latencia (>0 vs =0)': p_lat,
        })

df_th_comp = pd.DataFrame(th_comp_rows)
display(df_th_comp.style.format({
    'Accuracy': '{:.4f}', 'Macro-F1 (9)': '{:.4f}', 'Macro-F1 (pres.)': '{:.4f}',
    'Conf Media': '{:.4f}', 'Input Med (tok)': '{:.1f}',
    'Lat p50 (ms)': '{:.1f}', 'Lat p95 (ms)': '{:.1f}',
    'p Accuracy (>0 vs =0)': '{:.3e}', 'p Latencia (>0 vs =0)': '{:.3e}',
}))

# 3. Cruce por clase real, panel_agreement y longitud en Gemini Zero-Shot (donde hay suficiente n)
gem_zs = df_all[df_all['vid'].isin(['gemini_zs_a', 'gemini_zs_b'])].copy()
gem_zs['has_thoughts'] = (gem_zs['thoughts_tokens'] > 0).astype(int)

cls_th = []
for c in CANONICAL_9_CLASSES:
    sc = gem_zs[gem_zs['y_true'] == c]
    sc_pos = sc[sc['has_thoughts'] == 1]
    sc_zer = sc[sc['has_thoughts'] == 0]
    cls_th.append({
        'Clase Gold': c,
        'n total (A+B 0s)': len(sc),
        'n thoughts>0': len(sc_pos),
        '% thoughts>0': sc['has_thoughts'].mean() * 100.0,
        'Acc (th>0)': sc_pos['is_correct'].mean() if len(sc_pos) else np.nan,
        'Acc (th==0)': sc_zer['is_correct'].mean() if len(sc_zer) else np.nan,
        'Δ Acc (th>0 - th==0)': (sc_pos['is_correct'].mean() - sc_zer['is_correct'].mean()) if (len(sc_pos) and len(sc_zer)) else np.nan,
    })
df_cls_th = pd.DataFrame(cls_th).sort_values('% thoughts>0', ascending=False).reset_index(drop=True)
display(df_cls_th.style.format({
    '% thoughts>0': '{:.1f}%', 'Acc (th>0)': '{:.3f}', 'Acc (th==0)': '{:.3f}', 'Δ Acc (th>0 - th==0)': '{:+.3f}'
}))

# Cruce por consenso A2 (panel_agreement)
gem_zs['agree_bucket'] = np.where(
    gem_zs['panel_agreement'] >= 0.99, 'Unánime (1.0)',
    np.where(gem_zs['panel_agreement'] >= 0.60, 'Mayoría (0.67)', 'Empate (<0.5)')
)
agree_th = gem_zs.groupby(['agree_bucket', 'has_thoughts']).agg(
    n=('is_correct', 'size'),
    acc=('is_correct', 'mean'),
    conf=('confidence', 'mean'),
    lat_p50=('latency_ms', 'median'),
).reset_index()
display(agree_th)
"""))

    cells.append(_code("""
# Figura 9: Análisis integral de thoughts_tokens
fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.4))

# Panel A: Tasa de activación thoughts > 0 por variante Gemini y distribución condicional en 0-Shot
ax0 = axes[0]
gem_vids = ['gemini_zs_a', 'gemini_zs_b', 'gemini_fs_a', 'gemini_fs_b']
act_pcts = [df_all[df_all['vid'] == v]['thoughts_tokens'].gt(0).mean() * 100.0 for v in gem_vids]
act_ns   = [int(df_all[df_all['vid'] == v]['thoughts_tokens'].gt(0).sum()) for v in gem_vids]
bars = ax0.bar(range(len(gem_vids)), act_pcts, color=[COLORS[v] for v in gem_vids], alpha=0.88)
for i, (pct, n_act) in enumerate(zip(act_pcts, act_ns)):
    ax0.text(i, pct + 1.4, f'{pct:.2f}%\\n(n={n_act:,})', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
ax0.set_xticks(range(len(gem_vids)))
ax0.set_xticklabels(['Gem 0s-A', 'Gem 0s-B', 'Gem 9s-A', 'Gem 9s-B'], fontsize=9)
ax0.set_ylabel('% Instancias con thoughts_tokens > 0')
ax0.set_ylim(0, max(act_pcts) * 1.24)
ax0.set_title('Supresión del Razonamiento en 9-Shot (-99%)')
ax0.grid(axis='y', linestyle='--', alpha=0.35)

# Panel B: Accuracy y Latencia p50 en Gemini 0-Shot A y B (thoughts > 0 vs == 0)
ax1 = axes[1]
sub_pairs = [
    ('0s-A\\nth==0', 'gemini_zs_a', 0),
    ('0s-A\\nth>0',  'gemini_zs_a', 1),
    ('0s-B\\nth==0', 'gemini_zs_b', 0),
    ('0s-B\\nth>0',  'gemini_zs_b', 1),
]
accs_p, lats_p, ns_p = [], [], []
for _, v, flag in sub_pairs:
    s = df_all[(df_all['vid'] == v) & ((df_all['thoughts_tokens'] > 0).astype(int) == flag)]
    accs_p.append(s['is_correct'].mean() * 100.0)
    lats_p.append(s['latency_ms'].median() / 1000.0)
    ns_p.append(len(s))

xp = np.arange(len(sub_pairs))
b1 = ax1.bar(xp - 0.18, accs_p, width=0.35, color='#2b5c8f', label='Accuracy (%)')
ax1_r = ax1.twinx()
b2 = ax1_r.bar(xp + 0.18, lats_p, width=0.35, color='#e6550d', alpha=0.82, label='Latencia p50 (s)')
for i in xp:
    ax1.text(i - 0.18, accs_p[i] + 1.0, f'{accs_p[i]:.1f}%', ha='center', fontsize=8, color='#2b5c8f', fontweight='bold')
    ax1_r.text(i + 0.18, lats_p[i] + 0.12, f'{lats_p[i]:.2f}s', ha='center', fontsize=8, color='#a63603', fontweight='bold')
ax1.set_xticks(xp)
ax1.set_xticklabels([f'{lbl}\\n(n={n})' for (lbl, _, _), n in zip(sub_pairs, ns_p)], fontsize=8.3)
ax1.set_ylabel('Accuracy (%)', color='#2b5c8f')
ax1_r.set_ylabel('Latencia Mediana p50 (s)', color='#e6550d')
ax1.set_ylim(25, 62); ax1_r.set_ylim(0, 6.2)
ax1.set_title('Desempeño vs. Costo Temporal de thoughts > 0')
ax1_r.spines['top'].set_visible(False)

# Panel C: Tasa de activación y diferencia Accuracy por clase real (Gemini 0-Shot A+B)
ax2 = axes[2]
yc = np.arange(len(df_cls_th))
ax2.barh(yc, df_cls_th['% thoughts>0'], color='#9467bd', alpha=0.85)
for i, r in df_cls_th.iterrows():
    d_str = f"{r['Δ Acc (th>0 - th==0)']:+.3f}" if pd.notna(r['Δ Acc (th>0 - th==0)']) else 'n/a'
    ax2.text(r['% thoughts>0'] + 0.8, i, f"{r['% thoughts>0']:.1f}% (Δ={d_str})",
             va='center', fontsize=8)
ax2.set_yticks(yc)
ax2.set_yticklabels(df_cls_th['Clase Gold'], fontsize=8.8)
ax2.invert_yaxis()
ax2.set_xlim(0, max(df_cls_th['% thoughts>0']) * 1.48)
ax2.set_xlabel('% Activación thoughts_tokens > 0 (Gemini 0-Shot A+B)')
ax2.set_title('Activación y ΔAccuracy por Clase Canónica')
ax2.grid(axis='x', linestyle='--', alpha=0.35)

fig.tight_layout()
fig.savefig(FIG_DIR / 'fig09_razonamiento_thoughts_subgrupos.png', bbox_inches='tight')
plt.show()
"""))

    # ------------------------------------------------------------------
    # Sección 9: Eje 5 - Correlaciones Multivariadas
    # ------------------------------------------------------------------
    cells.append(_md("""
## 9. Eje 5: Análisis de Correlación Multivariada (Spearman $\\rho$ y Pearson $r$)

Dado que las variables de caché, razonamiento y latencia presentan comportamientos estructuralmente distintos por régimen (`0-Shot` vs `9-Shot`, OpenAI vs Gemini), calcular correlaciones únicamente sobre la mezcla heterogénea induciría falacias de Simpson. Por ello calculamos **Spearman $\\rho$** y **Pearson $r$** (con *p-values* exactos) tanto en los regímenes clave por separado como en el consolidado, enfocándonos en `input_tokens`, `cached_input_tokens`, `output_tokens`, `thoughts_tokens`, `latency_ms`, `confidence` y `is_correct`.
"""))

    cells.append(_code("""
CORR_VARS = ['input_tokens', 'cached_input_tokens', 'output_tokens', 'thoughts_tokens', 'latency_ms', 'confidence', 'is_correct']
CORR_SHORT = ['in_tok', 'cache_tok', 'out_tok', 'think_tok', 'latency_ms', 'confidence', 'is_correct']

def compute_corr_pair(df_sub, vars_list):
    k = len(vars_list)
    rho_m = np.eye(k)
    r_m = np.eye(k)
    p_rho_m = np.zeros((k, k))
    p_r_m = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            x = df_sub[vars_list[i]].to_numpy(dtype=float)
            y = df_sub[vars_list[j]].to_numpy(dtype=float)
            if np.std(x) == 0 or np.std(y) == 0:
                rho_m[i, j] = np.nan; r_m[i, j] = np.nan
                p_rho_m[i, j] = np.nan; p_r_m[i, j] = np.nan
            else:
                rho, prho = stats.spearmanr(x, y)
                r, pr = stats.pearsonr(x, y)
                rho_m[i, j] = rho; p_rho_m[i, j] = prho
                r_m[i, j] = r; p_r_m[i, j] = pr
    return (
        pd.DataFrame(rho_m, index=CORR_SHORT, columns=CORR_SHORT),
        pd.DataFrame(r_m, index=CORR_SHORT, columns=CORR_SHORT),
        pd.DataFrame(p_rho_m, index=CORR_SHORT, columns=CORR_SHORT),
        pd.DataFrame(p_r_m, index=CORR_SHORT, columns=CORR_SHORT),
    )

cohorts = [
    ('Global A5 (N=22,384)', df_all),
    ('Gemini 0-Shot A+B (n=5,596, thoughts activos)', df_all[df_all['vid'].isin(['gemini_zs_a', 'gemini_zs_b'])]),
    ('Gemini 9-Shot A+B (n=5,596, caché oportunista)', df_all[df_all['vid'].isin(['gemini_fs_a', 'gemini_fs_b'])]),
    ('GPT-4.1-mini 9-Shot A+B (n=5,596, caché 84%)', df_all[df_all['vid'].isin(['gpt41_fs_a', 'gpt41_fs_b'])]),
]

corr_results = {}
for cname, df_c in cohorts:
    rho_df, r_df, prho_df, pr_df = compute_corr_pair(df_c, CORR_VARS)
    corr_results[cname] = (rho_df, r_df, prho_df, pr_df)
    print(f'\\n=== Correlaciones: {cname} ===')
    print('Spearman ρ:')
    display(rho_df.style.format('{:+.3f}', na_rep='const').background_gradient(cmap='RdBu_r', vmin=-1, vmax=1))
    print('Pearson r:')
    display(r_df.style.format('{:+.3f}', na_rep='const').background_gradient(cmap='RdBu_r', vmin=-1, vmax=1))

# Imprimir top correlaciones significativas clave con p-values exactos
print('\\nCorrelaciones bivariadas clave seleccionadas (r, ρ, p-valor):')
key_pairs = [
    ('Gemini 0-Shot A+B (n=5,596, thoughts activos)', 'thoughts_tokens', 'latency_ms'),
    ('Gemini 0-Shot A+B (n=5,596, thoughts activos)', 'thoughts_tokens', 'is_correct'),
    ('Gemini 0-Shot A+B (n=5,596, thoughts activos)', 'thoughts_tokens', 'confidence'),
    ('Gemini 0-Shot A+B (n=5,596, thoughts activos)', 'thoughts_tokens', 'input_tokens'),
    ('Gemini 0-Shot A+B (n=5,596, thoughts activos)', 'confidence', 'is_correct'),
    ('GPT-4.1-mini 9-Shot A+B (n=5,596, caché 84%)', 'cached_input_tokens', 'latency_ms'),
    ('GPT-4.1-mini 9-Shot A+B (n=5,596, caché 84%)', 'input_tokens', 'latency_ms'),
    ('GPT-4.1-mini 9-Shot A+B (n=5,596, caché 84%)', 'confidence', 'is_correct'),
    ('Global A5 (N=22,384)', 'input_tokens', 'latency_ms'),
    ('Global A5 (N=22,384)', 'confidence', 'is_correct'),
]
for cname, v1, v2 in key_pairs:
    df_c = dict(cohorts)[cname]
    rho, prho = stats.spearmanr(df_c[v1], df_c[v2])
    r, pr = stats.pearsonr(df_c[v1], df_c[v2])
    print(f' • {cname[:26]:<26} | {v1} vs {v2:<18}: ρ={rho:+.4f} (p={prho:.2e}) | r={r:+.4f} (p={pr:.2e})')
"""))

    cells.append(_code("""
# Figura 10: Heatmaps Spearman vs Pearson + Hexbin/Scatter clave
fig = plt.figure(figsize=(18.2, 10.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.0], wspace=0.38, hspace=0.40)

# Fila 1: Spearman ρ (Gemini 0s), Pearson r (Gemini 0s), Spearman ρ (Global)
panels_hm = [
    (gs[0, 0], corr_results['Gemini 0-Shot A+B (n=5,596, thoughts activos)'][0], 'Spearman ρ · Gemini 0-Shot (n=5,596)'),
    (gs[0, 1], corr_results['Gemini 0-Shot A+B (n=5,596, thoughts activos)'][1], 'Pearson r · Gemini 0-Shot (n=5,596)'),
    (gs[0, 2], corr_results['Global A5 (N=22,384)'][0],                           'Spearman ρ · Global A5 (N=22,384)'),
]
for idx_hm, (gspec, df_m, ttl) in enumerate(panels_hm):
    ax = fig.add_subplot(gspec)
    arr_m = df_m.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(arr_m)
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color='#f0f0f0')
    im = ax.imshow(masked, cmap=cmap, vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(CORR_SHORT))); ax.set_yticks(range(len(CORR_SHORT)))
    ax.set_xticklabels(CORR_SHORT, rotation=35, ha='right', fontsize=8.3)
    ax.set_yticklabels(CORR_SHORT if idx_hm == 0 else [], fontsize=8.3)
    ax.set_title(ttl)
    for i in range(len(CORR_SHORT)):
        for j in range(len(CORR_SHORT)):
            v = arr_m[i, j]
            txt = 'const' if np.isnan(v) else f'{v:+.2f}'
            col = '#888' if np.isnan(v) else ('white' if abs(v) > 0.55 else 'black')
            ax.text(j, i, txt, ha='center', va='center', fontsize=7.5, color=col)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)

# Fila 2: 3 scatters/hexbin clave
# (a) Gemini 0s: thoughts_tokens vs latency_ms
ax_s1 = fig.add_subplot(gs[1, 0])
df_g0 = dict(cohorts)['Gemini 0-Shot A+B (n=5,596, thoughts activos)']
sub_s1 = df_g0[df_g0['latency_ms'] <= 15000]
hb1 = ax_s1.hexbin(sub_s1['thoughts_tokens'], sub_s1['latency_ms'], gridsize=30, cmap='YlOrRd', mincnt=1, bins='log')
rho1, _ = stats.spearmanr(df_g0['thoughts_tokens'], df_g0['latency_ms'])
r1, _   = stats.pearsonr(df_g0['thoughts_tokens'], df_g0['latency_ms'])
ax_s1.set_xlabel('thoughts_tokens')
ax_s1.set_ylabel('latency_ms (recorte visuales ≤15s)')
ax_s1.set_title(f'Gemini 0s: Thoughts vs Latencia (ρ={rho1:+.3f}, r={r1:+.3f})')
fig.colorbar(hb1, ax=ax_s1, label='log10(conteo)')

# (b) Global: input_tokens vs latency_ms coloreado por familia
ax_s2 = fig.add_subplot(gs[1, 1])
for mod_n, col_m, lbl_m in [('gpt-4.1-mini', '#1f77b4', 'GPT-4.1-mini'), ('gemini-3.8-flash', '#ff7f0e', 'Gemini-3.8-Flash')]:
    sub_m = df_all[(df_all['Modelo'] == mod_n) & (df_all['latency_ms'] <= 15000)]
    ax_s2.scatter(sub_m['input_tokens'], sub_m['latency_ms'], s=5, alpha=0.18, color=col_m, label=lbl_m)
ax_s2.set_xlabel('input_tokens')
ax_s2.set_ylabel('latency_ms (≤15s)')
ax_s2.set_title('Longitud Entrada vs Latencia por Proveedor')
ax_s2.grid(linestyle='--', alpha=0.3)
leg = ax_s2.legend(loc='upper right', fontsize=8.5, frameon=True)
for lh in leg.legend_handles:
    lh.set_alpha(1.0); lh.set_sizes([35])

# (c) GPT-4.1-mini 9-Shot: cached_input_tokens vs latency_ms
ax_s3 = fig.add_subplot(gs[1, 2])
df_o9 = dict(cohorts)['GPT-4.1-mini 9-Shot A+B (n=5,596, caché 84%)']
sub_s3 = df_o9[df_o9['latency_ms'] <= 5000]
hb3 = ax_s3.hexbin(sub_s3['cached_input_tokens'], sub_s3['latency_ms'], gridsize=30, cmap='Blues', mincnt=1, bins='log')
rho3, _ = stats.spearmanr(df_o9['cached_input_tokens'], df_o9['latency_ms'])
ax_s3.set_xlabel('cached_input_tokens')
ax_s3.set_ylabel('latency_ms (≤5s)')
ax_s3.set_title(f'GPT 9s: Caché vs Latencia (ρ={rho3:+.3f})')
fig.colorbar(hb3, ax=ax_s3, label='log10(conteo)')

fig.savefig(FIG_DIR / 'fig10_correlaciones_multivariadas_y_scatter.png', bbox_inches='tight')
plt.show()
"""))

    # ------------------------------------------------------------------
    # Sección 10: Eje 6 - Extensiones y Frontera Económica
    # ------------------------------------------------------------------
    cells.append(_md("""
## 10. Eje 6 (Extensiones Contextualizadas): Sensibilidad por Consenso de Jueces A2 y Proyección a Escala ($N = 10{,}000$)

Conectando A5 con los antecedentes del proyecto (`ADR-13`, `ARQ-08` y `ARQ-09`):
1. Estratificamos el acierto de las 8 variantes según el nivel de consenso previo registrado por el Panel de 3 Jueces LLM de A2 (`llm_panel` unánime $1.0$, `llm_panel` mayoría $0.67$, y `needs_review` rescatado desde reglas/NLI), cuantificando cuánto del techo de ~0.50 responde a ambigüedad epistémica del propio corpus de prueba.
2. Proyectamos costo USD y tiempo de inferencia secuencial a escala operativa de **$10{,}000$ citas** en una frontera de Pareto limpia y sin solapamientos.
"""))

    cells.append(_code("""
# 1. Sensibilidad por estrato de consenso previo A2 / origen en test.jsonl
def agreement_stratum(row):
    ls = row['label_source']
    pa = row['panel_agreement']
    if ls == 'human_gold':
        return 'Human Gold (MultiCite)'
    if ls == 'llm_panel' and pa is not None and pa >= 0.99:
        return 'Panel Unánime (3/3)'
    if ls == 'llm_panel':
        return 'Panel Mayoría (2/3)'
    return 'Needs Review (desacuerdo)'

strata_order = [
    'Human Gold (MultiCite)',
    'Panel Unánime (3/3)',
    'Panel Mayoría (2/3)',
    'Needs Review (desacuerdo)',
]
gold_df['stratum'] = gold_df.apply(agreement_stratum, axis=1)
df_all['stratum'] = df_all['test_idx'].map(gold_df['stratum'])
stratum_counts = gold_df['stratum'].value_counts()
print('Estratos de origen / consenso A2 en test.jsonl (N=2,798):')
for k in strata_order:
    v = int(stratum_counts[k])
    print(f' • {k:<28}: n={v:,} ({v/N_TEST*100:.2f}%)')

strat_rows = []
for sname in strata_order:
    n_s = int(stratum_counts[sname])
    for vid, label, *_ in VARIANTS:
        sub = df_all[(df_all['vid'] == vid) & (df_all['stratum'] == sname)]
        strat_rows.append({
            'Estrato': f'{sname} (n={n_s})',
            'stratum': sname,
            'Variante': label,
            'vid': vid,
            'Accuracy': sub['is_correct'].mean(),
            'Macro-F1': f1_score(sub['y_true_idx'], sub['y_pred_idx'], labels=list(range(9)), average='macro', zero_division=0),
        })
df_strat = pd.DataFrame(strat_rows)
print('\\nAccuracy por estrato:')
display(df_strat.pivot(index='Estrato', columns='Variante', values='Accuracy').reindex([f"{s} (n={int(stratum_counts[s])})" for s in strata_order])[[v[1] for v in VARIANTS]].style.format('{:.4f}'))
print('\\nMacro-F1 por estrato:')
display(df_strat.pivot(index='Estrato', columns='Variante', values='Macro-F1').reindex([f"{s} (n={int(stratum_counts[s])})" for s in strata_order])[[v[1] for v in VARIANTS]].style.format('{:.4f}'))

# 2. Tabla de proyección a escala (10,000 citas)
base_cost_10k = df_summary.loc[df_summary['vid'] == 'gpt41_zs_a', 'Costo Neto ($)'].iloc[0] / N_TEST * 10000.0
base_f1 = df_summary.loc[df_summary['vid'] == 'gpt41_zs_a', 'Macro-F1'].iloc[0]
scale_rows = []
for _, r in df_summary.iterrows():
    cost_10k_net = (r['Costo Neto ($)'] / N_TEST) * 10000.0
    cost_10k_gross = (r['Costo Bruto ($)'] / N_TEST) * 10000.0
    hours_10k_seq = (r['Lat Media (ms)'] * 10000.0) / 1000.0 / 3600.0
    delta_f1_pts = (r['Macro-F1'] - base_f1) * 100.0
    if r['vid'] == 'gpt41_zs_a':
        c_per_pt = 0.0
    elif delta_f1_pts > 0:
        c_per_pt = (cost_10k_net - base_cost_10k) / delta_f1_pts
    else:
        c_per_pt = np.nan
    scale_rows.append({
        'Variante': r['Variante'],
        'Macro-F1': r['Macro-F1'],
        'ECE': df_calib.loc[df_calib['vid'] == r['vid'], 'ECE'].iloc[0],
        'Costo Neto / 10k ($)': cost_10k_net,
        'Costo Bruto / 10k ($)': cost_10k_gross,
        'Tiempo Sec. 10k (h)': hours_10k_seq,
        'Costo USD / +0.01 F1 vs base GPT-0s-A': c_per_pt,
    })
df_scale = pd.DataFrame(scale_rows)
display(df_scale.style.format({
    'Macro-F1': '{:.4f}', 'ECE': '{:.4f}',
    'Costo Neto / 10k ($)': '${:.2f}', 'Costo Bruto / 10k ($)': '${:.2f}',
    'Tiempo Sec. 10k (h)': '{:.2f} h', 'Costo USD / +0.01 F1 vs base GPT-0s-A': '${:.2f}',
}, na_rep='— (Δ ≤ 0)'))

# Figura 11: Frontera de Pareto A5 sin solapamiento + Estratificación por consenso A2
fig, axes = plt.subplots(1, 2, figsize=(16.0, 6.0))

# Panel A: Pareto Costo USD / 10k citas vs Macro-F1
ax0 = axes[0]
offsets = {
    'gpt41_zs_a':  (-10, -22),
    'gpt41_zs_b':  (8, 8),
    'gpt41_fs_a':  (8, -16),
    'gpt41_fs_b':  (8, -6),
    'gemini_zs_a': (-20, -24),
    'gemini_zs_b': (8, 8),
    'gemini_fs_a': (-65, 8),
    'gemini_fs_b': (-75, -20),
}
for _, r in df_scale.iterrows():
    vid = df_summary.loc[df_summary['Variante'] == r['Variante'], 'vid'].iloc[0]
    mod = df_summary.loc[df_summary['vid'] == vid, 'Modelo'].iloc[0]
    sh = df_summary.loc[df_summary['vid'] == vid, 'Shots'].iloc[0]
    cfg = df_summary.loc[df_summary['vid'] == vid, 'Config'].iloc[0]
    marker = 'o' if mod == 'gpt-4.1-mini' else 'D'
    ax0.scatter(r['Costo Neto / 10k ($)'], r['Macro-F1'], s=150, marker=marker,
                color=COLORS[vid], edgecolors='black', linewidth=1.2, zorder=4)
    lbl_pt = f"{'GPT' if 'gpt' in vid else 'Gem'} {sh}s-{cfg}\\n(${r['Costo Neto / 10k ($)']:.2f}, {r['Macro-F1']:.3f})"
    dx, dy = offsets[vid]
    ax0.annotate(lbl_pt, xy=(r['Costo Neto / 10k ($)'], r['Macro-F1']), xytext=(dx, dy),
                 textcoords='offset points', fontsize=8.0, fontweight='bold',
                 arrowprops=dict(arrowstyle='-', color='#888', lw=0.7) if abs(dx) > 20 or abs(dy) > 15 else None)

# Unir puntos de la frontera de Pareto no dominada (GPT 0s-A -> GPT 0s-B -> Gem 0s-B -> Gem 9s-A)
pareto_vids = ['gpt41_zs_a', 'gpt41_zs_b', 'gemini_zs_b', 'gemini_fs_a']
p_coords = []
for pv in pareto_vids:
    lbl_pv = runs[pv]['label']
    rr = df_scale[df_scale['Variante'] == lbl_pv].iloc[0]
    p_coords.append((rr['Costo Neto / 10k ($)'], rr['Macro-F1']))
ax0.plot([p[0] for p in p_coords], [p[1] for p in p_coords], linestyle='--', color='#444', linewidth=1.3, label='Frontera Pareto A5', zorder=2)

ax0.set_xscale('log')
ax0.set_xlim(4.2, 78)
ax0.set_ylim(0.398, 0.502)
ax0.set_xlabel('Costo Neto Proyectado (USD por 10,000 citas, escala log)')
ax0.set_ylabel('Macro-F1 (9 Clases Canónicas)')
ax0.set_title('Frontera de Pareto A5: Costo a Escala vs. Macro-F1')
ax0.grid(True, which='both', linestyle='--', alpha=0.35)
ax0.legend(loc='lower right', fontsize=8.5, frameon=True)

# Panel B: Sensibilidad de Accuracy según Estrato Epistémico (Human Gold & Jueces A2)
ax1 = axes[1]
xs = np.arange(len(strata_order))
for vid in ['gpt41_zs_a', 'gpt41_zs_b', 'gpt41_fs_b', 'gemini_zs_b', 'gemini_fs_a', 'gemini_fs_b']:
    vals = [df_strat[(df_strat['vid'] == vid) & (df_strat['stratum'] == st)]['Accuracy'].iloc[0] * 100.0 for st in strata_order]
    ax1.plot(xs, vals, marker='o', linewidth=2.0, color=COLORS[vid], label=runs[vid]['label'])
ax1.set_xticks(xs)
short_x_lbls = {
    'Human Gold (MultiCite)': f"Human Gold\\nMultiCite (n={stratum_counts['Human Gold (MultiCite)']:,})",
    'Panel Unánime (3/3)': f"Panel Unánime 3/3\\n(n={stratum_counts['Panel Unánime (3/3)']:,})",
    'Panel Mayoría (2/3)': f"Panel Mayoría 2/3\\n(n={stratum_counts['Panel Mayoría (2/3)']:,})",
    'Needs Review (desacuerdo)': f"Needs Review\\n(n={stratum_counts['Needs Review (desacuerdo)']:,})",
}
ax1.set_xticklabels([short_x_lbls[st] for st in strata_order], fontsize=8.4)
ax1.set_ylabel('Accuracy (%)')
ax1.set_ylim(20, 82)
ax1.set_title('Techo Epistémico: Desempeño según Estrato de Etiquetado')
ax1.grid(linestyle='--', alpha=0.35)
ax1.legend(loc='upper right', fontsize=7.8, frameon=True)

fig.tight_layout()
fig.savefig(FIG_DIR / 'fig11_frontera_pareto_y_sensibilidad_panel.png', bbox_inches='tight')
plt.show()
"""))

    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {
        "name": "python",
        "version": "3.14.7",
    }
    out_path = Path("notebooks/analisis_quirurgico_a5.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook generado exitosamente en {out_path} ({len(cells)} celdas).")


if __name__ == "__main__":
    build_notebook()
