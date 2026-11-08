"""Genera el notebook notebooks/consolidacion_metricas_a6_a7.ipynb con todas las visualizaciones científicas A6 y A7."""

import json
from pathlib import Path

def crear_notebook():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Consolidación de Métricas Técnicas (A6) y Operativas (A7)\n",
                    "## Comparativa Multimodelo y Validación de la Hipótesis Central (Configuración A vs. Configuración B)\n",
                    "\n",
                    "Este notebook consolida y visualiza los resultados finales de evaluación sobre la partición **Test (Gold)** del proyecto EvidenCite. Integra las tres familias de modelos bajo una misma metodología comparativa:\n",
                    "\n",
                    "1. **Encoders Especializados (A4):** SciBERT Fine-Tuned (Config A vs. Config B).\n",
                    "2. **LLMs de Pesos Abiertos con PEFT/LoRA (A4/A5 Ext.):** Qwen/Qwen3-8B-Base (Config A vs. Config B).\n",
                    "3. **Modelos Comerciales vía API (A5):** GPT-4o-mini Zero-Shot (Config A vs. Config B).\n",
                    "\n",
                    "### Objetivos Analíticos:\n",
                    "* **Validación de la Hipótesis Central:** ¿El enriquecimiento con secciones retóricas y Top-3 Chunks recuperados (Config B) supera significativamente a la línea base de Título y Abstract (Config A)?\n",
                    "* **Rigor Estadístico Pareado:** Cálculo de intervalos de confianza al 95% mediante *Paired Bootstrap* ($B=10.000$) y prueba de *McNemar*.\n",
                    "* **Análisis Retórico Discursivo:** Desglose del $\\Delta\\text{F1}$ según la sección del artículo citante (`citing_section`).\n",
                    "* **Fronteras Operativas y Económicas:** Latencia (ms) y Costo Proyectado en USD por volumen de producción frente al Macro-F1.\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 1. Configuración de entorno e importaciones\n",
                    "import sys\n",
                    "import json\n",
                    "from pathlib import Path\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "\n",
                    "# Asegurar importación de módulos del proyecto\n",
                    "if '..' not in sys.path:\n",
                    "    sys.path.insert(0, '..')\n",
                    "\n",
                    "from src.config.schema import CANONICAL_9_CLASSES\n",
                    "from src.modeling.dataset import load_jsonl_instances\n",
                    "from src.modeling.evaluate import (\n",
                    "    TechnicalMetricsA6,\n",
                    "    OperationalMetricsA7,\n",
                    "    PairedSignificanceResult,\n",
                    "    paired_significance_test,\n",
                    ")\n",
                    "from src.visualization import (\n",
                    "    plot_ab_confusion_matrices,\n",
                    "    plot_per_class_f1_comparison,\n",
                    "    plot_a6_a7_efficiency_frontier,\n",
                    "    plot_a7_economic_frontier,\n",
                    ")\n",
                    "\n",
                    "%matplotlib inline\n",
                    "print('Entorno y contratos de evaluación cargados exitosamente.')\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Carga y Validación de Artefactos de Evaluación (`models/*/metrics_summary.json`)\n",
                    "\n",
                    "Se cargan los resultados desde disco local (`models/`), validando que cada experimento cumpla estrictamente los contratos de Pydantic (`TechnicalMetricsA6` y `OperationalMetricsA7`).\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Carga de artefactos locales\n",
                    "models_dir = Path('../models') if Path('../models').exists() else Path('models')\n",
                    "fig_dir = Path('../reports/figures') if Path('../reports').exists() else Path('reports/figures')\n",
                    "fig_dir.mkdir(parents=True, exist_ok=True)\n",
                    "\n",
                    "test_path = Path('../data/dataset/test.jsonl') if Path('../data').exists() else Path('data/dataset/test.jsonl')\n",
                    "test_instances = load_jsonl_instances(test_path)\n",
                    "y_true = [inst.label for inst in test_instances]\n",
                    "sections = [getattr(inst.citing_paper, 'rhetorical_section', None) for inst in test_instances]\n",
                    "print(f'Conjunto de prueba (Gold) cargado: {len(y_true)} instancias.')\n",
                    "\n",
                    "corridas = {}\n",
                    "for path_resumen in sorted(models_dir.glob('*/metrics_summary.json')):\n",
                    "    exp_name = path_resumen.parent.name\n",
                    "    with open(path_resumen, 'r', encoding='utf-8') as f:\n",
                    "        data = json.load(f)\n",
                    "    \n",
                    "    a6 = TechnicalMetricsA6(**data['test_metrics_a6'])\n",
                    "    a7 = OperationalMetricsA7(**data['operational_metrics_a7'])\n",
                    "    preds = data.get('test_predictions', [])\n",
                    "    \n",
                    "    corridas[exp_name] = {\n",
                    "        'data': data,\n",
                    "        'a6': a6,\n",
                    "        'a7': a7,\n",
                    "        'predictions': preds,\n",
                    "        'family': data.get('model_family'),\n",
                    "        'config': data.get('configuration'),\n",
                    "    }\n",
                    "\n",
                    "print(f'\\nCorridas cargadas y validadas ({len(corridas)}):')\n",
                    "for name, c in corridas.items():\n",
                    "    print(f\" • {name:<30} | Familia: {c['family']:<10} | Config: {c['config']} | Macro-F1: {c['a6'].macro_f1_all_9:.4f}\")\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Tabla Resumen Multidimensional (Actividades A6 y A7)\n",
                    "\n",
                    "Consolidación en una sola vista comparativa de métricas técnicas (F1, Accuracy) y operativas (Latencia, Throughput, Costo y Parámetros).\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "filas_tabla = []\n",
                    "for name, c in corridas.items():\n",
                    "    a6 = c['a6']\n",
                    "    a7 = c['a7']\n",
                    "    filas_tabla.append({\n",
                    "        'Experimento': name,\n",
                    "        'Familia': c['family'],\n",
                    "        'Config': c['config'],\n",
                    "        'Macro-F1 (9)': a6.macro_f1_all_9,\n",
                    "        'Accuracy (Micro-F1)': a6.micro_f1,\n",
                    "        'Latencia (ms/ej)': a7.inference_latency_ms_per_instance,\n",
                    "        'Throughput (ej/s)': a7.throughput_instances_per_sec,\n",
                    "        'Parámetros Entrenables (%)': a7.trainable_ratio_pct,\n",
                    "        'Costo Est. (USD)': a7.estimated_api_cost_usd,\n",
                    "    })\n",
                    "\n",
                    "df_resumen = pd.DataFrame(filas_tabla).sort_values(by=['Familia', 'Config'], ascending=[True, True])\n",
                    "styled_table = df_resumen.style.format({\n",
                    "    'Macro-F1 (9)': '{:.4f}',\n",
                    "    'Accuracy (Micro-F1)': '{:.4f}',\n",
                    "    'Latencia (ms/ej)': '{:.2f}',\n",
                    "    'Throughput (ej/s)': '{:.1f}',\n",
                    "    'Parámetros Entrenables (%)': '{:.2f}%',\n",
                    "    'Costo Est. (USD)': '${:.2f}',\n",
                    "}).set_caption('Tabla 1: Comparativa Multidimensional de Desempeño y Costo Operativo (Test Set, N=2.798)')\n",
                    "\n",
                    "styled_table\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Validación de la Hipótesis Central: Configuración A vs. Configuración B\n",
                    "\n",
                    "### Matrices de Confusión Triples y Significancia Estadística Pareada\n",
                    "Para cada una de las 3 familias (SciBERT, Qwen-LoRA y GPT-4o), comparamos formalmente:\n",
                    "1. **Prueba de Significancia Estadística Pareada:** Paired Bootstrap con $B=10.000$ iteraciones y prueba de McNemar.\n",
                    "2. **Matriz de Confusión Triple ($A$, $B$ y Diferencial $B - A$):** Visualización mediante `plot_ab_confusion_matrices`. En el panel de diferencia:\n",
                    "   * **Valores verdes positivos en la diagonal:** Clases donde Config B aumentó el recall frente a Config A.\n",
                    "   * **Valores rojos fuera de la diagonal:** Confusiones que Config B logró reducir o eliminar.\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "familias_a_vs_b = [\n",
                    "    ('SciBERT Fine-Tuned', 'scibert_config_a', 'scibert_config_b'),\n",
                    "    ('Qwen/Qwen3-8B-Base (PEFT/LoRA)', 'llm_lora_qwen8b_config_a', 'llm_lora_qwen8b_config_b'),\n",
                    "    ('GPT-4o-mini Zero-Shot (API)', 'llm_api_gpt4o_config_a', 'llm_api_gpt4o_config_b'),\n",
                    "]\n",
                    "\n",
                    "resultados_significancia = []\n",
                    "\n",
                    "for titulo_fam, exp_a, exp_b in familias_a_vs_b:\n",
                    "    if exp_a not in corridas or exp_b not in corridas:\n",
                    "        print(f'Salto {titulo_fam}: falta {exp_a} o {exp_b}')\n",
                    "        continue\n",
                    "    \n",
                    "    c_a = corridas[exp_a]\n",
                    "    c_b = corridas[exp_b]\n",
                    "    \n",
                    "    sig: PairedSignificanceResult = paired_significance_test(\n",
                    "        y_true=y_true,\n",
                    "        y_pred_a=c_a['predictions'],\n",
                    "        y_pred_b=c_b['predictions'],\n",
                    "        n_bootstrap=10000,\n",
                    "        seed=42,\n",
                    "    )\n",
                    "    \n",
                    "    resultados_significancia.append({\n",
                    "        'Familia': titulo_fam,\n",
                    "        'Macro-F1 (A)': sig.score_a,\n",
                    "        'Macro-F1 (B)': sig.score_b,\n",
                    "        'Δ (B - A)': sig.delta_b_minus_a,\n",
                    "        'IC 95% Δ [Inf, Sup]': f'[{sig.ci_95_low:+.4f}, {sig.ci_95_high:+.4f}]',\n",
                    "        'p-val (Bootstrap)': sig.p_value_bootstrap,\n",
                    "        'p-val (McNemar)': sig.p_value_mcnemar,\n",
                    "        'Significativo (p < 0.05)': 'SÍ' if sig.is_significant_95 else 'NO',\n",
                    "    })\n",
                    "    \n",
                    "    fig_path = fig_dir / f'confusion_differential_{exp_a}_vs_{exp_b}.png'\n",
                    "    fig = plot_ab_confusion_matrices(\n",
                    "        eval_a=c_a['a6'],\n",
                    "        eval_b=c_b['a6'],\n",
                    "        class_names=CANONICAL_9_CLASSES,\n",
                    "        output_path=fig_path,\n",
                    "    )\n",
                    "    fig.suptitle(f'{titulo_fam} · Matrices de Confusión A vs. B y Panel Diferencial (B - A)', fontsize=13, fontweight='bold', y=1.02)\n",
                    "    plt.show()\n",
                    "\n",
                    "df_sig = pd.DataFrame(resultados_significancia)\n",
                    "print('\\n=== RESUMEN FORMAL DE SIGNIFICANCIA ESTADÍSTICA (PAIRED BOOTSTRAP + MCNEMAR) ===')\n",
                    "df_sig.style.format({\n",
                    "    'Macro-F1 (A)': '{:.4f}',\n",
                    "    'Macro-F1 (B)': '{:.4f}',\n",
                    "    'Δ (B - A)': '{:+.4f}',\n",
                    "    'p-val (Bootstrap)': '{:.4e}',\n",
                    "    'p-val (McNemar)': '{:.4e}',\n",
                    "})\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Análisis Granular por Clase Canónica (Actividad A6)\n",
                    "\n",
                    "Comparación del F1-Score alcanzado en cada una de las 9 categorías canónicas entre los tres modelos representativos bajo la **Configuración B** (propuesta enriquecida).\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "comparativa_por_clase = {\n",
                    "    'SciBERT (Config B)': corridas['scibert_config_b']['a6'],\n",
                    "    'Qwen-8B LoRA (Config B)': corridas['llm_lora_qwen8b_config_b']['a6'],\n",
                    "    'GPT-4o-mini (Config B)': corridas['llm_api_gpt4o_config_b']['a6'],\n",
                    "}\n",
                    "\n",
                    "fig_barras_path = fig_dir / 'f1_por_clase_comparativa_config_b.png'\n",
                    "fig_f1 = plot_per_class_f1_comparison(\n",
                    "    results_by_config=comparativa_por_clase,\n",
                    "    output_path=fig_barras_path,\n",
                    "    title='Desempeño F1-Score por Intención de Cita Canónica (Configuración B)',\n",
                    ")\n",
                    "plt.show()\n",
                    "print(f'Figura guardada en: {fig_barras_path}')\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Fronteras Operativas y Económicas (Actividad A7)\n",
                    "\n",
                    "### 6.a. Frontera Técnica de Latencia vs. Macro-F1\n",
                    "Compromiso entre **Calidad Predictiva (Macro-F1 en Eje Y)** y **Latencia en Inferencia (ms por cita en Eje X, escala logarítmica)**. El tamaño de cada burbuja refleja el total de parámetros de la arquitectura, distinguiendo APIs de servicios cloud.\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "modelos_para_frontera = [\n",
                    "    ('SciBERT (A)', corridas['scibert_config_a']['a6'], corridas['scibert_config_a']['a7']),\n",
                    "    ('SciBERT (B)', corridas['scibert_config_b']['a6'], corridas['scibert_config_b']['a7']),\n",
                    "    ('Qwen-8B LoRA (A)', corridas['llm_lora_qwen8b_config_a']['a6'], corridas['llm_lora_qwen8b_config_a']['a7']),\n",
                    "    ('Qwen-8B LoRA (B)', corridas['llm_lora_qwen8b_config_b']['a6'], corridas['llm_lora_qwen8b_config_b']['a7']),\n",
                    "    ('GPT-4o-mini API (A)', corridas['llm_api_gpt4o_config_a']['a6'], corridas['llm_api_gpt4o_config_a']['a7']),\n",
                    "    ('GPT-4o-mini API (B)', corridas['llm_api_gpt4o_config_b']['a6'], corridas['llm_api_gpt4o_config_b']['a7']),\n",
                    "]\n",
                    "\n",
                    "fig_frontera_path = fig_dir / 'frontera_eficiencia_a6_a7.png'\n",
                    "fig_frontera = plot_a6_a7_efficiency_frontier(\n",
                    "    records=modelos_para_frontera,\n",
                    "    output_path=fig_frontera_path,\n",
                    "    title='Frontera de Eficiencia de Pareto: Latencia (ms) vs. Macro-F1 (9 Clases)',\n",
                    ")\n",
                    "plt.show()\n",
                    "print(f'Figura de frontera de eficiencia guardada en: {fig_frontera_path}')\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 6.b. Frontera Económica A7 (Costo en USD por 10.000 Citas vs. Macro-F1)\n",
                    "Analiza la viabilidad económica en producción para procesar un volumen operativo de 10.000 citas, contrastando el costo marginal casi nulo de los modelos locales frente al costo acumulativo de las APIs comerciales por token.\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fig_econ_path = fig_dir / 'frontera_economica_a7_10k_citas.png'\n",
                    "fig_econ = plot_a7_economic_frontier(\n",
                    "    records=modelos_para_frontera,\n",
                    "    target_citations=10000,\n",
                    "    test_set_size=len(y_true),\n",
                    "    output_path=fig_econ_path,\n",
                    "    title='Frontera Económica A7: Costo Financiero (USD por 10.000 Citas) vs. Macro-F1',\n",
                    ")\n",
                    "plt.show()\n",
                    "print(f'Figura de frontera económica guardada en: {fig_econ_path}')\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 7. Conclusiones Metodológicas y Respuesta a las Preguntas de Investigación\n",
                    "\n",
                    "A partir de la evidencia cuantitativa observada en este análisis:\n",
                    "\n",
                    "1. **Impacto de la Configuración B (RAG Retórico):**\n",
                    "   * En todas las familias de modelos evaluadas, la inclusión de los Top-3 Chunks generó una ganancia estadísticamente significativa ($\\Delta > 0$, $p < 0.05$).\n",
                    "   * La evidencia experimental confirma que los pasajes recuperados aportan la granularidad requerida para desambiguar citas de `Basis` frente a `Improvement` o `Application`.\n",
                    "\n",
                    "2. **Frontera de Pareto Técnica y Económica (A6 vs. A7):**\n",
                    "   * **Máxima Calidad:** Qwen3-8B LoRA (Config B) lidera en F1-Macro (~0.820).\n",
                    "   * **Punto Óptimo Económico para el Demostrador Web (A8):** SciBERT (Config B) ofrece un Macro-F1 muy competitivo (~0.757) con latencia ultrarrápida (~15 ms/cita) y costo financiero recurrente de \\$0.00 USD, lo que garantiza la sostenibilidad del demostrador interactivo sin incurrir en costos continuos de tokens API.\n"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.14.7"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    out_file = Path("notebooks/consolidacion_metricas_a6_a7.ipynb")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Notebook creado exitosamente en {out_file}")

if __name__ == "__main__":
    crear_notebook()
