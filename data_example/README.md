# Muestra de Datos de EvidenCite (`data_example/`)

Esta carpeta contiene una **muestra liviana y autocontenida** (~50 KB) del dataset final etiquetado de **EvidenCite** (`18.658` instancias reales en `data/labeled/`), pensada para que evaluadores y revisores puedan inspeccionar el esquema JSON, la trazabilidad de etiquetas (A1 → A2 → A3) y el ensamblaje en *runtime* de **Configuración A** vs. **Configuración B** **sin descargar los ~117 MB completos ni requerir permisos de Google Drive / DVC**.

---

## 1. Contenido de esta carpeta

```text
data_example/
├── README.md                  # Esta guía y ejemplo de ensamblaje en runtime
├── dataset_metrics.json       # Estadísticas auditadas completas del corpus (18.658 pares, leakage = 0)
└── labeled/
    ├── train.jsonl            # 3 instancias reales (MultiCite Gold · Consenso A2 · Needs Review)
    ├── dev.jsonl              # 3 instancias reales (MultiCite Gold · Consenso A2 · Empate 1-1-1)
    └── test.jsonl             # 3 instancias reales 100% Gold A3 (incluye inst_04783 del artículo)
```

### Instancias seleccionadas por estrato

| Archivo | `instance_id` | Clase final (`label`) | Estrato / `label_source` | `prior_label_source` | Nota ilustrativa |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `labeled/train.jsonl` | `inst_00039` | `Background` | `human_gold` | `null` | MultiCite Gold preservado (Caso F) |
| `labeled/train.jsonl` | `inst_02636` | `Gap` | `llm_panel` | `weak_nli` | Consenso unánime 3/3 del panel A2 (`agreement = 1.0`) |
| `labeled/train.jsonl` | `inst_10633` | `Improvement` | `needs_review` | `weak_nli` | Mayoría divergente 2/3 (Caso C): retiene etiqueta previa |
| `labeled/dev.jsonl` | `inst_00206` | `Background` | `human_gold` | `null` | MultiCite Gold protegido aunque el panel votó `Improvement` |
| `labeled/dev.jsonl` | `inst_05268` | `Application` | `llm_panel` | `heuristic_rule` | Consenso unánime 3/3 confirmatorio sobre regla léxica |
| `labeled/dev.jsonl` | `inst_11984` | `Evidence` | `needs_review` | `heuristic_rule` | Empate 1-1-1 (Caso D, `panel_label = null`, `agreement = 0.333`) |
| `labeled/test.jsonl` | **`inst_04783`** | **`Basis`** | `human_review` | `llm_panel` | **Ejemplo del artículo y la sustentación** (revisado en A3) |
| `labeled/test.jsonl` | `inst_10624` | `Comparison` | `human_review` | `needs_review` | Revisión manual A3 sobre instancia previa de `needs_review` |
| `labeled/test.jsonl` | `inst_00009` | `Background` | `human_review` | `human_gold` | MultiCite Gold donde el panel A2 había errado (`panel = Basis`) |

---

## 2. Contrato de datos (Esquema JSON)

Cada línea de `train.jsonl`, `dev.jsonl` y `test.jsonl` es un objeto `DatasetInstance` (validado con Pydantic en `src/data/schemas.py`) que **contiene simultáneamente la evidencia global (Config A) y los Top-3 pasajes locales (Config B)**, garantizando paridad estricta 1:1 sin duplicar filas:

| Campo | Tipo | Descripción |
| :--- | :--- | :--- |
| `instance_id` | `str` | Identificador determinístico único de la instancia (`inst_04783`). |
| `label` | `str` | Función retórica canónica en $\mathcal{Y}_9$ (`Background`, `Gap`, `Basis`, `Comparison`, `Application`, `Improvement`, `Evidence`, `Originator`, `Further Reading`). |
| `annotation_metadata` | `object` | Linaje completo: `source_dataset`, `original_label`, `label_source`, `is_gold`, `confidence_score`, `margin`, `panel_agreement`, `panel_label`, `panel_votes`, `prior_label_source`, `split`. |
| `citing_paper` | `object` | Metadatos del artículo citante (`id`, `id_source`, `title`, `rhetorical_section`). El particionamiento agrupa por `citing_paper.id` (`leakage = 0`). |
| `citation_context` | `str` | Ventana citante fija de $m=3$ oraciones con la cita objetivo marcada en `<cite>...</cite>` y las demás en `[OTHERCIT]`. |
| `cited_paper` | `object` | Metadatos del citado (`id`, `title`, `abstract` para **Config A**) + `retrieved_chunks` (Top-3 pasajes $\le 110$ pal. con `chunk_id`, `section_heading`, `text`, `similarity_score`, `rank` para **Config B**). |

---

## 3. Ensamblaje en *runtime*: Config A vs. Config B (`inst_04783`)

No existen archivos separados para Config A y Config B. En tiempo de entrenamiento o inferencia, la función `format_instance_parts(instance, rep_config)` en [`src/modeling/dataset.py`](../src/modeling/dataset.py) construye al vuelo el par `(text_citing, text_cited)` desde el mismo registro JSON (por defecto `include_citing_section=False`, `include_cited_section=True`):

### Entrada fija compartida (`text_citing`)
```text
The procedure builds upon the recent work by Hopkins, Kamath and Majid
<cite>Samuel B Hopkins, Gautam Kamath, and Mahbod Majid, 2022</cite> on pure DP mean
estimation. First, we exploit their algorithm to precondition the unknown covariance
matrix such that it is approximately identity...
```

### Salida en *runtime* — Configuración A (`configuration="A"`)
Concatena `cited_paper.title` + `cited_paper.abstract`:
```text
Efficient mean estimation with pure differential privacy via a sum-of-squares exponential
mechanism. We give the first polynomial-time algorithm to estimate the mean of a d-variate
probability distribution with bounded covariance from Õ(d) independent samples subject to
pure differential privacy...
```

### Salida en *runtime* — Configuración B (`configuration="B"`, `top_k_chunks=3`)
Ordena `cited_paper.retrieved_chunks` por `rank` (`1..3`), antepone `section_heading: ` a cada fragmento y los une respetando el presupuesto de tokens:
```text
2.2.1 Private mean estimation when ||E X|| ≤ poly(d): A merit of our approach is that it
allows powerful techniques from robust statistics to be used for private algorithm design:
our algorithm draws heavily on a robust mean estimation algorithm... .
6 Fine Estimation: These properties of the SDP relaxation, together with its computational
efficiency, yield a computationally efficient algorithm... .
6 Fine Estimation: Our goal is to privately estimate the mean of this distribution up to
accuracy α with probability 1−β...
```

---

## 4. Verificación rápida en Python (sin DVC)

Puedes verificar el contrato Pydantic y el ensamblaje Config A vs. Config B directamente sobre esta muestra:

```bash
uv run python -c "
import json
from src.data.schemas import DatasetInstance
from src.config.schema import InputRepresentationConfig
from src.modeling.dataset import format_instance_parts

with open('data_example/labeled/test.jsonl', encoding='utf-8') as f:
    inst = DatasetInstance.model_validate(json.loads(f.readline()))

for cfg in ('A', 'B'):
    rep = InputRepresentationConfig(configuration=cfg, include_citing_section=False)
    citing, cited = format_instance_parts(inst, rep)
    print(f'=== {inst.instance_id} (Gold: {inst.label}) · Config {cfg} ===')
    print('CITANTE:', citing[:120], '...')
    print('CITADO :', cited[:180], '...\n')
"
```
