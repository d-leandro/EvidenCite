# Paquete de Datos (`src/data`)

Este paquete implementa el pipeline de ingeniería de datos para el proyecto de clasificación de funciones de citas científicas. Transforma insumos crudos (`unarXive`, `MultiCite` y `Semantic Scholar`) en los conjuntos de datos consolidados para entrenamiento y evaluación bajo **paridad estricta 1:1** entre la **Configuración A** (Abstract global) y la **Configuración B** (Evidencia discursiva local con SciBERT).

---

## 1. Flujo de Datos y Capas Arquitectónicas

El pipeline opera en 4 capas desacopladas mediante procesamiento por streaming y tipado estricto con Pydantic:

```text
Capa 1: Raw (data/raw/)
├── unarxive/                    # Shards crudos JSONL de arXiv
├── multicite/                   # Particiones crudas de MultiCite
└── s2_cache.db                  # Base de datos SQLite de Semantic Scholar
         │
         ▼  [structuring.py + seminal_resolver.py]
Capa 2: Structured (data/structured/)
└── papers.jsonl                 # Corpus filtrado CS (Computer Science)
         │
         ▼  [mining.py + chunking.py]
Capa 3: Processed (data/processed/)
├── citing_contexts.jsonl        # Contextos citantes (ventana m=3 oraciones)
└── cited_chunks.jsonl           # Fragmentos discursivos de artículos citados (<=110 palabras)
         │
         ▼  [builder.py + retrieval.py (SciBERT Top-3)]
Capa 4: Dataset (data/dataset/)
├── instances.jsonl              # Dataset consolidado con paridad 1:1
├── train.jsonl (70%)            # Partición de entrenamiento (Group-Stratified Split)
├── dev.jsonl   (15%)            # Partición de validación
├── test.jsonl  (15%)            # Partición de prueba
└── dataset_metrics.json         # Métricas de balance y auditoría de fuga
```

---

## 2. Mapa de Módulos y Responsabilidades

| Módulo / Archivo | Responsabilidad Principal |
|---|---|
| [pipeline.py](pipeline.py) | **Punto de entrada maestro:** Orquesta las 5 etapas del pipeline con checkpointing automático e idempotencia. |
| [schemas.py](schemas.py) | **Contratos de datos Pydantic:** Modelos para cada capa (`StructuredPaper`, `CitingContext`, `CitedChunk`, `DatasetInstance`). |
| [structuring.py](structuring.py) | Normaliza secciones, limpia metadatos y filtra por disciplina Computer Science hacia `papers.jsonl`. |
| [seminal_resolver.py](seminal_resolver.py) | Resuelve el catálogo canónico (146 papers seminales) en S2 y anexa los textos completos faltantes. |
| [mining.py](mining.py) | Extrae contextos citantes en ventana $m=3$, enmascara citas y resuelve metadatos del artículo citado. |
| [chunking.py](chunking.py) | Segmenta los artículos citados respetando límites discursivos ($\le 110$ palabras) hacia `cited_chunks.jsonl`. |
| [retrieval.py](retrieval.py) | Motor de inferencia SciBERT: calcula similitud coseno densa y recupera los Top-3 chunks más relevantes. |
| [builder.py](builder.py) | Ensambla pares 1:1, mapea a la taxonomía canónica de 9 clases y aplica particionado anti-fuga agrupado por autor. |
| `ingestion/` | Adaptadores crudos para streaming de unarXive ([unarxive.py](ingestion/unarxive.py)), MultiCite ([multicite.py](ingestion/multicite.py)) y cliente S2 ([s2_client.py](ingestion/s2_client.py)). |

---

## 3. Guía de Ejecución Rápida

### Ejecución End-to-End (Recomendada)
Para construir o actualizar todo el dataset desde cero o de forma incremental:

```bash
# 1. Simulación e inspección de estado de artefactos (sin cómputo):
uv run python -m src.data.pipeline --dry-run

# 2. Ejecución incremental (omite automáticamente etapas con artefactos existentes):
uv run python -m src.data.pipeline

# 3. Forzar reconstrucción total desde cero:
uv run python -m src.data.pipeline --all --force
```

### Ejecución Selectiva por Etapas
```bash
# Ejecutar únicamente una etapa:
uv run python -m src.data.pipeline --stage mining

# Ejecutar a partir de una etapa en adelante:
uv run python -m src.data.pipeline --from-stage chunking

# Omitir la resolución en red de Semantic Scholar (modo offline):
uv run python -m src.data.pipeline --skip-seminal
```

---

## 4. Archivos de Configuración Asociados

Los hiperparámetros y rutas están externalizados en `configs/`:

* `configs/data_pipeline.yaml`: Rutas maestras de las 4 capas y flags de orquestación.
* `configs/builder.yaml`: Cuotas por clase, proporciones de splits (70/15/15) y semilla determinista (`seed=42`).
* `configs/retrieval.yaml`: Modelo transformer (`allenai/scibert_scivocab_uncased`), tamaño de lote y dispositivo (`auto`, `mps`, `cuda`, `cpu`).

---

## 5. Pruebas Unitarias

La integridad de cada componente y del orquestador se valida con:

```bash
uv run pytest tests/unit/
```
