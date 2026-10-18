# EvidenCite

> **Recomendación Local de Citas y Clasificación de Función Retórica en Literatura Científica mediante Recuperación Densa de Evidencia y Modelos de Lenguaje**  
> *Grupo de Investigación: FLAG – TICsW (DISC)*

[![Python 3.14](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![Package Manager: uv](https://img.shields.io/badge/package%20manager-uv-blueviolet.svg)](https://github.com/astral-sh/uv)
[![Embeddings: SciBERT](https://img.shields.io/badge/Embeddings-SciBERT-teal.svg)](https://huggingface.co/allenai/scibert_scivocab_uncased)
[![Tests: 73 passed](https://img.shields.io/badge/Tests-73%20passed-brightgreen.svg)]()

[🇬🇧 Read in English](README.md) | [🇪🇸 Español](README.es.md)

---

## 1. Visión General

Si bien los sistemas modernos de búsqueda académica (ej. Google Scholar, Semantic Scholar) destacan en la recuperación de documentos a nivel global, carecen de **resolución local**: no identifican el fragmento textual específico que sustenta una afirmación ni discriminan *por qué* se realizó una cita.

**EvidenCite** investiga la sinergia entre la **recomendación local de citas** y la **clasificación de la función de citas**. Evaluamos la hipótesis científica de que enriquecer la clasificación de citas con **evidencia discursiva local granular**, recuperada mediante codificadores densos de pasajes, supera significativamente a las representaciones convencionales basadas en resúmenes globales bajo paridad experimental estricta.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CONTEXTO CITANTE                               │
│  "Recent neural architectures rely on self-attention mechanisms             │
│   <cite>Vaswani et al., 2017</cite> for sequence transduction."             │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
┌───────────────────────────────┐     ┌───────────────────────────────────────┐
│     CONFIGURACIÓN A (Global)  │     │    CONFIGURACIÓN B (Evidencia Local)  │
│                               │     │                                       │
│  • Título del citado          │     │  • Top-3 Chunks discursivos (≤110 p)  │
│  • Abstract completo citado   │     │  • Recuperación densa coseno SciBERT  │
│                               │     │  • Preservación de encabezados autor  │
└───────────────┬───────────────┘     └───────────────────┬───────────────────┘
                │                                         │
                └──────────────────────┬──────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CLASIFICACIÓN DE FUNCIÓN DE CITA                         │
│   [Background, Uses, Comparison, Extends, Motivation, Future Work, ...]     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Diseño Experimental: Configuraciones Duales

Para garantizar una comparación científica no sesgada, todas las instancias de evaluación aplican **paridad estricta 1:1**: cada instancia cuenta simultáneamente con metadatos completos de abstract (Configuración A) y fragmentos discursivos estructurados verificados de texto completo (Configuración B).

| Dimensión | Configuración A (Línea Base Global) | Configuración B (Evidencia Local) |
|---|---|---|
| **Contexto Citante** | Ventana de $m=3$ oraciones con marcadores enmascarados | Ventana de $m=3$ oraciones con marcadores enmascarados |
| **Evidencia Objetivo** | Título + Abstract completo del citado | Top-3 chunks discursivos densos ($\le 110$ palabras c/u) |
| **Codificador / Modelo** | SciBERT / Transformer | SciBERT dense retriever + Chunking consciente de sección |
| **Presupuesto de Tokens** | Presupuesto estándar de 512 tokens | Acotado a 512 tokens (cero truncamiento forzado) |
| **Estrategia de Split** | Group-Stratified Split (70/15/15) por paper citante para eliminar fuga de datos |

### Taxonomía Canónica (9 Clases Científicas)
Derivada de la meta-síntesis empírica de [Lyu et al. (2021)](https://link.springer.com/article/10.1007/s11192-021-03908-z):
`Background`, `Uses`, `Comparison`, `Extends`, `Motivation`, `Future Work`, `Evidence`, `Originator`, `Further Reading`.

---

## 3. Estructura del Repositorio

```text
.
├── configs/                     # Archivos YAML centralizados de hiperparámetros
│   ├── data_pipeline.yaml       # Configuración maestra del orquestador de datos
│   ├── builder.yaml             # Ensamble de dataset, cuotas y ratios de split
│   └── retrieval.yaml           # Parámetros de recuperación SciBERT y dispositivo
├── data/                        # Arquitectura desacoplada en 4 capas (.gitkeep)
│   ├── raw/                     # Fuentes de ingesta (unarXive, MultiCite, s2_cache.db)
│   ├── structured/              # Corpus canónico filtrado CS (papers.jsonl)
│   ├── processed/               # Contextos minados y chunks retóricos
│   └── dataset/                 # Instancias con paridad 1:1 y particiones train/dev/test
├── models/                      # Checkpoints y pesos de experimentos base
├── notebooks/                   # Análisis exploratorio e inspección de errores
├── src/                         # Código fuente modular en Python
│   ├── data/                    # Pipeline end-to-end de ingeniería de datos
│   │   ├── ingestion/           # Adaptadores para unarXive, MultiCite y Semantic Scholar
│   │   ├── structuring.py       # Estructuración de artículos y filtrado CS
│   │   ├── seminal_resolver.py  # Catálogo seminal (146 papers) y caché S2
│   │   ├── mining.py            # Extracción de ventanas m=3 y resolución de citas
│   │   ├── chunking.py          # Chunking discursivo de oraciones (≤110 palabras)
│   │   ├── retrieval.py         # Motor de recuperación densa SciBERT y ranking Top-k
│   │   ├── builder.py           # Ensamble 1:1, mapeo de etiquetas y split anti-fuga
│   │   ├── pipeline.py          # Orquestador maestro con checkpointing automático
│   │   └── schemas.py           # Contratos estrictos Pydantic para todas las capas
│   ├── modeling/                # Clasificadores base (Config A vs Config B)
│   └── visualization/           # Métricas, matrices de confusión y mapas de atención
└── tests/                       # Suite completa de pruebas con pytest
    └── unit/                    # 73 pruebas unitarias validando componentes del pipeline
```

---

## 4. Inicio Rápido

### Requisitos Previos
- Python 3.14+
- [uv](https://github.com/astral-sh/uv) (gestor rápido de paquetes de Python)

### Instalación
```bash
# Clonar el repositorio
git clone https://github.com/leandrojimenez/EvidenCite.git
cd EvidenCite

# Instalar dependencias y sincronizar el entorno virtual
uv sync
```

---

## 5. Reproducción del Pipeline de Datos

El ciclo de vida completo de los datos—desde las fuentes crudas hasta las particiones finales con paridad 1:1—está unificado a través de `src.data.pipeline`:

```bash
# 1. Inspección previa (simulación sin cómputo)
uv run python -m src.data.pipeline --dry-run

# 2. Ejecución incremental (omite automáticamente etapas completadas)
uv run python -m src.data.pipeline

# 3. Forzar reconstrucción completa desde cero
uv run python -m src.data.pipeline --all --force

# 4. Ejecutar una etapa individual (ej. chunking o builder)
uv run python -m src.data.pipeline --stage chunking
```

Para consultar la descripción detallada de capas y contratos de datos, revisa la [Guía del Paquete de Datos (`src/data/README.md`)](src/data/README.md).

---

## 6. Pruebas Unitarias

Ejecución de la suite completa de pruebas:

```bash
uv run pytest tests/unit/
```

Las 73 pruebas unitarias validan:
- Estructuración de artículos y clasificación temática en CS.
- Caché SQLite y control de tasa (*rate-limiting*) del cliente S2.
- Extracción de ventanas de cita y enmascaramiento de marcadores.
- Límites de longitud de chunks WordPiece ($\le 110$ palabras) y preservación de encabezados.
- Recuperación densa y ordenamiento de relevancia Top-k con SciBERT.
- Paridad estricta 1:1 y agrupación por autor citante sin fuga en train/dev/test.
- Orquestación del DAG, checkpointing y recuperación ante errores en el pipeline.

---

## 7. Autores y Contexto Académico

### Equipo de Investigación
| Autor | Correo Institucional | Afiliación |
|---|---|---|
| **Daniel Ricardo Gallo Acevedo** | `dr.gallo@uniandes.edu.co` | Universidad de los Andes |
| **Dubán Leandro Jiménez Puerta** | `dl.jimenez@uniandes.edu.co` | Universidad de los Andes |
| **Julián Andrés Malaver Montoya** | `ja.malaverm1@uniandes.edu.co` | Universidad de los Andes |
| **Javier Mauricio Torres Ortega** | `jm.torreso1@uniandes.edu.co` | Universidad de los Andes |

* **Programa Académico:** Maestría en Inteligencia Artificial (MAIA)
* **Unidad Académica:** Departamento de Ingeniería de Sistemas y Computación (DISC), Universidad de los Andes (Bogotá, Colombia)
* **Grupo de Investigación:** Grupo FLAG – TICsW
* **Asesor / Experto de Dominio:** PhD. Juan Camilo Sanguino
