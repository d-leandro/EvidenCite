# EvidenCite

> **Local Citation Recommendation and Rhetorical Function Classification in Scientific Literature via Dense Evidence Retrieval and Language Models**  
> *Research Group: FLAG – TICsW (DISC)*

[![Python 3.14](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![Package Manager: uv](https://img.shields.io/badge/package%20manager-uv-blueviolet.svg)](https://github.com/astral-sh/uv)
[![Embeddings: SciBERT](https://img.shields.io/badge/Embeddings-SciBERT-teal.svg)](https://huggingface.co/allenai/scibert_scivocab_uncased)
[![Tests: 73 passed](https://img.shields.io/badge/Tests-73%20passed-brightgreen.svg)]()
---

## 1. Overview

While modern academic search systems (e.g., Google Scholar, Semantic Scholar) excel at retrieving documents at a global level, they lack **local resolution**: they neither identify the specific passage that supports a claim nor discern *why* a reference was cited.

**EvidenCite** investigates the synergy between **local citation recommendation** and **citation function classification**. We test the scientific hypothesis that augmenting citation classification with **fine-grained discursive evidence** retrieved via dense passage encoders significantly outperforms conventional global abstract representations under strict experimental parity.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CITING CONTEXT                                 │
│  "Recent neural architectures rely on self-attention mechanisms             │
│   <cite>Vaswani et al., 2017</cite> for sequence transduction."             │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
┌───────────────────────────────┐     ┌───────────────────────────────────────┐
│     CONFIGURATION A (Global)  │     │      CONFIGURATION B (Local Evidence) │
│                               │     │                                       │
│  • Cited Title                │     │  • Top-3 Discursive Chunks (≤110 w)   │
│  • Cited Full Abstract        │     │  • Dense SciBERT Cosine Retrieval     │
│                               │     │  • Preserved Author Section Headings  │
└───────────────┬───────────────┘     └───────────────────┬───────────────────┘
                │                                         │
                └──────────────────────┬──────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CITATION FUNCTION CLASSIFICATION                         │
│   [Background, Uses, Comparison, Extends, Motivation, Future Work, ...]     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Experimental Setup: Dual Configurations

To guarantee unbiased scientific comparison, all evaluation instances enforce **strict 1:1 parity**: every instance in the evaluation set has both complete abstract metadata (Config A) and verified full-text discursive chunks (Config B).

| Dimension | Configuration A (Global Baseline) | Configuration B (Local Evidence) |
|---|---|---|
| **Citing Context** | $m=3$ sentences window with masked markers | $m=3$ sentences window with masked markers |
| **Target Evidence** | Title + Full Abstract | Top-3 dense discursive chunks ($\le 110$ words each) |
| **Encoder / Model** | SciBERT / Transformer | SciBERT dense retriever + Section-aware chunking |
| **Token Budget** | Standard 512 token budget | Capped at 512 tokens (zero forced truncation) |
| **Split Strategy** | Group-Stratified Split (70/15/15) by citing paper ID to eliminate data leakage |

### Canonical Taxonomy (9 Scientific Classes)
Derived from the empirical meta-synthesis of [Lyu et al. (2021)](https://link.springer.com/article/10.1007/s11192-021-03908-z):
`Background`, `Uses`, `Comparison`, `Extends`, `Motivation`, `Future Work`, `Evidence`, `Originator`, `Further Reading`.

---

## 3. Repository Structure

```text
.
├── configs/                     # Centralized YAML hyperparameter files
│   ├── data_pipeline.yaml       # Master data orchestrator configuration
│   ├── builder.yaml             # Dataset assembly, quotas and split ratios
│   └── retrieval.yaml           # SciBERT retrieval parameters and compute device
├── data/                        # 4-layer decoupled data architecture (.gitkeep)
│   ├── raw/                     # Ingestion sources (unarXive, MultiCite, s2_cache.db)
│   ├── structured/              # Canonical CS-filtered corpus (papers.jsonl)
│   ├── processed/               # Mined contexts and rhetorical chunks
│   └── dataset/                 # 1:1 parity instances and train/dev/test splits
├── models/                      # Checkpoints and baseline experiment weights
├── notebooks/                   # Exploratory analysis and error inspections
├── src/                         # Production Python source code
│   ├── data/                    # End-to-end data engineering pipeline
│   │   ├── ingestion/           # Adapters for unarXive, MultiCite, and Semantic Scholar
│   │   ├── structuring.py       # Paper structuring and CS discipline filtering
│   │   ├── seminal_resolver.py  # 146-paper seminal catalog and S2 caching
│   │   ├── mining.py            # Window extraction and citation resolution
│   │   ├── chunking.py          # Discursive sentence chunking (≤110 words)
│   │   ├── retrieval.py         # SciBERT dense retriever and Top-k ranker
│   │   ├── builder.py           # 1:1 assembler, label mapping, and anti-leakage split
│   │   ├── pipeline.py          # Master orchestrator with automated checkpointing
│   │   └── schemas.py           # Strict Pydantic contracts across all data layers
│   ├── modeling/                # Baseline classifiers (Config A vs Config B)
│   └── visualization/           # Evaluation metrics, confusion matrices and attention maps
└── tests/                       # Complete pytest unit and integration test suite
    └── unit/                    # 73 unit tests validating pipeline components
```

---

## 4. Quickstart

### Prerequisites
- Python 3.14+
- [uv](https://github.com/astral-sh/uv) (fast Python package manager)

### Installation
```bash
# Clone the repository
git clone https://github.com/leandrojimenez/EvidenCite.git
cd EvidenCite

# Install dependencies and sync virtual environment
uv sync
```

---

## 5. Reproducing the Data Pipeline

The entire data lifecycle—from raw sources to final 1:1 parity dataset partitions—is unified through `src.data.pipeline`:

```bash
# 1. Dry-run inspection (evaluates existing artifacts without compute)
uv run python -m src.data.pipeline --dry-run

# 2. Incremental execution (automatically skips completed stages)
uv run python -m src.data.pipeline

# 3. Force full end-to-end reconstruction from scratch
uv run python -m src.data.pipeline --all --force

# 4. Execute a single stage (e.g., chunking or builder)
uv run python -m src.data.pipeline --stage chunking
```

For detailed layer descriptions and data contracts, consult the [Data Package Guide (`src/data/README.md`)](src/data/README.md).

---

## 6. Testing

Run the comprehensive unit test suite:

```bash
uv run pytest tests/unit/
```

All 73 unit tests validate:
- Paper structuring and CS classification.
- S2 client caching and rate-limiting.
- Citation window extraction and marker masking.
- WordPiece chunk length limits ($\le 110$ words) and heading preservation.
- SciBERT dense retrieval and top-k ranking.
- 1:1 parity and zero-leakage author grouping in train/dev/test splits.
- Data pipeline DAG orchestration, checkpointing, and error handling.

---

## 7. Authors & Academic Context

### Research Team
| Author | Institutional Email | Affiliation |
|---|---|---|
| **Daniel Ricardo Gallo Acevedo** | `dr.gallo@uniandes.edu.co` | Universidad de los Andes |
| **Dubán Leandro Jiménez Puerta** | `dl.jimenez@uniandes.edu.co` | Universidad de los Andes |
| **Julián Andrés Malaver Montoya** | `ja.malaverm1@uniandes.edu.co` | Universidad de los Andes |
| **Javier Mauricio Torres Ortega** | `jm.torreso1@uniandes.edu.co` | Universidad de los Andes |

* **Academic Program:** Maestría en Inteligencia Artificial (MAIA)
* **Academic Unit:** Departamento de Ingeniería de Sistemas y Computación (DISC), Universidad de los Andes (Bogotá, Colombia)
* **Research Group:** Grupo FLAG – TICsW
* **Advisor / Domain Expert:** PhD. Juan Camilo Sanguino

---