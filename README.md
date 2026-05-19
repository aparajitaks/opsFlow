# Industrial Operations Flow (opsFlow): ML Pipelines & RAG Assistants

Welcome to **opsFlow**, a unified enterprise intelligence codebase for industrial equipment prediction, safety automation, and technical operations support. This repository contains the implementations of two main predictive and cognitive maintenance systems:

1. **Task 3 — Equipment Failure Prediction Pipeline (ML):** An end-to-end Machine Learning pipeline utilizing the **AI4I 2020 Predictive Maintenance Dataset** to predict mechanical and structural breakdowns.
2. **Task 4 — Retrieval-Based AI Assistant (RAG):** A custom Retrieval-Augmented Generation system that ingests plant safety, operation manuals, and FAQs, executing semantic similarity matches to provide grounded answers to plant technicians.

---

## 1. Unified Project Folder Structure

```text
opsFlow/
├── task3/                                     # TASK 3: EQUIPMENT FAILURE ML PIPELINE
│   ├── v1/                                    # v1: Basic sequential ML stages
│   │   ├── stage2_preprocessing.py            # loads, drops, scales, and stratifies data
│   │   ├── stage4_model_training.py           # trains basic Logistic Regression & Random Forest
│   │   ├── stage5_model_evaluation.py         # calculates accuracy, ROC-AUC, and matrices
│   │   ├── stage6_overfitting_analysis.py     # compares train vs test ROC-AUC gap
│   │   └── explore_predictive_maintenance.py  # Stage 1: Exploratory Data Analysis (EDA)
│   ├── v2/                                    # v2: Class imbalance & feature engineering
│   │   ├── src/
│   │   │   ├── config.py                      # centralized hyperparameter directories
│   │   │   ├── preprocess.py                  # SMOTE class balancing + engineered columns
│   │   │   ├── train.py                       # class_weight='balanced' model fitting
│   │   │   └── evaluate.py                    # multi-metric reporting & ROC-AUC curves
│   │   ├── outputs/                           # preprocessed CSVs & performance plots
│   │   └── main.py                            # modular v2 runner orchestrator
│   ├── v3/                                    # v3: Production-grade ML & explainability
│   │   ├── src/
│   │   │   ├── config.py                      # hyperparameter configurations
│   │   │   ├── preprocess.py                  # advanced SMOTE scaling splits
│   │   │   ├── train.py                       # GridSearchCV tuning engine
│   │   │   ├── evaluate.py                    # production reporting helpers
│   │   │   └── explainability.py              # SHAP Beeswarm & Force explanations
│   │   ├── outputs/                           # MLflow sqlite databases, SHAP plots, and models
│   │   │   ├── models/                        # serialized joblib model artifacts
│   │   │   ├── plots/                         # SHAP beeswarm & force HTML visuals
│   │   │   └── mlflow/                        # local MLflow experiment metrics db
│   │   ├── Dockerfile                         # slim containerization config
│   │   ├── requirements.txt                   # v3-specific task dependencies
│   │   └── main.py                            # production execution orchestrator
│   └── data/                                  # Shared local datasets directory
│       └── ai4i2020.csv                       # AI4I 2020 Predictive Maintenance Dataset
├── task4/                                     # TASK 4: RETRIEVAL-BASED AI ASSISTANT (RAG)
│   ├── v1/                                    # v1: FAISS & Anthropic RAG Pipeline
│   │   ├── docs/                              # Ingested plain-text industrial manuals
│   │   │   ├── maintenance_guide.txt          # lubrication & schedules
│   │   │   ├── equipment_manual.txt           # motor speed & thermal specifications
│   │   │   ├── troubleshooting_faq.txt        # failure symptoms & codes
│   │   │   ├── safety_procedures.txt          # LOTO & PPE isolation procedures
│   │   │   └── preventive_maintenance.txt     # accelerometers & sensor limits
│   │   ├── outputs/                           # audit trails directory
│   │   │   └── retrieved_chunks.log           # append-only search and context logs
│   │   ├── rag_pipeline.py                    # fixed chunking + FAISS + Claude-3.5-Sonnet pipeline
│   │   └── requirements.txt                   # v1 RAG dependencies
│   └── v2/                                    # v2: Persistent ChromaDB & Groq RAG Pipeline
│       ├── docs/                              # Local copy of the 5 text manuals
│       ├── src/
│       │   ├── chunker.py                     # sliding-window overlap tokenization (300/50 split)
│       │   ├── embedder.py                    # sentence-transformers embedding helper
│       │   ├── vector_store.py                # serverless persistent SQLite-backed ChromaDB
│       │   ├── retriever.py                   # top-k retrieval calculations
│       │   ├── generator.py                   # deterministic grounding prompt & Groq caller
│       │   └── logger.py                      # rich terminal logging & query audits
│       ├── outputs/
│       │   ├── chroma_db/                     # persistent ChromaDB sqlite databases
│       │   └── retrieved_chunks.log           # audit trails of searches, indices, and answers
│       ├── main.py                            # RAG v2 orchestrator & interactive CLI
│       └── requirements.txt                   # v2 RAG dependencies
├── README.md                                  # Unified General Project Documentation (This File)
├── requirements.txt                           # General project dependencies
└── venv/                                      # Shared virtual sandboxed environment
```

---

## 2. Central Local Setup and Environment Guide

Follow these steps to initialize a clean sandboxed Python virtual environment on your local machine:

### Step 1: Create a Virtual Environment
Navigate to the repository root and initialize the environment:
```bash
# Navigate to project root
cd /Users/galaxy_grid/Desktop/opsFlow

# Create virtual environment
python3 -m venv venv
```

### Step 2: Activate the Virtual Environment
Activate the environment before executing any commands or scripts:
* **macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```
* **Windows:**
  ```cmd
  venv\Scripts\activate
  ```

### Step 3: Install Core Dependencies
Install the required packages:
```bash
pip install -r requirements.txt
```

---

## 3. Task 3 — Equipment Failure ML Pipeline Guide

The Predictive Maintenance ML Pipeline progresses from basic scripts (v1) to a balanced model (v2), and finally to a production-ready hyperparameter-tuned architecture (v3).

### Running basic v1 Stages sequentially:
```bash
# EDA (Exploratory Data Analysis)
python task3/v1/explore_predictive_maintenance.py

# Preprocessing
python task3/v1/stage2_preprocessing.py

# Model Training
python task3/v1/stage4_model_training.py

# Model Evaluation
python task3/v1/stage5_model_evaluation.py

# Overfitting Analysis
python task3/v1/stage6_overfitting_analysis.py
```

### Running modular v2 Pipeline:
The v2 implementation introduces **SMOTE class balancing** and **rich feature engineering** (`temp_diff`, `power`, and `wear_torque_ratio`) to handle severe class imbalance:
```bash
# Run modular v2 pipeline
python task3/v2/main.py
```

### Running Production v3 Pipeline (GridSearchCV, MLflow, SHAP, Docker):
The **v3** subdirectory utilizes `GridSearchCV` for hyperparameter optimization, logs parameters locally in **MLflow**, generates **SHAP** feature explanations, and exports serialized joblib models.

#### 1. Running v3 Locally:
```bash
# Navigate to task3/v3/
cd task3/v3

# Execute orchestrator
python main.py

# Launch MLflow UI to analyze runs and compare parameters
mlflow ui
```
Open `http://localhost:5000` to view metrics and parameter runs in the dashboard.

#### 2. Running with Docker:
```bash
# From within task3/v3/ directory, build the image
docker build -t task3-v3 .

# Run container mounting local data folder (avoids bloating container)
docker run -v $(pwd)/../data:/app/data task3-v3
```

---

## 4. Task 4 — Retrieval-Based AI Assistant (RAG) Guide

The RAG Assistant searches plant safety procedures, equipment limits, and FAQs to answer user queries with strictly grounded context.

### RAG v1 — FAISS & Anthropic SDK
Uses fixed-size chunking (300 words, no overlap), `sentence-transformers/all-MiniLM-L6-v2` dense embeddings, `faiss.IndexFlatIP` search index, and the **Anthropic Claude API** (`claude-3-5-sonnet-latest`).

#### Running RAG v1:
```bash
# Navigate to v1 folder
cd task4/v1

# Export your Anthropic API Key
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Run RAG v1 pipeline
python rag_pipeline.py
```

---

### RAG v2 — Persistent ChromaDB & Groq API
Upgrades the assistant with sliding-window overlap chunking, persistent SQLite vector storage, the high-speed **Groq API** (`llama-3.1-8b-instant`), strict grounding, and an interactive CLI loop.

#### Core Architectural Features:
* **Overlap Chunking (`src/chunker.py`):** Uses `chunk_size = 300` and `overlap = 50` words to duplicate boundaries, preserving context that would otherwise be split in half (e.g., LOTO procedures split at transition words).
* **Persistent ChromaDB Store (`src/vector_store.py`):** Replaces FAISS with `chromadb.PersistentClient` stored in `outputs/chroma_db/`. Computes embeddings on the first run and reloads them instantly from disk on subsequent runs, skipping re-embedding.
* **Strict Grounded completions (`src/generator.py`):** Utilizes Groq completions at `temperature = 0.0` for deterministic outputs. If the answer is missing from the retrieved context, it returns: *"I don't have enough information in my knowledge base to answer this question."*
* **Automatic Groq Decommissioning Fallback:** First attempts `llama3-8b-8192` as requested. If the API returns a model decommissioned error (Error 400), it instantly falls back to the direct successor `llama-3.1-8b-instant` to guarantee execution success.
* **Richer Audit Trail Logging (`src/logger.py`):** Saves exact timestamps, query strings, document matches, similarity scores, word scopes, and final answers in `outputs/retrieved_chunks.log`.

#### Running RAG v2:
```bash
# Navigate to v2 folder
cd task4/v2

# Export your Groq API Key
export GROQ_API_KEY="your-groq-api-key"

# Run RAG v2 orchestrator and enter interactive CLI
python main.py
```
*(On startup, 4 baseline queries will automatically run and verify the pipeline, followed by an interactive `input()` loop until you type `quit`.)*
